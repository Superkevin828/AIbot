from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import ConfigError, load_config
from .llm_client import LLMError
from .pdf_processor import PdfProcessingError
from .workflow import Workflow, WorkflowError, WorkflowOptions, WorkflowResult

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-workflow",
        description=(
            "Extract PDF content, summarize it with an LLM, notify Slack, "
            "and schedule a Google Calendar follow-up."
        ),
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        type=Path,
        help="One or more PDF files to process as a single document bundle.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to a .env file (defaults to ./.env when present).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run PDF extraction and LLM analysis, print the Slack preview and "
            "proposed meeting, but do not send or create anything."
        ),
    )
    parser.add_argument(
        "--skip-slack",
        action="store_true",
        help="Do not send a Slack notification.",
    )
    parser.add_argument(
        "--skip-calendar",
        action="store_true",
        help="Do not create a Google Calendar event.",
    )
    parser.add_argument(
        "--force-followup",
        action="store_true",
        help="Schedule a follow-up even when the LLM says it is unnecessary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for JSON and Markdown reports (overrides OUTPUT_DIR).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write local JSON/Markdown reports.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the workflow result as JSON instead of a human summary.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override LOG_LEVEL.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        config = load_config(
            args.env_file,
            slack_enabled=False if args.skip_slack else None,
            calendar_enabled=False if args.skip_calendar else None,
            validate_integrations=not args.dry_run,
        )
        if args.output_dir is not None:
            config = replace(config, output_dir=args.output_dir.expanduser())
        _set_log_level(args.log_level or config.log_level)

        workflow = Workflow(config)
        result = workflow.run(
            [path.expanduser() for path in args.pdfs],
            WorkflowOptions(
                dry_run=args.dry_run,
                force_followup=args.force_followup,
                skip_report=args.no_report,
            ),
        )
    except WorkflowError as exc:
        if exc.result is not None:
            _print_result(exc.result, as_json=args.json, dry_run=args.dry_run)
        _print_error(str(exc))
        return 1
    except (ConfigError, PdfProcessingError, LLMError) as exc:
        _print_error(str(exc))
        return 1
    except KeyboardInterrupt:
        _print_error("Interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected workflow failure")
        _print_error(f"Unexpected error: {exc}")
        return 1

    _print_result(result, as_json=args.json, dry_run=args.dry_run)
    return 0


def _print_result(
    result: WorkflowResult,
    *,
    as_json: bool,
    dry_run: bool,
) -> None:
    if as_json:
        payload = {
            "documents": [
                {
                    "source_name": document.source_name,
                    "source_path": document.source_path,
                    "page_count": document.page_count,
                    "word_count": document.word_count,
                    "table_count": document.table_count,
                    "headings": document.headings,
                    "warnings": document.warnings,
                    "sha256": document.sha256,
                }
                for document in result.documents
            ],
            "analysis": result.analysis.model_dump(mode="json"),
            "meeting_plan": (
                asdict(result.meeting_plan) if result.meeting_plan else None
            ),
            "scheduled_meeting": (
                asdict(result.scheduled_meeting)
                if result.scheduled_meeting
                else None
            ),
            "slack": (
                {
                    "mode": result.slack_delivery.mode,
                    "channel": result.slack_delivery.channel,
                    "message_ts": result.slack_delivery.message_ts,
                    "preview": result.slack_delivery.preview,
                }
                if result.slack_delivery
                else None
            ),
            "report_paths": (
                [str(path) for path in result.report_paths]
                if result.report_paths
                else None
            ),
            "errors": result.errors or [],
        }
        print(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False))
        return

    analysis = result.analysis
    slack_failed = _integration_failed(result.errors or [], "slack")
    calendar_failed = _integration_failed(result.errors or [], "calendar")
    print(f"Summary: {analysis.title}")
    print("Sources: " + ", ".join(doc.source_name for doc in result.documents))

    if dry_run:
        print("Mode: DRY RUN - no Slack message or calendar event was created")
        if result.slack_delivery and result.slack_delivery.preview:
            print("\n--- Slack preview ---")
            print(result.slack_delivery.preview)
            print("--- end preview ---")
        elif result.slack_delivery is None:
            print("Slack: failed" if slack_failed else "Slack: skipped")
    elif result.slack_delivery:
        target = result.slack_delivery.channel or "configured target"
        print(f"Slack: delivered via {result.slack_delivery.mode} to {target}")
    elif result.slack_delivery is None:
        print("Slack: failed" if slack_failed else "Slack: skipped")

    if result.scheduled_meeting:
        meeting = result.scheduled_meeting
        print(
            "Calendar: scheduled "
            f"{meeting.start.strftime('%Y-%m-%d %H:%M')} {meeting.timezone} "
            f"({meeting.duration_minutes} min)"
        )
        if meeting.html_link:
            print(f"Calendar link: {meeting.html_link}")
    elif result.meeting_plan:
        plan = result.meeting_plan
        suffix = " (dry run)" if dry_run else " (not created)"
        print(
            "Calendar: proposed "
            f"{plan.start.strftime('%Y-%m-%d %H:%M')} {plan.timezone} "
            f"({plan.duration_minutes} min){suffix}"
        )
    elif calendar_failed:
        print("Calendar: scheduling failed")
    else:
        print("Calendar: no follow-up meeting")

    if result.report_paths:
        print(f"Report JSON: {result.report_paths[0]}")
        print(f"Report Markdown: {result.report_paths[1]}")

    if result.errors:
        for error in result.errors:
            _print_error(error)


def _integration_failed(errors: list[str], name: str) -> bool:
    prefixes = (
        ("calendar", "google", "could not initialize google")
        if name == "calendar"
        else ("slack",)
    )
    return any(error.lower().startswith(prefixes) for error in errors)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _configure_logging(level: str | None) -> None:
    logging.basicConfig(
        level=getattr(logging, (level or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _set_log_level(level: str) -> None:
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def _print_error(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
