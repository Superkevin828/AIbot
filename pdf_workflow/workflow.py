from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .calendar_client import (
    CalendarScheduler,
    CalendarSchedulingError,
    MeetingPlan,
    ScheduledMeeting,
)
from .config import AppConfig
from .llm_client import LLMAnalyzer
from .models import Analysis, PdfDocument
from .pdf_processor import PdfProcessor, build_llm_content
from .reporting import ReportError, write_reports
from .slack_client import SlackDelivery, SlackNotificationError, SlackNotifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowOptions:
    dry_run: bool = False
    force_followup: bool = False
    skip_report: bool = False


@dataclass
class WorkflowResult:
    documents: list[PdfDocument]
    analysis: Analysis
    meeting_plan: MeetingPlan | None = None
    scheduled_meeting: ScheduledMeeting | None = None
    slack_delivery: SlackDelivery | None = None
    report_paths: tuple[Path, Path] | None = None
    errors: list[str] | None = None


class WorkflowError(RuntimeError):
    def __init__(self, message: str, result: WorkflowResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class Workflow:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(
        self,
        pdf_paths: list[str | Path],
        options: WorkflowOptions | None = None,
    ) -> WorkflowResult:
        options = options or WorkflowOptions()
        if not pdf_paths:
            raise WorkflowError("At least one PDF path is required")

        processor = PdfProcessor(max_pages=self.config.pdf.max_pages)
        logger.info("Processing %d PDF document(s)", len(pdf_paths))
        documents = [processor.extract(path) for path in pdf_paths]

        content = build_llm_content(documents, self.config.pdf.max_chars)
        logger.info(
            "Sending %d characters to %s (%s)",
            len(content),
            self.config.llm.provider,
            self.config.llm.model,
        )
        analyzer = LLMAnalyzer(self.config.llm)
        analysis = analyzer.analyze(
            documents,
            content,
            timezone=self.config.calendar.timezone,
        )
        analysis = self._ensure_title(analysis, documents)

        errors: list[str] = []
        scheduler = CalendarScheduler(self.config.calendar)
        try:
            meeting_plan = scheduler.plan(
                analysis,
                documents,
                force=options.force_followup,
            )
        except Exception as exc:  # noqa: BLE001
            meeting_plan = None
            errors.append(f"Calendar planning failed: {exc}")
            logger.exception("Calendar planning failed")

        scheduled_meeting: ScheduledMeeting | None = None
        if (
            self.config.calendar.enabled
            and not options.dry_run
            and meeting_plan is not None
        ):
            try:
                scheduled_meeting = scheduler.create(meeting_plan)
            except CalendarSchedulingError as exc:
                errors.append(str(exc))
                logger.error("Calendar scheduling failed: %s", exc)
        elif self.config.calendar.enabled and meeting_plan is None:
            logger.info("Calendar step skipped because no follow-up was requested")

        slack_delivery: SlackDelivery | None = None
        notifier = SlackNotifier(self.config.slack)
        if options.dry_run and self.config.slack.enabled:
            slack_delivery = notifier.notify(
                analysis,
                documents,
                meeting_plan=meeting_plan,
                scheduled_meeting=None,
                dry_run=True,
            )
        elif self.config.slack.enabled:
            try:
                slack_delivery = notifier.notify(
                    analysis,
                    documents,
                    meeting_plan=meeting_plan,
                    scheduled_meeting=scheduled_meeting,
                )
            except SlackNotificationError as exc:
                errors.append(str(exc))
                logger.error("Slack notification failed: %s", exc)

        report_paths: tuple[Path, Path] | None = None
        if not options.skip_report:
            try:
                report_paths = write_reports(
                    self.config.output_dir,
                    analysis=analysis,
                    documents=documents,
                    meeting_plan=meeting_plan,
                    scheduled_meeting=scheduled_meeting,
                    slack_delivery=slack_delivery,
                    errors=errors,
                    dry_run=options.dry_run,
                )
            except ReportError as exc:
                errors.append(str(exc))
                logger.error("Report writing failed: %s", exc)

        result = WorkflowResult(
            documents=documents,
            analysis=analysis,
            meeting_plan=meeting_plan,
            scheduled_meeting=scheduled_meeting,
            slack_delivery=slack_delivery,
            report_paths=report_paths,
            errors=errors,
        )
        if errors:
            raise WorkflowError(
                "Workflow completed with one or more errors", result=result
            )
        return result

    @staticmethod
    def _ensure_title(
        analysis: Analysis, documents: list[PdfDocument]
    ) -> Analysis:
        if analysis.title and analysis.title != "Document Summary":
            return analysis
        for document in documents:
            metadata_title = document.metadata.get("Title", "").strip()
            if metadata_title:
                return analysis.model_copy(update={"title": metadata_title})
        stem = Path(documents[0].source_name).stem.replace("_", " ").strip()
        if stem:
            return analysis.model_copy(update={"title": stem})
        return analysis
