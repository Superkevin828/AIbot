from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calendar_client import MeetingPlan, ScheduledMeeting
from .models import Analysis, PdfDocument
from .slack_client import SlackDelivery


class ReportError(RuntimeError):
    """Raised when local report files cannot be written."""


def write_reports(
    output_dir: Path,
    *,
    analysis: Analysis,
    documents: list[PdfDocument],
    meeting_plan: MeetingPlan | None,
    scheduled_meeting: ScheduledMeeting | None,
    slack_delivery: SlackDelivery | None,
    errors: list[str],
    dry_run: bool,
) -> tuple[Path, Path]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now(timezone.utc)
        stem = _safe_filename(
            f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}_{analysis.title}"
        )
        json_path = output_dir / f"{stem}.json"
        markdown_path = output_dir / f"{stem}.md"

        payload = {
            "generated_at": generated_at.isoformat(),
            "dry_run": dry_run,
            "documents": [_document_dict(document) for document in documents],
            "analysis": analysis.model_dump(mode="json"),
            "meeting_plan": _meeting_plan_dict(meeting_plan),
            "scheduled_meeting": _scheduled_meeting_dict(scheduled_meeting),
            "slack": _slack_dict(slack_delivery),
            "errors": errors,
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        markdown_path.write_text(
            _render_markdown(
                analysis=analysis,
                documents=documents,
                meeting_plan=meeting_plan,
                scheduled_meeting=scheduled_meeting,
                slack_delivery=slack_delivery,
                errors=errors,
                dry_run=dry_run,
                generated_at=generated_at,
            ),
            encoding="utf-8",
        )
        return json_path, markdown_path
    except Exception as exc:  # noqa: BLE001
        raise ReportError(f"Could not write local report: {exc}") from exc


def _document_dict(document: PdfDocument) -> dict[str, Any]:
    return {
        "source_name": document.source_name,
        "source_path": document.source_path,
        "page_count": document.page_count,
        "word_count": document.word_count,
        "char_count": document.char_count,
        "table_count": document.table_count,
        "headings": document.headings,
        "metadata": document.metadata,
        "warnings": document.warnings,
        "sha256": document.sha256,
    }


def _meeting_plan_dict(plan: MeetingPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "summary": plan.summary,
        "description": plan.description,
        "start": plan.start.isoformat(),
        "end": plan.end.isoformat(),
        "timezone": plan.timezone,
        "attendees": list(plan.attendees),
        "duration_minutes": plan.duration_minutes,
        "source_id": plan.source_id,
    }


def _scheduled_meeting_dict(
    meeting: ScheduledMeeting | None,
) -> dict[str, Any] | None:
    if meeting is None:
        return None
    return {
        "event_id": meeting.event_id,
        "html_link": meeting.html_link,
        "start": meeting.start.isoformat(),
        "end": meeting.end.isoformat(),
        "timezone": meeting.timezone,
        "duration_minutes": meeting.duration_minutes,
        "already_existed": meeting.already_existed,
    }


def _slack_dict(delivery: SlackDelivery | None) -> dict[str, Any] | None:
    if delivery is None:
        return None
    return {
        "mode": delivery.mode,
        "channel": delivery.channel,
        "message_ts": delivery.message_ts,
        "preview": delivery.preview,
    }


def _render_markdown(
    *,
    analysis: Analysis,
    documents: list[PdfDocument],
    meeting_plan: MeetingPlan | None,
    scheduled_meeting: ScheduledMeeting | None,
    slack_delivery: SlackDelivery | None,
    errors: list[str],
    dry_run: bool,
    generated_at: datetime,
) -> str:
    lines = [f"# {analysis.title}", ""]
    lines.append(f"Generated: {generated_at.isoformat()}")
    lines.append(
        "Sources: " + ", ".join(document.source_name for document in documents)
    )
    if dry_run:
        lines.append("")
        lines.append("> Dry run: no Slack message was sent and no calendar event was created.")
    lines.extend(["", "## Executive summary", "", analysis.executive_summary])

    lines.extend(["", "## Core insights"])
    if analysis.core_insights:
        lines.extend(f"- {item}" for item in analysis.core_insights)
    else:
        lines.append("- None identified.")
    lines.extend(["", "## Key takeaways"])
    if analysis.key_takeaways:
        lines.extend(f"- {item}" for item in analysis.key_takeaways)
    else:
        lines.append("- None identified.")

    lines.extend(["", "## Action items"])
    if analysis.action_items:
        for item in analysis.action_items:
            bits = [item.task]
            if item.owner:
                bits.append(f"owner: {item.owner}")
            if item.due_date:
                bits.append(f"due: {item.due_date}")
            if item.priority:
                bits.append(f"priority: {item.priority}")
            lines.append(f"- [ ] {' | '.join(bits)}")
    else:
        lines.append("- None identified.")

    if analysis.important_dates:
        lines.extend(["", "## Important dates"])
        for item in analysis.important_dates:
            bits = [item.label]
            if item.date:
                bits.append(item.date)
            if item.time:
                bits.append(item.time)
            if item.context:
                bits.append(item.context)
            lines.append(f"- {' | '.join(bits)}")

    lines.extend(["", "## Follow-up"])
    if scheduled_meeting:
        lines.append(
            f"- Scheduled: {scheduled_meeting.start.strftime('%Y-%m-%d %H:%M')} "
            f"{scheduled_meeting.timezone} ({scheduled_meeting.duration_minutes} min)"
        )
        if scheduled_meeting.html_link:
            lines.append(f"- Calendar: {scheduled_meeting.html_link}")
        if scheduled_meeting.already_existed:
            lines.append("- An equivalent event already existed and was reused.")
    elif meeting_plan:
        lines.append(
            f"- Proposed: {meeting_plan.start.strftime('%Y-%m-%d %H:%M')} "
            f"{meeting_plan.timezone} ({meeting_plan.duration_minutes} min)"
        )
    else:
        lines.append("- Not scheduled.")
    if analysis.follow_up.reason:
        lines.append(f"- Reason: {analysis.follow_up.reason}")
    if analysis.follow_up.agenda:
        lines.append("- Agenda:")
        lines.extend(f"  - {item}" for item in analysis.follow_up.agenda)

    if slack_delivery:
        lines.extend(["", "## Slack delivery"])
        lines.append(f"- Mode: {slack_delivery.mode}")
        if slack_delivery.channel:
            lines.append(f"- Channel: {slack_delivery.channel}")
        if slack_delivery.message_ts:
            lines.append(f"- Message timestamp: {slack_delivery.message_ts}")

    warnings = [
        warning for document in documents for warning in document.warnings
    ]
    if warnings:
        lines.extend(["", "## PDF warnings"])
        lines.extend(f"- {warning}" for warning in warnings)

    if errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in errors)

    return "\n".join(lines).rstrip() + "\n"


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (cleaned or "pdf-summary")[:120]
