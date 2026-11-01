from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .config import SlackConfig
from .models import Analysis, PdfDocument

if TYPE_CHECKING:
    from .calendar_client import MeetingPlan, ScheduledMeeting

logger = logging.getLogger(__name__)

_MAX_BLOCK_CHARS = 2900
_MAX_HEADER_CHARS = 150
_MAX_BLOCKS = 50


class SlackNotificationError(RuntimeError):
    """Raised when a Slack notification cannot be delivered."""


@dataclass(frozen=True)
class SlackDelivery:
    mode: str
    channel: str | None = None
    message_ts: str | None = None
    preview: str | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)


class SlackNotifier:
    def __init__(self, config: SlackConfig) -> None:
        self.config = config

    def notify(
        self,
        analysis: Analysis,
        documents: list[PdfDocument],
        *,
        meeting_plan: MeetingPlan | None = None,
        scheduled_meeting: ScheduledMeeting | None = None,
        dry_run: bool = False,
    ) -> SlackDelivery:
        fallback, blocks = build_slack_message(
            analysis,
            documents,
            meeting_plan=meeting_plan,
            scheduled_meeting=scheduled_meeting,
        )

        if dry_run:
            logger.info("Dry run: Slack message was not sent")
            return SlackDelivery(
                mode="dry_run",
                channel=self._target_channel(),
                preview=fallback,
                blocks=blocks,
            )

        try:
            if self.config.mode == "webhook":
                return self._send_webhook(fallback, blocks)
            return self._send_bot(fallback, blocks)
        except SlackNotificationError:
            raise
        except Exception as exc:  # noqa: BLE001
            detail = _slack_error_detail(exc)
            raise SlackNotificationError(f"Slack delivery failed: {detail}") from exc

    def _send_webhook(
        self, fallback: str, blocks: list[dict[str, Any]]
    ) -> SlackDelivery:
        if not self.config.webhook_url:
            raise SlackNotificationError("SLACK_WEBHOOK_URL is not configured")

        try:
            from slack_sdk.webhook import WebhookClient
        except ImportError as exc:
            raise SlackNotificationError(
                "slack-sdk is not installed. Run: pip install -r requirements.txt"
            ) from exc

        client = WebhookClient(
            self.config.webhook_url,
            timeout=self.config.timeout_seconds,
        )
        response = client.send(text=fallback, blocks=blocks)
        if response.status_code != 200:
            raise SlackNotificationError(
                f"Slack webhook returned HTTP {response.status_code}: "
                f"{(response.body or '')[:300]}"
            )
        logger.info("Slack message delivered through incoming webhook")
        return SlackDelivery(mode="webhook", channel=self._target_channel())

    def _send_bot(self, fallback: str, blocks: list[dict[str, Any]]) -> SlackDelivery:
        if not self.config.bot_token or not self.config.channel:
            raise SlackNotificationError(
                "SLACK_BOT_TOKEN and SLACK_CHANNEL_ID are required for bot mode"
            )

        try:
            from slack_sdk import WebClient
        except ImportError as exc:
            raise SlackNotificationError(
                "slack-sdk is not installed. Run: pip install -r requirements.txt"
            ) from exc

        client = WebClient(
            token=self.config.bot_token,
            timeout=self.config.timeout_seconds,
        )
        response = client.chat_postMessage(
            channel=self.config.channel,
            text=fallback,
            blocks=blocks,
            unfurl_links=False,
            unfurl_media=False,
        )
        if not response.get("ok"):
            raise SlackNotificationError(
                f"Slack API rejected the message: {response.get('error', 'unknown')}"
            )
        channel = response.get("channel") or self.config.channel
        message_ts = response.get("ts")
        logger.info("Slack message delivered to %s (ts=%s)", channel, message_ts)
        return SlackDelivery(
            mode="bot",
            channel=channel,
            message_ts=message_ts,
        )

    def _target_channel(self) -> str | None:
        if self.config.mode == "webhook":
            return "incoming webhook"
        return self.config.channel


def build_slack_message(
    analysis: Analysis,
    documents: list[PdfDocument],
    *,
    meeting_plan: MeetingPlan | None = None,
    scheduled_meeting: ScheduledMeeting | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": _truncate(analysis.title, _MAX_HEADER_CHARS),
                "emoji": True,
            },
        }
    )

    source_names = ", ".join(document.source_name for document in documents)
    total_pages = sum(document.page_count for document in documents)
    context = (
        f"*Sources:* {_escape(source_names)} | "
        f"*Pages:* {total_pages} | "
        f"*Generated:* {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    blocks.append(
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": _truncate(context, _MAX_BLOCK_CHARS)}
            ],
        }
    )

    blocks.extend(_section_blocks("Executive summary", [analysis.executive_summary]))
    blocks.extend(
        _section_blocks(
            "Core insights",
            _bullets(_limit(analysis.core_insights, 10, "insights")),
        )
    )
    blocks.extend(
        _section_blocks(
            "Key takeaways",
            _bullets(_limit(analysis.key_takeaways, 8, "takeaways")),
        )
    )

    action_lines: list[str] = []
    for item in analysis.action_items[:15]:
        details = [item.task]
        if item.owner:
            details.append(f"Owner: {item.owner}")
        if item.due_date:
            details.append(f"Due: {item.due_date}")
        if item.priority:
            details.append(f"Priority: {item.priority}")
        action_lines.append(" - ".join(details))
    if len(analysis.action_items) > 15:
        action_lines.append(
            f"…and {len(analysis.action_items) - 15} more."
        )
    blocks.extend(
        _section_blocks(
            "Action items",
            _bullets(action_lines) if action_lines else ["_No action items identified._"],
        )
    )

    date_lines = []
    for item in analysis.important_dates[:10]:
        bits = [item.label]
        if item.date:
            bits.append(item.date)
        if item.time:
            bits.append(item.time)
        if item.context:
            bits.append(item.context)
        date_lines.append(" - ".join(bits))
    if len(analysis.important_dates) > 10:
        date_lines.append(
            f"…and {len(analysis.important_dates) - 10} more."
        )
    if date_lines:
        blocks.extend(_section_blocks("Important dates", _bullets(date_lines)))

    follow_up_lines = _follow_up_lines(analysis, meeting_plan, scheduled_meeting)
    blocks.extend(_section_blocks("Follow-up meeting", follow_up_lines))

    if len(blocks) > _MAX_BLOCKS:
        logger.warning(
            "Slack message has %d blocks; truncating to %d",
            len(blocks),
            _MAX_BLOCKS,
        )
        blocks = blocks[:_MAX_BLOCKS]

    fallback = _plain_message(analysis, documents, meeting_plan, scheduled_meeting)
    return _truncate(fallback, 3000), blocks


def _follow_up_lines(
    analysis: Analysis,
    meeting_plan: MeetingPlan | None,
    scheduled_meeting: ScheduledMeeting | None,
) -> list[str]:
    lines: list[str] = []
    follow_up = analysis.follow_up

    if scheduled_meeting is not None:
        start = scheduled_meeting.start.strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"Scheduled: {start} {scheduled_meeting.timezone} "
            f"({scheduled_meeting.duration_minutes} min)"
        )
        if scheduled_meeting.html_link:
            lines.append(
                f"<{scheduled_meeting.html_link}|Open Google Calendar event>"
            )
        if scheduled_meeting.already_existed:
            lines.append("_An equivalent event already existed; reused it._")
    elif meeting_plan is not None:
        start = meeting_plan.start.strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"Proposed: {start} {meeting_plan.timezone} "
            f"({meeting_plan.duration_minutes} min)"
        )
        if follow_up.required:
            lines.append("_The Google Calendar event was not created._")
    else:
        lines.append("_No follow-up meeting was scheduled._")
        if follow_up.reason:
            lines.append(f"Reason: {_escape(follow_up.reason)}")

    if follow_up.agenda:
        lines.append("*Agenda:*")
        lines.extend(
            f"• {_escape(item)}" for item in _limit(follow_up.agenda, 8, "agenda items")
        )
    if follow_up.attendees:
        lines.append(
            "*People mentioned:* " + _escape(", ".join(follow_up.attendees))
        )
    if follow_up.reason and (meeting_plan or scheduled_meeting):
        lines.append(f"*Why:* {_escape(follow_up.reason)}")
    return lines


def _section_blocks(heading: str, lines: list[str]) -> list[dict[str, Any]]:
    prefix = f"*{heading}*\n"
    if not lines:
        lines = ["_None._"]

    chunks: list[str] = []
    current = prefix
    for raw_line in lines:
        line = _truncate(raw_line, _MAX_BLOCK_CHARS - len(prefix) - 1)
        candidate = current + line + "\n"
        if len(candidate) > _MAX_BLOCK_CHARS and current != prefix:
            chunks.append(current.rstrip())
            current = f"*{heading} (continued)*\n{line}\n"
        else:
            current = candidate
    chunks.append(current.rstrip())

    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": chunk}}
        for chunk in chunks
    ]


def _plain_message(
    analysis: Analysis,
    documents: list[PdfDocument],
    meeting_plan: MeetingPlan | None,
    scheduled_meeting: ScheduledMeeting | None,
) -> str:
    parts = [
        analysis.title,
        "",
        analysis.executive_summary,
        "",
        "Core insights:",
        *_bullets(analysis.core_insights),
        "",
        "Key takeaways:",
        *_bullets(analysis.key_takeaways),
    ]
    if analysis.action_items:
        parts.extend(["", "Action items:"])
        for item in analysis.action_items:
            details = [item.task]
            if item.owner:
                details.append(f"owner: {item.owner}")
            if item.due_date:
                details.append(f"due: {item.due_date}")
            parts.append("• " + " - ".join(details))
    if analysis.follow_up.agenda:
        parts.extend(["", "Follow-up agenda:"])
        parts.extend(f"• {item}" for item in analysis.follow_up.agenda)

    if scheduled_meeting:
        parts.append(
            f"\nFollow-up scheduled: "
            f"{scheduled_meeting.start.strftime('%Y-%m-%d %H:%M')} "
            f"{scheduled_meeting.timezone}"
        )
    elif meeting_plan:
        parts.append(
            f"\nProposed follow-up: "
            f"{meeting_plan.start.strftime('%Y-%m-%d %H:%M')} "
            f"{meeting_plan.timezone}"
        )

    parts.append("\nSources: " + ", ".join(doc.source_name for doc in documents))
    return "\n".join(parts)


def _bullets(items: list[str]) -> list[str]:
    return [f"• {_escape(item)}" for item in items if item.strip()]


def _limit(items: list[str], maximum: int, noun: str) -> list[str]:
    if len(items) <= maximum:
        return items
    remaining = len(items) - maximum
    return items[:maximum] + [f"…and {remaining} more {noun}."]


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _slack_error_detail(error: Exception) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        slack_error = response.get("error")
        if slack_error:
            return str(slack_error)
    return " ".join(str(error).split())[:500] or type(error).__name__
