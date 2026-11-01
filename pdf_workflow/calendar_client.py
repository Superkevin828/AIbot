from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import CalendarConfig
from .models import Analysis, FollowUp, PdfDocument

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


class CalendarSchedulingError(RuntimeError):
    """Raised when a Google Calendar event cannot be planned or created."""


@dataclass(frozen=True)
class MeetingPlan:
    summary: str
    description: str
    start: datetime
    end: datetime
    timezone: str
    attendees: tuple[str, ...]
    source_id: str
    duration_minutes: int


@dataclass(frozen=True)
class ScheduledMeeting:
    event_id: str
    html_link: str | None
    start: datetime
    end: datetime
    timezone: str
    duration_minutes: int
    already_existed: bool = False


class CalendarScheduler:
    def __init__(self, config: CalendarConfig) -> None:
        self.config = config

    def plan(
        self,
        analysis: Analysis,
        documents: list[PdfDocument],
        *,
        force: bool = False,
    ) -> MeetingPlan | None:
        follow_up = analysis.follow_up
        should_schedule = (
            self.config.always_schedule or force or follow_up.required
        )
        if not should_schedule:
            logger.info("No follow-up meeting is required; calendar step skipped")
            return None

        timezone, timezone_name, notes = self._resolve_timezone(follow_up.timezone)
        start, time_notes = self._resolve_start(follow_up, timezone)
        notes.extend(time_notes)

        duration = (
            follow_up.duration_minutes
            if follow_up.duration_minutes is not None
            else self.config.duration_minutes
        )
        duration = max(15, min(int(duration), 480))
        end = start + timedelta(minutes=duration)

        fingerprint = hashlib.sha256(
            "|".join(document.sha256 for document in documents).encode("utf-8")
        ).hexdigest()
        source_id = hashlib.sha256(
            f"{fingerprint}|{analysis.title}|{start.isoformat()}".encode("utf-8")
        ).hexdigest()

        summary = _truncate(f"Follow-up: {analysis.title}", 250)
        description = self._build_description(
            analysis, documents, timezone_name, notes
        )

        return MeetingPlan(
            summary=summary,
            description=description,
            start=start,
            end=end,
            timezone=timezone_name,
            attendees=self.config.attendees,
            source_id=source_id,
            duration_minutes=duration,
        )

    def create(self, plan: MeetingPlan) -> ScheduledMeeting:
        if not self.config.enabled:
            raise CalendarSchedulingError("Google Calendar integration is disabled")

        service = self._build_service()

        try:
            if self.config.dedupe:
                try:
                    existing = self._find_existing(service, plan.source_id)
                except Exception as exc:  # noqa: BLE001
                    existing = None
                    logger.warning(
                        "Calendar deduplication lookup failed; creating a new "
                        "event instead: %s",
                        _google_error_detail(exc),
                    )
                if existing is not None and existing.get("id"):
                    logger.info(
                        "Reusing existing calendar event %s", existing.get("id")
                    )
                    return ScheduledMeeting(
                        event_id=str(existing.get("id")),
                        html_link=existing.get("htmlLink"),
                        start=plan.start,
                        end=plan.end,
                        timezone=plan.timezone,
                        duration_minutes=plan.duration_minutes,
                        already_existed=True,
                    )

            body = {
                "summary": plan.summary,
                "description": plan.description,
                "start": {
                    "dateTime": plan.start.isoformat(),
                    "timeZone": plan.timezone,
                },
                "end": {
                    "dateTime": plan.end.isoformat(),
                    "timeZone": plan.timezone,
                },
                "reminders": {
                    "useDefault": False,
                    "overrides": [{"method": "popup", "minutes": 10}],
                },
                "extendedProperties": {
                    "private": {
                        "pdfWorkflowSource": plan.source_id,
                        "pdfWorkflowVersion": "1",
                    }
                },
            }
            if plan.attendees:
                body["attendees"] = [{"email": email} for email in plan.attendees]

            event = (
                service.events()
                .insert(
                    calendarId=self.config.calendar_id,
                    body=body,
                    sendUpdates="all" if plan.attendees else "none",
                )
                .execute()
            )
        except CalendarSchedulingError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize Google API failures
            raise CalendarSchedulingError(
                f"Google Calendar request failed: {_google_error_detail(exc)}"
            ) from exc

        event_id = str(event.get("id", ""))
        if not event_id:
            raise CalendarSchedulingError(
                "Google Calendar did not return an event ID"
            )

        logger.info(
            "Created calendar event %s at %s %s",
            event_id,
            plan.start.isoformat(),
            plan.timezone,
        )
        return ScheduledMeeting(
            event_id=event_id,
            html_link=event.get("htmlLink"),
            start=plan.start,
            end=plan.end,
            timezone=plan.timezone,
            duration_minutes=plan.duration_minutes,
        )

    def _resolve_timezone(
        self, suggested_timezone: str | None
    ) -> tuple[ZoneInfo, str, list[str]]:
        if suggested_timezone:
            try:
                return (
                    ZoneInfo(suggested_timezone),
                    suggested_timezone,
                    [],
                )
            except ZoneInfoNotFoundError:
                logger.warning(
                    "LLM suggested unknown timezone %r; using %s",
                    suggested_timezone,
                    self.config.timezone,
                )
                return (
                    ZoneInfo(self.config.timezone),
                    self.config.timezone,
                    [
                        f"Document timezone {suggested_timezone!r} was invalid; "
                        f"using {self.config.timezone}."
                    ],
                )
        return ZoneInfo(self.config.timezone), self.config.timezone, []

    def _resolve_start(
        self,
        follow_up: FollowUp,
        timezone: ZoneInfo,
    ) -> tuple[datetime, list[str]]:
        notes: list[str] = []
        now = datetime.now(timezone)
        default_clock = _parse_time(self.config.default_time, fallback=dt_time(10, 0))
        clock = _parse_time(
            follow_up.suggested_time,
            fallback=default_clock,
            notes=notes,
        )

        planned: datetime | None = None
        if follow_up.suggested_date:
            planned = _parse_datetime_or_date(
                follow_up.suggested_date,
                timezone,
                clock,
                notes,
            )

        if planned is None:
            target_date = _add_business_days(now.date(), self.config.default_days)
            planned = datetime.combine(target_date, clock, tzinfo=timezone)
            notes.append(
                "No valid follow-up date was provided; used configured fallback "
                f"timing ({self.config.default_days} business day(s), "
                f"{self.config.default_time})."
            )
        elif planned <= now + timedelta(minutes=5):
            original = planned
            target_date = _add_business_days(now.date(), 1)
            planned = datetime.combine(target_date, original.timetz(), tzinfo=timezone)
            notes.append(
                "The suggested follow-up time was in the past; moved to "
                f"{planned.date().isoformat()}."
            )

        if planned <= now:
            planned = datetime.combine(
                _add_business_days(now.date(), 1),
                planned.timetz(),
                tzinfo=timezone,
            )
            notes.append("Adjusted the follow-up to the next business day.")

        return planned, notes

    def _build_description(
        self,
        analysis: Analysis,
        documents: list[PdfDocument],
        timezone_name: str,
        notes: list[str],
    ) -> str:
        lines = [
            analysis.executive_summary,
            "",
            "Created automatically from PDF analysis.",
            "Sources: " + ", ".join(doc.source_name for doc in documents),
            f"Timezone: {timezone_name}",
        ]
        if analysis.follow_up.reason:
            lines.extend(["", f"Why this follow-up: {analysis.follow_up.reason}"])
        if analysis.follow_up.agenda:
            lines.extend(["", "Agenda:"])
            lines.extend(f"- {item}" for item in analysis.follow_up.agenda)
        if analysis.action_items:
            lines.extend(["", "Action items:"])
            for item in analysis.action_items:
                bits = [item.task]
                if item.owner:
                    bits.append(f"owner: {item.owner}")
                if item.due_date:
                    bits.append(f"due: {item.due_date}")
                lines.append("- " + " | ".join(bits))
        if analysis.follow_up.attendees:
            lines.extend(
                [
                    "",
                    "People mentioned: "
                    + ", ".join(analysis.follow_up.attendees),
                ]
            )
        if notes:
            lines.extend(["", "Scheduling notes:"])
            lines.extend(f"- {note}" for note in notes)
        return _truncate("\n".join(lines), 8000)

    def _build_service(self):
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise CalendarSchedulingError(
                "google-api-python-client is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        try:
            credentials = self._credentials()
            return build(
                "calendar",
                "v3",
                credentials=credentials,
                cache_discovery=False,
            )
        except CalendarSchedulingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CalendarSchedulingError(
                f"Could not initialize Google Calendar client: "
                f"{_google_error_detail(exc)}"
            ) from exc

    def _credentials(self):
        if self.config.auth_mode == "service_account":
            return self._service_account_credentials()
        return self._oauth_credentials()

    def _service_account_credentials(self):
        try:
            from google.oauth2 import service_account
        except ImportError as exc:
            raise CalendarSchedulingError(
                "google-auth is not installed. Run: pip install -r requirements.txt"
            ) from exc

        path = self.config.service_account_file
        if path is None or not path.is_file():
            raise CalendarSchedulingError(
                f"Google service account file not found: {path}"
            )

        credentials = service_account.Credentials.from_service_account_file(
            str(path),
            scopes=_SCOPES,
        )
        if self.config.service_account_subject:
            credentials = credentials.with_subject(
                self.config.service_account_subject
            )
        return credentials

    def _oauth_credentials(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise CalendarSchedulingError(
                "google-auth-oauthlib is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        token_path = self.config.oauth_token_file
        client_path = self.config.oauth_client_secret_file
        credentials = None

        if token_path.is_file():
            try:
                credentials = Credentials.from_authorized_user_file(
                    str(token_path), _SCOPES
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load Google OAuth token: %s", exc)

        if credentials and credentials.valid:
            return credentials

        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                self._save_oauth_token(credentials, token_path)
                return credentials
            except Exception as exc:  # noqa: BLE001
                logger.warning("Google OAuth token refresh failed: %s", exc)

        if client_path is None or not client_path.is_file():
            raise CalendarSchedulingError(
                "GOOGLE_OAUTH_CLIENT_SECRET_FILE is required to authorize "
                "Google Calendar"
            )

        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_path), _SCOPES
            )
            credentials = flow.run_local_server(
                port=0,
                access_type="offline",
                prompt="consent",
                open_browser=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise CalendarSchedulingError(
                "Google OAuth authorization failed. Run this program in a "
                "browser-capable environment, or use a service account. "
                f"Details: {_google_error_detail(exc)}"
            ) from exc

        self._save_oauth_token(credentials, token_path)
        return credentials

    @staticmethod
    def _save_oauth_token(credentials, token_path: Path) -> None:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")
        try:
            os.chmod(token_path, 0o600)
        except OSError:
            pass

    def _find_existing(self, service, source_id: str) -> dict | None:
        response = (
            service.events()
            .list(
                calendarId=self.config.calendar_id,
                privateExtendedProperty=f"pdfWorkflowSource={source_id}",
                maxResults=1,
                singleEvents=True,
                showDeleted=False,
            )
            .execute()
        )
        items = response.get("items") or []
        return items[0] if items else None


def _parse_datetime_or_date(
    raw: str,
    timezone: ZoneInfo,
    clock: dt_time,
    notes: list[str],
) -> datetime | None:
    text = raw.strip()
    if not text:
        return None

    if "T" in text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
                return datetime.combine(parsed.date(), clock, tzinfo=timezone)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone)
            return parsed.astimezone(timezone)
        except ValueError:
            notes.append(f"Could not parse suggested datetime {text!r}.")

    parsed_date = _parse_date(text)
    if parsed_date is None:
        notes.append(f"Could not parse suggested date {text!r}.")
        return None
    return datetime.combine(parsed_date, clock, tzinfo=timezone)


def _parse_date(raw: str) -> date | None:
    text = raw.strip()
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _parse_time(
    raw: str | None,
    *,
    fallback: dt_time,
    notes: list[str] | None = None,
) -> dt_time:
    if not raw:
        return fallback
    text = raw.strip()
    formats = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I %p")
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    if notes is not None:
        notes.append(f"Could not parse suggested time {text!r}; used {fallback}.")
    return fallback


def _add_business_days(start: date, days: int) -> date:
    current = start
    added = 0
    while added < days or current.weekday() >= 5:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def _google_error_detail(error: Exception) -> str:
    reason = getattr(error, "reason", None)
    if reason:
        return str(reason)[:500]
    return " ".join(str(error).split())[:500] or type(error).__name__


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
