from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    max_attempts: int


@dataclass(frozen=True)
class PdfConfig:
    max_chars: int
    max_pages: int


@dataclass(frozen=True)
class SlackConfig:
    enabled: bool
    mode: str
    webhook_url: str | None
    bot_token: str | None
    channel: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class CalendarConfig:
    enabled: bool
    auth_mode: str
    service_account_file: Path | None
    service_account_subject: str | None
    oauth_client_secret_file: Path | None
    oauth_token_file: Path
    calendar_id: str
    timezone: str
    default_days: int
    default_time: str
    duration_minutes: int
    attendees: tuple[str, ...]
    always_schedule: bool
    dedupe: bool


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    pdf: PdfConfig
    slack: SlackConfig
    calendar: CalendarConfig
    output_dir: Path
    log_level: str


def load_config(
    env_file: str | Path | None = None,
    *,
    slack_enabled: bool | None = None,
    calendar_enabled: bool | None = None,
    validate_integrations: bool = True,
) -> AppConfig:
    """Load and validate environment configuration.

    Existing process environment variables take precedence over .env values.
    Integration credentials are only required when the integration is enabled
    and ``validate_integrations`` is true.
    """

    if env_file is not None:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise ConfigError(f"Environment file not found: {path}")
        load_dotenv(dotenv_path=path, override=False)
    else:
        default_env = Path(".env")
        if not default_env.is_file():
            default_env = Path(__file__).resolve().parent.parent / ".env"
        if default_env.is_file():
            load_dotenv(dotenv_path=default_env, override=False)

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    provider = {"google": "gemini", "claude": "anthropic"}.get(provider, provider)
    if provider not in {"openai", "anthropic", "gemini"}:
        raise ConfigError(
            "LLM_PROVIDER must be one of: openai, anthropic, gemini"
        )

    default_models = {
        "openai": "gpt-4.1-mini",
        "anthropic": "claude-sonnet-4-5",
        "gemini": "gemini-2.5-flash",
    }
    api_key_name = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }[provider]
    api_key = os.getenv(api_key_name, "").strip()
    if provider == "gemini" and not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            f"{api_key_name} is required for LLM_PROVIDER={provider}"
        )

    model = os.getenv("LLM_MODEL", "").strip() or default_models[provider]
    llm = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=_float_env("LLM_TEMPERATURE", 0.2, 0.0, 2.0),
        max_output_tokens=_int_env("LLM_MAX_OUTPUT_TOKENS", 4096, 256, 65536),
        timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 90.0, 5.0, 600.0),
        max_attempts=_int_env("LLM_MAX_ATTEMPTS", 3, 1, 10),
    )

    pdf = PdfConfig(
        max_chars=_int_env("PDF_MAX_CHARS", 80000, 2000, 2_000_000),
        max_pages=_int_env("PDF_MAX_PAGES", 0, 0, 10000),
    )

    slack_configured = _bool_env("SLACK_ENABLED", True)
    slack_is_enabled = (
        slack_configured if slack_enabled is None else bool(slack_enabled)
    )
    webhook_url = _optional_env("SLACK_WEBHOOK_URL")
    bot_token = _optional_env("SLACK_BOT_TOKEN")
    channel = _optional_env("SLACK_CHANNEL_ID")
    requested_mode = os.getenv("SLACK_MODE", "auto").strip().lower()
    if requested_mode not in {"auto", "webhook", "bot"}:
        raise ConfigError("SLACK_MODE must be one of: auto, webhook, bot")
    if requested_mode == "auto":
        slack_mode = "webhook" if webhook_url else "bot"
    else:
        slack_mode = requested_mode

    if slack_is_enabled and validate_integrations:
        if slack_mode == "webhook":
            if not webhook_url:
                raise ConfigError(
                    "SLACK_WEBHOOK_URL is required when SLACK_MODE=webhook"
                )
            if not webhook_url.startswith(("https://", "http://")):
                raise ConfigError("SLACK_WEBHOOK_URL must be an http(s) URL")
        else:
            if not bot_token:
                raise ConfigError(
                    "SLACK_BOT_TOKEN is required when SLACK_MODE=bot"
                )
            if not channel:
                raise ConfigError(
                    "SLACK_CHANNEL_ID is required when SLACK_MODE=bot"
                )

    slack = SlackConfig(
        enabled=slack_is_enabled,
        mode=slack_mode,
        webhook_url=webhook_url,
        bot_token=bot_token,
        channel=channel,
        timeout_seconds=_float_env("SLACK_TIMEOUT_SECONDS", 20.0, 5.0, 120.0),
    )

    calendar_configured = _bool_env("CALENDAR_ENABLED", True)
    calendar_is_enabled = (
        calendar_configured if calendar_enabled is None else bool(calendar_enabled)
    )
    auth_mode = os.getenv("GOOGLE_AUTH_MODE", "service_account").strip().lower()
    if auth_mode not in {"service_account", "oauth"}:
        raise ConfigError(
            "GOOGLE_AUTH_MODE must be one of: service_account, oauth"
        )

    service_account_path = _optional_path(
        "GOOGLE_SERVICE_ACCOUNT_FILE"
    ) or _optional_path("GOOGLE_APPLICATION_CREDENTIALS")
    oauth_client_path = _optional_path("GOOGLE_OAUTH_CLIENT_SECRET_FILE")
    oauth_token_path = Path(
        os.getenv("GOOGLE_OAUTH_TOKEN_FILE", "./credentials/google-token.json")
    ).expanduser()

    timezone_name = os.getenv("TIMEZONE", "UTC").strip() or "UTC"
    _validate_timezone(timezone_name)
    default_time = os.getenv("FOLLOWUP_DEFAULT_TIME", "10:00").strip()
    try:
        datetime.strptime(default_time, "%H:%M")
    except ValueError as exc:
        raise ConfigError("FOLLOWUP_DEFAULT_TIME must use HH:MM (24-hour)") from exc

    attendees = _parse_attendees(os.getenv("FOLLOWUP_ATTENDEES", ""))

    if calendar_is_enabled and validate_integrations:
        if auth_mode == "service_account":
            if service_account_path is None:
                raise ConfigError(
                    "GOOGLE_SERVICE_ACCOUNT_FILE is required for "
                    "GOOGLE_AUTH_MODE=service_account"
                )
            if not service_account_path.is_file():
                raise ConfigError(
                    f"Google service account file not found: {service_account_path}"
                )
        elif not oauth_token_path.is_file():
            if oauth_client_path is None:
                raise ConfigError(
                    "GOOGLE_OAUTH_CLIENT_SECRET_FILE is required for "
                    "GOOGLE_AUTH_MODE=oauth"
                )
            if not oauth_client_path.is_file():
                raise ConfigError(
                    f"Google OAuth client secret file not found: {oauth_client_path}"
                )

    calendar = CalendarConfig(
        enabled=calendar_is_enabled,
        auth_mode=auth_mode,
        service_account_file=service_account_path,
        service_account_subject=_optional_env("GOOGLE_SERVICE_ACCOUNT_SUBJECT"),
        oauth_client_secret_file=oauth_client_path,
        oauth_token_file=oauth_token_path,
        calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary").strip() or "primary",
        timezone=timezone_name,
        default_days=_int_env("FOLLOWUP_DEFAULT_DAYS", 3, 0, 365),
        default_time=default_time,
        duration_minutes=_int_env("FOLLOWUP_DURATION_MINUTES", 30, 15, 480),
        attendees=attendees,
        always_schedule=_bool_env("FOLLOWUP_ALWAYS_SCHEDULE", False),
        dedupe=_bool_env("CALENDAR_DEDUPE", True),
    )

    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")).expanduser()
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")

    return AppConfig(
        llm=llm,
        pdf=pdf,
        slack=slack,
        calendar=calendar,
        output_dir=output_dir,
        log_level=log_level,
    )


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _optional_path(name: str) -> Path | None:
    value = _optional_env(name)
    return Path(value).expanduser() if value else None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false")


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _parse_attendees(raw: str) -> tuple[str, ...]:
    attendees: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        email = part.strip()
        if not email:
            continue
        if not _EMAIL_RE.match(email):
            raise ConfigError(f"Invalid attendee email in FOLLOWUP_ATTENDEES: {email}")
        key = email.lower()
        if key not in seen:
            seen.add(key)
            attendees.append(email)
    return tuple(attendees)


def _validate_timezone(name: str) -> None:
    try:
        ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ConfigError(f"Unknown TIMEZONE: {name}") from exc
