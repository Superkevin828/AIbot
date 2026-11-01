from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AnalysisParseError(RuntimeError):
    """Raised when the LLM response cannot be parsed into an Analysis."""


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: str
    owner: str | None = None
    due_date: str | None = None
    priority: str | None = None

    @field_validator("task", mode="before")
    @classmethod
    def _task_text(cls, value: Any) -> str:
        text = _clean_text(value)
        if not text:
            raise ValueError("action item task cannot be empty")
        return text

    @field_validator("owner", "due_date", "priority", mode="before")
    @classmethod
    def _optional_text(cls, value: Any) -> str | None:
        return _none_if_unknown(value)


class ImportantDate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    date: str | None = None
    time: str | None = None
    context: str | None = None

    @field_validator("label", mode="before")
    @classmethod
    def _label_text(cls, value: Any) -> str:
        text = _clean_text(value)
        return text or "Important date"

    @field_validator("date", "time", "context", mode="before")
    @classmethod
    def _optional_text(cls, value: Any) -> str | None:
        return _none_if_unknown(value)


class FollowUp(BaseModel):
    model_config = ConfigDict(extra="ignore")

    required: bool = False
    suggested_date: str | None = None
    suggested_time: str | None = None
    timezone: str | None = None
    duration_minutes: int | None = None
    reason: str | None = None
    agenda: list[str] = Field(default_factory=list)
    attendees: list[str] = Field(default_factory=list)

    @field_validator("required", mode="before")
    @classmethod
    def _coerce_required(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @field_validator(
        "suggested_date", "suggested_time", "timezone", "reason", mode="before"
    )
    @classmethod
    def _optional_text(cls, value: Any) -> str | None:
        return _none_if_unknown(value)

    @field_validator("duration_minutes", mode="before")
    @classmethod
    def _coerce_duration(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            minutes = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None
        return max(15, min(minutes, 480))

    @field_validator("agenda", "attendees", mode="before")
    @classmethod
    def _coerce_string_list(cls, value: Any) -> list[str]:
        return _coerce_string_list(value)


class Analysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = "Document Summary"
    executive_summary: str
    core_insights: list[str] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    important_dates: list[ImportantDate] = Field(default_factory=list)
    follow_up: FollowUp = Field(default_factory=FollowUp)

    @field_validator("title", mode="before")
    @classmethod
    def _title_text(cls, value: Any) -> str:
        text = _clean_text(value)
        return text or "Document Summary"

    @field_validator("executive_summary", mode="before")
    @classmethod
    def _summary_text(cls, value: Any) -> str:
        text = _clean_text(value)
        if not text:
            raise ValueError("executive_summary cannot be empty")
        return text

    @field_validator("core_insights", "key_takeaways", mode="before")
    @classmethod
    def _coerce_string_list(cls, value: Any) -> list[str]:
        return _coerce_string_list(value)

    @field_validator("action_items", mode="before")
    @classmethod
    def _coerce_action_items(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            return [{"task": value}]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [
                {"task": item} if isinstance(item, str) else item
                for item in value
                if item is not None
            ]
        return []

    @field_validator("important_dates", mode="before")
    @classmethod
    def _coerce_important_dates(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            return [{"label": value, "date": value}]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [
                {"label": item, "date": item} if isinstance(item, str) else item
                for item in value
                if item is not None
            ]
        return []


@dataclass
class PdfPage:
    number: int
    text: str
    headings: list[str] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)


@dataclass
class PdfDocument:
    source_path: str
    source_name: str
    page_count: int
    metadata: dict[str, str]
    pages: list[PdfPage]
    sha256: str
    word_count: int
    char_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def headings(self) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for page in self.pages:
            for heading in page.headings:
                key = heading.casefold()
                if key not in seen:
                    seen.add(key)
                    output.append(heading)
        return output

    @property
    def table_count(self) -> int:
        return sum(len(page.tables) for page in self.pages)


def parse_analysis(raw: str) -> Analysis:
    """Parse JSON from an LLM response, tolerating markdown fences and preamble."""

    text = raw.strip()
    if not text:
        raise AnalysisParseError("The LLM returned an empty response")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _extract_json_object(text)

    if not isinstance(payload, dict):
        raise AnalysisParseError("The LLM response must be a JSON object")

    try:
        return Analysis.model_validate(payload)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:8]
        )
        raise AnalysisParseError(f"Invalid analysis JSON: {errors}") from exc


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise AnalysisParseError("No valid JSON object found in the LLM response")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split()).strip()


def _none_if_unknown(value: Any) -> str | None:
    text = _clean_text(value)
    if not text or text.casefold() in {
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "not specified",
        "not provided",
    }:
        return None
    return text


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    if isinstance(value, dict):
        value = [value]

    items: list[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, dict):
                item = (
                    item.get("text")
                    or item.get("title")
                    or item.get("item")
                    or item.get("description")
                    or item.get("insight")
                    or item.get("takeaway")
                )
            text = _clean_text(item)
            if text:
                items.append(text)
    return items
