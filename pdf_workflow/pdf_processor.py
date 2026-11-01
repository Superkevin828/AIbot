from __future__ import annotations

import hashlib
import logging
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import PdfDocument, PdfPage

logger = logging.getLogger(__name__)
_WORD_RE = re.compile(r"\S+")


class PdfProcessingError(RuntimeError):
    """Raised when a PDF cannot be read or contains no usable text."""


class PdfProcessor:
    def __init__(self, *, max_pages: int = 0) -> None:
        self.max_pages = max_pages

    def extract(self, path: str | Path) -> PdfDocument:
        pdf_path = Path(path).expanduser()
        if not pdf_path.is_file():
            raise PdfProcessingError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise PdfProcessingError(f"Not a PDF file: {pdf_path}")

        try:
            import pdfplumber
        except ImportError as exc:
            raise PdfProcessingError(
                "pdfplumber is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            file_hash = _sha256_file(pdf_path)
        except OSError as exc:
            raise PdfProcessingError(
                f"Could not read {pdf_path.name}: {exc}"
            ) from exc

        warnings: list[str] = []
        pages: list[PdfPage] = []
        metadata: dict[str, str] = {}

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total_pages = len(pdf.pages)
                metadata = _clean_metadata(pdf.metadata)
                pages_to_read = pdf.pages
                if self.max_pages > 0 and total_pages > self.max_pages:
                    pages_to_read = pdf.pages[: self.max_pages]
                    warnings.append(
                        f"Only the first {self.max_pages} of {total_pages} pages "
                        "were processed (PDF_MAX_PAGES)."
                    )

                failed_pages = 0
                for page in pages_to_read:
                    page_number = int(page.page_number)
                    page_text = ""
                    headings: list[str] = []
                    tables: list[list[list[str]]] = []

                    try:
                        page_text = page.extract_text() or ""
                    except Exception as exc:  # noqa: BLE001 - third-party parser errors vary
                        failed_pages += 1
                        warnings.append(
                            f"Page {page_number}: text extraction failed "
                            f"({type(exc).__name__})."
                        )

                    try:
                        headings = _extract_headings(page)
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(
                            f"Page {page_number}: heading detection failed "
                            f"({type(exc).__name__})."
                        )

                    try:
                        raw_tables = page.extract_tables() or []
                        tables = [
                            cleaned
                            for table in raw_tables
                            if (cleaned := _clean_table(table))
                        ]
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(
                            f"Page {page_number}: table extraction failed "
                            f"({type(exc).__name__})."
                        )

                    pages.append(
                        PdfPage(
                            number=page_number,
                            text=page_text.strip(),
                            headings=headings,
                            tables=tables,
                        )
                    )

                if failed_pages and failed_pages == len(pages):
                    raise PdfProcessingError(
                        f"All pages failed text extraction in {pdf_path.name}. "
                        "The PDF may be encrypted or corrupt."
                    )
        except PdfProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PdfProcessingError(
                f"Could not read {pdf_path.name}: {exc}. "
                "It may be encrypted, password-protected, or corrupt."
            ) from exc

        full_text = "\n".join(page.text for page in pages)
        if not full_text.strip() and not any(page.tables for page in pages):
            raise PdfProcessingError(
                f"No extractable text was found in {pdf_path.name}. "
                "Scanned/image-only PDFs must be OCR-processed first."
            )

        document = PdfDocument(
            source_path=str(pdf_path.resolve()),
            source_name=pdf_path.name,
            page_count=total_pages,
            metadata=metadata,
            pages=pages,
            sha256=file_hash,
            word_count=len(_WORD_RE.findall(full_text)),
            char_count=len(full_text),
            warnings=warnings,
        )
        logger.info(
            "Extracted %s: %d page(s), %d word(s), %d table(s)",
            document.source_name,
            document.page_count,
            document.word_count,
            document.table_count,
        )
        return document


def build_llm_content(documents: list[PdfDocument], max_chars: int) -> str:
    """Render extracted documents into a bounded prompt payload."""

    if not documents:
        raise PdfProcessingError("At least one PDF is required")

    chunks: list[str] = []
    remaining = max_chars
    for index, document in enumerate(documents):
        if remaining <= 0:
            chunks.append(
                f"[Additional document omitted because PDF_MAX_CHARS was reached: "
                f"{document.source_name}]"
            )
            continue

        documents_left = len(documents) - index
        budget = max(500, remaining // max(1, documents_left))
        rendered = _render_document(document, index + 1)
        if len(rendered) > budget:
            omitted = len(rendered) - budget
            rendered = (
                rendered[:budget].rstrip()
                + f"\n[... {omitted} characters truncated from this document ...]"
            )
        chunks.append(rendered)
        remaining -= len(rendered)

    return "\n\n".join(chunks)


def _render_document(document: PdfDocument, index: int) -> str:
    parts = [f"=== DOCUMENT {index}: {document.source_name} ==="]
    parts.append(
        f"Pages: {document.page_count} | Words: {document.word_count} | "
        f"Tables: {document.table_count}"
    )

    metadata_bits = []
    for key in ("Title", "Author", "Subject", "CreationDate", "ModDate"):
        value = document.metadata.get(key)
        if value:
            metadata_bits.append(f"{key}: {value}")
    if metadata_bits:
        parts.append("Metadata: " + " | ".join(metadata_bits))
    if document.warnings:
        parts.append("Warnings: " + " | ".join(document.warnings))

    headings = document.headings[:40]
    if headings:
        parts.append("Detected headings:")
        parts.extend(f"- {heading}" for heading in headings)

    for page in document.pages:
        parts.append(f"\n--- Page {page.number} ---")
        if page.text:
            parts.append(page.text)
        for table_number, table in enumerate(page.tables, start=1):
            parts.append(
                f"[Table {table_number} on page {page.number}]\n"
                + _render_table(table)
            )

    return "\n".join(parts)


def _render_table(table: list[list[str]], max_rows: int = 40) -> str:
    lines = []
    for row in table[:max_rows]:
        cells = [cell.replace("|", "/").strip()[:200] for cell in row]
        lines.append(" | ".join(cells))
    if len(table) > max_rows:
        lines.append(f"[... {len(table) - max_rows} table rows omitted ...]")
    return "\n".join(lines)


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    output: dict[str, str] = {}
    if not metadata:
        return output
    for raw_key, raw_value in metadata.items():
        if raw_value is None:
            continue
        key = str(raw_key).lstrip("/")
        value = " ".join(str(raw_value).split())
        if value:
            output[key] = value
    return output


def _clean_table(table: list[list[Any]] | None) -> list[list[str]]:
    if not table:
        return []
    rows: list[list[str]] = []
    for row in table:
        if row is None:
            continue
        cells = [
            " ".join(str(cell).replace("\x00", " ").split()) if cell is not None else ""
            for cell in row
        ]
        if any(cells):
            rows.append(cells)
    return rows


def _extract_headings(page: Any) -> list[str]:
    words = page.extract_words(extra_attrs=["fontname", "size"])
    if not words:
        return []

    sizes = [
        float(word["size"])
        for word in words
        if isinstance(word.get("size"), (int, float))
    ]
    median_size = statistics.median(sizes) if sizes else 0.0
    size_threshold = max(median_size * 1.18, median_size + 1.0)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for word in words:
        top = float(word.get("top", 0.0))
        grouped[round(top / 3.0)].append(word)

    headings: list[str] = []
    seen: set[str] = set()
    for _, line_words in sorted(grouped.items()):
        line_words.sort(key=lambda item: float(item.get("x0", 0.0)))
        text = " ".join(str(word.get("text", "")) for word in line_words)
        text = " ".join(text.split()).strip()
        if not 3 <= len(text) <= 120 or len(line_words) > 16:
            continue

        line_sizes = [
            float(word["size"])
            for word in line_words
            if isinstance(word.get("size"), (int, float))
        ]
        average_size = sum(line_sizes) / len(line_sizes) if line_sizes else 0.0
        bold_count = sum(
            1 for word in line_words if _is_bold(str(word.get("fontname", "")))
        )
        bold_ratio = bold_count / len(line_words)
        looks_uppercase = text.isupper() and len(text) >= 4 and len(line_words) <= 10

        if (
            average_size >= size_threshold
            or bold_ratio >= 0.7
            or looks_uppercase
        ):
            key = text.casefold()
            if key not in seen:
                seen.add(key)
                headings.append(text)

    return headings[:25]


def _is_bold(fontname: str) -> bool:
    name = fontname.casefold()
    return any(
        token in name
        for token in ("bold", "black", "heavy", "semibold", "demi", "extrabold")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
