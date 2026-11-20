from __future__ import annotations

import csv
from contextlib import closing
from pathlib import Path

from .chunker import chunk_text_block
from .config import IMAGE_EXTENSIONS, MAX_FILE_SIZE_BYTES, PDF_OCR_PAGE_LIMIT
from .models import TextChunk
from .ocr import OcrService


class ParseError(Exception):
    pass


class DocumentParser:
    def __init__(
        self,
        *,
        ocr_service: OcrService | None = None,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        pdf_ocr_page_limit: int = PDF_OCR_PAGE_LIMIT,
    ) -> None:
        self.ocr_service = ocr_service or OcrService(enabled=False)
        self.max_file_size = max_file_size
        self.pdf_ocr_page_limit = pdf_ocr_page_limit

    def parse(self, path: str | Path) -> tuple[list[TextChunk], str, str]:
        file_path = Path(path)
        if not file_path.exists():
            raise ParseError(f"File does not exist: {file_path}")
        if file_path.stat().st_size > self.max_file_size:
            raise ParseError(f"File is larger than {self.max_file_size // 1024 // 1024}MB")

        ext = file_path.suffix.lower()
        if ext in {".txt", ".md"}:
            return self._parse_plain_text(file_path), "parsed", "not_required"
        if ext == ".csv":
            return self._parse_csv(file_path), "parsed", "not_required"
        if ext == ".docx":
            return self._parse_docx(file_path), "parsed", "not_required"
        if ext == ".xlsx":
            return self._parse_xlsx(file_path), "parsed", "not_required"
        if ext == ".pptx":
            return self._parse_pptx(file_path), "parsed", "not_required"
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        if ext in IMAGE_EXTENSIONS:
            return self._parse_image(file_path)
        raise ParseError(f"Unsupported file extension: {ext}")

    def _read_text(self, path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")

    def _parse_plain_text(self, path: Path) -> list[TextChunk]:
        text = self._read_text(path)
        return chunk_text_block(text, source_type=path.suffix.lower().lstrip(".") or "text")

    def _parse_csv(self, path: Path) -> list[TextChunk]:
        text = self._read_text(path)
        rows: list[str] = []
        reader = csv.reader(text.splitlines())
        for index, row in enumerate(reader):
            if index >= 2000:
                rows.append("... CSV 内容过长，已截断")
                break
            rows.append(" | ".join(cell.strip() for cell in row if cell.strip()))
        return chunk_text_block("\n".join(rows), source_type="csv")

    def _parse_docx(self, path: Path) -> list[TextChunk]:
        try:
            from docx import Document
        except Exception as exc:
            raise ParseError("python-docx is not installed") from exc

        doc = Document(str(path))
        parts: list[str] = []
        parts.extend(p.text.strip() for p in doc.paragraphs if p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return chunk_text_block("\n".join(parts), source_type="docx")

    def _parse_xlsx(self, path: Path) -> list[TextChunk]:
        try:
            from openpyxl import load_workbook
        except Exception as exc:
            raise ParseError("openpyxl is not installed") from exc

        chunks: list[TextChunk] = []
        with path.open("rb") as stream, closing(load_workbook(stream, read_only=True, data_only=True)) as workbook:
            for sheet in workbook.worksheets:
                lines: list[str] = []
                # Early row limits must also close the worksheet's streaming XML reader.
                with closing(sheet.iter_rows(values_only=True)) as rows:
                    for row_index, row in enumerate(rows, start=1):
                        if row_index > 2000:
                            lines.append("... Sheet 内容过长，已截断")
                            break
                        values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                        if values:
                            lines.append(f"R{row_index}: " + " | ".join(values))
                chunks.extend(chunk_text_block("\n".join(lines), source_type="xlsx", sheet_name=sheet.title))
        return chunks

    def _parse_pptx(self, path: Path) -> list[TextChunk]:
        try:
            from pptx import Presentation
        except Exception as exc:
            raise ParseError("python-pptx is not installed") from exc

        presentation = Presentation(str(path))
        chunks: list[TextChunk] = []
        for slide_index, slide in enumerate(presentation.slides, start=1):
            texts: list[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            chunks.extend(chunk_text_block("\n".join(texts), source_type="pptx", page_no=slide_index))
        return chunks

    def _parse_pdf(self, path: Path) -> tuple[list[TextChunk], str, str]:
        try:
            import pymupdf
        except Exception as exc:
            raise ParseError("PyMuPDF is not installed") from exc

        chunks: list[TextChunk] = []
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    chunks.extend(chunk_text_block(text, source_type="pdf", page_no=index))

        total_text_len = sum(len(chunk.chunk_text) for chunk in chunks)
        if total_text_len >= 30:
            return chunks, "parsed", "not_required"

        ocr_results = self.ocr_service.ocr_pdf_pages(path, max_pages=self.pdf_ocr_page_limit)
        ocr_chunks: list[TextChunk] = []
        ocr_status = "done"
        for result in ocr_results:
            if result.status not in {"done", "empty"}:
                ocr_status = result.status
            ocr_chunks.extend(
                chunk_text_block(
                    result.text,
                    source_type="ocr_pdf",
                    page_no=result.page_no,
                    ocr_confidence=result.confidence,
                )
            )
        return ocr_chunks or chunks, "parsed" if ocr_chunks else "empty", ocr_status

    def _parse_image(self, path: Path) -> tuple[list[TextChunk], str, str]:
        result = self.ocr_service.ocr_image(path)
        chunks = chunk_text_block(
            result.text,
            source_type="ocr_image",
            page_no=result.page_no,
            ocr_confidence=result.confidence,
        )
        parse_status = "parsed" if chunks else "empty"
        return chunks, parse_status, result.status
