"""Manual-file extraction with resumable Telugu/Nepali PDF OCR fallback."""

from __future__ import annotations

import gzip
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

from .common import PreprocessError, atomic_write_text
from .html_extract import extract_html_text
from .text import target_script_count


MANUAL_FILE_SUFFIXES = {".epub", ".htm", ".html", ".jsonl", ".md", ".pdf", ".txt"}
DEFAULT_MANUAL_CHUNK_CHARACTERS = 20_000
MAX_EPUB_MEMBER_BYTES = 50 * 1024 * 1024
OCR_MODES = {"off", "auto", "force"}
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_PSM = 3
DEFAULT_OCR_LANGUAGES = {"tel": "tel+eng", "nep": "nep+eng"}


def manual_input_directory(data_root: Path, source: dict[str, Any]) -> Path:
    """Resolve a source directory while confining it to raw/manual/."""
    manual_root = (data_root / "raw" / "manual").resolve()
    configured = source.get("input_directory", f"raw/manual/{source['id']}")
    if not isinstance(configured, str) or not configured.strip():
        raise PreprocessError(f"manual source {source['id']} has an invalid input_directory")
    directory = (data_root / configured).resolve()
    try:
        directory.relative_to(manual_root)
    except ValueError as exc:
        raise PreprocessError(
            f"manual source {source['id']} must read below {manual_root}, not {directory}"
        ) from exc
    return directory


def manual_input_files(data_root: Path, source: dict[str, Any]) -> Iterator[Path]:
    directory = manual_input_directory(data_root, source)
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise PreprocessError(f"manual source {source['id']} may not contain symlinks: {path}")
        suffix = path.suffix.casefold()
        if path.is_file() and (suffix in MANUAL_FILE_SUFFIXES or path.name.casefold().endswith(".jsonl.gz")):
            yield path


def _decode_text(value: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", "replace")


def chunk_text(text: str, max_characters: int) -> Iterator[str]:
    """Split long books into deterministic paragraph-aware documents."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return
    blocks = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
    if not blocks:
        blocks = [text]
    current: list[str] = []
    current_size = 0

    def pieces(block: str) -> Iterator[str]:
        while len(block) > max_characters:
            boundary = block.rfind(" ", 0, max_characters + 1)
            if boundary < max_characters // 2:
                boundary = max_characters
            yield block[:boundary].strip()
            block = block[boundary:].strip()
        if block:
            yield block

    for block in blocks:
        for piece in pieces(block):
            extra = len(piece) + (2 if current else 0)
            if current and current_size + extra > max_characters:
                yield "\n\n".join(current)
                current = []
                current_size = 0
            current.append(piece)
            current_size += len(piece) + (2 if len(current) > 1 else 0)
    if current:
        yield "\n\n".join(current)


def _chunk_records(text: str, prefix: str, document_type: str, max_characters: int, **metadata: Any) -> Iterator[dict[str, Any]]:
    for index, chunk in enumerate(chunk_text(text, max_characters), 1):
        yield {
            "document_id": f"{prefix}:chunk-{index:05d}",
            "text": chunk,
            "document_type": document_type,
            **metadata,
        }


def _epub_spine(archive: zipfile.ZipFile) -> list[str]:
    """Return EPUB reading order, falling back to sorted XHTML members."""
    names = set(archive.namelist())
    try:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(element for element in container.iter() if element.tag.endswith("rootfile"))
        package_path = rootfile.attrib["full-path"]
        package = ET.fromstring(archive.read(package_path))
        manifest = {
            element.attrib["id"]: element.attrib["href"]
            for element in package.iter()
            if element.tag.endswith("item") and "id" in element.attrib and "href" in element.attrib
        }
        base = PurePosixPath(package_path).parent
        ordered = [
            str(base / manifest[element.attrib["idref"]])
            for element in package.iter()
            if element.tag.endswith("itemref") and element.attrib.get("idref") in manifest
        ]
        result = [name for name in ordered if name in names]
        if result:
            return result
    except (ET.ParseError, KeyError, StopIteration, OSError, zipfile.BadZipFile):
        pass
    return sorted(name for name in names if PurePosixPath(name).suffix.casefold() in {".htm", ".html", ".xhtml"})


def _epub_documents(path: Path, source: dict[str, Any], max_characters: int, audit: dict[str, Any]) -> Iterator[dict[str, Any]]:
    try:
        with zipfile.ZipFile(path) as archive:
            for chapter_index, member in enumerate(_epub_spine(archive), 1):
                info = archive.getinfo(member)
                if info.file_size > MAX_EPUB_MEMBER_BYTES:
                    audit["oversized_epub_members"] += 1
                    continue
                raw_html = _decode_text(archive.read(member))
                text, strategy = extract_html_text(raw_html, source)
                for record in _chunk_records(
                    text,
                    f"{path.name}:{chapter_index:05d}",
                    "manual_epub_chapter",
                    max_characters,
                    extraction_method=strategy,
                    epub_member=member,
                ):
                    audit["text_characters"] += len(record["text"])
                    audit["records"] += 1
                    yield record
    except (OSError, zipfile.BadZipFile) as exc:
        raise PreprocessError(f"could not read EPUB {path}: {exc}") from exc


def _load_ocr_modules() -> tuple[Any, Any]:
    try:
        import pypdfium2 as pdfium  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise PreprocessError(
            "PDF OCR requires pypdfium2 and pytesseract. Install requirements-colab.txt."
        ) from exc
    return pdfium, pytesseract


def _close_if_supported(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _ocr_pdf_page(
    pdfium_document: Any,
    page_number: int,
    language_spec: str,
    dpi: int,
    psm: int,
    checkpoint_directory: Path | None,
    pytesseract: Any,
) -> tuple[str, bool]:
    """OCR one page and durably cache only its text, never the rendered image."""
    checkpoint = checkpoint_directory / f"page-{page_number:06d}.json" if checkpoint_directory else None
    if checkpoint and checkpoint.exists():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                saved.get("schema_version") == 1
                and saved.get("page_number") == page_number
                and saved.get("languages") == language_spec
                and saved.get("dpi") == dpi
                and saved.get("psm") == psm
                and isinstance(saved.get("text"), str)
            ):
                return saved["text"], True
        except (OSError, json.JSONDecodeError):
            pass

    page = pdfium_document[page_number - 1]
    bitmap = image = None
    try:
        bitmap = page.render(scale=dpi / 72.0)
        image = bitmap.to_pil()
        text = str(
            pytesseract.image_to_string(
                image,
                lang=language_spec,
                config=f"--oem 1 --psm {psm}",
            )
            or ""
        ).strip()
    except Exception as exc:
        raise PreprocessError(f"OCR failed for {page_number=} in {pdfium_document}: {exc}") from exc
    finally:
        _close_if_supported(image)
        _close_if_supported(bitmap)
        _close_if_supported(page)

    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            checkpoint,
            json.dumps(
                {
                    "schema_version": 1,
                    "page_number": page_number,
                    "languages": language_spec,
                    "dpi": dpi,
                    "psm": psm,
                    "text": text,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
    return text, False


def _pdf_documents(
    path: Path,
    source: dict[str, Any],
    max_characters: int,
    audit: dict[str, Any],
    ocr_checkpoint_directory: Path | None = None,
    ocr_progress: Callable[[int, int, str], None] | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise PreprocessError("pypdf is required for manual PDF text extraction. Install requirements-colab.txt.") from exc
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise PreprocessError(f"could not open PDF {path}: {exc}") from exc
    language = str(source["language"])
    ocr_mode = str(source.get("ocr_mode", "off")).casefold()
    if ocr_mode not in OCR_MODES:
        raise PreprocessError(f"ocr_mode for {source['id']} must be one of {sorted(OCR_MODES)}")
    ocr_dpi = int(source.get("ocr_dpi", DEFAULT_OCR_DPI))
    ocr_psm = int(source.get("ocr_psm", DEFAULT_OCR_PSM))
    ocr_languages = str(source.get("ocr_languages", DEFAULT_OCR_LANGUAGES.get(language, language)))
    pdfium_document = pytesseract = None
    current_pages: list[str] = []
    current_methods: list[str] = []
    current_start = 0
    current_end = 0
    current_size = 0

    def flush() -> dict[str, Any] | None:
        nonlocal current_pages, current_methods, current_start, current_end, current_size
        if not current_pages:
            return None
        text = "\n\n".join(current_pages)
        methods = set(current_methods)
        extraction_method = (
            next(iter(methods)) if len(methods) == 1 else "mixed_pdf_text_and_tesseract_ocr"
        )
        record = {
            "document_id": f"{path.name}:pages-{current_start:05d}-{current_end:05d}",
            "text": text,
            "document_type": "manual_pdf_text",
            "page_start": current_start,
            "page_end": current_end,
            "extraction_method": extraction_method,
        }
        if "tesseract_ocr" in methods:
            record.update({"ocr_languages": ocr_languages, "ocr_dpi": ocr_dpi, "ocr_psm": ocr_psm})
        audit["records"] += 1
        audit["text_characters"] += len(text)
        current_pages = []
        current_methods = []
        current_start = current_end = current_size = 0
        return record

    audit.update({
        "pdf_pages": len(reader.pages),
        "ocr_mode": ocr_mode,
        "ocr_languages": ocr_languages if ocr_mode != "off" else None,
        "ocr_dpi": ocr_dpi if ocr_mode != "off" else None,
        "ocr_psm": ocr_psm if ocr_mode != "off" else None,
        "ocr_completed_pages": 0,
        "ocr_reused_pages": 0,
        "ocr_failed_pages": 0,
        "embedded_pdf_pages": 0,
    })
    try:
        for page_number, page in enumerate(reader.pages, 1):
            try:
                embedded_text = (page.extract_text() or "").strip()
            except Exception:
                embedded_text = ""
            use_ocr = ocr_mode == "force" or target_script_count(embedded_text, language) < 40
            text = embedded_text
            method = "embedded_pdf_text"
            if use_ocr:
                audit["ocr_candidate_pages"] += 1
                if ocr_mode == "off":
                    continue
                if pdfium_document is None:
                    pdfium, pytesseract = _load_ocr_modules()
                    try:
                        available = set(pytesseract.get_languages(config=""))
                    except Exception as exc:
                        raise PreprocessError(f"could not query installed Tesseract languages: {exc}") from exc
                    missing = [code for code in ocr_languages.split("+") if code not in available]
                    if missing:
                        packages = " ".join(f"tesseract-ocr-{code}" for code in missing)
                        raise PreprocessError(
                            f"missing Tesseract language data {missing}; install with: apt-get install -y {packages}"
                        )
                    try:
                        pdfium_document = pdfium.PdfDocument(str(path))
                    except Exception as exc:
                        raise PreprocessError(f"could not open PDF renderer for {path}: {exc}") from exc
                text, reused = _ocr_pdf_page(
                    pdfium_document,
                    page_number,
                    ocr_languages,
                    ocr_dpi,
                    ocr_psm,
                    ocr_checkpoint_directory,
                    pytesseract,
                )
                audit["ocr_reused_pages" if reused else "ocr_completed_pages"] += 1
                if target_script_count(text, language) < 40:
                    audit["ocr_failed_pages"] += 1
                    status = "cached-low-text" if reused else "low-text"
                    if ocr_progress:
                        ocr_progress(page_number, len(reader.pages), status)
                    continue
                method = "tesseract_ocr"
                status = "cached" if reused else "complete"
                if ocr_progress:
                    ocr_progress(page_number, len(reader.pages), status)
            else:
                audit["embedded_pdf_pages"] += 1
            if current_pages and current_size + len(text) + 2 > max_characters:
                record = flush()
                if record:
                    yield record
            if not current_pages:
                current_start = page_number
            current_pages.append(text)
            current_methods.append(method)
            current_end = page_number
            current_size += len(text) + (2 if len(current_pages) > 1 else 0)
        record = flush()
        if record:
            yield record
    finally:
        _close_if_supported(pdfium_document)


def manual_document_records(
    path: Path,
    source: dict[str, Any],
    audit: dict[str, Any],
    ocr_checkpoint_directory: Path | None = None,
    ocr_progress: Callable[[int, int, str], None] | None = None,
) -> Iterator[dict[str, Any]]:
    max_characters = int(source.get("manual_chunk_characters", DEFAULT_MANUAL_CHUNK_CHARACTERS))
    if not 5_000 <= max_characters <= 100_000:
        raise PreprocessError(f"manual_chunk_characters for {source['id']} must be between 5,000 and 100,000")
    suffix = path.suffix.casefold()
    audit.update({
        "format": suffix.removeprefix("."),
        "records": 0,
        "text_characters": 0,
        "ocr_candidate_pages": 0,
        "oversized_epub_members": 0,
    })
    if suffix == ".pdf":
        yield from _pdf_documents(
            path,
            source,
            max_characters,
            audit,
            ocr_checkpoint_directory=ocr_checkpoint_directory,
            ocr_progress=ocr_progress,
        )
        return
    if suffix == ".epub":
        yield from _epub_documents(path, source, max_characters, audit)
        return
    if suffix == ".jsonl" or path.name.casefold().endswith(".jsonl.gz"):
        opener = gzip.open if path.name.casefold().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreprocessError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                text = item.get("text") if isinstance(item, dict) else None
                if not isinstance(text, str):
                    continue
                prefix = str(item.get("document_id", f"line-{line_number:08d}"))
                for record in _chunk_records(text, prefix, "manual_jsonl_document", max_characters):
                    audit["records"] += 1
                    audit["text_characters"] += len(record["text"])
                    yield record
        return
    raw = _decode_text(path.read_bytes())
    document_type = "manual_text_document"
    strategy = "plain_text"
    if suffix in {".htm", ".html"}:
        raw, strategy = extract_html_text(raw, source)
        document_type = "manual_html_document"
    for record in _chunk_records(raw, path.name, document_type, max_characters, extraction_method=strategy):
        audit["records"] += 1
        audit["text_characters"] += len(record["text"])
        yield record
