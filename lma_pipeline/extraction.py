"""Incremental Parquet, Wikimedia, and crawled-web extraction."""

from __future__ import annotations

import bz2
import base64
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

from .common import PreprocessError, atomic_gzip_jsonl, atomic_write_text, read_jsonl_gzip, sha256_bytes, sha256_file
from .html_extract import extract_html_text
from .manual_documents import manual_document_records, manual_input_files
from .pulling.mediawiki import page_content
from .source_policy import HPLT_COURSE_MANUAL_DATASETS, is_manual_source


class ExtractionMixin:
    @staticmethod
    def _preselection_rate(source: dict[str, Any]) -> float | None:
        """Return the HPLT rate applied before candidate shards are written.

        HPLT is the only very-large source with an explicit fixed source rate.
        Applying its stable selection here keeps the unselected 70--80% from
        ever becoming Drive-backed candidate JSONL, while normal downloaded
        sources still reach calibration unchanged.
        """
        key = f"{source.get('dataset')}:{source.get('dataset_config')}"
        if key not in HPLT_COURSE_MANUAL_DATASETS:
            return None
        rate = source.get("sample_rate")
        if not isinstance(rate, (int, float)) or not 0 < float(rate) <= 1:
            raise PreprocessError(f"HPLT source {source.get('id')} needs a sample_rate in (0, 1]")
        return float(rate)

    def raw_files(self, source: dict[str, Any]) -> Iterator[Path]:
        root = self.root / "raw"
        method = source.get("collection_method")
        if method == "huggingface_parquet":
            directory = root / "assets" / source["id"]
            yield from sorted(directory.rglob("*.parquet")) if directory.exists() else ()
        elif source.get("source_type") == "wikimedia_dump":
            for asset in source.get("assets", []):
                rel = asset.get("destination") if isinstance(asset, dict) else None
                if rel and (self.root / rel).exists():
                    yield self.root / rel
        elif method in {"mediawiki_api", "mediawiki_category_html", "sitemap_crawl"}:
            directory = root / "web" / source["id"]
            yield from sorted(directory.glob("*.jsonl.gz")) if directory.exists() else ()
        elif method == "manual_files":
            yield from manual_input_files(self.root, source)

    def _parquet_documents(self, path: Path, source: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise PreprocessError("PyArrow is required for Sangraha Parquet. Install requirements-colab.txt.") from exc
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema.names)
        text_column = "text" if "text" in columns else None
        if not text_column:
            raise PreprocessError(f"{path} has no text column")
        identity_columns = ("doc_id", "document_id", "id")
        hplt = source.get("dataset") == "HPLT/HPLT2.0_cleaned"
        # HPLT schemas have changed between releases.  Read only fields present
        # in a shard, and preserve the durable provenance fields without assuming
        # a particular Parquet schema.
        hplt_columns = (
            "url", "source_url", "crawl", "crawl_collection", "warc",
            "warc_reference", "quality", "quality_metadata", "date",
            "document_date", "language_score", "score",
        )
        wanted = [text_column] + [name for name in (*identity_columns, "type", *hplt_columns) if name in columns and name != text_column]
        row_number = 0
        for batch in parquet.iter_batches(columns=wanted, batch_size=4096):
            rows = batch.to_pylist()
            for row in rows:
                fallback_id = row_number
                row_number += 1
                text = row.get(text_column)
                if not isinstance(text, str):
                    continue
                doc_id = next(
                    (row.get(name) for name in identity_columns if row.get(name) is not None),
                    fallback_id,
                )
                record: dict[str, Any] = {"document_id": str(doc_id), "text": text, "document_type": str(row.get("type", "dataset_document"))}
                if hplt:
                    provenance = {
                        key: row[key] for key in hplt_columns
                        if key in row and row[key] is not None
                    }
                    record.update({
                        "hplt_url": provenance.get("url") or provenance.get("source_url"),
                        "hplt_crawl_collection": provenance.get("crawl_collection") or provenance.get("crawl"),
                        "hplt_warc_reference": provenance.get("warc_reference") or provenance.get("warc"),
                        "hplt_quality_metadata": {
                            key: value for key, value in provenance.items()
                            if key in {"quality", "quality_metadata", "language_score", "score"}
                        },
                    })
                    if provenance.get("date") or provenance.get("document_date"):
                        record["publication_date"] = str(provenance.get("document_date") or provenance["date"])
                yield record

    def _wiki_documents(self, path: Path, source: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            import mwparserfromhell  # type: ignore
        except ImportError as exc:
            raise PreprocessError("mwparserfromhell is required for Wikimedia dumps. Install requirements-colab.txt.") from exc
        with bz2.BZ2File(path, "rb") as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if not element.tag.endswith("page"):
                    continue
                child = lambda name: next((item for item in element if item.tag.endswith(name)), None)
                ns = child("ns")
                redirect = child("redirect")
                title = child("title")
                revision = child("revision")
                text_element = next((item for item in revision if item.tag.endswith("text")), None) if revision is not None else None
                # Empty MediaWiki pages commonly use ``<text />``; ElementTree
                # exposes that as ``None`` rather than an empty string.
                raw = (text_element.text if text_element is not None else "") or ""
                page_id = child("id")
                is_disambiguation = "{{disambig" in raw.casefold() or "{{dab" in raw.casefold()
                if (ns is not None and (ns.text or "") != "0") or redirect is not None or raw.casefold().lstrip().startswith("#redirect") or is_disambiguation:
                    element.clear()
                    continue
                text = mwparserfromhell.parse(raw).strip_code(normalize=True, collapse=True)
                if text.strip():
                    yield {"document_id": (page_id.text if page_id is not None and page_id.text else title.text if title is not None else "page"), "text": text, "document_type": "wikimedia_article"}
                element.clear()

    def _response_body(self, record: dict[str, Any]) -> bytes | None:
        encoded = record.get("raw_body_base64")
        if not isinstance(encoded, str):
            return None
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None

    def _response_text(self, record: dict[str, Any]) -> str | None:
        raw_html = record.get("raw_html")
        if isinstance(raw_html, str):
            return raw_html
        body = self._response_body(record)
        if body is None:
            return None
        headers = record.get("headers", {})
        content_type = next(
            (value for key, value in headers.items() if str(key).casefold() == "content-type"),
            "",
        ) if isinstance(headers, dict) else ""
        charset = re.search(r"charset=([A-Za-z0-9._-]+)", str(content_type), re.IGNORECASE)
        encoding = charset.group(1) if charset else "utf-8"
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            return body.decode("utf-8", "replace")

    def _web_documents(self, path: Path, source: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for index, record in enumerate(read_jsonl_gzip(path)):
            if record.get("job_kind") not in {None, "article", "pib_article"}:
                continue
            raw_html = self._response_text(record)
            if raw_html is None:
                continue
            if raw_html.lstrip().casefold().startswith(("<?xml", "<urlset", "<sitemapindex")):
                continue
            text, strategy = extract_html_text(raw_html, source)
            if text:
                metadata = record.get("manual_metadata")
                provenance = metadata if isinstance(metadata, dict) else {}
                yield {
                    "document_id": str(record.get("url", index)),
                    "text": text,
                    "document_type": "crawled_web_article",
                    "extraction_method": strategy,
                    "attribution_url": str(record.get("url", "")),
                    **{key: value for key, value in provenance.items() if value is not None},
                }

    def _mediawiki_documents(self, path: Path, source: dict[str, Any]) -> Iterator[dict[str, Any]]:
        raw_file = str(path.resolve().relative_to(self.root.resolve()))
        accepted = rejected = malformed = 0
        rejection_reasons: dict[str, int] = {}
        for record in read_jsonl_gzip(path):
            if record.get("job_kind") != "mediawiki_page":
                continue
            metadata = record.get("manual_metadata")
            if not isinstance(metadata, dict):
                malformed += 1
                continue
            if metadata.get("policy_decision") != "accepted":
                rejected += 1
                reason = str(metadata.get("policy_reason", "missing_policy_decision"))
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue
            body = self._response_body(record)
            if body is None:
                malformed += 1
                continue
            try:
                page = page_content(body)
            except Exception as exc:
                raise PreprocessError(f"malformed accepted MediaWiki response in {raw_file}: {exc}") from exc
            if page.get("rendered_html"):
                render_source = {
                    **source,
                    "content_selectors": source.get("content_selectors", [".mw-parser-output"]),
                }
                text, extraction_method = extract_html_text(page["rendered_html"], render_source)
                extraction_method = f"mediawiki_rendered_{extraction_method}"
            else:
                try:
                    import mwparserfromhell  # type: ignore
                except ImportError as exc:
                    raise PreprocessError(
                        "mwparserfromhell is required for legacy MediaWiki responses. Install requirements-colab.txt."
                    ) from exc
                text = mwparserfromhell.parse(page["wikitext"]).strip_code(normalize=True, collapse=True).strip()
                extraction_method = "mediawiki_revision_wikitext"
            if not text:
                malformed += 1
                continue
            accepted += 1
            yield {
                "document_id": str(page["page_id"]),
                "text": text,
                "document_type": "manually_collected_mediawiki_page",
                "extraction_method": extraction_method,
                **metadata,
            }
        audit = {
            "source_id": source["id"],
            "language": source["language"],
            "raw_file": raw_file,
            "accepted_by_item_policy": accepted,
            "rejected_by_item_policy": rejected,
            "malformed_or_empty": malformed,
            "rejection_reasons": rejection_reasons,
        }
        audit_name = sha256_bytes(raw_file.encode())[:16] + ".json"
        audit_path = self.root / "audit" / self.run_id / "manual-policy" / source["language"] / source["id"] / audit_name
        atomic_write_text(audit_path, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        self.progress.emit(
            f"extract-policy: {source['id']}: {accepted:,} accepted, {rejected:,} rights-excluded, "
            f"{malformed:,} malformed/empty in {path.name}"
        )

    def _mediawiki_html_documents(self, path: Path, source: dict[str, Any]) -> Iterator[dict[str, Any]]:
        raw_file = str(path.resolve().relative_to(self.root.resolve()))
        accepted = rejected = malformed = 0
        rejection_reasons: dict[str, int] = {}
        for index, record in enumerate(read_jsonl_gzip(path)):
            if record.get("job_kind") != "mediawiki_html_page":
                continue
            metadata = record.get("manual_metadata")
            if not isinstance(metadata, dict):
                malformed += 1
                continue
            if metadata.get("policy_decision") != "accepted":
                rejected += 1
                reason = str(metadata.get("policy_reason", "missing_policy_decision"))
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue
            raw_html = self._response_text(record)
            if raw_html is None:
                malformed += 1
                continue
            text, strategy = extract_html_text(raw_html, source)
            if not text:
                malformed += 1
                continue
            accepted += 1
            yield {
                "document_id": str(metadata.get("mediawiki_page_id") or record.get("url", index)),
                "text": text,
                "document_type": "manually_collected_mediawiki_page",
                "extraction_method": f"mediawiki_html_{strategy}",
                **metadata,
            }
        audit = {
            "source_id": source["id"],
            "language": source["language"],
            "raw_file": raw_file,
            "accepted_by_item_policy": accepted,
            "rejected_by_item_policy": rejected,
            "malformed_or_empty": malformed,
            "rejection_reasons": rejection_reasons,
        }
        audit_name = sha256_bytes(raw_file.encode())[:16] + ".json"
        audit_path = self.root / "audit" / self.run_id / "manual-policy" / source["language"] / source["id"] / audit_name
        atomic_write_text(audit_path, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        self.progress.emit(
            f"extract-policy: {source['id']}: {accepted:,} accepted, {rejected:,} rights-excluded, "
            f"{malformed:,} malformed/empty in {path.name}"
        )

    def _manual_documents(self, path: Path, source: dict[str, Any], raw_sha: str | None = None) -> Iterator[dict[str, Any]]:
        raw_file = str(path.resolve().relative_to(self.root.resolve()))
        audit: dict[str, Any] = {
            "source_id": source["id"],
            "language": source["language"],
            "raw_file": raw_file,
        }
        ocr_configuration = json.dumps(
            {
                "ocr_mode": source.get("ocr_mode", "off"),
                "ocr_languages": source.get("ocr_languages"),
                "ocr_dpi": source.get("ocr_dpi", 300),
                "ocr_psm": source.get("ocr_psm", 3),
            },
            sort_keys=True,
        )
        checkpoint_key = sha256_bytes(f"{raw_sha or raw_file}:{ocr_configuration}".encode())[:16]
        ocr_checkpoint_directory = (
            self.root
            / "audit"
            / self.run_id
            / "ocr-checkpoints"
            / source["language"]
            / source["id"]
            / checkpoint_key
        )

        def ocr_progress(page_number: int, page_total: int, status: str) -> None:
            if page_number == 1 or page_number == page_total or page_number % 10 == 0:
                self.progress.emit(
                    f"ocr: {raw_file}: page {page_number:,} / {page_total:,} {status}"
                )

        yield from manual_document_records(
            path,
            source,
            audit,
            ocr_checkpoint_directory=ocr_checkpoint_directory,
            ocr_progress=ocr_progress,
        )
        audit_name = sha256_bytes(raw_file.encode())[:16] + ".json"
        audit_path = self.root / "audit" / self.run_id / "manual-extraction" / source["language"] / source["id"] / audit_name
        atomic_write_text(audit_path, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        if audit.get("ocr_candidate_pages"):
            self.progress.emit(
                f"extract: {audit['raw_file']}: {audit['ocr_candidate_pages']:,} / "
                f"{audit.get('pdf_pages', 0):,} OCR candidates; "
                f"{audit.get('ocr_completed_pages', 0):,} newly OCRed, "
                f"{audit.get('ocr_reused_pages', 0):,} resumed, "
                f"{audit.get('ocr_failed_pages', 0):,} below target-script threshold"
            )

    def extracted_documents(self, path: Path, source: dict[str, Any], raw_sha: str | None = None) -> Iterator[dict[str, Any]]:
        if path.suffix == ".parquet":
            yield from self._parquet_documents(path, source)
        elif path.name.endswith(".xml.bz2"):
            yield from self._wiki_documents(path, source)
        elif path.name.endswith(".jsonl.gz"):
            if source.get("collection_method") == "manual_files":
                yield from self._manual_documents(path, source, raw_sha)
            elif source.get("collection_method") == "mediawiki_api":
                yield from self._mediawiki_documents(path, source)
            elif source.get("collection_method") == "mediawiki_category_html":
                yield from self._mediawiki_html_documents(path, source)
            else:
                yield from self._web_documents(path, source)
        elif source.get("collection_method") == "manual_files":
            yield from self._manual_documents(path, source, raw_sha)
        else:
            raise PreprocessError(f"unsupported raw input {path}")

    def task_output(self, stage: str, language: str, source_id: str, raw_sha: str) -> Path:
        directory = "candidates" if stage == "extract" else "clean"
        return Path("processed") / self.run_id / language / directory / f"{source_id}-{raw_sha[:16]}.jsonl.gz"

    def extract(self, sources: list[dict[str, Any]]) -> int:
        done = 0
        for source in sources:
            manual = is_manual_source(source, self.config_root)
            preselection_rate = self._preselection_rate(source)
            for path in self.raw_files(source):
                raw_ref = str(path.resolve().relative_to(self.root.resolve()))
                configurable_extraction = source.get("collection_method") in {
                    "manual_files",
                    "mediawiki_api",
                    "mediawiki_category_html",
                    "sitemap_crawl",
                } or preselection_rate is not None
                # Raw downloads are immutable and pull.py already verifies them.
                # On a restart, consulting the durable extraction task prevents a
                # multi-gigabyte SHA-256 scan solely to discover a completed shard.
                existing = self.state.existing_task(self.run_id, "extract", source["id"], raw_ref)
                if not configurable_extraction and existing and existing["status"] == "complete" and (self.root / existing["output_path"]).exists():
                    self.progress.emit(f"extract: skipping completed {raw_ref}")
                    continue
                raw_sha = sha256_file(path)
                input_sha = raw_sha
                if configurable_extraction:
                    extraction_configuration = json.dumps(
                        {
                            "version": 5 if source.get("collection_method") == "manual_files" else 4,
                            "collection_method": source.get("collection_method"),
                            "content_selectors": source.get("content_selectors", []),
                            "excluded_content_selectors": source.get("excluded_content_selectors", []),
                            "manual_chunk_characters": source.get("manual_chunk_characters"),
                            "ocr_mode": source.get("ocr_mode", "off"),
                            "ocr_languages": source.get("ocr_languages"),
                            "ocr_dpi": source.get("ocr_dpi", 300),
                            "ocr_psm": source.get("ocr_psm", 3),
                            "mediawiki_discovery": source.get("mediawiki_discovery"),
                            "mediawiki_category": source.get("mediawiki_category"),
                            "excluded_templates": source.get("excluded_templates", []),
                            "required_license_templates": source.get("required_license_templates", []),
                            "required_license_categories": source.get("required_license_categories", []),
                            "excluded_categories": source.get("excluded_categories", []),
                            "license_expression": source.get("license_expression"),
                            "attribution_text": source.get("attribution_text"),
                            "hplt_preselection_rate": preselection_rate,
                        },
                        sort_keys=True,
                    )
                    input_sha = sha256_bytes(f"{raw_sha}:{extraction_configuration}".encode())
                output = self.task_output("extract", source["language"], source["id"], input_sha)
                task = self.state.task(self.run_id, "extract", source["id"], raw_ref, input_sha, str(output))
                if task["status"] == "complete" and (self.root / output).exists():
                    self.progress.emit(f"extract: skipping completed {raw_ref}")
                    continue
                try:
                    self.progress.emit(f"extract: starting {raw_ref}")
                    extracted = preselected = 0
                    def records() -> Iterator[dict[str, Any]]:
                        nonlocal extracted, preselected
                        for number, item in enumerate(self.extracted_documents(path, source, raw_sha), 1):
                            extracted += 1
                            self.progress.tick("extract", raw_ref, number)
                            if preselection_rate is not None:
                                selection_item = {
                                    **item,
                                    "source_id": source["id"],
                                    "raw_file": raw_ref,
                                }
                                if not self._selected_for_budget(selection_item, source["language"], preselection_rate):
                                    continue
                                preselected += 1
                                item = {
                                    **item,
                                    "budget_preselected": True,
                                    "budget_selection_rate": preselection_rate,
                                }
                            manual_provenance = {}
                            if manual:
                                manual_provenance = {
                                    "license_expression": source.get("license_expression"),
                                    "attribution_text": source.get("attribution_text"),
                                    "attribution_url": source.get("url"),
                                }
                            yield {
                                **manual_provenance,
                                **item,
                                "source_id": source["id"],
                                "source_type": source.get("source_type", "unknown"),
                                "raw_file": raw_ref,
                            }
                    count = atomic_gzip_jsonl(self.root / output, records())
                    self.state.complete_task(task, count)
                    if preselection_rate is not None:
                        sampling_path = self.root / "audit" / self.run_id / "extraction-sampling" / source["language"] / f"{output.stem}.json"
                        atomic_write_text(sampling_path, json.dumps({
                            "source_id": source["id"],
                            "raw_file": raw_ref,
                            "input_records": extracted,
                            "candidate_records": count,
                            "budget_skipped_before_candidates": extracted - preselected,
                            "sample_rate": preselection_rate,
                            "selection_key": "selection-v1",
                        }, indent=2, sort_keys=True))
                    self.remove_replaced_output(existing["output_path"] if existing else None, str(output))
                    self.progress.emit(f"extract: complete {raw_ref}: {count:,} candidate records")
                    done += 1
                except Exception as exc:
                    self.state.fail_task(task, exc)
                    self.progress.emit(f"extract: failed {raw_ref}: {type(exc).__name__}: {exc}")
                    raise
        return done
