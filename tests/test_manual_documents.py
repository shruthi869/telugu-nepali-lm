"""Fixtures for policy-gated, low-OCR manual document extraction."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import preprocess
from lma_pipeline.html_extract import extract_html_text
from lma_pipeline.manual_documents import manual_document_records, manual_input_directory, manual_input_files


TELUGU = "కఖగఘఙచఛజఝఞ" * 80


class ManualDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "data"
        self.root.mkdir()
        self.config_dir = self.base / "configs"
        policy_dir = self.config_dir / "policy_snapshots"
        policy_dir.mkdir(parents=True)
        (policy_dir / "manual.md").write_text(
            "Dated public-domain rights review for these exact manually selected documents.",
            encoding="utf-8",
        )
        self.source = {
            "id": "manual_tel",
            "approved": True,
            "language": "tel",
            "source_type": "manually_curated_documents",
            "collection_method": "manual_files",
            "input_directory": "raw/manual/manual_tel",
            "manual_chunk_characters": 5_000,
            "manual_collection": True,
            "rights_basis": "public_domain",
            "license_expression": "Public domain",
            "attribution_text": "Credit the fixture author and title.",
            "policy_snapshot": "configs/policy_snapshots/manual.md",
        }
        self.config = self.config_dir / "sources.json"
        self.config.write_text(json.dumps({"sources": [self.source]}), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_html_prefers_configured_or_structured_article_content(self) -> None:
        configured = {**self.source, "content_selectors": [".story-body"]}
        html = "<nav>menu</nav><div class='story-body'>selected story</div><article>semantic story</article>"
        text, strategy = extract_html_text(html, configured)
        self.assertEqual((text, strategy), ("selected story", "configured_selector"))

        configured["excluded_content_selectors"] = [".license-box", ".noprint"]
        html = (
            "<div class='story-body'>kept opening"
            "<table class='license-box'><tr><td>long English legal notice</td></tr></table>"
            "<span class='noprint'>hidden navigation</span>kept ending</div>"
        )
        text, strategy = extract_html_text(html, configured)
        self.assertEqual((text, strategy), ("kept opening\nkept ending", "configured_selector"))

        pib = {**self.source, "content_selectors": ["#ContentPlaceHolder1_StoryBody"], "excluded_content_selectors": [".third-party-content", ".twitter-tweet"]}
        html = "<div id='ContentPlaceHolder1_StoryBody'>PIB authored <div class='third-party-content'>licensed elsewhere</div><blockquote class='twitter-tweet'>embed</blockquote> ending</div>"
        text, strategy = extract_html_text(html, pib)
        self.assertEqual((text, strategy), ("PIB authored\nending", "configured_selector"))

        json_ld = """<script type="application/ld+json">{"articleBody":"structured story"}</script><article>fallback</article>"""
        text, strategy = extract_html_text(json_ld, self.source)
        self.assertEqual((text, strategy), ("structured story", "json_ld_article_body"))

    def test_web_extraction_reads_compact_and_legacy_response_records(self) -> None:
        path = self.root / "responses.jsonl.gz"
        html = f"<nav>menu</nav><article>{TELUGU}</article>"
        preprocess.atomic_gzip_jsonl(path, [
            {
                "job_kind": "article",
                "url": "https://example.org/new",
                "headers": {"Content-Type": "text/html; charset=utf-8"},
                "raw_body_base64": base64.b64encode(html.encode()).decode(),
            },
            {
                "url": "https://example.org/legacy",
                "raw_html": html,
            },
        ])
        processor = preprocess.Preprocessor(self.root, self.config, "manual-fixture")
        records = list(processor._web_documents(path, self.source))
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["text"] == TELUGU for record in records))
        processor.close()

    def test_mediawiki_extraction_enforces_item_policy_and_keeps_attribution(self) -> None:
        path = self.root / "wiki.jsonl.gz"

        def response(page_id: int, decision: str, template: str = "") -> dict:
            body = {
                "query": {
                    "pages": [{
                        "pageid": page_id,
                        "title": f"Page {page_id}",
                        "fullurl": f"https://example.org/wiki/{page_id}",
                        "revisions": [{
                            "revid": page_id * 10,
                            "timestamp": "2026-08-15T00:00:00Z",
                            "slots": {"main": {"content": template + TELUGU}},
                        }],
                    }]
                }
            }
            return {
                "job_kind": "mediawiki_page",
                "raw_body_base64": base64.b64encode(json.dumps(body, ensure_ascii=False).encode()).decode(),
                "manual_metadata": {
                    "policy_decision": decision,
                    "policy_reason": "approved_source_policy" if decision == "accepted" else "excluded_template:non-free dli",
                    "mediawiki_page_id": page_id,
                    "mediawiki_title": f"Page {page_id}",
                    "mediawiki_revision_id": page_id * 10,
                    "mediawiki_revision_timestamp": "2026-08-15T00:00:00Z",
                    "license_templates": [template] if template else [],
                    "license_expression": "CC-BY-SA-4.0",
                    "attribution_url": f"https://example.org/wiki/{page_id}",
                    "attribution_text": "Credit contributors.",
                },
            }

        preprocess.atomic_gzip_jsonl(path, [response(1, "accepted"), response(2, "rejected", "{{Non-free DLI}}")])
        source = {
            **self.source,
            "collection_method": "mediawiki_api",
            "api_url": "https://example.org/w/api.php",
            "allowed_domains": ["example.org"],
            "allowed_paths": ["/w/api.php"],
            "pilot_max_documents": 10,
            "license_expression": "CC-BY-SA-4.0",
        }
        processor = preprocess.Preprocessor(self.root, self.config, "manual-fixture")
        parser = types.ModuleType("mwparserfromhell")

        class Parsed:
            def __init__(self, text: str):
                self.text = text

            def strip_code(self, **_kwargs: object) -> str:
                return self.text.replace("{{Non-free DLI}}", "")

        parser.parse = lambda text: Parsed(text)  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"mwparserfromhell": parser}):
            records = list(processor._mediawiki_documents(path, source))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["document_id"], "1")
        self.assertEqual(records[0]["attribution_url"], "https://example.org/wiki/1")
        audit_path = next((self.root / "audit" / "manual-fixture" / "manual-policy").rglob("*.json"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["rejected_by_item_policy"], 1)
        processor.close()

    def test_mediawiki_html_extraction_uses_rendered_content_and_policy(self) -> None:
        path = self.root / "wiki-html.jsonl.gz"
        html = f"<nav>menu</nav><div class='mw-parser-output'><p>{TELUGU}</p></div>"
        records = []
        for page_id, decision in ((1, "accepted"), (2, "rejected")):
            records.append({
                "job_kind": "mediawiki_html_page",
                "url": f"https://example.org/wiki/{page_id}",
                "headers": {"Content-Type": "text/html; charset=utf-8"},
                "raw_body_base64": base64.b64encode(html.encode()).decode(),
                "manual_metadata": {
                    "policy_decision": decision,
                    "policy_reason": "approved_category_and_item_policy" if decision == "accepted" else "excluded_rights_marker",
                    "mediawiki_page_id": page_id,
                    "mediawiki_revision_id": page_id * 10,
                    "license_categories": ["pd india"],
                    "matched_required_license_categories": ["pd india"],
                    "license_expression": "Public domain; contributor edits CC-BY-SA-4.0",
                    "attribution_url": f"https://example.org/wiki/{page_id}",
                    "attribution_text": "Credit contributors.",
                },
            })
        preprocess.atomic_gzip_jsonl(path, records)
        source = {
            **self.source,
            "collection_method": "mediawiki_category_html",
            "content_selectors": [".mw-parser-output"],
            "excluded_content_selectors": [".licensetpl"],
        }
        processor = preprocess.Preprocessor(self.root, self.config, "manual-fixture")
        extracted = list(processor._mediawiki_html_documents(path, source))
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["text"], TELUGU)
        self.assertEqual(extracted[0]["mediawiki_revision_id"], 10)
        processor.close()

    def test_manual_txt_and_html_files_use_resumable_extraction_tasks(self) -> None:
        directory = self.root / "raw" / "manual" / "manual_tel"
        directory.mkdir(parents=True)
        (directory / "book.txt").write_text(TELUGU, encoding="utf-8")
        (directory / "article.html").write_text(
            f"<nav>MENU_SHOULD_BE_REMOVED</nav><article>{TELUGU}</article>", encoding="utf-8"
        )
        processor = preprocess.Preprocessor(
            self.root,
            self.config,
            "manual-fixture",
            language_predictor=lambda text: ("te", 0.99),
            sample_rate=0.000001,
        )
        sources = processor.selected_sources(None)
        self.assertTrue(preprocess.is_manual_source(sources[0], self.config.parent.parent))
        self.assertEqual(processor.extract(sources), 2)
        task_rows = list(processor.state.conn.execute(
            "SELECT status,records FROM tasks WHERE run_id='manual-fixture' AND stage='extract'"
        ))
        self.assertEqual([(row["status"], row["records"]) for row in task_rows], [("complete", 1), ("complete", 1)])
        candidates = [record for path in processor.candidate_paths("tel") for record in preprocess.read_jsonl_gzip(path)]
        self.assertEqual(len(candidates), 2)
        html_record = next(record for record in candidates if record["document_type"] == "manual_html_document")
        self.assertEqual(html_record["extraction_method"], "semantic_article")
        self.assertNotIn("MENU_SHOULD_BE_REMOVED", html_record["text"])
        audits = list((self.root / "audit" / "manual-fixture" / "manual-extraction").rglob("*.json"))
        self.assertEqual(len(audits), 2)
        self.assertEqual(processor.clean(sources), 2)
        cleaned = [record for path in processor.clean_paths("tel") for record in preprocess.read_jsonl_gzip(path)]
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(all(record["manual_collection"] for record in cleaned))
        self.assertTrue(all(record["rights_basis"] == "public_domain" for record in cleaned))
        self.assertTrue(all(record["license_expression"] == "Public domain" for record in cleaned))
        report = json.loads(
            (self.root / "audit" / "manual-fixture" / "manual-yield" / "tel" / "manual_tel.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["accepted_documents"], 2)
        self.assertGreater(report["accepted_whitespace_tokens"], 0)
        processor.close()

    def test_epub_uses_spine_order_and_extracts_chapters(self) -> None:
        path = self.root / "book.epub"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "META-INF/container.xml",
                "<container><rootfiles><rootfile full-path='OPS/content.opf'/></rootfiles></container>",
            )
            archive.writestr(
                "OPS/content.opf",
                "<package><manifest><item id='two' href='two.xhtml'/><item id='one' href='one.xhtml'/></manifest>"
                "<spine><itemref idref='one'/><itemref idref='two'/></spine></package>",
            )
            archive.writestr("OPS/one.xhtml", f"<article>one {TELUGU}</article>")
            archive.writestr("OPS/two.xhtml", f"<article>two {TELUGU}</article>")
        audit: dict = {}
        records = list(manual_document_records(path, self.source, audit))
        self.assertEqual([record["epub_member"] for record in records], ["OPS/one.xhtml", "OPS/two.xhtml"])
        self.assertEqual(audit["records"], 2)

    def test_manual_rule_change_replaces_only_its_authoritative_checkpoint(self) -> None:
        directory = self.root / "raw" / "manual" / "manual_tel"
        directory.mkdir(parents=True)
        (directory / "book.txt").write_text(TELUGU, encoding="utf-8")
        first = preprocess.Preprocessor(self.root, self.config, "manual-fixture")
        first.extract(first.selected_sources(None))
        old_paths = first.candidate_paths("tel")
        self.assertEqual(len(old_paths), 1)
        first.close()

        changed = {**self.source, "manual_chunk_characters": 6_000}
        self.config.write_text(json.dumps({"sources": [changed]}), encoding="utf-8")
        restarted = preprocess.Preprocessor(self.root, self.config, "manual-fixture")
        restarted.extract(restarted.selected_sources(None))
        new_paths = restarted.candidate_paths("tel")
        self.assertEqual(len(new_paths), 1)
        self.assertNotEqual(old_paths, new_paths)
        self.assertFalse(old_paths[0].exists())
        self.assertTrue(new_paths[0].exists())
        restarted.close()

    def test_pdf_extracts_text_layer_and_flags_only_scan_candidates(self) -> None:
        path = self.root / "book.pdf"
        path.write_bytes(b"fixture")

        class Page:
            def __init__(self, text: str):
                self.text = text
            def extract_text(self) -> str:
                return self.text

        class PdfReader:
            def __init__(self, path: Path):
                self.pages = [Page(TELUGU), Page("image-only cover")]

        pypdf = types.ModuleType("pypdf")
        pypdf.PdfReader = PdfReader  # type: ignore[attr-defined]
        audit: dict = {}
        with patch.dict(sys.modules, {"pypdf": pypdf}):
            records = list(manual_document_records(path, self.source, audit))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["extraction_method"], "embedded_pdf_text")
        self.assertEqual(audit["pdf_pages"], 2)
        self.assertEqual(audit["ocr_candidate_pages"], 1)

    def test_pdf_ocr_uses_telugu_and_resumes_from_page_text_checkpoint(self) -> None:
        path = self.root / "scanned-book.pdf"
        path.write_bytes(b"fixture")

        class EmbeddedPage:
            def extract_text(self) -> str:
                return "image-only scan"

        class PdfReader:
            def __init__(self, path: Path):
                self.pages = [EmbeddedPage()]

        class Image:
            def close(self) -> None:
                pass

        class Bitmap:
            def to_pil(self) -> Image:
                return Image()
            def close(self) -> None:
                pass

        class RenderPage:
            def render(self, scale: float) -> Bitmap:
                self.scale = scale
                return Bitmap()
            def close(self) -> None:
                pass

        class PdfDocument:
            def __init__(self, path: str):
                self.page = RenderPage()
            def __getitem__(self, index: int) -> RenderPage:
                return self.page
            def close(self) -> None:
                pass

        calls = {"ocr": 0}
        pypdf = types.ModuleType("pypdf")
        pypdf.PdfReader = PdfReader  # type: ignore[attr-defined]
        pdfium = types.ModuleType("pypdfium2")
        pdfium.PdfDocument = PdfDocument  # type: ignore[attr-defined]
        pytesseract = types.ModuleType("pytesseract")
        pytesseract.get_languages = lambda config="": ["eng", "tel"]  # type: ignore[attr-defined]

        def image_to_string(image: Image, lang: str, config: str) -> str:
            calls["ocr"] += 1
            self.assertEqual(lang, "tel+eng")
            self.assertIn("--psm 3", config)
            return TELUGU

        pytesseract.image_to_string = image_to_string  # type: ignore[attr-defined]
        source = {
            **self.source,
            "ocr_mode": "auto",
            "ocr_languages": "tel+eng",
            "ocr_dpi": 300,
            "ocr_psm": 3,
        }
        checkpoint = self.root / "audit" / "ocr-checkpoint"
        modules = {"pypdf": pypdf, "pypdfium2": pdfium, "pytesseract": pytesseract}
        with patch.dict(sys.modules, modules):
            first_audit: dict = {}
            first = list(
                manual_document_records(
                    path,
                    source,
                    first_audit,
                    ocr_checkpoint_directory=checkpoint,
                )
            )
            second_audit: dict = {}
            second = list(
                manual_document_records(
                    path,
                    source,
                    second_audit,
                    ocr_checkpoint_directory=checkpoint,
                )
            )
        self.assertEqual(first[0]["extraction_method"], "tesseract_ocr")
        self.assertEqual(first[0]["text"], TELUGU)
        self.assertEqual(second[0]["text"], TELUGU)
        self.assertEqual(first_audit["ocr_completed_pages"], 1)
        self.assertEqual(second_audit["ocr_reused_pages"], 1)
        self.assertEqual(calls["ocr"], 1)

    def test_manual_directory_cannot_escape_raw_manual(self) -> None:
        source = {**self.source, "input_directory": "../outside"}
        with self.assertRaises(preprocess.PreprocessError):
            manual_input_directory(self.root, source)

    def test_manual_directory_rejects_symlinks(self) -> None:
        directory = self.root / "raw" / "manual" / "manual_tel"
        directory.mkdir(parents=True)
        outside = self.base / "outside.txt"
        outside.write_text(TELUGU, encoding="utf-8")
        (directory / "outside.txt").symlink_to(outside)
        with self.assertRaisesRegex(preprocess.PreprocessError, "symlink"):
            list(manual_input_files(self.root, self.source))

    def test_approved_manual_source_requires_declared_rights_basis(self) -> None:
        source = {**self.source}
        source.pop("rights_basis")
        self.config.write_text(json.dumps({"sources": [source]}), encoding="utf-8")
        processor = preprocess.Preprocessor(self.root, self.config, "manual-fixture")
        with self.assertRaisesRegex(preprocess.PreprocessError, "rights_basis"):
            processor.selected_sources(None)
        processor.close()


if __name__ == "__main__":
    unittest.main()
