"""Local HTTP fixture coverage for the collector's recovery-critical paths."""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pull


PAYLOAD = b"0123456789abcdefghijklmnopqrstuvwxyz"


class FixtureHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str | None]] = []
    failures = 0

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP handler API
        path = self.path.split("?", 1)[0]
        FixtureHandler.requests.append((path, self.headers.get("Range")))
        if path == "/robots.txt":
            self._send(200, b"User-agent: *\nAllow: /\n")
        elif path == "/flaky":
            if FixtureHandler.failures == 0:
                FixtureHandler.failures += 1
                self._send(503, b"later", {"Retry-After": "120"})
            else:
                self._send(200, PAYLOAD, {"ETag": '"fixture"'})
        elif path == "/asset":
            start = 0
            range_header = self.headers.get("Range")
            if range_header:
                start = int(range_header.removeprefix("bytes=").removesuffix("-"))
                self._send(206, PAYLOAD[start:], {"Content-Range": f"bytes {start}-{len(PAYLOAD)-1}/{len(PAYLOAD)}", "ETag": '"fixture"'})
            else:
                self._send(200, PAYLOAD, {"ETag": '"fixture"'})
        elif path == "/interrupted":
            range_header = self.headers.get("Range")
            if range_header:
                start = int(range_header.removeprefix("bytes=").removesuffix("-"))
                self._send(206, PAYLOAD[start:], {"Content-Range": f"bytes {start}-{len(PAYLOAD)-1}/{len(PAYLOAD)}", "ETag": '"fixture"'})
            else:
                # Advertise the full object but cut this connection short. The
                # downloader must retain the bytes already streamed to .part.
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.send_header("ETag", '"fixture"')
                self.end_headers()
                self.wfile.write(PAYLOAD[:10])
                self.wfile.flush()
                self.close_connection = True
        elif path == "/sitemap.xml":
            self._send(200, b"<sitemapindex><sitemap><loc>/child.xml</loc></sitemap></sitemapindex>")
        elif path == "/pilot-sitemap.xml":
            self._send(200, b"<urlset><url><loc>/news/a</loc></url><url><loc>/news/b</loc></url></urlset>")
        elif path == "/pib-rss":
            self._send(200, b"<rss><channel><item><link>/PressReleasePage.aspx?PRID=42&amp;lang=1</link></item></channel></rss>")
        elif path == "/PressReleasePage.aspx":
            language = parse_qs(urlsplit(self.path).query).get("lang", [""])[0]
            if language == "1":
                self._send(200, b'<html><body><a href="/PressReleasePage.aspx?PRID=42&amp;lang=16">Telugu</a></body></html>')
            else:
                self._send(200, "<html><body><div id='ContentPlaceHolder1_StoryBody'>కఖగఘఙచఛజఝఞ</div></body></html>".encode())
        elif path == "/child.xml":
            self._send(200, b"<urlset><url><loc>/news/a</loc></url><url><loc>/news/a</loc></url><url><loc>/other</loc></url></urlset>")
        elif path == "/news/a":
            self._send(200, b"<html><body>article</body></html>")
        elif path == "/news/b":
            self._send(200, b"<html><body>second article</body></html>")
        elif path == "/wiki/Category:PD_India":
            self._send(
                200,
                b'<html><body><div id="mw-pages"><a href="/wiki/Open_page">Open</a>'
                b'<a href="/wiki/Restricted_page">Restricted</a></div></body></html>',
                {"Content-Type": "text/html; charset=utf-8"},
            )
        elif path in {"/wiki/Open_page", "/wiki/Restricted_page"}:
            restricted = path.endswith("Restricted_page")
            categories = '["PD India"]'
            marker = "<div>Non-free DLI</div>" if restricted else ""
            html = (
                '<html><head><script>mw.config.set({"wgRevisionId":901,"wgArticleId":90,'
                f'"wgCategories":{categories}}});</script></head><body>'
                f"{marker}<div class='mw-parser-output'><p>{'కఖగఘఙచఛజఝఞ ' * 100}</p></div></body></html>"
            )
            self._send(200, html.encode(), {"Content-Type": "text/html; charset=utf-8"})
        elif path == "/w/api.php":
            query = parse_qs(urlsplit(self.path).query)
            if query.get("list") == ["allpages"]:
                payload = {
                    "query": {
                        "allpages": [
                            {"pageid": 101, "title": "Open page"},
                            {"pageid": 102, "title": "Restricted page"},
                            {"pageid": 103, "title": "Later page"},
                        ]
                    }
                }
            else:
                page_id = int(query["pageid"][0])
                restricted = page_id == 102
                prefix = "{{Non-free DLI}}\n" if restricted else "{{PD-India}}\n"
                templates = [{"title": "Template:Non-free DLI" if restricted else "Template:PD-India"}]
                payload = {
                    "parse": {
                        "pageid": page_id,
                        "title": f"Page {page_id}",
                        "revid": page_id * 10,
                        "wikitext": prefix + "కఖగఘఙచఛజఝఞ " * 100,
                        "text": "<div class='mw-parser-output'><p>" + "కఖగఘఙచఛజఝఞ " * 100 + "</p></div>",
                        "templates": templates,
                    }
                }
            self._send(200, json.dumps(payload, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
        else:
            self._send(404, b"missing")


class Server:
    def __enter__(self) -> "Server":
        FixtureHandler.requests = []
        FixtureHandler.failures = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        self.server.base_url = f"http://127.0.0.1:{self.server.server_port}"  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


class CollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "data"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, sources: list[dict]) -> Path:
        path = Path(self.temp.name) / "sources.json"
        path.write_text(json.dumps({"sources": sources}), encoding="utf-8")
        return path

    def collector(self, config: Path) -> pull.Collector:
        self.root.mkdir(parents=True, exist_ok=True)
        return pull.Collector(self.root, config, timeout=2)

    def direct_source(self, base: str, url: str = "/asset") -> dict:
        return {
            "id": "asset",
            "approved": True,
            "language": "tel",
            "source_type": "fixture",
            "collection_method": "direct_assets",
            "assets": [{"url": base + url, "destination": "raw/assets/asset.bin", "expected_bytes": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest()}],
            "crawl_delay": 0,
            "allowed_paths": ["/"],
            "policy_snapshot": "fixture",
        }

    def test_interrupted_range_download_resumes_from_part_file(self) -> None:
        with Server() as server:
            collector = self.collector(self.config([self.direct_source(server.base_url)]))
            collector.schedule()
            job = collector.ledger.runnable()[0]
            partial = self.root / "raw/assets/asset.bin.part"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(PAYLOAD[:10])
            collector.ledger.progress(job["id"], 10, '"fixture"', len(PAYLOAD))
            collector.process_job(job)
            self.assertEqual((self.root / "raw/assets/asset.bin").read_bytes(), PAYLOAD)
            self.assertFalse(partial.exists())
            self.assertIn(("/asset", "bytes=10-"), FixtureHandler.requests)
            self.assertEqual(collector.ledger.job(job["id"])["status"], "complete")
            collector.close()

    def test_real_connection_drop_keeps_streamed_bytes_for_range_resume(self) -> None:
        with Server() as server:
            collector = self.collector(self.config([self.direct_source(server.base_url, "/interrupted")]))
            collector.schedule()
            job = collector.ledger.runnable()[0]
            collector.process_job(job)
            partial = self.root / "raw/assets/asset.bin.part"
            self.assertEqual(partial.read_bytes(), PAYLOAD[:10])
            self.assertEqual(collector.ledger.job(job["id"])["status"], "deferred")
            # A future normal run will pick this job up; call it directly here so
            # the test does not wait for the backoff time.
            collector.process_job(collector.ledger.job(job["id"]))
            self.assertEqual((self.root / "raw/assets/asset.bin").read_bytes(), PAYLOAD)
            self.assertIn(("/interrupted", "bytes=10-"), FixtureHandler.requests)
            collector.close()

    def test_corrupt_complete_partial_is_restarted_after_checksum_check(self) -> None:
        with Server() as server:
            collector = self.collector(self.config([self.direct_source(server.base_url)]))
            collector.schedule()
            job = collector.ledger.runnable()[0]
            partial = self.root / "raw/assets/asset.bin.part"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"x" * len(PAYLOAD))
            collector.process_job(job)
            self.assertEqual((self.root / "raw/assets/asset.bin").read_bytes(), PAYLOAD)
            self.assertIn(("/asset", None), FixtureHandler.requests)
            collector.close()

    def test_recursive_sitemap_discovers_one_deduplicated_article(self) -> None:
        with Server() as server:
            source = {
                "id": "news",
                "approved": True,
                "language": "tel",
                "source_type": "news",
                "collection_method": "sitemap_crawl",
                "url": server.base_url + "/sitemap.xml",
                "allowed_domains": ["127.0.0.1"],
                "allowed_paths": ["/news/"],
                "crawl_delay": 0,
                "robots": True,
                "policy_snapshot": "fixture",
            }
            collector = self.collector(self.config([source]))
            collector.run()
            rows = list(collector.ledger.conn.execute("SELECT kind, status, url FROM jobs ORDER BY id"))
            articles = [row for row in rows if row["kind"] == "article"]
            self.assertEqual(len(articles), 1)
            self.assertEqual(articles[0]["status"], "complete")
            self.assertTrue(list((self.root / "raw/web/news").glob("*.jsonl.gz")))
            self.assertNotIn(("/other", None), FixtureHandler.requests)
            collector.close()

    def test_sitemap_pilot_stops_and_resumes_by_completed_article_count(self) -> None:
        with Server() as server:
            source = {
                "id": "pilot_news",
                "approved": True,
                "language": "nep",
                "source_type": "news",
                "collection_method": "sitemap_crawl",
                "url": server.base_url + "/pilot-sitemap.xml",
                "allowed_domains": ["127.0.0.1"],
                "allowed_paths": ["/news/"],
                "pilot_max_documents": 1,
                "crawl_delay": 0,
                "robots": True,
                "policy_snapshot": "fixture",
            }
            collector = self.collector(self.config([source]))
            collector.run("pilot_news")
            self.assertEqual(collector.ledger.count_jobs("pilot_news", "article", ("complete",)), 1)
            self.assertEqual(collector.ledger.count_jobs("pilot_news", "article", ("pending",)), 1)
            collector.close()

            resumed = pull.Collector(self.root, self.config([source]), timeout=2, pilot_max_documents=2)
            resumed.run("pilot_news")
            self.assertEqual(resumed.ledger.count_jobs("pilot_news", "article", ("complete",)), 2)
            resumed.close()

    def test_pib_rss_follows_only_the_linked_telugu_release_with_a_page_cap(self) -> None:
        with Server() as server:
            policy = Path(self.temp.name) / "pib-policy.md"
            policy.write_text("Reviewed PIB terms permit attributed reproduction excluding third-party material.", encoding="utf-8")
            source = {
                "id": "pib_tel", "approved": True, "language": "tel", "source_type": "government_press_release",
                "collection_method": "pib_rss_crawl", "url": server.base_url + "/pib-rss",
                "allowed_domains": ["127.0.0.1"], "allowed_paths": ["/PressReleasePage.aspx"],
                "pilot_max_documents": 1, "crawl_delay": 0, "robots": False,
                "rights_basis": "written_permission", "license_expression": "PIB attribution policy",
                "attribution_text": "Credit Press Information Bureau and link the release.",
                "policy_snapshot": str(policy), "manual_collection": True,
            }
            collector = self.collector(self.config([source]))
            collector.run("pib_tel")
            self.assertEqual(collector.ledger.count_jobs("pib_tel", "pib_article", ("complete",)), 1)
            self.assertIn(("/PressReleasePage.aspx", None), FixtureHandler.requests)
            records = []
            for path in (self.root / "raw" / "web" / "pib_tel").glob("*.jsonl.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle)
            discovery = next(record for record in records if record["job_kind"] == "pib_release")
            self.assertTrue(discovery["manual_metadata"]["pib_telugu_alternate_url"].endswith("lang=16"))
            collector.close()

    def test_reconciles_batch_created_before_database_commit(self) -> None:
        with Server() as server:
            source = self.direct_source(server.base_url)
            # An article job lets this test exercise the immutable response writer.
            source.update({"id": "news", "collection_method": "sitemap_crawl", "url": server.base_url + "/sitemap.xml", "allowed_domains": ["127.0.0.1"], "allowed_paths": ["/news/"], "robots": False})
            source.pop("assets")
            config = self.config([source])
            collector = self.collector(config)
            job_id = collector.ledger.add_job("news", "article", server.base_url + "/news/a")
            job = collector.ledger.job(job_id)
            response = collector.http.fetch(server.base_url + "/news/a")
            batch_path, _ = collector._write_response_batch(job, response)
            self.assertEqual(collector.ledger.job(job_id)["status"], "pending")
            collector.close()
            restarted = self.collector(config)
            self.assertEqual(restarted.ledger.reconcile_batches(self.root), 1)
            self.assertEqual(restarted.ledger.job(job_id)["status"], "complete")
            self.assertTrue((self.root / batch_path).exists())
            restarted.close()

    def test_retry_is_deferred_and_offline_restart_does_not_forget_it(self) -> None:
        with Server() as server:
            collector = self.collector(self.config([self.direct_source(server.base_url, "/flaky")]))
            collector.schedule()
            job = collector.ledger.runnable()[0]
            collector.process_job(job)
            deferred = collector.ledger.job(job["id"])
            self.assertEqual(deferred["status"], "deferred")
            self.assertEqual(deferred["retry_count"], 1)
            self.assertGreater(deferred["retry_at"], pull.unix_now() + 100)
            collector.close()
            # No network is needed after restart: the future retry remains in the ledger.
            restarted = self.collector(self.config([self.direct_source(server.base_url, "/flaky")]))
            self.assertEqual(restarted.run(), 0)
            self.assertEqual(restarted.ledger.job(job["id"])["status"], "deferred")
            restarted.close()

    def test_response_batch_is_valid_gzip_jsonl(self) -> None:
        with Server() as server:
            source = {
                "id": "news",
                "approved": True,
                "language": "tel",
                "source_type": "news",
                "collection_method": "sitemap_crawl",
                "url": server.base_url + "/sitemap.xml",
                "allowed_domains": ["127.0.0.1"],
                "allowed_paths": ["/news/"],
                "crawl_delay": 0,
                "robots": False,
                "policy_snapshot": "fixture",
            }
            collector = self.collector(self.config([source]))
            collector.run()
            records = []
            for batch in (self.root / "raw/web/news").glob("*.jsonl.gz"):
                with gzip.open(batch, "rt", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle)
            self.assertTrue(all(record["source_id"] == "news" for record in records))
            self.assertEqual({record["job_kind"] for record in records}, {"sitemap", "article"})
            self.assertTrue(all(record["schema_version"] == 2 for record in records))
            self.assertTrue(all("raw_body_base64" in record for record in records))
            self.assertTrue(all("raw_html" not in record for record in records))
            collector.close()

    def test_article_responses_are_committed_in_one_bounded_batch(self) -> None:
        with Server() as server:
            source = {
                "id": "news",
                "approved": True,
                "language": "tel",
                "source_type": "news",
                "collection_method": "sitemap_crawl",
                "url": server.base_url + "/sitemap.xml",
                "allowed_domains": ["127.0.0.1"],
                "allowed_paths": ["/news/"],
                "crawl_delay": 0,
                "robots": False,
                "policy_snapshot": "fixture",
            }
            collector = self.collector(self.config([source]))
            job_ids = [
                collector.ledger.add_job("news", "article", server.base_url + f"/news/a?id={index}")
                for index in range(3)
            ]
            for job_id in job_ids:
                collector.process_job(collector.ledger.job(job_id))
            self.assertTrue(all(collector.ledger.job(job_id)["status"] == "in_progress" for job_id in job_ids))
            collector._flush_response_batch()
            rows = [collector.ledger.job(job_id) for job_id in job_ids]
            self.assertTrue(all(row["status"] == "complete" for row in rows))
            self.assertEqual(len({row["batch_path"] for row in rows}), 1)
            batches = list((self.root / "raw" / "web" / "news").glob("*.jsonl.gz"))
            self.assertEqual(len(batches), 1)
            with gzip.open(batches[0], "rt", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 3)
            collector.close()

    def test_gzip_sitemap_is_parsed_without_writing_an_expanded_copy(self) -> None:
        config = self.config([])
        collector = self.collector(config)
        body = gzip.compress(b"<urlset><url><loc>https://example.org/a</loc></url></urlset>")
        self.assertEqual(collector._parse_sitemap(body), ["https://example.org/a"])
        collector.close()

    def test_disabled_selected_source_does_not_process_old_queued_jobs(self) -> None:
        with Server() as server:
            source = {**self.direct_source(server.base_url), "approved": False}
            collector = self.collector(self.config([source]))
            job_id = collector.ledger.add_job("asset", "download", server.base_url + "/asset", "raw/assets/asset.bin")
            with self.assertRaises(pull.CollectorError):
                collector.run("asset")
            self.assertEqual(collector.ledger.job(job_id)["status"], "pending")
            self.assertNotIn(("/asset", None), FixtureHandler.requests)
            collector.close()

    def test_selected_asset_is_not_blocked_by_another_mediawiki_source(self) -> None:
        """Explicit pulls remain usable when bounded sources are configured."""
        with Server() as server:
            mediawiki = {
                "id": "manual_wiki_tel",
                "approved": True,
                "language": "tel",
                "source_type": "wikisource",
                "collection_method": "mediawiki_api",
                "url": server.base_url + "/wiki/Main_Page",
                "api_url": server.base_url + "/w/api.php",
                "allowed_domains": ["127.0.0.1"],
                "allowed_paths": ["/w/api.php"],
                "policy_snapshot": "fixture",
            }
            collector = self.collector(self.config([self.direct_source(server.base_url), mediawiki]))
            collector.run("asset")
            self.assertEqual((self.root / "raw/assets/asset.bin").read_bytes(), PAYLOAD)
            collector.close()

    def test_mediawiki_pilot_is_bounded_and_saves_item_policy_decisions(self) -> None:
        with Server() as server:
            policy = Path(self.temp.name) / "policy.md"
            policy.write_text(
                "Dated fixture review approving open MediaWiki pages while excluding non-free templates.",
                encoding="utf-8",
            )
            source = {
                "id": "manual_wiki_tel",
                "approved": True,
                "language": "tel",
                "source_type": "wikisource",
                "collection_method": "mediawiki_api",
                "url": server.base_url + "/wiki/Main_Page",
                "api_url": server.base_url + "/w/api.php",
                "allowed_domains": ["127.0.0.1"],
                "allowed_paths": ["/w/api.php"],
                "mediawiki_discovery": "allpages",
                "mediawiki_namespace": 0,
                "excluded_templates": ["Non-free DLI"],
                "required_license_templates": ["PD-India"],
                "pilot_max_documents": 3,
                "crawl_delay": 0,
                "robots": True,
                "rights_basis": "open_license",
                "license_expression": "CC-BY-SA-4.0",
                "attribution_text": "Credit contributors and link the page.",
                "policy_snapshot": str(policy),
                "manual_collection": True,
            }
            config = self.config([source])
            self.root.mkdir(parents=True, exist_ok=True)
            collector = pull.Collector(self.root, config, timeout=2, pilot_max_documents=2)
            collector.run("manual_wiki_tel")
            self.assertEqual(collector.ledger.count_jobs("manual_wiki_tel", "mediawiki_page", ("complete",)), 2)
            self.assertEqual(collector.ledger.count_jobs("manual_wiki_tel", "mediawiki_page", ("pending",)), 1)
            records = []
            for path in (self.root / "raw" / "web" / "manual_wiki_tel").glob("*.jsonl.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle)
            page_records = [record for record in records if record["job_kind"] == "mediawiki_page"]
            self.assertEqual(
                [record["manual_metadata"]["policy_decision"] for record in page_records],
                ["accepted", "rejected"],
            )
            self.assertIn("non-free dli", page_records[1]["manual_metadata"]["policy_reason"])
            collector.close()

    def test_mediawiki_category_html_respects_robots_and_item_categories(self) -> None:
        with Server() as server:
            policy = Path(self.temp.name) / "policy.md"
            policy.write_text(
                "Dated fixture rights review approving only pages in a public-domain category.",
                encoding="utf-8",
            )
            source = {
                "id": "manual_wiki_html_tel",
                "approved": True,
                "language": "tel",
                "source_type": "wikisource",
                "collection_method": "mediawiki_category_html",
                "url": server.base_url + "/wiki/Category:PD_India",
                "allowed_domains": ["127.0.0.1"],
                "allowed_paths": ["/wiki/"],
                "required_license_categories": ["PD India"],
                "excluded_categories": ["Non-free DLI books"],
                "excluded_templates": ["Non-free DLI"],
                "pilot_max_documents": 2,
                "crawl_delay": 0,
                "robots": True,
                "rights_basis": "open_license",
                "license_expression": "Public domain; contributor edits CC-BY-SA-4.0",
                "attribution_text": "Credit contributors and link the page.",
                "policy_snapshot": str(policy),
                "manual_collection": True,
            }
            self.root.mkdir(parents=True, exist_ok=True)
            collector = pull.Collector(self.root, self.config([source]), timeout=2)
            collector.run("manual_wiki_html_tel")
            self.assertEqual(
                collector.ledger.count_jobs("manual_wiki_html_tel", "mediawiki_html_page", ("complete",)),
                2,
            )
            records = []
            for path in (self.root / "raw" / "web" / "manual_wiki_html_tel").glob("*.jsonl.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle)
            decisions = [
                record["manual_metadata"]["policy_decision"]
                for record in records
                if record["job_kind"] == "mediawiki_html_page"
            ]
            self.assertEqual(decisions, ["accepted", "rejected"])
            collector.close()


if __name__ == "__main__":
    unittest.main()
