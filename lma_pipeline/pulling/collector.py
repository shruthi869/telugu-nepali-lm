"""Source scheduling, immutable downloads, and policy-aware crawling."""

from __future__ import annotations

import base64
import contextlib
import gzip
import io
import json
import os
import re
import sqlite3
import shutil
import sys
import time
import urllib.robotparser
import uuid
import xml.etree.ElementTree as ET
from http.client import HTTPException
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit

from ..source_policy import manual_source_issue
from .common import (
    CHECKPOINT_BYTES,
    DOWNLOAD_CHUNK_BYTES,
    PROGRESS_INTERVAL_SECONDS,
    USER_AGENT,
    CollectorError,
    Response,
    TransientError,
    canonical_url,
    dt,
    fsync_directory,
    human_size,
    safe_filename,
    sha256_file,
    utcnow,
)
from .http_client import HttpClient
from .ledger import Ledger
from .mediawiki import discovery_url, page_policy_metadata, page_url, parse_discovery
from .mediawiki_html import page_policy_metadata as html_page_policy_metadata
from .mediawiki_html import parse_category_page


MAX_SITEMAP_UNCOMPRESSED_BYTES = 60 * 1024 * 1024
RESPONSE_BATCH_RECORDS = 100
RESPONSE_BATCH_BYTES = 32 * 1024 * 1024

class Collector:
    """Schedules configured sources and processes each unfinished ledger job."""

    def __init__(
        self,
        data_root: Path,
        config_path: Path,
        timeout: float = 45.0,
        pilot_max_documents: int | None = None,
    ):
        if pilot_max_documents is not None and pilot_max_documents < 1:
            raise CollectorError("--pilot-max-documents must be at least 1")
        self.data_root = data_root
        self.config_path = config_path
        self.config_root = config_path.parent.parent
        with config_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        self.sources = {source["id"]: source for source in loaded["sources"]}
        self.ledger = Ledger(data_root / "pull_state.sqlite")
        self.http = HttpClient(timeout)
        self.pilot_max_documents = pilot_max_documents
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._pending_response_source: str | None = None
        self._pending_responses: list[dict[str, Any]] = []
        self._pending_response_bytes = 0

    def close(self) -> None:
        self.ledger.close()

    def source(self, source_id: str) -> dict[str, Any]:
        return self.sources[source_id]

    def approved_sources(self, selected: str | None) -> Iterable[dict[str, Any]]:
        if selected and selected not in self.sources:
            raise CollectorError(f"unknown source id: {selected}")
        for source in self.sources.values():
            if source.get("approved", False) and (selected is None or source["id"] == selected):
                if source.get("manual_collection"):
                    issue = manual_source_issue(source, self.config_root)
                    if issue:
                        raise CollectorError(f"manual source {source['id']} is not eligible: {issue}")
                yield source

    def _require_source_space(self, source: dict[str, Any]) -> None:
        """Fail closed before a large configured source starts downloading."""
        minimum_gib = source.get("minimum_free_gib")
        if minimum_gib is None:
            return
        if not isinstance(minimum_gib, (int, float)) or minimum_gib <= 0:
            raise CollectorError(f"source {source['id']} has invalid minimum_free_gib")
        free = shutil.disk_usage(self.data_root).free
        required = int(float(minimum_gib) * 1024**3)
        if free < required:
            raise CollectorError(
                f"refusing {source['id']}: {free / 1024**3:.1f} GiB free, but its "
                f"raw plus derived checkpoints require at least {minimum_gib:.1f} GiB free"
            )

    def schedule(self, selected: str | None = None) -> None:
        for source in self.approved_sources(selected):
            method = source["collection_method"]
            if method == "direct_assets":
                for asset in source.get("assets", []):
                    url = asset["url"] if isinstance(asset, dict) else asset
                    rel = asset.get("destination") if isinstance(asset, dict) else None
                    if not rel:
                        rel = str(Path("raw") / "assets" / source["id"] / safe_filename(url))
                    expected = asset.get("expected_bytes") if isinstance(asset, dict) else None
                    checksum = asset.get("sha256") if isinstance(asset, dict) else None
                    self.ledger.add_job(source["id"], "download", url, rel, expected, checksum)
            elif method == "huggingface_parquet":
                self.ledger.add_job(source["id"], "hf_manifest", source["manifest_url"])
            elif method == "sitemap_crawl":
                self.ledger.add_job(source["id"], "sitemap", source["url"])
            elif method == "pib_rss_crawl":
                self.ledger.add_job(source["id"], "pib_rss", source["url"])
            elif method == "mediawiki_api":
                self.ledger.add_job(source["id"], "mediawiki_list", discovery_url(source))
            elif method == "mediawiki_category_html":
                self.ledger.add_job(source["id"], "mediawiki_category", source["url"])
            elif method == "manual_files":
                # Files are deliberately placed by the student below raw/manual/;
                # no network work is scheduled for a locally curated source.
                continue
            else:
                raise CollectorError(f"unsupported collection method {method!r} for {source['id']}")

    def allowed_article_url(self, source: dict[str, Any], url: str) -> bool:
        parsed = urlsplit(url)
        if not self.allowed_source_domain(source, url):
            return False
        paths = source.get("allowed_paths", ["/"])
        return isinstance(paths, list) and any(
            isinstance(prefix, str) and prefix.startswith("/") and parsed.path.startswith(prefix)
            for prefix in paths
        )

    def allowed_source_domain(self, source: dict[str, Any], url: str) -> bool:
        parsed = urlsplit(url)
        domains = source.get("allowed_domains") or [urlsplit(source["url"]).hostname]
        if not isinstance(domains, list):
            return False
        approved = {str(domain).strip().casefold() for domain in domains if domain}
        return parsed.scheme.casefold() in {"http", "https"} and (parsed.hostname or "").casefold() in approved

    def _robots_allowed(self, source: dict[str, Any], url: str) -> bool:
        if source.get("robots", True) is False:
            return True
        parsed = urlsplit(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        parser = self.robots.get(root)
        if parser is None:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(urljoin(root, "/robots.txt"))
            try:
                reply = self.http.fetch(parser.url, delay=float(source.get("crawl_delay", 0)))
                parser.parse(reply.body.decode("utf-8", "replace").splitlines())
            except TransientError:
                raise
            except CollectorError:
                # A missing/forbidden robots response is not permission to crawl.
                return False
            self.robots[root] = parser
        return parser.can_fetch(USER_AGENT, url)

    def _response_record(
        self,
        job: sqlite3.Row,
        response: Response,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "schema_version": 2,
            "job_id": job["id"],
            "job_kind": job["kind"],
            "source_id": job["source_id"],
            "url": job["url"],
            "fetched_at": utcnow(),
            "http_status": response.status,
            "headers": response.headers,
            "raw_body_base64": base64.b64encode(response.body).decode("ascii"),
        }
        if metadata:
            record["manual_metadata"] = metadata
        return record

    def _write_response_records(self, source_id: str, records: list[dict[str, Any]]) -> tuple[str, str]:
        if not records:
            raise CollectorError("cannot write an empty response batch")
        directory = self.data_root / "raw" / "web" / source_id
        directory.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        first_id = int(records[0]["job_id"])
        last_id = int(records[-1]["job_id"])
        final = directory / f"{stamp}-{first_id}-{last_id}-{uuid.uuid4().hex[:10]}.jsonl.gz"
        temporary = final.with_suffix(final.suffix + ".tmp")
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for record in records:
                    zipped.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, final)
        # fsync the directory so the rename survives a sudden runtime interruption.
        fsync_directory(directory)
        return str(final.relative_to(self.data_root)), sha256_file(final)

    def _write_response_batch(self, job: sqlite3.Row, response: Response) -> tuple[str, str]:
        """Compatibility helper for a durable one-response recovery fixture."""
        return self._write_response_records(job["source_id"], [self._response_record(job, response)])

    def _flush_response_batch(self) -> None:
        if not self._pending_responses or self._pending_response_source is None:
            return
        records = self._pending_responses
        batch_path, digest = self._write_response_records(self._pending_response_source, records)
        self.ledger.complete_responses([int(record["job_id"]) for record in records], batch_path, digest)
        self._pending_response_source = None
        self._pending_responses = []
        self._pending_response_bytes = 0

    def _queue_response(
        self,
        job: sqlite3.Row,
        response: Response,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        source_id = str(job["source_id"])
        if self._pending_response_source not in {None, source_id}:
            self._flush_response_batch()
        if self._pending_response_source is None:
            self._pending_response_source = source_id
        self._pending_responses.append(self._response_record(job, response, metadata))
        self._pending_response_bytes += len(response.body)
        if (
            len(self._pending_responses) >= RESPONSE_BATCH_RECORDS
            or self._pending_response_bytes >= RESPONSE_BATCH_BYTES
        ):
            self._flush_response_batch()

    def _print_download_progress(
        self,
        job: sqlite3.Row,
        size: int,
        total: int | None,
        resumed_from: int,
        started_at: float,
        final: bool = False,
    ) -> None:
        """Render one overwritable progress line for a streamed asset download."""
        elapsed = max(time.monotonic() - started_at, 0.001)
        speed = human_size(int(max(size - resumed_from, 0) / elapsed)) + "/s"
        progress = f"{human_size(size)} / {human_size(total)}"
        if total:
            progress += f" ({100 * size / total:5.1f}%)"
        line = f"[{job['source_id']}] {safe_filename(job['url'])}: {progress} at {speed}"
        print("\r" + line.ljust(120), end="\n" if final else "", flush=True)

    def _download(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        if not job["destination"]:
            raise CollectorError("download job has no destination")
        destination = self.data_root / job["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        existing = partial.stat().st_size if partial.exists() else 0
        expected = job["expected_bytes"]
        if expected is not None and existing > expected:
            partial.unlink()
            existing = 0
        if expected is not None and existing == expected:
            digest = sha256_file(partial)
            expected_digest = job["expected_sha256"]
            if expected_digest and digest.lower() != expected_digest.lower():
                partial.unlink()
                existing = 0
            else:
                os.replace(partial, destination)
                self.ledger.complete_download(job["id"], existing, job["etag"], digest)
                print(f"[{job['source_id']}] verified existing download: {safe_filename(job['url'])}")
                return
        headers: dict[str, str] = {"Range": f"bytes={existing}-"} if existing else {}
        print(f"[{job['source_id']}] downloading: {safe_filename(job['url'])}")
        response = self.http.open_stream(job["url"], headers=headers, delay=float(source.get("crawl_delay", 0)))
        response_headers = dict(response.headers.items())
        etag = response_headers.get("ETag")
        # Servers that ignore Range must not have their full body appended to a partial.
        append = existing > 0 and response.status == 206
        if existing > 0 and not append:
            existing = 0
        content_range = response_headers.get("Content-Range", "")
        if append and content_range and not content_range.startswith(f"bytes {existing}-"):
            response.close()
            raise TransientError(f"invalid Content-Range for {job['url']}")
        total = expected
        if "/" in content_range:
            with contextlib.suppress(ValueError):
                total = int(content_range.rsplit("/", 1)[1])
        elif response_headers.get("Content-Length") and not append:
            with contextlib.suppress(ValueError):
                total = int(response_headers["Content-Length"])
        resumed_from = existing
        size = existing
        checkpoint_at = size + CHECKPOINT_BYTES
        started_at = time.monotonic()
        last_display = started_at
        mode = "ab" if append else "wb"
        try:
            with partial.open(mode) as handle:
                while True:
                    block = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not block:
                        break
                    handle.write(block)
                    size += len(block)
                    now = time.monotonic()
                    if size >= checkpoint_at:
                        handle.flush()
                        os.fsync(handle.fileno())
                        self.ledger.progress(job["id"], size, etag, total)
                        checkpoint_at = size + CHECKPOINT_BYTES
                    if now - last_display >= PROGRESS_INTERVAL_SECONDS:
                        self._print_download_progress(job, size, total, resumed_from, started_at)
                        last_display = now
                handle.flush()
                os.fsync(handle.fileno())
        except (HTTPException, TimeoutError, OSError) as exc:
            size = partial.stat().st_size if partial.exists() else 0
            self.ledger.progress(job["id"], size, etag, total)
            self._print_download_progress(job, size, total, resumed_from, started_at, final=True)
            raise TransientError(f"interrupted after {human_size(size)} for {job['url']}: {exc}") from exc
        finally:
            response.close()
        size = partial.stat().st_size
        self.ledger.progress(job["id"], size, etag, total)
        self._print_download_progress(job, size, total, resumed_from, started_at, final=True)
        if total is not None and size != total:
            raise TransientError(f"partial download ({size}/{total} bytes) for {job['url']}")
        digest = sha256_file(partial)
        expected_digest = job["expected_sha256"]
        if expected_digest and digest.lower() != expected_digest.lower():
            partial.unlink(missing_ok=True)
            raise TransientError(f"SHA-256 mismatch for {job['url']}")
        os.replace(partial, destination)
        self.ledger.complete_download(job["id"], size, etag, digest)

    def _discover_hf(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        response = self.http.fetch(job["url"], delay=float(source.get("crawl_delay", 0)))
        try:
            files = json.loads(response.body)
        except json.JSONDecodeError as exc:
            raise TransientError(f"invalid Hugging Face manifest for {source['id']}") from exc
        if not isinstance(files, list):
            raise TransientError(f"unexpected Hugging Face manifest for {source['id']}")
        prefix = source["path_prefix"].rstrip("/") + "/"
        parquet_entries = [
            entry for entry in files
            if isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and entry["path"].startswith(prefix)
            and entry["path"].endswith(".parquet")
        ]
        expected_shards = source.get("expected_parquet_shards")
        expected_bytes = source.get("expected_raw_bytes")
        if expected_shards is not None and len(parquet_entries) != expected_shards:
            raise CollectorError(
                f"Hugging Face manifest for {source['id']} has {len(parquet_entries)} Parquet shards; "
                f"expected {expected_shards}. Refuse changed upstream snapshot."
            )
        total_bytes = sum(int(entry.get("size") or 0) for entry in parquet_entries)
        if expected_bytes is not None and total_bytes != expected_bytes:
            raise CollectorError(
                f"Hugging Face manifest for {source['id']} totals {total_bytes:,} bytes; "
                f"expected {expected_bytes:,}. Refuse changed upstream snapshot."
            )
        for entry in parquet_entries:
            path = entry["path"]
            url = f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source.get('revision', 'main')}/{path}"
            relative = Path("raw") / "assets" / source["id"] / path.removeprefix(prefix)
            size = entry.get("size") if isinstance(entry, dict) else None
            lfs = entry.get("lfs") if isinstance(entry, dict) else None
            digest = lfs.get("oid") if isinstance(lfs, dict) else None
            # Hugging Face's LFS oid is a SHA-256 content digest.  Do not accept
            # arbitrary algorithms here: this is the immutable raw-shard gate.
            if not isinstance(digest, str) or len(digest) != 64:
                digest = None
            self.ledger.add_job(
                source["id"], "download", url, str(relative),
                size if isinstance(size, int) else None, digest,
            )
        self._queue_response(job, response)

    def _parse_sitemap(self, body: bytes) -> list[str]:
        if body.startswith(b"\x1f\x8b"):
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
                    body = compressed.read(MAX_SITEMAP_UNCOMPRESSED_BYTES + 1)
            except OSError as exc:
                raise TransientError("invalid gzip-compressed sitemap") from exc
            if len(body) > MAX_SITEMAP_UNCOMPRESSED_BYTES:
                raise CollectorError("expanded sitemap exceeds the 60 MiB safety limit")
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise TransientError("invalid sitemap XML") from exc
        return [element.text.strip() for element in root.iter() if element.tag.endswith("loc") and element.text]

    def _parse_rss_links(self, body: bytes) -> list[str]:
        """Return only RSS item links; feeds are never treated as generic HTML."""
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise TransientError("invalid PIB RSS XML") from exc
        links: list[str] = []
        for item in (element for element in root.iter() if element.tag.endswith("item")):
            link = next((child.text for child in item if child.tag.endswith("link") and child.text), None)
            if link:
                links.append(link.strip())
        return links

    def _pib_telugu_alternate(self, body: bytes, base_url: str, source: dict[str, Any]) -> str | None:
        """Find the explicitly linked Telugu release, never guess a URL path."""
        html = body.decode("utf-8", "replace")
        language_code = str(source.get("pib_telugu_language_code", "16"))
        for href in re.findall(r"\bhref\s*=\s*['\"]([^'\"]+)['\"]", html, re.IGNORECASE):
            candidate = canonical_url(urljoin(base_url, href.replace("&amp;", "&")))
            if re.search(rf"[?&]lang={re.escape(language_code)}(?:&|$)", candidate, re.IGNORECASE) and self.allowed_article_url(source, candidate):
                return candidate
        return None

    def _crawl_pib_rss(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        if not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"robots.txt disallows PIB RSS URL {job['url']}")
        response = self.http.fetch(job["url"], delay=float(source.get("crawl_delay", 2.0)))
        if not self.allowed_source_domain(source, response.url):
            raise CollectorError(f"PIB RSS redirected outside approved domains: {response.url}")
        for link in self._parse_rss_links(response.body):
            url = canonical_url(urljoin(job["url"], link))
            if self.allowed_article_url(source, url):
                self.ledger.add_job(source["id"], "pib_release", url)
        self._queue_response(job, response)

    def _crawl_pib_release(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        """Discover a release's linked Telugu version before collecting text."""
        if not self.allowed_article_url(source, job["url"]) or not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"PIB release URL is outside the approved crawl scope: {job['url']}")
        response = self.http.fetch(job["url"], delay=float(source.get("crawl_delay", 2.0)), max_bytes=int(source.get("max_article_bytes", 5 * 1024 * 1024)))
        if not self.allowed_article_url(source, response.url):
            raise CollectorError(f"PIB release redirected outside approved paths: {response.url}")
        alternate = self._pib_telugu_alternate(response.body, response.url, source)
        if alternate:
            self.ledger.add_job(source["id"], "pib_article", alternate)
        self._queue_response(job, response, {
            "pib_telugu_alternate_url": alternate,
            "policy_decision": "pib_telugu_alternate_found" if alternate else "pib_no_telugu_alternate",
        })

    def _crawl_sitemap(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        if not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"robots.txt disallows sitemap URL {job['url']}")
        response = self.http.fetch(job["url"], delay=float(source.get("crawl_delay", 0)))
        if not self.allowed_source_domain(source, response.url):
            raise CollectorError(f"sitemap redirected outside approved domains: {response.url}")
        locations = self._parse_sitemap(response.body)
        for location in locations:
            url = canonical_url(urljoin(job["url"], location))
            # A nested sitemap is identified by its XML naming convention; all other
            # permitted entries become independent article jobs.
            if urlsplit(url).path.lower().endswith((".xml", ".xml.gz")) or "sitemap" in urlsplit(url).path.lower():
                self.ledger.add_job(source["id"], "sitemap", url)
            elif self.allowed_article_url(source, url):
                self.ledger.add_job(source["id"], "article", url)
        self._queue_response(job, response)

    def _crawl_article(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        if not self.allowed_article_url(source, job["url"]):
            raise CollectorError(f"URL outside configured policy: {job['url']}")
        if not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"robots.txt disallows article URL {job['url']}")
        response = self.http.fetch(
            job["url"],
            delay=float(source.get("crawl_delay", 0)),
            max_bytes=int(source.get("max_article_bytes", 5 * 1024 * 1024)),
        )
        if not self.allowed_article_url(source, response.url):
            raise CollectorError(f"article redirected outside approved paths: {response.url}")
        content_type = next(
            (value.split(";", 1)[0].strip().casefold() for key, value in response.headers.items() if key.casefold() == "content-type"),
            "",
        )
        configured_types = source.get("allowed_content_types", ["text/html", "application/xhtml+xml"])
        if not isinstance(configured_types, list):
            raise CollectorError(f"allowed_content_types must be a list for {source['id']}")
        allowed_types = {str(value).strip().casefold() for value in configured_types}
        if content_type and content_type not in allowed_types:
            raise CollectorError(f"content type {content_type!r} is not approved for {job['url']}")
        lower = response.body[:8192].lower()
        if b"cf-chl-" in lower or b"cloudflare" in lower or b"paywall" in lower:
            raise CollectorError(f"access wall detected for {job['url']}; manual approval required")
        self._queue_response(job, response)

    def _crawl_pib_article(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        """Fetch a Telugu PIB release with a bounded, attributable response."""
        if not self.allowed_article_url(source, job["url"]):
            raise CollectorError(f"PIB release URL outside configured policy: {job['url']}")
        if not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"robots.txt disallows PIB release URL {job['url']}")
        response = self.http.fetch(job["url"], delay=float(source.get("crawl_delay", 2.0)), max_bytes=int(source.get("max_article_bytes", 5 * 1024 * 1024)))
        if not self.allowed_article_url(source, response.url):
            raise CollectorError(f"PIB release redirected outside approved paths: {response.url}")
        text = response.body.decode("utf-8", "replace")
        prid_match = re.search(r"[?&]PRID=(\d+)", response.url, re.IGNORECASE)
        date_match = re.search(r"(?:Release\s+Date|ప్రచురణ తేదీ)[^<]{0,80}</[^>]+>\s*([^<]+)", text, re.IGNORECASE)
        ministry_match = re.search(r"(?:Ministry|మంత్రిత్వ శాఖ)[^<]{0,80}</[^>]+>\s*([^<]+)", text, re.IGNORECASE)
        self._queue_response(job, response, {
            "pib_release_id": prid_match.group(1) if prid_match else None,
            "publication_date": date_match.group(1).strip() if date_match else None,
            "ministry": ministry_match.group(1).strip() if ministry_match else None,
            "attribution_url": response.url,
            "policy_decision": "pib_authored_release_only",
        })

    def _crawl_mediawiki_list(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        if not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"robots.txt disallows MediaWiki API URL {job['url']}")
        response = self.http.fetch(job["url"], delay=float(source.get("crawl_delay", 0.2)))
        if not self.allowed_source_domain(source, response.url):
            raise CollectorError(f"MediaWiki API redirected outside approved domains: {response.url}")
        pages, continuation = parse_discovery(response.body, source)
        for page_id, _title in pages:
            self.ledger.add_job(source["id"], "mediawiki_page", page_url(source, page_id))
        if continuation:
            self.ledger.add_job(source["id"], "mediawiki_list", discovery_url(source, continuation))
        self._queue_response(job, response, {
            "policy_decision": "discovery",
            "discovered_pages": len(pages),
            "continuation": continuation,
        })

    def _crawl_mediawiki_page(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        if not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"robots.txt disallows MediaWiki API URL {job['url']}")
        response = self.http.fetch(
            job["url"],
            delay=float(source.get("crawl_delay", 0.2)),
            max_bytes=int(source.get("max_article_bytes", 5 * 1024 * 1024)),
        )
        if not self.allowed_source_domain(source, response.url):
            raise CollectorError(f"MediaWiki API redirected outside approved domains: {response.url}")
        metadata = page_policy_metadata(response.body, source)
        self._queue_response(job, response, metadata)

    def _crawl_mediawiki_category(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        if not self.allowed_article_url(source, job["url"]):
            raise CollectorError(f"MediaWiki category URL outside configured policy: {job['url']}")
        if not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"robots.txt disallows MediaWiki category URL {job['url']}")
        response = self.http.fetch(
            job["url"],
            delay=float(source.get("crawl_delay", 0.25)),
            max_bytes=int(source.get("max_category_bytes", 2 * 1024 * 1024)),
        )
        if not self.allowed_article_url(source, response.url):
            raise CollectorError(f"MediaWiki category redirected outside approved paths: {response.url}")
        pages, continuations = parse_category_page(response.body, response.url, source)
        for url in pages:
            self.ledger.add_job(source["id"], "mediawiki_html_page", url)
        for url in continuations:
            self.ledger.add_job(source["id"], "mediawiki_category", url)
        self._queue_response(job, response, {
            "policy_decision": "discovery",
            "discovered_pages": len(pages),
            "continuations": continuations,
        })

    def _crawl_mediawiki_html_page(self, job: sqlite3.Row, source: dict[str, Any]) -> None:
        if not self.allowed_article_url(source, job["url"]):
            raise CollectorError(f"MediaWiki page URL outside configured policy: {job['url']}")
        if not self._robots_allowed(source, job["url"]):
            raise CollectorError(f"robots.txt disallows MediaWiki page URL {job['url']}")
        response = self.http.fetch(
            job["url"],
            delay=float(source.get("crawl_delay", 0.25)),
            max_bytes=int(source.get("max_article_bytes", 10 * 1024 * 1024)),
        )
        if not self.allowed_article_url(source, response.url):
            raise CollectorError(f"MediaWiki page redirected outside approved paths: {response.url}")
        metadata = html_page_policy_metadata(response.body, response.url, source)
        self._queue_response(job, response, metadata)

    def process_job(self, job: sqlite3.Row) -> None:
        source = self.source(job["source_id"])
        self.ledger.mark_in_progress(job["id"])
        try:
            if job["kind"] == "download":
                self._download(job, source)
            elif job["kind"] == "hf_manifest":
                self._discover_hf(job, source)
            elif job["kind"] == "sitemap":
                self._crawl_sitemap(job, source)
            elif job["kind"] == "pib_rss":
                self._crawl_pib_rss(job, source)
            elif job["kind"] == "pib_release":
                self._crawl_pib_release(job, source)
            elif job["kind"] == "article":
                self._crawl_article(job, source)
            elif job["kind"] == "pib_article":
                self._crawl_pib_article(job, source)
            elif job["kind"] == "mediawiki_list":
                self._crawl_mediawiki_list(job, source)
            elif job["kind"] == "mediawiki_page":
                self._crawl_mediawiki_page(job, source)
            elif job["kind"] == "mediawiki_category":
                self._crawl_mediawiki_category(job, source)
            elif job["kind"] == "mediawiki_html_page":
                self._crawl_mediawiki_html_page(job, source)
            else:
                raise CollectorError(f"unknown job kind {job['kind']}")
        except TransientError as exc:
            self.ledger.defer(job["id"], str(exc), exc.retry_after)
            print(f"Deferred {job['kind']} job {job['id']}: {exc}")
        except CollectorError as exc:
            # Access/policy failures are deliberately deferred, not attacked with retries.
            self.ledger.defer(job["id"], str(exc), 24 * 3600)
            print(f"Deferred {job['kind']} job {job['id']}: {exc}")
        except Exception as exc:  # leave evidence without losing the rest of the queue
            self.ledger.fail(job["id"], f"unexpected {type(exc).__name__}: {exc}")
            print(f"Failed {job['kind']} job {job['id']}: {type(exc).__name__}: {exc}", file=sys.stderr)

    def run(self, selected: str | None = None) -> int:
        recovered = self.ledger.reconcile_batches(self.data_root)
        selected_source = self.source(selected) if selected else None
        if selected_source is not None and not selected_source.get("approved", False):
            raise CollectorError(f"source {selected_source['id']} is not approved; no queued jobs were processed")
        if selected_source is None and any(
            source.get("collection_method") in {"mediawiki_api", "mediawiki_category_html", "pib_rss_crawl"}
            or source.get("minimum_free_gib") is not None
            for source in self.approved_sources(None)
        ):
            raise CollectorError("collect bounded or large approved sources explicitly with --source SOURCE_ID")
        if selected_source is not None:
            self._require_source_space(selected_source)
        self.schedule(selected)
        processed = 0
        last_progress = time.monotonic()
        document_limit: int | None = None
        documents_fetched = 0
        mediawiki_methods = {"mediawiki_api", "mediawiki_category_html"}
        document_kind = None
        document_label = "documents"
        if selected_source and selected_source.get("collection_method") in mediawiki_methods:
            configured_limit = int(selected_source.get("pilot_max_documents", 0))
            document_limit = self.pilot_max_documents or configured_limit
            document_kind = (
                "mediawiki_page"
                if selected_source.get("collection_method") == "mediawiki_api"
                else "mediawiki_html_page"
            )
            document_label = "MediaWiki pages"
        elif selected_source and selected_source.get("collection_method") in {"sitemap_crawl", "pib_rss_crawl"}:
            configured_limit = int(selected_source.get("pilot_max_documents", 0))
            document_limit = self.pilot_max_documents or configured_limit or None
            document_kind = "pib_article" if selected_source.get("collection_method") == "pib_rss_crawl" else "article"
            document_label = "articles"
        if document_kind is not None and document_limit is not None:
            documents_fetched = self.ledger.count_jobs(
                selected_source["id"], document_kind, ("complete",)
            )
            if documents_fetched >= document_limit:
                print(
                    f"pilot limit already reached: {documents_fetched:,} / "
                    f"{document_limit:,} {document_label} fetched",
                    flush=True,
                )
                return 0
        elif selected_source is None and any(
            source.get("collection_method") in mediawiki_methods
            for source in self.approved_sources(None)
        ):
            raise CollectorError("collect each approved MediaWiki source with --source SOURCE_ID")
        elif selected_source is None and any(
            source.get("collection_method") == "pib_rss_crawl"
            for source in self.approved_sources(None)
        ):
            raise CollectorError("collect each approved PIB RSS source with --source SOURCE_ID so its page cap is enforced")
        # New article/download jobs can be scheduled by a job completed in this run.
        pilot_reached = False
        while True:
            jobs = self.ledger.runnable(selected)
            if not jobs:
                break
            for job in jobs:
                if job["kind"] == document_kind and document_limit is not None:
                    if documents_fetched >= document_limit:
                        pilot_reached = True
                        break
                self.process_job(job)
                processed += 1
                if job["kind"] == document_kind and document_limit is not None:
                    status = self.ledger.job(job["id"])["status"]
                    if status in {"in_progress", "complete"}:
                        documents_fetched += 1
                now = time.monotonic()
                if processed == 1 or processed % 100 == 0 or now - last_progress >= 30:
                    counts: dict[str, int] = {}
                    for row in self.ledger.status_counts(selected):
                        counts[row["status"]] = counts.get(row["status"], 0) + int(row["count"])
                    print(
                        f"collection progress: {processed:,} processed this run; "
                        f"{counts.get('complete', 0):,} complete, "
                        f"{counts.get('in_progress', 0):,} awaiting batch commit, "
                        f"{counts.get('pending', 0):,} pending, "
                        f"{counts.get('deferred', 0):,} deferred, "
                        f"{counts.get('failed', 0):,} failed",
                        flush=True,
                    )
                    last_progress = now
            # Sitemap children become runnable only after this query batch. Flush
            # their parent responses before querying the ledger again; a crash
            # between the durable rename and this transaction is reconciled.
            self._flush_response_batch()
            if pilot_reached or (document_limit is not None and documents_fetched >= document_limit):
                print(
                    f"pilot limit reached: {documents_fetched:,} / "
                    f"{document_limit:,} {document_label} fetched; rerun with a higher "
                    "--pilot-max-documents value to continue",
                    flush=True,
                )
                break
        if recovered:
            print(f"Recovered {recovered} response batch(es) written before a prior interruption.")
        return processed

    def print_status(self, selected: str | None = None) -> None:
        rows = self.ledger.status_counts(selected)
        if not rows:
            print("No jobs recorded.")
            return
        for row in rows:
            print(f"{row['source_id']:20} {row['kind']:12} {row['status']:12} {row['count']}")
