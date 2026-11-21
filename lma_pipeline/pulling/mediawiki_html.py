"""Robots-compliant Wikimedia category discovery and HTML item policy."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlsplit, urlunsplit

from .common import CollectorError


CONFIG_RE = re.compile(r'"wgCategories"\s*:\s*(\[[^\]]*\])')
PAGE_ID_RE = re.compile(r'"wgArticleId"\s*:\s*(\d+)')
REVISION_ID_RE = re.compile(r'"wgRevisionId"\s*:\s*(\d+)')
REDIRECT_RE = re.compile(r'"wgIsRedirect"\s*:\s*true')


def normalize_category(value: str) -> str:
    return " ".join(value.replace("_", " ").strip().casefold().split())


class CategoryPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.div_depth = 0
        self.pages_depth: int | None = None
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "div":
            self.div_depth += 1
            if values.get("id") == "mw-pages":
                self.pages_depth = self.div_depth
        if tag == "a" and self.pages_depth is not None:
            href = values.get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag != "div":
            return
        if self.pages_depth == self.div_depth:
            self.pages_depth = None
        self.div_depth = max(0, self.div_depth - 1)


def _category_continuation(seed_url: str, href: str) -> str | None:
    parsed = urlsplit(urljoin(seed_url, href))
    query = parse_qs(parsed.query)
    continuation = {
        key: values[-1]
        for key, values in query.items()
        if key in {"pagefrom", "subcatfrom", "filefrom"} and values
    }
    if not continuation:
        return None
    seed = urlsplit(seed_url)
    return urlunsplit((seed.scheme, seed.netloc, seed.path, urlencode(continuation), ""))


def parse_category_page(body: bytes, page_url: str, source: dict[str, Any]) -> tuple[list[str], list[str]]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CollectorError("MediaWiki category page is not valid UTF-8") from exc
    parser = CategoryPageParser()
    parser.feed(text)
    pages: set[str] = set()
    continuations: set[str] = set()
    for href in parser.links:
        continuation = _category_continuation(source["url"], href)
        if continuation:
            continuations.add(continuation)
            continue
        candidate = urljoin(page_url, href)
        parsed = urlsplit(candidate)
        if parsed.netloc.casefold() != urlsplit(source["url"]).netloc.casefold():
            continue
        title_path = unquote(parsed.path.removeprefix("/wiki/"))
        if not parsed.path.startswith("/wiki/") or ":" in title_path:
            continue
        pages.add(urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")))
    return sorted(pages), sorted(continuations)


def page_policy_metadata(body: bytes, page_url: str, source: dict[str, Any]) -> dict[str, Any]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CollectorError("MediaWiki article page is not valid UTF-8") from exc
    match = CONFIG_RE.search(text)
    try:
        categories = json.loads(match.group(1)) if match else []
    except json.JSONDecodeError as exc:
        raise CollectorError("MediaWiki page has malformed category metadata") from exc
    normalized = {normalize_category(str(value)) for value in categories}
    required = {
        normalize_category(value)
        for value in source.get("required_license_categories", [])
        if isinstance(value, str)
    }
    excluded = {
        normalize_category(value)
        for value in source.get("excluded_categories", [])
        if isinstance(value, str)
    }
    required_matched = sorted(normalized & required)
    excluded_matched = sorted(normalized & excluded)
    lowered = normalize_category(text)
    excluded_markers = [
        normalize_category(value)
        for value in source.get("excluded_templates", [])
        if isinstance(value, str)
    ]
    marker_matches = sorted({value for value in excluded_markers if value and value in lowered})
    if REDIRECT_RE.search(text):
        decision = "rejected"
        reason = "mediawiki_redirect"
    elif excluded_matched or marker_matches:
        decision = "rejected"
        reason = "excluded_rights_marker:" + ",".join(excluded_matched + marker_matches)
    elif required and not required_matched:
        decision = "rejected"
        reason = "missing_required_license_category"
    else:
        decision = "accepted"
        reason = "approved_category_and_item_policy"
    page_id = PAGE_ID_RE.search(text)
    revision_id = REVISION_ID_RE.search(text)
    return {
        "policy_decision": decision,
        "policy_reason": reason,
        "mediawiki_page_id": int(page_id.group(1)) if page_id else 0,
        "mediawiki_revision_id": int(revision_id.group(1)) if revision_id else 0,
        "mediawiki_revision_timestamp": "",
        "license_categories": sorted(normalized),
        "matched_required_license_categories": required_matched,
        "license_expression": source["license_expression"],
        "attribution_url": page_url,
        "attribution_text": source["attribution_text"],
    }
