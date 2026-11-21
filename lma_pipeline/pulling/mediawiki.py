"""Bounded MediaWiki discovery and conservative page-policy decisions."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urlencode, urljoin

from .common import CollectorError, TransientError


MANDATORY_EXCLUDED_TEMPLATES = {"non-free dli", "non free dli"}
TEMPLATE_RE = re.compile(r"\{\{\s*([^|}\n]+)")


def _payload(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransientError("invalid MediaWiki API JSON") from exc
    if not isinstance(value, dict):
        raise TransientError("unexpected MediaWiki API response")
    error = value.get("error")
    if isinstance(error, dict):
        code = str(error.get("code", "unknown"))
        message = str(error.get("info", code))
        if code in {"maxlag", "readonly", "ratelimited"}:
            raise TransientError(f"MediaWiki {code}: {message}", retry_after=30.0)
        raise CollectorError(f"MediaWiki API error {code}: {message}")
    return value


def normalize_template(value: str) -> str:
    value = value.replace("_", " ").strip().casefold()
    for namespace in ("template:", "మూస:"):
        if value.startswith(namespace):
            value = value.removeprefix(namespace).strip()
            break
    return " ".join(value.split())


def wikitext_templates(wikitext: str) -> list[str]:
    return sorted({normalize_template(match) for match in TEMPLATE_RE.findall(wikitext) if match.strip()})


def discovery_url(source: dict[str, Any], continuation: str | None = None) -> str:
    mode = source.get("mediawiki_discovery", "allpages")
    namespace = int(source.get("mediawiki_namespace", 0))
    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "maxlag": 5,
    }
    if mode == "allpages":
        params.update({"list": "allpages", "apnamespace": namespace, "aplimit": "max"})
        if continuation:
            params["apcontinue"] = continuation
    elif mode == "categorymembers":
        params.update({
            "list": "categorymembers",
            "cmtitle": source["mediawiki_category"],
            "cmnamespace": namespace,
            "cmlimit": "max",
        })
        if continuation:
            params["cmcontinue"] = continuation
    else:
        raise CollectorError(f"unsupported mediawiki_discovery {mode!r}")
    return f"{source['api_url']}?{urlencode(params)}"


def parse_discovery(body: bytes, source: dict[str, Any]) -> tuple[list[tuple[int, str]], str | None]:
    payload = _payload(body)
    mode = source.get("mediawiki_discovery", "allpages")
    key = "allpages" if mode == "allpages" else "categorymembers"
    values = payload.get("query", {}).get(key, [])
    pages = [
        (int(item["pageid"]), str(item.get("title", item["pageid"])))
        for item in values
        if isinstance(item, dict) and isinstance(item.get("pageid"), int)
    ]
    continuation_key = "apcontinue" if mode == "allpages" else "cmcontinue"
    continuation = payload.get("continue", {}).get(continuation_key)
    return pages, str(continuation) if continuation is not None else None


def page_url(source: dict[str, Any], page_id: int) -> str:
    params = {
        "action": "parse",
        "format": "json",
        "formatversion": 2,
        "maxlag": 5,
        "pageid": page_id,
        "prop": "text|wikitext|templates|revid|displaytitle",
    }
    return f"{source['api_url']}?{urlencode(params)}"


def page_content(body: bytes) -> dict[str, Any]:
    payload = _payload(body)
    parsed = payload.get("parse")
    if isinstance(parsed, dict):
        wikitext = parsed.get("wikitext", "")
        rendered_html = parsed.get("text", "")
        if not isinstance(wikitext, str) or not isinstance(rendered_html, str):
            raise CollectorError("MediaWiki parse response has no text")
        templates = parsed.get("templates", [])
        return {
            "page_id": int(parsed["pageid"]),
            "title": str(parsed.get("title", parsed["pageid"])),
            "canonical_url": "",
            "revision_id": int(parsed.get("revid", 0)),
            "revision_timestamp": "",
            "wikitext": wikitext,
            "rendered_html": rendered_html,
            "api_templates": [
                str(item.get("title", item.get("*", "")))
                for item in templates
                if isinstance(item, dict)
            ],
        }
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list) or not pages or not isinstance(pages[0], dict):
        raise CollectorError("MediaWiki page response has no page")
    page = pages[0]
    if page.get("missing") is True:
        raise CollectorError("MediaWiki page is missing")
    revisions = page.get("revisions", [])
    if not isinstance(revisions, list) or not revisions or not isinstance(revisions[0], dict):
        raise CollectorError("MediaWiki page has no revision text")
    revision = revisions[0]
    slots = revision.get("slots", {})
    main = slots.get("main", {}) if isinstance(slots, dict) else {}
    wikitext = main.get("content") if isinstance(main, dict) else None
    if not isinstance(wikitext, str):
        # Older API shapes expose revision content directly.
        wikitext = revision.get("content")
    if not isinstance(wikitext, str):
        raise CollectorError("MediaWiki revision has no wikitext content")
    return {
        "page_id": int(page["pageid"]),
        "title": str(page.get("title", page["pageid"])),
        "canonical_url": str(page.get("fullurl", "")),
        "revision_id": int(revision.get("revid", 0)),
        "revision_timestamp": str(revision.get("timestamp", "")),
        "wikitext": wikitext,
        "rendered_html": "",
        "api_templates": [],
    }


def page_policy_metadata(body: bytes, source: dict[str, Any]) -> dict[str, Any]:
    page = page_content(body)
    templates = sorted({
        *wikitext_templates(page["wikitext"]),
        *(normalize_template(value) for value in page.get("api_templates", []) if value),
    })
    configured = source.get("excluded_templates", [])
    excluded = MANDATORY_EXCLUDED_TEMPLATES | {
        normalize_template(value) for value in configured if isinstance(value, str)
    }
    matched = sorted(set(templates) & excluded)
    required = {
        normalize_template(value)
        for value in source.get("required_license_templates", [])
        if isinstance(value, str)
    }
    required_matched = sorted(set(templates) & required)
    redirect = bool(re.match(r"^\s*#redirect\b", page["wikitext"], re.IGNORECASE))
    if matched:
        decision = "rejected"
        reason = "excluded_template:" + ",".join(matched)
    elif required and not required_matched:
        decision = "rejected"
        reason = "missing_required_license_template"
    elif redirect:
        decision = "rejected"
        reason = "mediawiki_redirect"
    else:
        decision = "accepted"
        reason = "approved_source_policy"
    return {
        "policy_decision": decision,
        "policy_reason": reason,
        "mediawiki_page_id": page["page_id"],
        "mediawiki_title": page["title"],
        "mediawiki_revision_id": page["revision_id"],
        "mediawiki_revision_timestamp": page["revision_timestamp"],
        "license_templates": templates,
        "matched_required_license_templates": required_matched,
        "license_expression": source["license_expression"],
        "attribution_url": page["canonical_url"] or urljoin(
            source["url"],
            "/wiki/" + quote(page["title"].replace(" ", "_"), safe="()_-"),
        ),
        "attribution_text": source["attribution_text"],
    }
