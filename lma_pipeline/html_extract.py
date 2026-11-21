"""Dependency-free main-content extraction for manually crawled HTML pages."""

from __future__ import annotations

import html.parser
import json
import re
from typing import Any, Iterable


IGNORED_TAGS = {
    "aside", "button", "canvas", "footer", "form", "iframe", "nav",
    "noscript", "script", "style", "svg", "template",
}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
SIMPLE_SELECTOR_RE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9_-]*)?(?:[.#][A-Za-z_][A-Za-z0-9_-]*)?$")


def selector_supported(selector: str) -> bool:
    return bool(selector and SIMPLE_SELECTOR_RE.fullmatch(selector.strip()))


def _matches_selector(tag: str, attrs: dict[str, str], selector: str) -> bool:
    """Match the small selector subset needed by source-specific article rules."""
    selector = selector.strip()
    if not selector_supported(selector):
        return False
    identifier = attrs.get("id", "")
    classes = set(attrs.get("class", "").split())
    if selector.startswith("#"):
        return identifier == selector[1:]
    if selector.startswith("."):
        return selector[1:] in classes
    if "#" in selector:
        expected_tag, expected_id = selector.split("#", 1)
        return tag == expected_tag.casefold() and identifier == expected_id
    if "." in selector:
        expected_tag, expected_class = selector.split(".", 1)
        return tag == expected_tag.casefold() and expected_class in classes
    return tag == selector.casefold()


def _article_bodies(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        body = value.get("articleBody")
        if isinstance(body, str) and body.strip():
            yield body
        for child in value.values():
            yield from _article_bodies(child)
    elif isinstance(value, list):
        for child in value:
            yield from _article_bodies(child)


class MainContentHTMLParser(html.parser.HTMLParser):
    """Collect configured, JSON-LD, semantic article, and fallback body text."""

    def __init__(
        self,
        selectors: Iterable[str] = (),
        excluded_selectors: Iterable[str] = (),
    ) -> None:
        super().__init__(convert_charrefs=True)
        self.selectors = tuple(selectors)
        self.excluded_selectors = tuple(excluded_selectors)
        self.selected_parts: list[str] = []
        self.semantic_parts: list[str] = []
        self.fallback_parts: list[str] = []
        self.json_ld_chunks: list[str] = []
        self._stack: list[tuple[str, bool, bool, bool, bool, bool]] = []
        self._ignored_depth = 0
        self._excluded_depth = 0
        self._selected_depth = 0
        self._semantic_depth = 0
        self._json_ld_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = {key.casefold(): value or "" for key, value in attrs}
        is_json_ld = tag == "script" and "ld+json" in attributes.get("type", "").casefold()
        is_ignored = tag in IGNORED_TAGS and not is_json_ld
        is_selected = any(_matches_selector(tag, attributes, selector) for selector in self.selectors)
        is_excluded = any(
            _matches_selector(tag, attributes, selector)
            for selector in self.excluded_selectors
        )
        is_semantic = tag in {"article", "main"}
        if tag in VOID_TAGS:
            return
        self._stack.append((tag, is_ignored, is_excluded, is_selected, is_semantic, is_json_ld))
        self._ignored_depth += int(is_ignored)
        self._excluded_depth += int(is_excluded)
        self._selected_depth += int(is_selected)
        self._semantic_depth += int(is_semantic)
        self._json_ld_depth += int(is_json_ld)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in VOID_TAGS:
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        while self._stack:
            opened_tag, is_ignored, is_excluded, is_selected, is_semantic, is_json_ld = self._stack.pop()
            self._ignored_depth -= int(is_ignored)
            self._excluded_depth -= int(is_excluded)
            self._selected_depth -= int(is_selected)
            self._semantic_depth -= int(is_semantic)
            self._json_ld_depth -= int(is_json_ld)
            if opened_tag == tag:
                break

    def handle_data(self, data: str) -> None:
        if self._json_ld_depth:
            self.json_ld_chunks.append(data)
            return
        value = data.strip()
        if not value or self._ignored_depth or self._excluded_depth:
            return
        self.fallback_parts.append(value)
        if self._semantic_depth:
            self.semantic_parts.append(value)
        if self._selected_depth:
            self.selected_parts.append(value)

    def json_ld_text(self) -> str:
        bodies: list[str] = []
        for chunk in self.json_ld_chunks:
            try:
                parsed = json.loads(chunk)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            bodies.extend(_article_bodies(parsed))
        return "\n".join(bodies)


def extract_html_text(raw_html: str, source: dict[str, Any]) -> tuple[str, str]:
    """Return main text and the extraction strategy used for auditability."""
    selectors = source.get("content_selectors", ())
    if not isinstance(selectors, list) or not all(isinstance(value, str) and selector_supported(value) for value in selectors):
        selectors = ()
    excluded_selectors = source.get("excluded_content_selectors", ())
    if not isinstance(excluded_selectors, list) or not all(
        isinstance(value, str) and selector_supported(value)
        for value in excluded_selectors
    ):
        excluded_selectors = ()
    parser = MainContentHTMLParser(selectors, excluded_selectors)
    parser.feed(raw_html)
    parser.close()
    candidates = (
        (parser.selected_parts, "configured_selector"),
        ([parser.json_ld_text()], "json_ld_article_body"),
        (parser.semantic_parts, "semantic_article"),
        (parser.fallback_parts, "visible_text_fallback"),
    )
    for parts, strategy in candidates:
        text = re.sub(r"[ \t\r\f\v]+", " ", "\n".join(parts)).strip()
        if text:
            return text, strategy
    return "", "empty"
