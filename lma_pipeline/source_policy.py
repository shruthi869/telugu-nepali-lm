"""Policy gates that decide whether a source may count as manual collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .html_extract import selector_supported


MANUAL_COLLECTION_METHODS = {
    "manual_files",
    "mediawiki_api",
    "mediawiki_category_html",
    "sitemap_crawl",
    "pib_rss_crawl",
}
MANUAL_RIGHTS_BASES = {"open_license", "public_domain", "written_permission"}
COURSE_COLLECTION_CATEGORIES = {"manual", "downloaded"}
COURSE_MANUAL_WIKI_TYPES = {"wikimedia_dump", "wikisource_public_domain"}
# This is intentionally a narrow exception, not a general way to relabel a
# downloaded dataset.  The supporting clarification note is tracked with the
# source registry.
HPLT_COURSE_MANUAL_DATASETS = {
    "HPLT/HPLT2.0_cleaned:tel_Telu",
    "HPLT/HPLT2.0_cleaned:npi_Deva",
}


def manual_source_issue(source: dict[str, Any], config_root: Path) -> str | None:
    """Return why a declared manual source is ineligible, or ``None``."""
    if not source.get("manual_collection"):
        return "manual_collection is not true"
    method = source.get("collection_method")
    if method not in MANUAL_COLLECTION_METHODS:
        return f"collection_method must be one of {sorted(MANUAL_COLLECTION_METHODS)}"
    if method in {"mediawiki_api", "mediawiki_category_html", "sitemap_crawl", "pib_rss_crawl"}:
        domains = source.get("allowed_domains")
        paths = source.get("allowed_paths")
        if not isinstance(domains, list) or not domains or not all(isinstance(value, str) and value for value in domains):
            return "network manual sources require a non-empty allowed_domains list"
        if not isinstance(paths, list) or not paths or not all(
            isinstance(value, str) and value.startswith("/") for value in paths
        ):
            return "network manual sources require a non-empty allowed_paths list"
    if method == "mediawiki_api":
        api_url = source.get("api_url")
        if not isinstance(api_url, str) or not api_url.startswith(("https://", "http://")):
            return "mediawiki_api sources require an HTTP(S) api_url"
        discovery = source.get("mediawiki_discovery", "allpages")
        if discovery not in {"allpages", "categorymembers"}:
            return "mediawiki_discovery must be allpages or categorymembers"
        if discovery == "categorymembers" and not source.get("mediawiki_category"):
            return "categorymembers discovery requires mediawiki_category"
        pilot = source.get("pilot_max_documents")
        if not isinstance(pilot, int) or pilot < 1:
            return "mediawiki_api sources require pilot_max_documents of at least 1"
        required = source.get("required_license_templates")
        if not isinstance(required, list) or not required or not all(
            isinstance(value, str) and value.strip() for value in required
        ):
            return "mediawiki_api sources require a non-empty required_license_templates list"
    if method == "mediawiki_category_html":
        url = source.get("url")
        if not isinstance(url, str) or "/wiki/Category:" not in url:
            return "mediawiki_category_html sources require a /wiki/ category URL"
        required_categories = source.get("required_license_categories")
        if not isinstance(required_categories, list) or not required_categories or not all(
            isinstance(value, str) and value.strip() for value in required_categories
        ):
            return "mediawiki_category_html sources require required_license_categories"
        excluded_categories = source.get("excluded_categories", [])
        if not isinstance(excluded_categories, list) or not all(
            isinstance(value, str) and value.strip() for value in excluded_categories
        ):
            return "excluded_categories must be a list of category names"
        pilot = source.get("pilot_max_documents")
        if not isinstance(pilot, int) or pilot < 1:
            return "mediawiki_category_html sources require pilot_max_documents of at least 1"
    if method in {"sitemap_crawl", "pib_rss_crawl"}:
        pilot = source.get("pilot_max_documents")
        if not isinstance(pilot, int) or pilot < 1:
            return "manual sitemap/RSS sources require pilot_max_documents of at least 1"
    if method == "pib_rss_crawl" and source.get("language") != "tel":
        return "pib_rss_crawl is currently approved only for Telugu"
    if method == "manual_files":
        ocr_mode = source.get("ocr_mode", "off")
        if ocr_mode not in {"off", "auto", "force"}:
            return "manual_files ocr_mode must be off, auto, or force"
        ocr_dpi = source.get("ocr_dpi", 300)
        if not isinstance(ocr_dpi, int) or not 150 <= ocr_dpi <= 450:
            return "manual_files ocr_dpi must be an integer from 150 through 450"
        ocr_psm = source.get("ocr_psm", 3)
        if not isinstance(ocr_psm, int) or not 0 <= ocr_psm <= 13:
            return "manual_files ocr_psm must be an integer from 0 through 13"
        ocr_languages = source.get("ocr_languages")
        if ocr_languages is not None and (
            not isinstance(ocr_languages, str)
            or not ocr_languages
            or any(not part.isalpha() for part in ocr_languages.split("+"))
        ):
            return "manual_files ocr_languages must look like tel+eng or nep+eng"
    if method in {"mediawiki_api", "mediawiki_category_html"}:
        excluded = source.get("excluded_templates", [])
        if not isinstance(excluded, list) or not all(isinstance(value, str) for value in excluded):
            return "excluded_templates must be a list of template names"
    rights_basis = source.get("rights_basis")
    if rights_basis not in MANUAL_RIGHTS_BASES:
        return f"rights_basis must be one of {sorted(MANUAL_RIGHTS_BASES)}"
    license_expression = source.get("license_expression")
    if (
        not isinstance(license_expression, str)
        or len(license_expression.strip()) < 3
        or any(marker in license_expression.upper() for marker in ("ADD_", "SET_", "PLACEHOLDER"))
    ):
        return "manual sources require a license_expression"
    attribution_text = source.get("attribution_text")
    if (
        not isinstance(attribution_text, str)
        or len(attribution_text.strip()) < 3
        or any(marker in attribution_text.upper() for marker in ("ADD_", "SET_", "PLACEHOLDER"))
    ):
        return "manual sources require attribution_text"
    target = source.get("target_clean_tokens")
    if target is not None and (not isinstance(target, int) or target < 1):
        return "target_clean_tokens must be a positive integer when provided"
    selectors = source.get("content_selectors", [])
    if not isinstance(selectors, list) or not all(
        isinstance(selector, str) and selector_supported(selector) for selector in selectors
    ):
        return "content_selectors must use simple tag, .class, #id, tag.class, or tag#id selectors"
    snapshot = source.get("policy_snapshot")
    if not isinstance(snapshot, str) or not snapshot or "ADD_POLICY_SNAPSHOT" in snapshot:
        return "policy_snapshot is missing or still a placeholder"
    resolved_root = config_root.resolve()
    snapshot_path = (resolved_root / snapshot).resolve()
    try:
        snapshot_path.relative_to(resolved_root)
    except ValueError:
        return "policy_snapshot must stay within the tracked project"
    try:
        contents = snapshot_path.read_text(encoding="utf-8")
    except OSError:
        return f"policy snapshot cannot be read: {snapshot_path}"
    if "ADD_POLICY_SNAPSHOT" in contents or len(contents.strip()) <= 40:
        return "policy snapshot is empty or still contains a placeholder"
    return None


def is_course_manual_source(source: dict[str, Any], config_root: Path) -> bool:
    """Return the category used for the project's manual-token requirement.

    The project convention classifies Wiki material as manual even
    when an official dump was fetched with ``direct_assets``.  The separate
    ``collection_method`` field continues to record how the bytes were actually
    acquired; this function only controls project-category sampling/reporting.
    """
    course_category = source.get("course_collection_category")
    if course_category == "downloaded":
        return False
    if course_category == "manual" and source.get("source_type") in COURSE_MANUAL_WIKI_TYPES:
        return True
    hplt_key = f"{source.get('dataset')}:{source.get('dataset_config')}"
    if course_category == "manual" and hplt_key in HPLT_COURSE_MANUAL_DATASETS:
        return (
            source.get("collection_method") == "huggingface_parquet"
            and source.get("manual_collection") is False
            and source.get("course_classification_basis") == "ta_whatsapp_hplt_manual_exception_2026_08"
        )
    return manual_source_issue(source, config_root) is None


def is_budget_exempt_source(source: dict[str, Any], config_root: Path) -> bool:
    """Return whether deterministic downloaded-budget sampling is bypassed.

    Category reporting and byte acquisition are separate concerns.  In particular,
    HPLT is classified as manual under the project's allow-listed exception but remains
    sampled so a backfill cannot accidentally consume every Parquet row.
    """
    if not is_course_manual_source(source, config_root):
        return False
    return f"{source.get('dataset')}:{source.get('dataset_config')}" not in HPLT_COURSE_MANUAL_DATASETS


def is_manual_source(source: dict[str, Any], config_root: Path) -> bool:
    """Backward-compatible alias for the manual classification."""
    return is_course_manual_source(source, config_root)
