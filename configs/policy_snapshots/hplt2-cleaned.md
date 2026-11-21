# HPLT v2 cleaned provenance and classification

Reviewed 2026-08-20.

Physical acquisition is recorded truthfully as Hugging Face Parquet downloads
from `HPLT/HPLT2.0_cleaned`. Raw shards stay immutable under `raw/assets/` and
the collector records Hugging Face LFS sizes and SHA-256 OIDs when supplied by the
repository API. HPLT URL, crawl collection, WARC reference, document identifier,
date, and quality fields are retained when present in a shard schema.

The recovery registry is pinned to Hugging Face revision
`d1324a5283f762ee62c2a5c81de08fc6450ea540`. At review it contains 37 Telugu
Parquet shards (6,706,642,351 bytes) and 42 Nepali shards (7,439,462,246 bytes).
The collector rejects a manifest with a different count or total size instead of
silently downloading a changed upstream snapshot.

The collection category is a separate, narrowly scoped decision: the project's
rules allowed HPLT to count toward the manual-collection share. The registry
applies that exception only to `tel_Telu` and `npi_Deva`,
with `course_classification_basis: ta_whatsapp_hplt_manual_exception_2026_08`.
It does not claim HPLT was personally crawled, does not change its
`collection_method`, and does not bypass deterministic budget sampling.
