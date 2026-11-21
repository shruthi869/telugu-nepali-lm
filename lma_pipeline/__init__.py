"""Stable public API for the resumable Telugu/Nepali corpus pipeline."""

from .cli import download_model, main, parse_args
from .common import (
    DEFAULT_MAX_SHINGLES,
    DEFAULT_SEED,
    LANGUAGE_CODES,
    LSH_BANDS,
    LSH_ROWS,
    MINHASH_SIZE,
    MODEL_NAME,
    MODEL_URL,
    MODEL_VERSION,
    PARTITIONED_LSH_ALGORITHM,
    PreprocessError,
    ProgressReporter,
    atomic_clean_pairs,
    atomic_clean_shards,
    atomic_gzip_jsonl,
    atomic_write_text,
    default_data_root,
    read_jsonl_gzip,
    sha256_bytes,
    sha256_file,
    valid_gzip_jsonl,
)
from .processor import Preprocessor, run_source_stage
from .source_policy import is_budget_exempt_source, is_course_manual_source, is_manual_source, manual_source_issue
from .state import State
from .text import (
    TextHTMLParser,
    _fast_hash64,
    jaccard_similarity,
    minhash_bucket_batch,
    minhash_signature,
    normalize_text,
    quality_rejection,
    script_ratio,
    signature_buckets,
    signature_shingles,
    split_for,
    target_script_count,
    word_shingles,
)

__all__ = [
    "DEFAULT_MAX_SHINGLES", "DEFAULT_SEED", "LANGUAGE_CODES", "LSH_BANDS",
    "LSH_ROWS", "MINHASH_SIZE", "MODEL_NAME", "MODEL_URL", "MODEL_VERSION",
    "PARTITIONED_LSH_ALGORITHM", "PreprocessError", "Preprocessor",
    "ProgressReporter", "State", "TextHTMLParser", "_fast_hash64",
    "atomic_clean_pairs", "atomic_clean_shards", "atomic_gzip_jsonl",
    "atomic_write_text", "default_data_root", "download_model",
    "is_budget_exempt_source", "is_course_manual_source", "is_manual_source", "jaccard_similarity", "main", "manual_source_issue", "minhash_bucket_batch",
    "minhash_signature", "normalize_text", "parse_args", "quality_rejection",
    "read_jsonl_gzip", "run_source_stage", "script_ratio", "sha256_bytes",
    "sha256_file", "signature_buckets", "signature_shingles", "split_for",
    "target_script_count", "valid_gzip_jsonl", "word_shingles",
]
