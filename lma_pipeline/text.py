"""Text normalization, filtering, hashing, and split helpers."""

from __future__ import annotations

import functools
import hashlib
import html.parser
import re
import unicodedata
from collections import Counter
from typing import Iterable, Iterator

from .common import (
    DEFAULT_MAX_SHINGLES,
    DEFAULT_SEED,
    EMAIL_RE,
    ID_RE,
    LSH_BANDS,
    LSH_ROWS,
    MINHASH_SIZE,
    PHONE_RE,
    SCRIPT_RANGES,
    SENTENCE_RE,
    TOKEN_RE,
    URL_RE,
    _xxhash,
)
def normalize_text(text: str) -> str:
    """NFC normalization plus conservative PII redaction without stripping Indic joiners."""
    text = unicodedata.normalize("NFC", text)
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    text = ID_RE.sub("[ID_REDACTED]", text)
    # Keep ZWJ/ZWNJ, which can be meaningful in Indic text.  Other control codes
    # are replaced with whitespace rather than silently joining neighbouring words.
    text = "".join(
        " " if unicodedata.category(char) == "Cc" and char not in "\n\t\r" else char
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def target_script_count(text: str, language: str) -> int:
    lower, upper = SCRIPT_RANGES[language]
    return sum(lower <= ord(char) <= upper for char in text)


def script_ratio(text: str, language: str) -> float:
    lower, upper = SCRIPT_RANGES[language]
    alphabetic = [char for char in text if char.isalpha()]
    if not alphabetic:
        return 0.0
    return sum(lower <= ord(char) <= upper for char in alphabetic) / len(alphabetic)


def repeated_content(text: str) -> bool:
    paragraphs = [part.strip().casefold() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs and max(Counter(paragraphs).values()) >= 3:
        return True
    sentences = [part.strip().casefold() for part in SENTENCE_RE.findall(text) if len(part.strip()) >= 20]
    counts = Counter(sentences)
    if counts and max(counts.values()) >= 4:
        return True
    return bool(sentences and sum(count - 1 for count in counts.values()) / len(sentences) > 0.5)


def quality_rejection(text: str, language: str) -> str | None:
    if len(text) < 200:
        return "too_short"
    if target_script_count(text, language) < 80:
        return "too_few_target_script_characters"
    if script_ratio(text, language) < 0.70:
        return "low_target_script_ratio"
    if text.count("\ufffd") / max(len(text), 1) > 0.01:
        return "replacement_character_rate"
    urls = URL_RE.findall(text)
    if len(urls) >= 4 or sum(len(item) for item in urls) / max(len(text), 1) > 0.10:
        return "url_heavy"
    symbols = sum(unicodedata.category(char)[0] in {"S", "P"} for char in text)
    if symbols / max(len(text), 1) > 0.35:
        return "symbol_heavy"
    if repeated_content(text):
        return "repeated_content"
    return None


def word_shingles(text: str, width: int = 5, max_shingles: int | None = None) -> set[str]:
    words = TOKEN_RE.findall(text.casefold())
    if not words:
        return {""}
    if len(words) <= width:
        return {" ".join(words)}
    count = len(words) - width + 1
    if max_shingles is None or count <= max_shingles:
        indices = range(count)
    else:
        # Cover a long document from beginning to end rather than retaining only
        # its opening.  This cap affects only the MinHash candidate stage; the
        # final Jaccard verification still uses all shingles.
        indices = {round(index * (count - 1) / (max_shingles - 1)) for index in range(max_shingles)}
    return {" ".join(words[index : index + width]) for index in indices}


def signature_shingles(text: str, width: int = 5, max_shingles: int = DEFAULT_MAX_SHINGLES) -> Iterator[str]:
    """Yield the same candidate shingles without building a second Python set.

    Duplicate shingles do not change a MinHash minimum, so streaming them is
    signature-equivalent while avoiding a large set allocation and Python's
    additional string-hash pass for ordinary, mostly unique documents.
    """
    words = TOKEN_RE.findall(text.casefold())
    if not words:
        yield ""
        return
    if len(words) <= width:
        yield " ".join(words)
        return
    count = len(words) - width + 1
    if count <= max_shingles:
        indices: Iterable[int] = range(count)
    else:
        indices = (round(index * (count - 1) / (max_shingles - 1)) for index in range(max_shingles))
    for index in indices:
        yield " ".join(words[index : index + width])


def _fast_hash64(value: str | bytes, seed: int = 0) -> int:
    """Deterministic xxHash64, with a stdlib fallback for developer test runs."""
    if _xxhash is None:
        data = value.encode("utf-8") if isinstance(value, str) else value
        return int.from_bytes(
            hashlib.blake2b(data, digest_size=8, key=seed.to_bytes(8, "big", signed=False)).digest(),
            "big",
        )
    return _xxhash.xxh64_intdigest(value, seed=seed)


@functools.lru_cache(maxsize=32)
def _minhash_permutation_seeds(seed: str, size: int) -> tuple[int, ...]:
    return tuple(_fast_hash64(f"{seed}:permutation:{index}") for index in range(size))


def minhash_signature(text: str, seed: str = DEFAULT_SEED, size: int = MINHASH_SIZE, max_shingles: int = DEFAULT_MAX_SHINGLES) -> tuple[int, ...]:
    """Densified one-permutation MinHash in O(number of shingles + signature size).

    Conventional MinHash hashes every shingle once for each of 128 permutations.
    That is far too expensive for a billion-scale corpus.  This standard MinHash
    variant hashes each shingle once, places it in one of ``size`` bins, and
    deterministically densifies empty bins.  It retains a 128-value signature for
    the same LSH layout.  xxHash is only a candidate accelerator: SHA-256 remains
    the document identity and full Jaccard remains the deletion decision.
    """
    empty = (1 << 64) - 1
    values = [empty] * size
    # One stable permutation is enough: use its low bits to choose a bin and the
    # remaining bits as the rank within that bin.
    base_seed = _fast_hash64(f"{seed}:doph-v1")
    for shingle in signature_shingles(text, max_shingles=max_shingles):
        hashed = _fast_hash64(shingle, base_seed)
        index = hashed % size
        rank = hashed // size
        if rank < values[index]:
            values[index] = rank
    if all(value == empty for value in values):
        return tuple(_fast_hash64(f"{seed}:empty:{index}") for index in range(size))
    # Densified one-permutation hashing: map an empty bin to the next populated
    # bin, then rehash with its destination index so empty bins do not create a
    # universal collision between short documents.
    for index, value in enumerate(values):
        if value != empty:
            continue
        step = 1
        while values[(index + step) % size] == empty:
            step += 1
        values[index] = _fast_hash64(values[(index + step) % size].to_bytes(8, "big"), index + 1)
    return tuple(values)


def signature_buckets(signature: tuple[int, ...], bands: int = LSH_BANDS, rows: int = LSH_ROWS) -> Iterator[tuple[int, int]]:
    if len(signature) != bands * rows:
        raise ValueError("signature size must equal bands * rows")
    for band in range(bands):
        block = signature[band * rows : (band + 1) * rows]
        packed = b"".join(value.to_bytes(8, "big") for value in block)
        yield band, _fast_hash64(packed, band)


def minhash_bucket_batch(
    records: list[tuple[int, str]], seed: str, max_shingles: int
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Pickle-friendly CPU worker for a bounded MinHash/LSH batch."""
    return [
        (document_id, list(signature_buckets(minhash_signature(text, seed, max_shingles=max_shingles))))
        for document_id, text in records
    ]


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a, b = word_shingles(text_a), word_shingles(text_b)
    intersection = len(a.intersection(b))
    return intersection / max(len(a) + len(b) - intersection, 1)


def split_for(document_id: str, seed: str = DEFAULT_SEED) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).digest()[:8], "big") % 10_000
    return "train" if value < 9800 else "validation" if value < 9900 else "test"


class TextHTMLParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)
