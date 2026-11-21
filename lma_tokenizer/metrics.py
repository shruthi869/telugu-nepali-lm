"""Held-out tokenizer metrics and deterministic vocabulary selection."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


def evaluate_tokenizer(
    processor: Any,
    records: Iterable[dict[str, Any]],
    maximum_whitespace_tokens: int = 0,
    num_threads: int = 2,
) -> dict[str, Any]:
    """Measure fertility, unknowns, character efficiency, and token frequencies."""
    frequency: Counter[int] = Counter()
    documents = whitespace_tokens = subword_tokens = unknown_tokens = 0
    byte_fallback_tokens = documents_with_byte_fallback = 0
    characters = non_whitespace_characters = 0
    examples: list[dict[str, Any]] = []
    unknown_examples: list[dict[str, Any]] = []
    byte_fallback_examples: list[dict[str, Any]] = []
    unk_id = int(processor.unk_id())
    byte_ids = {
        token_id
        for token_id in range(int(processor.vocab_size()))
        if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", str(processor.id_to_piece(token_id)))
    }
    batch: list[tuple[str, str]] = []

    def consume(values: list[tuple[str, str]]) -> None:
        nonlocal documents, whitespace_tokens, subword_tokens, unknown_tokens
        nonlocal byte_fallback_tokens, documents_with_byte_fallback
        nonlocal characters, non_whitespace_characters
        texts = [text for _, text in values]
        encoded_batch = processor.encode(texts, out_type=int, num_threads=num_threads)
        for (document_id, text), ids in zip(values, encoded_batch):
            words = len(text.split())
            documents += 1
            whitespace_tokens += words
            characters += len(text)
            non_whitespace_characters += sum(not character.isspace() for character in text)
            subword_tokens += len(ids)
            frequency.update(ids)
            unknown = sum(token_id == unk_id for token_id in ids)
            byte_count = sum(token_id in byte_ids for token_id in ids)
            unknown_tokens += unknown
            byte_fallback_tokens += byte_count
            documents_with_byte_fallback += int(byte_count > 0)
            if len(examples) < 10:
                snippet = text[:300]
                example_ids = processor.encode(snippet, out_type=int)
                examples.append({
                    "document_id": document_id,
                    "text": snippet,
                    "token_ids": example_ids[:120],
                    "pieces": [processor.id_to_piece(token_id) for token_id in example_ids[:120]],
                    "truncated": len(example_ids) > 120 or len(text) > len(snippet),
                })
            if unknown and len(unknown_examples) < 20:
                unknown_examples.append({"document_id": document_id, "text": text[:300], "unknown_tokens": unknown})
            if byte_count and len(byte_fallback_examples) < 20:
                byte_fallback_examples.append(
                    {
                        "document_id": document_id,
                        "text": text[:300],
                        "byte_fallback_tokens": byte_count,
                    }
                )

    for index, record in enumerate(records, 1):
        text = record["cleaned_text"]
        if maximum_whitespace_tokens and whitespace_tokens + sum(len(value.split()) for _, value in batch) >= maximum_whitespace_tokens:
            break
        batch.append((str(record.get("document_id", index)), text))
        if len(batch) >= 128:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    top_tokens = [
        {
            "id": token_id,
            "piece": processor.id_to_piece(token_id),
            "count": count,
            "fraction": count / max(subword_tokens, 1),
        }
        for token_id, count in frequency.most_common(100)
    ]
    return {
        "documents": documents,
        "whitespace_tokens": whitespace_tokens,
        "subword_tokens": subword_tokens,
        "characters": characters,
        "non_whitespace_characters": non_whitespace_characters,
        "fertility_subwords_per_whitespace_token": subword_tokens / max(whitespace_tokens, 1),
        "average_characters_per_token": non_whitespace_characters / max(subword_tokens, 1),
        "unknown_tokens": unknown_tokens,
        "unknown_token_rate": unknown_tokens / max(subword_tokens, 1),
        "byte_fallback_tokens": byte_fallback_tokens,
        "byte_fallback_token_rate": byte_fallback_tokens / max(subword_tokens, 1),
        "documents_with_byte_fallback": documents_with_byte_fallback,
        "documents_with_byte_fallback_rate": documents_with_byte_fallback / max(documents, 1),
        "observed_vocabulary_size": len(frequency),
        "vocabulary_utilization": len(frequency) / max(int(processor.vocab_size()), 1),
        "tokens_observed_once": sum(count == 1 for count in frequency.values()),
        "top_tokens": top_tokens,
        "tokenization_examples": examples,
        "unknown_examples": unknown_examples,
        "byte_fallback_examples": byte_fallback_examples,
    }


def choose_candidate(
    candidates: list[dict[str, Any]],
    maximum_unknown_rate: float,
    fertility_tolerance: float,
) -> dict[str, Any]:
    """Choose the smallest vocabulary within tolerance of the best valid fertility."""
    if not candidates:
        raise ValueError("no tokenizer candidates were evaluated")
    valid = [candidate for candidate in candidates if candidate["metrics"]["unknown_token_rate"] <= maximum_unknown_rate]
    if not valid:
        best_unknown = min(candidate["metrics"]["unknown_token_rate"] for candidate in candidates)
        raise ValueError(
            "no tokenizer candidate met the maximum unknown-token rate "
            f"of {maximum_unknown_rate:.6%}; best candidate was {best_unknown:.6%}"
        )
    pool = valid
    best_fertility = min(candidate["metrics"]["fertility_subwords_per_whitespace_token"] for candidate in pool)
    tolerated = [
        candidate
        for candidate in pool
        if candidate["metrics"]["fertility_subwords_per_whitespace_token"] <= best_fertility * (1.0 + fertility_tolerance)
    ]
    return min(tolerated, key=lambda candidate: int(candidate["vocabulary_size"]))
