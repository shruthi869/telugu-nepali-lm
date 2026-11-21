"""Focused checks for the final SentencePiece-token audit helpers."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import preprocess


SPEC = importlib.util.spec_from_file_location(
    "count_training_tokens",
    Path(__file__).resolve().parents[1] / "report" / "count_training_tokens.py",
)
assert SPEC and SPEC.loader
counter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(counter)


class ExactTokenTests(unittest.TestCase):
    def test_per_source_sentencepiece_counts_sum_to_shard_total(self) -> None:
        class FakeSentencePiece:
            def encode(self, texts: list[str], **_: object) -> list[list[int]]:
                return [list(range(len(text.split()))) for text in texts]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl.gz"
            preprocess.atomic_gzip_jsonl(path, [
                {"cleaned_text": "one two", "source_id": "hplt2_tel_cleaned", "course_collection_category": "manual"},
                {"cleaned_text": "three four five", "source_id": "sangraha_tel_verified", "course_collection_category": "downloaded"},
            ])
            result = counter.count_shard(FakeSentencePiece(), path, batch_size=1)
        self.assertEqual(result["sentencepiece_tokens"], 5)
        self.assertEqual(sum(item["sentencepiece_tokens"] for item in result["sources"].values()), 5)
        self.assertEqual(result["sources"]["hplt2_tel_cleaned"]["course_collection_category"], "manual")


if __name__ == "__main__":
    unittest.main()
