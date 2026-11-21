"""Pipeline orchestration and the public Preprocessor class."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .cleaning import CleaningMixin
from .common import (
    DEFAULT_MAX_SHINGLES,
    DEFAULT_SEED,
    LANGUAGE_CODES,
    MODEL_NAME,
    PreprocessError,
    ProgressReporter,
)
from .deduplication import DeduplicationMixin
from .extraction import ExtractionMixin
from .source_policy import (
    COURSE_COLLECTION_CATEGORIES,
    COURSE_MANUAL_WIKI_TYPES,
    HPLT_COURSE_MANUAL_DATASETS,
    is_course_manual_source,
    manual_source_issue,
)
from .state import State


class Preprocessor(ExtractionMixin, CleaningMixin, DeduplicationMixin):
    def __init__(self, data_root: Path, config: Path, run_id: str, seed: str = DEFAULT_SEED, shard_records: int = 10_000, model_path: Path | None = None, language_predictor: Callable[[str], tuple[str, float]] | None = None, workers: int = 1, sample_rate: float | None = None, target_downloaded_tokens: int = 400_000_000, calibration_records_per_shard: int = 5_000, max_shingles: int = DEFAULT_MAX_SHINGLES, minimum_train_tokens: int | None = None):
        if workers < 1:
            raise PreprocessError("--workers must be at least 1")
        if sample_rate is not None and not 0 < sample_rate <= 1:
            raise PreprocessError("--sample-rate must be greater than 0 and at most 1")
        if target_downloaded_tokens < 1 or calibration_records_per_shard < 1 or max_shingles < 5:
            raise PreprocessError("target tokens, calibration records, and max shingles must be positive")
        if minimum_train_tokens is not None and minimum_train_tokens < 1:
            raise PreprocessError("--minimum-train-tokens must be positive")
        self.root = data_root
        self.config_path = config
        self.config_root = config.parent.parent
        self.run_id = run_id
        self.seed = seed
        self.shard_records = shard_records
        with config.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        self.registry = loaded
        self.sources = {source["id"]: source for source in loaded["sources"]}
        self.state = State(data_root / "preprocess_state.sqlite")
        self.progress = ProgressReporter(data_root, run_id)
        self.model_path = model_path or data_root / "lid.176.ftz"
        self._language_predictor = language_predictor
        self._fasttext_model: Any | None = None
        self.workers = workers
        self.sample_rate = sample_rate
        self.target_downloaded_tokens = target_downloaded_tokens
        self.calibration_records_per_shard = calibration_records_per_shard
        self.max_shingles = max_shingles
        self.minimum_train_tokens = minimum_train_tokens

    def close(self) -> None:
        self.state.close()

    def remove_replaced_output(self, previous_output: str | None, current_output: str) -> None:
        """Delete an unreferenced derived shard after its replacement is durable."""
        if not previous_output or previous_output == current_output:
            return
        references = self.state.conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id=? AND status='complete' AND output_path=?",
            (self.run_id, previous_output),
        ).fetchone()[0]
        if references:
            return
        allowed_root = (self.root / "processed" / self.run_id).resolve()
        candidate = (self.root / previous_output).resolve()
        try:
            candidate.relative_to(allowed_root)
        except ValueError:
            return
        if candidate.is_file():
            candidate.unlink()

    def selected_sources(self, selected: str | None) -> list[dict[str, Any]]:
        if selected and selected not in self.sources:
            raise PreprocessError(f"unknown source id: {selected}")
        values = [self.sources[selected]] if selected else list(self.sources.values())
        approved = [source for source in values if source.get("approved") and source.get("language") in LANGUAGE_CODES]
        for source in approved:
            course_category = source.get("course_collection_category")
            if course_category is not None and course_category not in COURSE_COLLECTION_CATEGORIES:
                raise PreprocessError(
                    f"source {source['id']} has invalid course_collection_category {course_category!r}"
                )
            hplt_key = f"{source.get('dataset')}:{source.get('dataset_config')}"
            if (
                course_category == "manual"
                and not source.get("manual_collection")
                and source.get("source_type") not in COURSE_MANUAL_WIKI_TYPES
                and hplt_key not in HPLT_COURSE_MANUAL_DATASETS
            ):
                raise PreprocessError(
                    f"source {source['id']} may use the manual category only for Wiki material, "
                    "an allow-listed HPLT exception, or an eligible manual collection"
                )
            if course_category == "manual" and hplt_key in HPLT_COURSE_MANUAL_DATASETS and not is_course_manual_source(source, self.config_root):
                raise PreprocessError(
                    f"source {source['id']} has an invalid HPLT manual-exception declaration"
                )
            if source.get("manual_collection"):
                issue = manual_source_issue(source, self.config_root)
                if issue:
                    raise PreprocessError(f"manual source {source['id']} is not eligible: {issue}")
        return approved

    def _require_preprocess_space(self, sources: list[dict[str, Any]]) -> None:
        """Protect Drive before HPLT-derived shards can consume free space."""
        for source in sources:
            minimum_gib = source.get("minimum_preprocess_free_gib")
            if minimum_gib is None:
                continue
            if not isinstance(minimum_gib, (int, float)) or minimum_gib <= 0:
                raise PreprocessError(f"source {source['id']} has invalid minimum_preprocess_free_gib")
            free = shutil.disk_usage(self.root).free
            required = int(float(minimum_gib) * 1024**3)
            if free < required:
                raise PreprocessError(
                    f"refusing {source['id']}: {free / 1024**3:.1f} GiB free, but "
                    f"preprocessing requires at least {minimum_gib:.1f} GiB free"
                )

    def predictor(self) -> Callable[[str], tuple[str, float]]:
        if self._language_predictor is not None:
            return self._language_predictor
        if not self.model_path.exists():
            raise PreprocessError(f"Missing {MODEL_NAME} at {self.model_path}. Run with --download-model once, then rerun.")
        try:
            import fasttext  # type: ignore
        except ImportError as exc:
            raise PreprocessError("fastText is required for filtering. Install requirements-colab.txt.") from exc
        model = fasttext.load_model(str(self.model_path))
        self._fasttext_model = model

        def predict(text: str) -> tuple[str, float]:
            labels, scores = model.predict(text.replace("\n", " "), k=1)
            return labels[0].removeprefix("__label__"), float(scores[0])

        self._language_predictor = predict
        return predict

    def predict_many(self, texts: list[str]) -> list[tuple[str, float]]:
        """Use fastText's batch API; fallback preserves deterministic test doubles."""
        if not texts:
            return []
        predictor = self.predictor()
        if self._fasttext_model is None:
            return [predictor(text) for text in texts]
        labels, scores = self._fasttext_model.predict([text.replace("\n", " ") for text in texts], k=1)
        return [(label[0].removeprefix("__label__"), float(score[0])) for label, score in zip(labels, scores)]

    def _parallel_source_stage(self, stage: str, sources: list[dict[str, Any]]) -> None:
        """Run independent sources concurrently without concurrent work on one source.

        Every child uses ``--workers 1`` and owns a distinct source's task rows and
        output shards.  The parent remains the only process that runs assembly,
        where a single SQLite LSH database is intentionally required.
        """
        active_workers = min(self.workers, len(sources))
        if active_workers <= 1:
            if stage == "extract":
                self.extract(sources)
            elif stage == "clean":
                self.clean(sources)
            else:
                raise ValueError(stage)
            return
        self.progress.emit(f"{stage}: starting {len(sources)} independent source(s) with {active_workers} CPU worker processes")
        with ProcessPoolExecutor(max_workers=active_workers) as executor:
            futures = {
                executor.submit(
                    run_source_stage,
                    str(Path(__file__).resolve().parent.parent / "preprocess.py"),
                    str(self.root),
                    str(self.config_path),
                    self.run_id,
                    self.seed,
                    self.shard_records,
                    str(self.model_path),
                    self.sample_rate,
                    self.target_downloaded_tokens,
                    self.calibration_records_per_shard,
                    self.max_shingles,
                    stage,
                    source["id"],
                ): source["id"]
                for source in sources
            }
            for future in as_completed(futures):
                source_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.progress.emit(f"{stage}: worker failed for {source_id}: {type(exc).__name__}: {exc}")
                    raise PreprocessError(f"{stage} worker failed for {source_id}: {exc}") from exc
                self.progress.emit(f"{stage}: worker complete for {source_id}")

    def run(self, selected: str | None = None, stages: set[str] | None = None, finalize: bool = False, allow_low_manual_fraction: bool = False) -> dict[str, dict[str, Any]]:
        stages = stages or {"extract", "calibrate", "clean", "assemble"}
        recovered = self.state.reconcile(self.root, self.run_id)
        if recovered:
            print(f"Recovered {recovered} durable shard(s) written before a prior interruption.")
            self.progress.emit(f"recovered {recovered} durable shard(s) from a prior interruption")
        sources = self.selected_sources(selected)
        if {"extract", "clean", "assemble"}.intersection(stages):
            self._require_preprocess_space(sources)
        if "extract" in stages:
            self._parallel_source_stage("extract", sources)
        if "calibrate" in stages:
            self.calibrate(sources)
        if "clean" in stages:
            self._parallel_source_stage("clean", sources)
        result: dict[str, dict[str, Any]] = {}
        if "assemble" in stages:
            for language in sorted({source["language"] for source in sources}):
                manifest = self.assemble(language, finalize, allow_low_manual_fraction)
                if manifest:
                    result[language] = manifest
        if finalize:
            failures = []
            # A recovery run commonly refreshes one HPLT language at a time.
            # Do not make a Telugu-only finalization fail solely because Nepali
            # was intentionally left for the next sequential run.
            for language in sorted({source["language"] for source in sources}):
                manifest = result.get(language)
                if not manifest:
                    failures.append(f"{language}: no manifest was assembled")
                elif manifest["training_status"] != "final":
                    failures.append(f"{language}: {manifest.get('finalization_error', 'finalization requirements were not met')}")
            if failures:
                raise PreprocessError("Refusing final-for-training status: " + "; ".join(failures))
        return result

    def print_status(self) -> None:
        rows = self.state.status(self.run_id)
        if not rows:
            print("No preprocessing work recorded.")
        for row in rows:
            print(f"{row['run_id']:20} {row['stage']:10} {row['source_id']:28} {row['status']:10} {row['count']}")
        stage_rows = self.state.conn.execute(
            "SELECT language,stage,status,updated_at FROM stages WHERE run_id=? ORDER BY language,stage",
            (self.run_id,),
        )
        for row in stage_rows:
            print(f"{self.run_id:20} {row['stage']:10} {row['language']:28} {row['status']:10} {row['updated_at']}")




def run_source_stage(
    script: str,
    data_root: str,
    config: str,
    run_id: str,
    seed: str,
    shard_records: int,
    model_path: str,
    sample_rate: float | None,
    target_downloaded_tokens: int,
    calibration_records_per_shard: int,
    max_shingles: int,
    stage: str,
    source_id: str,
) -> None:
    """Child-process entrypoint for one independent source/stage pair.

    It deliberately invokes the same script with one worker.  This retains the
    normal durable-output reconciliation behaviour if a Colab runtime stops while
    a worker is writing a shard.
    """
    command = [
        sys.executable, "-u", script,
        "--data-root", data_root,
        "--config", config,
        "--run-id", run_id,
        "--seed", seed,
        "--shard-records", str(shard_records),
        "--model-path", model_path,
        "--target-downloaded-tokens", str(target_downloaded_tokens),
        "--calibration-records-per-shard", str(calibration_records_per_shard),
        "--max-shingles", str(max_shingles),
        "--source", source_id,
        "--stage", stage,
        "--workers", "1",
    ]
    if sample_rate is not None:
        command.extend(["--sample-rate", str(sample_rate)])
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"subprocess exited with status {completed.returncode}")
