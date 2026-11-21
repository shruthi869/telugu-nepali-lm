"""Resumable MinHash/LSH near-duplicate detection."""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import sqlite3
import struct
import uuid
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from .common import (
    LSH_BANDS,
    LSH_PARTITIONS,
    LSH_ROWS,
    MAX_CACHED_VERIFICATION_SHINGLES,
    MAX_CANDIDATE_PAIRS,
    MAX_LSH_BUCKET_DOCUMENTS,
    MINHASH_SIZE,
    PARTITIONED_LSH_ALGORITHM,
    SIGNATURE_BATCH_SIZE,
    SIGNATURE_CHECKPOINT_DOCUMENTS,
    SIGNATURE_CHUNK_MAGIC,
    SIGNATURE_PROGRESS_SCHEMA,
    PreprocessError,
    atomic_write_text,
)
from .text import minhash_bucket_batch, word_shingles


class NearDedupMixin:
    def _near_deduplicate(self, language: str, drive_db_path: Path) -> tuple[Path, int]:
        db_path = self._near_work_db(language, drive_db_path)
        db = sqlite3.connect(db_path)
        if db_path != drive_db_path:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
        db.execute("CREATE TABLE IF NOT EXISTS dedup_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        configuration = json.dumps(
            {
                "algorithm": PARTITIONED_LSH_ALGORITHM,
                "minhash_size": MINHASH_SIZE,
                "bands": LSH_BANDS,
                "rows": LSH_ROWS,
                "max_shingles": self.max_shingles,
                "max_lsh_bucket_documents": MAX_LSH_BUCKET_DOCUMENTS,
                "partitions": LSH_PARTITIONS,
            },
            sort_keys=True,
        )
        previous_configuration = db.execute("SELECT value FROM dedup_metadata WHERE key='configuration'").fetchone()
        if previous_configuration and previous_configuration[0] != configuration:
            # The previous table-based LSH state is derived and cannot be mixed
            # with partitioned records.  Dropping it does not affect exact dedup.
            db.execute("DROP TABLE IF EXISTS buckets")
            db.execute("DROP TABLE IF EXISTS minhash_done")
            db.execute("DROP TABLE IF EXISTS pairs")
            db.execute("DELETE FROM dedup_metadata")
            db.commit()
        db.execute("INSERT OR REPLACE INTO dedup_metadata(key,value) VALUES('configuration',?)", (configuration,))
        db.commit()
        complete = db.execute("SELECT value FROM dedup_metadata WHERE key='near_complete'").fetchone()
        if complete:
            self.progress.emit(f"near-dedup: {language}: reusing completed LSH state")
            db.close()
            if db_path != drive_db_path and self._db_near_complete(drive_db_path) != int(complete[0]):
                self.progress.emit(f"near-dedup: {language}: checkpointing completed local LSH state to Drive")
                self._sqlite_backup(db_path, drive_db_path)
            shutil.rmtree(db_path.with_name(db_path.stem + ".lsh-partitions"), ignore_errors=True)
            shutil.rmtree(self.root / "audit" / self.run_id / "lsh-checkpoints" / language, ignore_errors=True)
            return db_path, int(complete[0])
        # `dropped` changes and `near_complete` are committed together by this
        # version. Reset any orphaned flags from an older interrupted version so
        # documents can never silently disappear on resume.
        db.execute("UPDATE docs SET dropped=0 WHERE dropped<>0")
        db.commit()
        document_total = db.execute("SELECT COUNT(*) FROM docs WHERE dropped=0").fetchone()[0]
        partition_root = db_path.with_name(db_path.stem + ".lsh-partitions")
        marker = partition_root / "signatures-complete.json"
        progress_marker = partition_root / "signatures-progress.json"
        drive_checkpoint_root = self.root / "audit" / self.run_id / "lsh-checkpoints" / language
        drive_progress_marker = drive_checkpoint_root / "signatures-progress.json"
        record_format = struct.Struct(">BQQ")  # band, 64-bit bucket, docs.rowid
        expected_partition_bytes = document_total * LSH_BANDS * record_format.size
        partition_paths = [partition_root / f"part-{index:03d}.bin" for index in range(LSH_PARTITIONS)]

        def write_audit(relative: Path, payload: dict[str, Any]) -> None:
            """Audit files must not make preprocessing fail when Drive is transient."""
            try:
                target = self.root / relative
                atomic_write_text(target, json.dumps(payload, sort_keys=True))
            except OSError as exc:
                self.progress.emit(f"near-dedup: {language}: audit warning: {exc}")

        def valid_partition_checkpoint() -> bool:
            try:
                details = json.loads(marker.read_text(encoding="utf-8"))
                return (
                    details.get("configuration") == json.loads(configuration)
                    and details.get("documents") == document_total
                    and all(path.exists() and path.stat().st_size % record_format.size == 0 for path in partition_paths)
                    and sum(path.stat().st_size for path in partition_paths) == expected_partition_bytes
                )
            except (OSError, ValueError, json.JSONDecodeError):
                return False

        def valid_progress_details(details: dict[str, Any]) -> bool:
            try:
                sizes = details["partition_sizes"]
                written = int(details["documents_written"])
                last_rowid = int(details["last_rowid"])
                if not (
                    details.get("schema_version") == SIGNATURE_PROGRESS_SCHEMA
                    and details.get("configuration") == json.loads(configuration)
                    and details.get("documents_total") == document_total
                    and 0 <= written <= document_total
                    and last_rowid >= 0
                    and isinstance(sizes, list)
                    and len(sizes) == LSH_PARTITIONS
                    and all(isinstance(size, int) and size >= 0 and size % record_format.size == 0 for size in sizes)
                    and sum(sizes) == written * LSH_BANDS * record_format.size
                ):
                    return False
                frontier_count = db.execute(
                    "SELECT COUNT(*) FROM docs WHERE dropped=0 AND rowid<=?",
                    (last_rowid,),
                ).fetchone()[0]
                return frontier_count == written
            except (KeyError, TypeError, ValueError, sqlite3.Error):
                return False

        def resumable_signature_progress() -> dict[str, Any] | None:
            """Validate the local rowid frontier and its per-partition byte sizes."""
            try:
                details = json.loads(progress_marker.read_text(encoding="utf-8"))
                if not valid_progress_details(details):
                    return None
                if not all(
                    path.exists() and path.stat().st_size >= size
                    for path, size in zip(partition_paths, details["partition_sizes"])
                ):
                    return None
                return details
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return None

        def drive_signature_progress() -> dict[str, Any] | None:
            """Validate a cross-runtime Drive frontier built from incremental chunks."""
            try:
                details = json.loads(drive_progress_marker.read_text(encoding="utf-8"))
                chunks = details.get("chunks")
                if not valid_progress_details(details) or not isinstance(chunks, list):
                    return None
                if not all(
                    isinstance(name, str)
                    and re.fullmatch(r"checkpoint-\d{9}\.bin", name)
                    and (drive_checkpoint_root / name).is_file()
                    for name in chunks
                ):
                    return None
                return details
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return None

        def restore_drive_progress(details: dict[str, Any]) -> bool:
            """Reconstruct bounded local partitions from Drive chunk checkpoints."""
            temporary_root = partition_root.with_name(partition_root.name + f".{uuid.uuid4().hex}.restore")
            shutil.rmtree(temporary_root, ignore_errors=True)
            try:
                temporary_root.mkdir(parents=True)
                outputs = []
                try:
                    for path in partition_paths:
                        outputs.append((temporary_root / path.name).open("wb", buffering=1024 * 1024))
                    for name in details["chunks"]:
                        with (drive_checkpoint_root / name).open("rb") as chunk:
                            if chunk.read(len(SIGNATURE_CHUNK_MAGIC)) != SIGNATURE_CHUNK_MAGIC:
                                raise PreprocessError(f"invalid durable signature chunk {name}")
                            for output in outputs:
                                length_bytes = chunk.read(8)
                                if len(length_bytes) != 8:
                                    raise PreprocessError(f"truncated durable signature chunk {name}")
                                remaining = struct.unpack(">Q", length_bytes)[0]
                                while remaining:
                                    block = chunk.read(min(1024 * 1024, remaining))
                                    if not block:
                                        raise PreprocessError(f"truncated durable signature chunk {name}")
                                    output.write(block)
                                    remaining -= len(block)
                            if chunk.read(1):
                                raise PreprocessError(f"unexpected trailing data in signature chunk {name}")
                finally:
                    for output in outputs:
                        output.close()
                restored_sizes = [(temporary_root / path.name).stat().st_size for path in partition_paths]
                if restored_sizes != details["partition_sizes"]:
                    raise PreprocessError("durable signature chunks do not match their recorded partition sizes")
                shutil.rmtree(partition_root, ignore_errors=True)
                os.replace(temporary_root, partition_root)
                atomic_write_text(progress_marker, json.dumps(details, sort_keys=True))
                self.progress.emit(
                    f"near-dedup: {language}: restored {details['documents_written']:,} compact signatures from Drive"
                )
                return True
            except (OSError, PreprocessError) as exc:
                shutil.rmtree(temporary_root, ignore_errors=True)
                self.progress.emit(f"near-dedup: {language}: Drive signature restore warning: {exc}")
                return False

        if not valid_partition_checkpoint():
            resume = resumable_signature_progress()
            durable_resume = drive_signature_progress()
            if durable_resume and (
                resume is None
                or int(durable_resume["documents_written"]) > int(resume["documents_written"])
            ):
                if restore_drive_progress(durable_resume):
                    resume = resumable_signature_progress()
            if resume is None:
                shutil.rmtree(partition_root, ignore_errors=True)
                partition_root.mkdir(parents=True, exist_ok=True)
                for path in partition_paths:
                    path.touch()
                number = 0
                last_rowid = 0
                checkpointed_number = 0
            else:
                number = int(resume["documents_written"])
                last_rowid = int(resume["last_rowid"])
                checkpointed_number = number
                for path, size in zip(partition_paths, resume["partition_sizes"]):
                    with path.open("r+b") as handle:
                        handle.truncate(size)
                self.progress.emit(
                    f"near-dedup: {language}: resuming compact signatures at {number:,} / {document_total:,} documents"
                )
            free = shutil.disk_usage(partition_root).free
            durable_bytes = sum(path.stat().st_size for path in partition_paths)
            required = max(0, expected_partition_bytes - durable_bytes) + 512 * 1024 * 1024
            if free < required:
                raise PreprocessError(
                    f"Not enough local disk for compact {language} LSH partitions: "
                    f"need {required / 2**30:.1f} GiB, have {free / 2**30:.1f} GiB"
                )
            self.progress.emit(
                f"near-dedup: {language}: writing compact partitioned LSH records for {document_total:,} documents with {min(self.workers, 2)} CPU worker(s)"
            )
            writers = []
            try:
                for path in partition_paths:
                    writers.append(path.open("ab", buffering=1024 * 1024))
            except Exception:
                for writer in writers:
                    writer.close()
                db.close()
                raise

            def batches(after_rowid: int) -> Iterator[list[tuple[int, str]]]:
                # SQLite's built-in JSON extraction avoids decoding every other
                # provenance field in Python before sending text to CPU workers.
                cursor = db.execute(
                    "SELECT rowid,json_extract(record_json,'$.cleaned_text') "
                    "FROM docs WHERE dropped=0 AND rowid>? ORDER BY rowid",
                    (after_rowid,),
                )
                while rows := cursor.fetchmany(SIGNATURE_BATCH_SIZE):
                    batch = [(int(rowid), text) for rowid, text in rows]
                    if any(not isinstance(text, str) for _, text in batch):
                        raise PreprocessError("exact-dedup database contains a document without cleaned_text")
                    yield batch

            def write_batch(results: list[tuple[int, list[tuple[int, int]]]], written: int) -> None:
                for document_id, buckets in results:
                    for band, bucket in buckets:
                        writers[bucket & (LSH_PARTITIONS - 1)].write(record_format.pack(band, bucket, document_id))
                self.progress.tick("near-dedup", language, written, " compact signatures written")

            def checkpoint_signatures_to_drive(written: int, frontier_rowid: int, sizes: list[int]) -> bool:
                """Append only new partition bytes to one atomic Drive chunk."""
                existing = drive_signature_progress()
                if existing and int(existing["documents_written"]) >= written:
                    return True
                previous_sizes = list(existing["partition_sizes"]) if existing else [0] * LSH_PARTITIONS
                chunks = list(existing["chunks"]) if existing else []
                if any(previous > current for previous, current in zip(previous_sizes, sizes)):
                    previous_sizes = [0] * LSH_PARTITIONS
                    chunks = []
                delta_bytes = sum(current - previous for previous, current in zip(previous_sizes, sizes))
                chunk_name = f"checkpoint-{written:09d}.bin"
                chunk_path = drive_checkpoint_root / chunk_name
                temporary = chunk_path.with_name(chunk_path.name + f".{uuid.uuid4().hex}.tmp")
                try:
                    drive_checkpoint_root.mkdir(parents=True, exist_ok=True)
                    free = shutil.disk_usage(drive_checkpoint_root).free
                    if free < delta_bytes + 256 * 1024 * 1024:
                        raise PreprocessError(
                            f"not enough Drive space for signature checkpoint: need "
                            f"{(delta_bytes + 256 * 1024 * 1024) / 2**30:.1f} GiB, have {free / 2**30:.1f} GiB"
                        )
                    with temporary.open("wb") as output:
                        output.write(SIGNATURE_CHUNK_MAGIC)
                        for path, start, end in zip(partition_paths, previous_sizes, sizes):
                            length = end - start
                            output.write(struct.pack(">Q", length))
                            with path.open("rb") as source:
                                source.seek(start)
                                remaining = length
                                while remaining:
                                    block = source.read(min(1024 * 1024, remaining))
                                    if not block:
                                        raise PreprocessError(f"local signature partition {path.name} ended unexpectedly")
                                    output.write(block)
                                    remaining -= len(block)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, chunk_path)
                    updated_chunks = chunks + [chunk_name]
                    durable_details = {
                        "schema_version": SIGNATURE_PROGRESS_SCHEMA,
                        "configuration": json.loads(configuration),
                        "documents_total": document_total,
                        "documents_written": written,
                        "last_rowid": frontier_rowid,
                        "partition_sizes": sizes,
                        "chunks": updated_chunks,
                    }
                    atomic_write_text(drive_progress_marker, json.dumps(durable_details, sort_keys=True))
                    referenced = set(updated_chunks)
                    for stale in drive_checkpoint_root.glob("checkpoint-*.bin"):
                        if stale.name not in referenced:
                            stale.unlink(missing_ok=True)
                    for stale in drive_checkpoint_root.glob("*.tmp"):
                        stale.unlink(missing_ok=True)
                    return True
                except (OSError, PreprocessError) as exc:
                    temporary.unlink(missing_ok=True)
                    self.progress.emit(
                        f"near-dedup: {language}: Drive signature checkpoint warning: {exc}; local checkpoint remains usable"
                    )
                    return False

            def checkpoint_signatures(written: int, frontier_rowid: int) -> None:
                for writer in writers:
                    writer.flush()
                    os.fsync(writer.fileno())
                sizes = [writer.tell() for writer in writers]
                drive_durable = checkpoint_signatures_to_drive(written, frontier_rowid, sizes)
                atomic_write_text(
                    progress_marker,
                    json.dumps(
                        {
                            "schema_version": SIGNATURE_PROGRESS_SCHEMA,
                            "configuration": json.loads(configuration),
                            "documents_total": document_total,
                            "documents_written": written,
                            "last_rowid": frontier_rowid,
                            "partition_sizes": sizes,
                        },
                        sort_keys=True,
                    ),
                )
                self.progress.emit(
                    f"near-dedup: {language}: {'durable Drive' if drive_durable else 'local'} signature checkpoint "
                    f"{written:,} / {document_total:,}"
                )

            try:
                if self.workers > 1:
                    iterator = iter(batches(last_rowid))
                    with ProcessPoolExecutor(max_workers=min(self.workers, 2)) as executor:
                        pending: dict[Any, int] = {}
                        ready: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
                        submitted_sequence = 0
                        write_sequence = 0
                        prefetch = min(self.workers, 2) * 2

                        def submit_available() -> None:
                            nonlocal submitted_sequence
                            while len(pending) + len(ready) < prefetch:
                                batch = next(iterator, None)
                                if not batch:
                                    return
                                future = executor.submit(minhash_bucket_batch, batch, self.seed, self.max_shingles)
                                pending[future] = submitted_sequence
                                submitted_sequence += 1

                        submit_available()
                        while pending:
                            future = next(as_completed(pending))
                            sequence = pending.pop(future)
                            ready[sequence] = future.result()
                            while write_sequence in ready:
                                results = ready.pop(write_sequence)
                                number += len(results)
                                last_rowid = results[-1][0]
                                write_batch(results, number)
                                write_sequence += 1
                                if number - checkpointed_number >= SIGNATURE_CHECKPOINT_DOCUMENTS:
                                    checkpoint_signatures(number, last_rowid)
                                    checkpointed_number = number
                            submit_available()
                else:
                    for batch in batches(last_rowid):
                        results = minhash_bucket_batch(batch, self.seed, self.max_shingles)
                        number += len(results)
                        last_rowid = results[-1][0]
                        write_batch(results, number)
                        if number - checkpointed_number >= SIGNATURE_CHECKPOINT_DOCUMENTS:
                            checkpoint_signatures(number, last_rowid)
                            checkpointed_number = number
                if number != checkpointed_number:
                    checkpoint_signatures(number, last_rowid)
            except BaseException:
                db.close()
                raise
            finally:
                for writer in writers:
                    writer.close()
            actual_partition_bytes = sum(path.stat().st_size for path in partition_paths)
            if number != document_total or actual_partition_bytes != expected_partition_bytes:
                raise PreprocessError(
                    f"compact {language} signature checkpoint is incomplete: "
                    f"{number:,}/{document_total:,} documents, {actual_partition_bytes:,}/{expected_partition_bytes:,} bytes"
                )
            atomic_write_text(
                marker,
                json.dumps({"configuration": json.loads(configuration), "documents": number}, sort_keys=True),
            )
            progress_marker.unlink(missing_ok=True)

        self.progress.emit(f"near-dedup: {language}: grouping {LSH_PARTITIONS} compact LSH partitions")
        candidate_pairs: set[tuple[int, int]] = set()
        oversized_buckets = 0
        for partition in range(LSH_PARTITIONS):
            path = partition_root / f"part-{partition:03d}.bin"
            records: list[tuple[int, int, int]] = []
            with path.open("rb") as handle:
                while block := handle.read(record_format.size * 65_536):
                    for offset in range(0, len(block), record_format.size):
                        records.append(record_format.unpack_from(block, offset))
            records.sort()
            index = 0
            while index < len(records):
                band, bucket, _ = records[index]
                end = index + 1
                while end < len(records) and records[end][:2] == (band, bucket):
                    end += 1
                ids = [entry[2] for entry in records[index:end]]
                if len(ids) > MAX_LSH_BUCKET_DOCUMENTS:
                    oversized_buckets += 1
                elif len(ids) > 1:
                    for left in range(len(ids)):
                        for right in range(left + 1, len(ids)):
                            candidate_pairs.add((ids[left], ids[right]) if ids[left] < ids[right] else (ids[right], ids[left]))
                            if len(candidate_pairs) > MAX_CANDIDATE_PAIRS:
                                raise PreprocessError(
                                    f"{language} produced more than {MAX_CANDIDATE_PAIRS:,} LSH candidate pairs; "
                                    "stopping before RAM exhaustion. Inspect boilerplate filters or lower the LSH bucket limit."
                                )
                index = end
            self.progress.tick("near-dedup-group", language, partition + 1, f" / {LSH_PARTITIONS} partitions; {len(candidate_pairs):,} candidate pairs")
        parent = {rowid: rowid for (rowid,) in db.execute("SELECT rowid FROM docs WHERE dropped=0")}
        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value
        def union(a: int, b: int) -> None:
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a
        verified = 0
        pair_total = len(candidate_pairs)
        self.progress.emit(f"near-dedup: {language}: verifying {pair_total:,} LSH candidate pairs")
        @functools.lru_cache(maxsize=20_000)
        def document_record(rowid: int) -> dict[str, Any]:
            row = db.execute("SELECT record_json FROM docs WHERE rowid=?", (rowid,)).fetchone()
            return json.loads(row[0])

        shingle_cache: OrderedDict[int, set[str]] = OrderedDict()
        cached_shingle_count = 0

        def document_shingles(rowid: int) -> set[str]:
            nonlocal cached_shingle_count
            cached = shingle_cache.get(rowid)
            if cached is not None:
                shingle_cache.move_to_end(rowid)
                return cached
            shingles = word_shingles(document_record(rowid)["cleaned_text"])
            # Keep reuse benefits for clustered candidates while enforcing a
            # corpus-independent memory ceiling measured in retained shingles.
            if len(shingles) <= MAX_CACHED_VERIFICATION_SHINGLES:
                while shingle_cache and cached_shingle_count + len(shingles) > MAX_CACHED_VERIFICATION_SHINGLES:
                    _, removed_shingles = shingle_cache.popitem(last=False)
                    cached_shingle_count -= len(removed_shingles)
                shingle_cache[rowid] = shingles
                cached_shingle_count += len(shingles)
            return shingles

        for number, (a, b) in enumerate(candidate_pairs, 1):
            a_shingles = document_shingles(a)
            b_shingles = document_shingles(b)
            size_ratio = min(len(a_shingles), len(b_shingles)) / max(len(a_shingles), len(b_shingles), 1)
            if size_ratio >= 0.80:
                intersection = len(a_shingles.intersection(b_shingles))
                similarity = intersection / max(len(a_shingles) + len(b_shingles) - intersection, 1)
            else:
                similarity = 0.0
            if similarity >= 0.80:
                union(a, b)
                verified += 1
            self.progress.tick("near-dedup-verify", language, number, f" / {pair_total:,} pairs; {verified:,} verified")
        groups: dict[int, list[int]] = defaultdict(list)
        for rowid in parent:
            groups[find(rowid)].append(rowid)
        removed = 0
        for rowids in groups.values():
            if len(rowids) < 2:
                continue
            representatives = [(self._representative_key(document_record(rowid)), rowid) for rowid in rowids]
            keep = max(representatives)[1]
            for rowid in rowids:
                if rowid != keep:
                    db.execute("UPDATE docs SET dropped=1 WHERE rowid=?", (rowid,))
                    removed += 1
        db.execute("INSERT OR REPLACE INTO dedup_metadata(key,value) VALUES('near_complete',?)", (str(removed),))
        db.commit()
        write_audit(Path("audit") / self.run_id / "lsh" / language / "summary.json", {"candidate_pairs": pair_total, "verified_pairs": verified, "near_removed": removed, "oversized_lsh_buckets_skipped": oversized_buckets, "storage": "partitioned_binary_local"})
        db.close()
        if db_path != drive_db_path:
            self.progress.emit(f"near-dedup: {language}: checkpointing completed LSH state to Drive")
            self._sqlite_backup(db_path, drive_db_path)
        shutil.rmtree(partition_root, ignore_errors=True)
        shutil.rmtree(drive_checkpoint_root, ignore_errors=True)
        self.progress.emit(f"near-dedup: {language}: complete; {removed:,} documents removed")
        return db_path, removed
