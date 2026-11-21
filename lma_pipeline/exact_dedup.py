"""Exact-hash deduplication and SQLite checkpoint utilities."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .common import PARTITIONED_LSH_ALGORITHM, PreprocessError, read_jsonl_gzip, sha256_file


class ExactDedupMixin:
    @staticmethod
    def _representative_key(record: dict[str, Any]) -> tuple[int, float, int, str]:
        return (int(bool(record.get("manual_collection"))), float(record.get("language_score", 0)), len(record["cleaned_text"]), record["document_id"])

    def _fingerprint(self, paths: list[Path]) -> str:
        digest = hashlib.sha256(self.seed.encode())
        for path in paths:
            digest.update(str(path.relative_to(self.root)).encode())
            digest.update(sha256_file(path).encode())
        return digest.hexdigest()

    @staticmethod
    def _resolve_document_id_collision(db: sqlite3.Connection, record: dict[str, Any]) -> bool:
        """Give different content with a reused source ID a stable unique ID."""
        desired = str(record["document_id"])
        content_hash = str(record["content_hash"])
        occupied = db.execute(
            "SELECT content_hash FROM docs WHERE document_id=?",
            (desired,),
        ).fetchone()
        if not occupied or occupied[0] == content_hash:
            return False
        # A full content hash makes the replacement deterministic and avoids
        # depending on SQLite row order or a mutable counter.
        replacement = f"{desired}:content-{content_hash}"
        replacement_occupied = db.execute(
            "SELECT content_hash FROM docs WHERE document_id=?",
            (replacement,),
        ).fetchone()
        if replacement_occupied and replacement_occupied[0] != content_hash:
            raise PreprocessError(f"could not resolve repeated document ID {desired!r}")
        record["document_id_collision_original"] = desired
        record["document_id"] = replacement
        return True

    def _assemble_db(self, language: str, paths: list[Path], fingerprint: str | None = None) -> tuple[Path, dict[str, int]]:
        work = self.root / "audit" / self.run_id / "work"
        work.mkdir(parents=True, exist_ok=True)
        db_path = work / f"{language}.sqlite"
        fingerprint = fingerprint or self._fingerprint(paths)
        if db_path.exists():
            previous = sqlite3.connect(db_path)
            try:
                metadata = previous.execute("SELECT value FROM metadata WHERE key='input_fingerprint'").fetchone()
                stats = previous.execute("SELECT value FROM metadata WHERE key='exact_stats'").fetchone()
                if metadata and metadata[0] == fingerprint and stats:
                    return db_path, json.loads(stats[0])
            except sqlite3.Error:
                pass
            finally:
                previous.close()
            # A prior crash can leave SQLite sidecars next to an incomplete DB.
            # They are derived work files and must not be reused with a rebuilt DB.
            for stale in (
                db_path,
                db_path.with_name(db_path.name + "-journal"),
                db_path.with_name(db_path.name + "-wal"),
                db_path.with_name(db_path.name + "-shm"),
            ):
                stale.unlink(missing_ok=True)
        db = sqlite3.connect(db_path)
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE docs (document_id TEXT PRIMARY KEY, content_hash TEXT UNIQUE, record_json TEXT NOT NULL, dropped INTEGER NOT NULL DEFAULT 0)")
        exact_removed = 0
        id_collisions_resolved = 0
        accepted = 0
        for path in paths:
            path_records = 0
            self.progress.emit(f"exact-dedup: {language}: reading {path.relative_to(self.root)}")
            for record in read_jsonl_gzip(path):
                accepted += 1
                path_records += 1
                self.progress.tick("exact-dedup", str(path.relative_to(self.root)), path_records, f" ({exact_removed:,} exact duplicates removed)")
                existing = db.execute("SELECT document_id, record_json FROM docs WHERE content_hash=?", (record["content_hash"],)).fetchone()
                if existing:
                    exact_removed += 1
                    old = json.loads(existing[1])
                    if self._representative_key(record) > self._representative_key(old):
                        id_collisions_resolved += int(self._resolve_document_id_collision(db, record))
                        db.execute("UPDATE docs SET document_id=?, record_json=? WHERE content_hash=?", (record["document_id"], json.dumps(record, ensure_ascii=False, sort_keys=True), record["content_hash"]))
                    continue
                id_collisions_resolved += int(self._resolve_document_id_collision(db, record))
                db.execute("INSERT INTO docs(document_id, content_hash, record_json) VALUES(?,?,?)", (record["document_id"], record["content_hash"], json.dumps(record, ensure_ascii=False, sort_keys=True)))
        db.commit()
        stats = {
            "accepted_before_dedup": accepted,
            "exact_removed": exact_removed,
            "document_id_collisions_resolved": id_collisions_resolved,
        }
        db.execute("INSERT INTO metadata(key,value) VALUES('input_fingerprint',?)", (fingerprint,))
        db.execute("INSERT INTO metadata(key,value) VALUES('exact_stats',?)", (json.dumps(stats, sort_keys=True),))
        db.commit()
        db.close()
        self.progress.emit(
            f"exact-dedup: {language}: complete; {accepted:,} input, "
            f"{exact_removed:,} duplicates removed, {id_collisions_resolved:,} repeated IDs resolved"
        )
        return db_path, stats

    @staticmethod
    def _db_fingerprint(path: Path) -> str | None:
        """Return an exact-input fingerprint without trusting a stale local DB."""
        if not path.exists():
            return None
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                row = connection.execute("SELECT value FROM metadata WHERE key='input_fingerprint'").fetchone()
                return str(row[0]) if row else None
            finally:
                connection.close()
        except sqlite3.Error:
            return None

    @staticmethod
    def _db_near_complete(path: Path) -> int | None:
        """Read a completed near-dedup count without mutating SQLite state."""
        if not path.exists():
            return None
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                row = connection.execute("SELECT value FROM dedup_metadata WHERE key='near_complete'").fetchone()
                return int(row[0]) if row else None
            finally:
                connection.close()
        except (sqlite3.Error, TypeError, ValueError):
            return None

    @staticmethod
    def _sqlite_backup(source: Path, destination: Path) -> None:
        """Atomically copy SQLite through its backup API, including all committed pages."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
        source_size = source.stat().st_size
        free = shutil.disk_usage(destination.parent).free
        if free < source_size + 256 * 1024 * 1024:
            raise PreprocessError(
                f"Not enough free space to checkpoint {source.name}: need at least "
                f"{(source_size + 256 * 1024 * 1024) / 2**30:.1f} GiB, have {free / 2**30:.1f} GiB"
            )
        try:
            with contextlib.closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as source_connection:
                with contextlib.closing(sqlite3.connect(temporary)) as destination_connection:
                    source_connection.backup(destination_connection)
            for sidecar in (destination.with_name(destination.name + "-wal"), destination.with_name(destination.name + "-shm")):
                sidecar.unlink(missing_ok=True)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            temporary.with_name(temporary.name + "-wal").unlink(missing_ok=True)
            temporary.with_name(temporary.name + "-shm").unlink(missing_ok=True)
            raise

    def _near_work_db(self, language: str, drive_db_path: Path) -> Path:
        """Use Colab's local SSD for LSH writes, retaining a Drive checkpoint.

        The exact-dedup DB originates on Drive so it survives runtime resets.  The
        LSH phase then makes millions of small SQLite writes; Drive FUSE is far too
        slow for that access pattern.  A local copy is reused during the same
        runtime and copied back after a completed near-dedup phase.
        """
        local_db_path = self._local_work_db_path(language)
        if local_db_path is None:
            return drive_db_path
        local_root = local_db_path.parent
        drive_fingerprint = self._db_fingerprint(drive_db_path)
        # The previous bucket-table implementation can consume the entire Colab
        # disk.  Its contents are derived and safely discarded before switching to
        # compact partition files.  This branch only ever targets local scratch.
        if local_db_path.exists():
            try:
                connection = sqlite3.connect(f"file:{local_db_path}?mode=ro", uri=True)
                try:
                    row = connection.execute("SELECT value FROM dedup_metadata WHERE key='configuration'").fetchone()
                    old_configuration = json.loads(row[0]) if row else {}
                finally:
                    connection.close()
            except (sqlite3.Error, json.JSONDecodeError):
                old_configuration = {}
            if old_configuration.get("algorithm") != PARTITIONED_LSH_ALGORITHM:
                self.progress.emit(f"near-dedup: {language}: removing obsolete local bucket-table scratch data")
                for path in (local_db_path, local_db_path.with_name(local_db_path.name + "-wal"), local_db_path.with_name(local_db_path.name + "-shm")):
                    path.unlink(missing_ok=True)
                shutil.rmtree(local_db_path.with_name(local_db_path.stem + ".lsh-partitions"), ignore_errors=True)
        local_fingerprint = self._db_fingerprint(local_db_path)
        if local_fingerprint == drive_fingerprint and drive_fingerprint is not None:
            # Prefer the durable Drive copy if it reached near_complete after the
            # local copy (for example, checkpoint succeeded and final shard output
            # was interrupted). Otherwise the local copy contains equal or newer
            # resumable work and avoids another multi-GB copy.
            if self._db_near_complete(drive_db_path) is None or self._db_near_complete(local_db_path) is not None:
                self.progress.emit(f"near-dedup: {language}: reusing local SSD work database {local_db_path}")
                return local_db_path
            self.progress.emit(f"near-dedup: {language}: Drive checkpoint is newer; refreshing local SSD work database")
        if local_db_path.exists():
            self.progress.emit(f"near-dedup: {language}: replacing stale local SSD work database")
            for path in (local_db_path, local_db_path.with_name(local_db_path.name + "-wal"), local_db_path.with_name(local_db_path.name + "-shm")):
                path.unlink(missing_ok=True)
            shutil.rmtree(local_db_path.with_name(local_db_path.stem + ".lsh-partitions"), ignore_errors=True)
        local_root.mkdir(parents=True, exist_ok=True)
        source_size = drive_db_path.stat().st_size
        free = shutil.disk_usage(local_root).free
        # Exact DB copy + compact partition records + generous transient margin.
        required = source_size + 2 * 1024**3
        if free < required:
            raise PreprocessError(
                f"Not enough local disk for {language} near-dedup: need at least "
                f"{required / 2**30:.1f} GiB free, have {free / 2**30:.1f} GiB"
            )
        self.progress.emit(f"near-dedup: {language}: copying exact-dedup database to local SSD")
        self._sqlite_backup(drive_db_path, local_db_path)
        return local_db_path

    def _local_work_db_path(self, language: str) -> Path | None:
        """Return the generated local-scratch DB path when local SSD is available."""
        local_default = Path("/content/lma_preprocess_work")
        configured = os.environ.get("LMA_LOCAL_WORK_ROOT")
        if configured:
            local_root = Path(configured) / self.run_id
        else:
            if not local_default.parent.exists():
                return None
            local_root = local_default / self.run_id
        return local_root / f"{language}.sqlite"

    def _cleanup_local_work(self, language: str) -> None:
        """Remove only regenerable local scratch after a durable run completes."""
        db_path = self._local_work_db_path(language)
        if db_path is None:
            return
        partition_root = db_path.with_name(db_path.stem + ".lsh-partitions")
        for path in (
            db_path,
            db_path.with_name(db_path.name + "-journal"),
            db_path.with_name(db_path.name + "-wal"),
            db_path.with_name(db_path.name + "-shm"),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                self.progress.emit(f"cleanup: {language}: could not remove local scratch {path.name}: {exc}")
        try:
            shutil.rmtree(partition_root, ignore_errors=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.progress.emit(f"cleanup: {language}: could not remove local LSH partitions: {exc}")
        try:
            db_path.parent.rmdir()
        except OSError:
            # Another language or harmless runtime scratch still occupies it.
            pass
