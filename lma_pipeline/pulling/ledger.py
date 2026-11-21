"""Persistent download and crawl job ledger."""

from __future__ import annotations

import gzip
import json
import random
import sqlite3
from pathlib import Path
from typing import Any

from .common import canonical_url, sha256_file, unix_now, utcnow

class Ledger:
    """Persistent state for source discovery, downloads, and crawl responses."""

    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                url TEXT NOT NULL,
                destination TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','in_progress','complete','deferred','failed')),
                expected_bytes INTEGER,
                expected_sha256 TEXT,
                bytes_written INTEGER NOT NULL DEFAULT 0,
                etag TEXT,
                sha256 TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                retry_at REAL,
                error TEXT,
                batch_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_id, kind, url)
            );
            CREATE INDEX IF NOT EXISTS jobs_runnable
                ON jobs(status, retry_at, source_id);
            """
        )
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(jobs)")}
        if "expected_sha256" not in columns:
            self.conn.execute("ALTER TABLE jobs ADD COLUMN expected_sha256 TEXT")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_job(
        self,
        source_id: str,
        kind: str,
        url: str,
        destination: str | None = None,
        expected_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> int:
        url = canonical_url(url)
        now = utcnow()
        with self.conn:
            self.conn.execute(
                """INSERT INTO jobs(source_id,kind,url,destination,expected_bytes,expected_sha256,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(source_id,kind,url) DO UPDATE SET
                     destination=COALESCE(jobs.destination, excluded.destination),
                     expected_bytes=COALESCE(jobs.expected_bytes, excluded.expected_bytes),
                     expected_sha256=COALESCE(jobs.expected_sha256, excluded.expected_sha256)""",
                (source_id, kind, url, destination, expected_bytes, expected_sha256, now, now),
            )
        row = self.conn.execute(
            "SELECT id FROM jobs WHERE source_id=? AND kind=? AND url=?", (source_id, kind, url)
        ).fetchone()
        return int(row["id"])

    def job(self, job_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return row

    def runnable(self, source_id: str | None = None, limit: int = 1_000) -> list[sqlite3.Row]:
        statement = """
            SELECT * FROM jobs WHERE
              (status='pending' OR status='in_progress' OR (status='deferred' AND COALESCE(retry_at, 0)<=?))
        """
        args: list[Any] = [unix_now()]
        if source_id:
            statement += " AND source_id=?"
            args.append(source_id)
        statement += " ORDER BY id LIMIT ?"
        args.append(limit)
        return list(self.conn.execute(statement, args))

    def mark_in_progress(self, job_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE jobs SET status='in_progress', updated_at=?, error=NULL WHERE id=?",
                (utcnow(), job_id),
            )

    def progress(self, job_id: int, bytes_written: int, etag: str | None, expected: int | None) -> None:
        with self.conn:
            self.conn.execute(
                """UPDATE jobs SET bytes_written=?, etag=COALESCE(?,etag),
                   expected_bytes=COALESCE(?,expected_bytes), updated_at=? WHERE id=?""",
                (bytes_written, etag, expected, utcnow(), job_id),
            )

    def complete_download(self, job_id: int, bytes_written: int, etag: str | None, digest: str) -> None:
        with self.conn:
            self.conn.execute(
                """UPDATE jobs SET status='complete', bytes_written=?, etag=COALESCE(?,etag), sha256=?,
                   error=NULL, retry_at=NULL, updated_at=? WHERE id=?""",
                (bytes_written, etag, digest, utcnow(), job_id),
            )

    def complete_response(self, job_id: int, batch_path: str, digest: str) -> None:
        """Commit a durable response batch to the ledger in one transaction."""
        self.complete_responses([job_id], batch_path, digest)

    def complete_responses(self, job_ids: list[int], batch_path: str, digest: str) -> None:
        """Commit all jobs represented by one immutable response batch."""
        if not job_ids:
            return
        with self.conn:
            self.conn.executemany(
                """UPDATE jobs SET status='complete', batch_path=?, sha256=?, error=NULL,
                   retry_at=NULL, updated_at=? WHERE id=?""",
                [(batch_path, digest, utcnow(), job_id) for job_id in job_ids],
            )

    def defer(self, job_id: int, error: str, retry_after: float | None = None) -> None:
        old = self.job(job_id)
        retries = int(old["retry_count"]) + 1
        delay = retry_after if retry_after is not None else min(3600.0, (2 ** min(retries, 10)) + random.random())
        with self.conn:
            self.conn.execute(
                """UPDATE jobs SET status='deferred', retry_count=?, retry_at=?, error=?, updated_at=?
                   WHERE id=?""",
                (retries, unix_now() + delay, error[:2000], utcnow(), job_id),
            )

    def fail(self, job_id: int, error: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?",
                (error[:2000], utcnow(), job_id),
            )

    def status_counts(self, source_id: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT source_id, kind, status, COUNT(*) AS count FROM jobs"
        args: list[Any] = []
        if source_id:
            query += " WHERE source_id=?"
            args.append(source_id)
        query += " GROUP BY source_id, kind, status ORDER BY source_id, kind, status"
        return list(self.conn.execute(query, args))

    def count_jobs(
        self,
        source_id: str,
        kind: str,
        statuses: tuple[str, ...] = ("complete",),
    ) -> int:
        if not statuses:
            return 0
        placeholders = ",".join("?" for _ in statuses)
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE source_id=? AND kind=? AND status IN ({placeholders})",
            (source_id, kind, *statuses),
        ).fetchone()
        return int(row[0])

    def reconcile_batches(self, data_root: Path) -> int:
        """Recover response files written before an interruption prevented DB completion."""
        recovered = 0
        batches = data_root / "raw" / "web"
        if not batches.exists():
            return 0
        for path in batches.rglob("*.jsonl.gz"):
            try:
                recoverable: list[int] = []
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        job_id = int(record["job_id"])
                        job = self.job(job_id)
                        if job["status"] != "complete" and canonical_url(record["url"]) == job["url"]:
                            recoverable.append(job_id)
                if recoverable:
                    self.complete_responses(
                        recoverable,
                        str(path.relative_to(data_root)),
                        sha256_file(path),
                    )
                    recovered += len(recoverable)
            except (OSError, ValueError, KeyError, json.JSONDecodeError, sqlite3.Error):
                # A temporary/corrupt response is never treated as collected.
                continue
        return recovered
