"""SQLite task and stage ledger for resumable preprocessing."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .common import read_jsonl_gzip, utcnow, valid_gzip_jsonl

class State:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                source_id TEXT NOT NULL,
                raw_ref TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                output_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','complete','failed')),
                records INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, stage, source_id, raw_ref)
            );
            CREATE TABLE IF NOT EXISTS stages (
                run_id TEXT NOT NULL,
                language TEXT NOT NULL,
                stage TEXT NOT NULL,
                input_fingerprint TEXT,
                output_path TEXT,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','complete','failed')),
                details_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, language, stage)
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def task(self, run_id: str, stage: str, source_id: str, raw_ref: str, input_sha: str, output_path: str) -> sqlite3.Row:
        with self.conn:
            self.conn.execute(
                """INSERT INTO tasks(run_id,stage,source_id,raw_ref,input_sha256,output_path,updated_at)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(run_id,stage,source_id,raw_ref) DO UPDATE SET
                  input_sha256=excluded.input_sha256, output_path=excluded.output_path,
                  status=CASE WHEN tasks.input_sha256<>excluded.input_sha256 THEN 'pending' ELSE tasks.status END,
                  updated_at=excluded.updated_at""",
                (run_id, stage, source_id, raw_ref, input_sha, output_path, utcnow()),
            )
        return self.conn.execute("SELECT * FROM tasks WHERE run_id=? AND stage=? AND source_id=? AND raw_ref=?", (run_id, stage, source_id, raw_ref)).fetchone()

    def existing_task(self, run_id: str, stage: str, source_id: str, raw_ref: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM tasks WHERE run_id=? AND stage=? AND source_id=? AND raw_ref=?",
            (run_id, stage, source_id, raw_ref),
        ).fetchone()

    def complete_task(self, row: sqlite3.Row, records: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE tasks SET status='complete', records=?, error=NULL, updated_at=? WHERE run_id=? AND stage=? AND source_id=? AND raw_ref=?", (records, utcnow(), row["run_id"], row["stage"], row["source_id"], row["raw_ref"]))

    def fail_task(self, row: sqlite3.Row, error: Exception) -> None:
        with self.conn:
            self.conn.execute("UPDATE tasks SET status='failed', error=?, updated_at=? WHERE run_id=? AND stage=? AND source_id=? AND raw_ref=?", (f"{type(error).__name__}: {error}"[:2000], utcnow(), row["run_id"], row["stage"], row["source_id"], row["raw_ref"]))

    def reconcile(self, root: Path, run_id: str) -> int:
        recovered = 0
        rows = self.conn.execute("SELECT * FROM tasks WHERE run_id=? AND status!='complete'", (run_id,))
        for row in rows:
            path = root / row["output_path"]
            if path.exists() and valid_gzip_jsonl(path):
                self.complete_task(row, sum(1 for _ in read_jsonl_gzip(path)))
                recovered += 1
        return recovered

    def stage_complete(self, run_id: str, language: str, stage: str, fingerprint: str, root: Path) -> bool:
        row = self.conn.execute("SELECT * FROM stages WHERE run_id=? AND language=? AND stage=?", (run_id, language, stage)).fetchone()
        return bool(row and row["status"] == "complete" and row["input_fingerprint"] == fingerprint and row["output_path"] and (root / row["output_path"]).exists())

    def complete_stage(self, run_id: str, language: str, stage: str, fingerprint: str, output_path: str, details: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("""INSERT INTO stages(run_id,language,stage,input_fingerprint,output_path,status,details_json,updated_at)
                VALUES(?,?,?,?,?,'complete',?,?) ON CONFLICT(run_id,language,stage) DO UPDATE SET
                input_fingerprint=excluded.input_fingerprint, output_path=excluded.output_path, status='complete',
                details_json=excluded.details_json, updated_at=excluded.updated_at""", (run_id, language, stage, fingerprint, output_path, json.dumps(details, sort_keys=True), utcnow()))

    def status(self, run_id: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT run_id,stage,source_id,status,COUNT(*) count FROM tasks"
        args: list[Any] = []
        if run_id:
            query += " WHERE run_id=?"
            args.append(run_id)
        query += " GROUP BY run_id,stage,source_id,status ORDER BY run_id,stage,source_id,status"
        return list(self.conn.execute(query, args))


