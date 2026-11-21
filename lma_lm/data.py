"""Resumable SentencePiece packing of final JSONL shards into uint16 memmaps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lma_tokenizer.common import sha256_file as tokenizer_sha256_file
from lma_tokenizer.corpus import iter_split_records

from .common import (
    LMError,
    SPLITS,
    atomic_write_json,
    copy_to_local_ssd,
    emit,
    local_work_root,
    manifest_path,
    packed_bin_path,
    packed_meta_path,
    read_json,
    sha256_file,
    sha256_json,
    tokenized_dir,
    tokenizer_dir,
    tokenizer_model_path,
)
from .config import ModelConfig, load_language_config


class Encoder(Protocol):
    def vocab_size(self) -> int: ...
    def eos_id(self) -> int: ...
    def pad_id(self) -> int: ...
    def bos_id(self) -> int: ...
    def unk_id(self) -> int: ...
    def encode(self, text: str, out_type: type = int) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


def load_sentencepiece(model_path: Path) -> Encoder:
    try:
        import sentencepiece as sentencepiece
    except ImportError as exc:
        raise LMError("sentencepiece is required; install requirements-colab.txt") from exc
    if not model_path.is_file():
        raise LMError(f"missing tokenizer model: {model_path}")
    return sentencepiece.SentencePieceProcessor(model_file=str(model_path))


def load_tokenizer(data_root: Path, run_id: str, language: str) -> tuple[Encoder, dict[str, Any]]:
    directory = tokenizer_dir(data_root, run_id, language)
    model_path = tokenizer_model_path(data_root, run_id, language)
    manifest_path = directory / "tokenizer_manifest.json"
    if not model_path.is_file():
        raise LMError(f"missing Phase 1 tokenizer: {model_path}")
    processor = load_sentencepiece(model_path)
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    recorded = manifest.get("model_sha256")
    actual = sha256_file(model_path)
    if recorded and recorded != actual:
        raise LMError(f"tokenizer.model hash mismatch for {language}: manifest {recorded} vs file {actual}")
    if processor.vocab_size() >= 65535:
        raise LMError("vocabulary does not fit in uint16 packed bins")
    return processor, manifest


def load_preprocessing_manifest(data_root: Path, run_id: str, language: str) -> dict[str, Any]:
    path = manifest_path(data_root, run_id, language)
    if not path.is_file():
        raise LMError(f"missing preprocessing manifest: {path}")
    manifest = read_json(path)
    manifest["_manifest_path"] = str(path)
    manifest["_manifest_sha256"] = tokenizer_sha256_file(path)
    return manifest


def encode_document(processor: Encoder, text: str, eos_id: int) -> list[int]:
    pieces = [int(token_id) for token_id in processor.encode(text, out_type=int)]
    if not pieces:
        return []
    if pieces[-1] != eos_id:
        pieces.append(eos_id)
    return pieces


def encode_documents(processor: Encoder, texts: list[str], eos_id: int) -> list[np.ndarray]:
    if not texts:
        return []
    try:
        encoded = processor.encode(texts, out_type=int)
    except TypeError:
        encoded = [processor.encode(text, out_type=int) for text in texts]
    packed: list[np.ndarray] = []
    for pieces in encoded:
        ids = np.asarray(pieces, dtype=np.uint16)
        if ids.size == 0:
            packed.append(ids)
            continue
        if int(ids[-1]) != eos_id:
            ids = np.concatenate((ids, np.asarray([eos_id], dtype=np.uint16)))
        packed.append(ids)
    return packed


def _pack_state_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}.pack-state.json"


def _part_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}.bin.part"


def _append_uint16(path: Path, token_ids: list[int] | np.ndarray, fsync: bool = False) -> None:
    if isinstance(token_ids, np.ndarray):
        if token_ids.size == 0:
            return
        array = np.ascontiguousarray(token_ids, dtype=np.uint16)
    else:
        if not token_ids:
            return
        array = np.asarray(token_ids, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(array.tobytes())
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def pack_split(
    data_root: Path,
    run_id: str,
    language: str,
    split: str,
    processor: Encoder,
    manifest: dict[str, Any],
    tokenizer_sha256: str,
    progress=emit,
    flush_every: int = 1_000_000,
    encode_batch_size: int = 256,
) -> dict[str, Any]:
    """Encode one split into a resumable uint16 bin. Completed shards are not re-read."""
    if split not in SPLITS:
        raise LMError(f"unknown split {split}")
    output_dir = tokenized_dir(data_root, run_id, language)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = packed_bin_path(data_root, run_id, language, split)
    state_path = _pack_state_path(output_dir, split)
    part_path = _part_path(output_dir, split)
    shards = list(manifest.get("outputs", {}).get(split) or [])
    if not shards:
        raise LMError(f"manifest has no {split} shards for {language}")
    eos_id = int(processor.eos_id())
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "language": language,
        "split": split,
        "tokenizer_sha256": tokenizer_sha256,
        "manifest_sha256": manifest["_manifest_sha256"],
        "eos_id": eos_id,
        "dtype": "uint16",
    }
    state: dict[str, Any]
    if state_path.is_file():
        state = read_json(state_path)
        if sha256_json({key: state.get(key) for key in identity}) != sha256_json(identity):
            progress(f"pack: {language}/{split}: tokenizer or manifest changed; restarting this split")
            part_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
            state = {**identity, "completed_shards": [], "token_count": 0, "document_count": 0, "utf8_bytes": 0}
        if final_path.is_file() and set(state.get("completed_shards") or []) == set(shards):
            progress(f"pack: {language}/{split}: already complete ({int(state['token_count']):,} tokens)")
            return {**state, "path": str(final_path.relative_to(data_root)), "status": "complete"}
        if final_path.is_file() and not part_path.exists():
            os.replace(final_path, part_path)
        expected_bytes = int(state["token_count"]) * 2
        if part_path.exists() and part_path.stat().st_size != expected_bytes:
            actual = part_path.stat().st_size
            if actual > expected_bytes and expected_bytes >= 0 and actual % 2 == 0:
                extra = actual - expected_bytes
                progress(
                    f"pack: {language}/{split}: truncating {extra:,} extra bytes from an interrupted shard "
                    f"(keeping {int(state['token_count']):,} tokens)"
                )
                with part_path.open("r+b") as handle:
                    handle.truncate(expected_bytes)
            else:
                progress(
                    f"pack: {language}/{split}: part size {actual} != 2 * token_count "
                    f"{state['token_count']}; restarting this split"
                )
                part_path.unlink(missing_ok=True)
                final_path.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
                state = {
                    **identity,
                    "completed_shards": [],
                    "token_count": 0,
                    "document_count": 0,
                    "utf8_bytes": 0,
                }
    else:
        state = {**identity, "completed_shards": [], "token_count": 0, "document_count": 0, "utf8_bytes": 0}
        part_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)

    completed = set(state["completed_shards"])
    pending = [shard for shard in shards if shard not in completed]
    progress(
        f"pack: {language}/{split}: {len(completed)}/{len(shards)} shards done; "
        f"{int(state['token_count']):,} tokens so far"
    )
    buffer_chunks: list[np.ndarray] = []
    buffer_len = 0

    def flush(durable: bool = False) -> None:
        nonlocal buffer_chunks, buffer_len
        if not buffer_chunks:
            return
        array = np.concatenate(buffer_chunks) if len(buffer_chunks) > 1 else buffer_chunks[0]
        _append_uint16(part_path, array, fsync=durable)
        buffer_chunks = []
        buffer_len = 0

    for index, relative in enumerate(pending, 1):
        shard_tokens = 0
        shard_docs = 0
        shard_bytes = 0
        fake_manifest = {"outputs": {split: [relative]}}
        pending_texts: list[str] = []
        pending_bytes: list[int] = []

        def drain_texts() -> None:
            nonlocal shard_tokens, shard_docs, shard_bytes, pending_texts, pending_bytes, buffer_len
            if not pending_texts:
                return
            for tokens, nbytes in zip(encode_documents(processor, pending_texts, eos_id), pending_bytes):
                if tokens.size == 0:
                    continue
                buffer_chunks.append(tokens)
                buffer_len += int(tokens.size)
                shard_tokens += int(tokens.size)
                shard_docs += 1
                shard_bytes += nbytes
            pending_texts = []
            pending_bytes = []
            if buffer_len >= flush_every:
                flush(durable=False)

        for record in iter_split_records(data_root, fake_manifest, split):
            text = record["cleaned_text"]
            pending_texts.append(text)
            pending_bytes.append(len(text.encode("utf-8")))
            if len(pending_texts) >= encode_batch_size:
                drain_texts()
        drain_texts()
        flush(durable=True)
        state["completed_shards"] = list(completed | {relative})
        state["token_count"] = int(state["token_count"]) + shard_tokens
        state["document_count"] = int(state["document_count"]) + shard_docs
        state["utf8_bytes"] = int(state["utf8_bytes"]) + shard_bytes
        completed.add(relative)
        atomic_write_json(state_path, state)
        progress(
            f"pack: {language}/{split}: shard {index}/{len(pending)} {relative} "
            f"+{shard_tokens:,} tokens (total {int(state['token_count']):,})"
        )

    if part_path.exists():
        os.replace(part_path, final_path)
    elif not final_path.exists():
        _append_uint16(final_path, [])
    expected_bytes = int(state["token_count"]) * 2
    if final_path.stat().st_size != expected_bytes:
        raise LMError(f"packed {final_path} has {final_path.stat().st_size} bytes, expected {expected_bytes}")
    state["path"] = str(final_path.relative_to(data_root))
    state["bin_bytes"] = final_path.stat().st_size
    state["status"] = "complete"
    atomic_write_json(state_path, state)
    progress(f"pack: {language}/{split}: complete {int(state['token_count']):,} tokens -> {state['path']}")
    return state


def pack_language(
    data_root: Path,
    run_id: str,
    language: str,
    splits: tuple[str, ...] = SPLITS,
    progress=emit,
) -> dict[str, Any]:
    processor, tokenizer_manifest = load_tokenizer(data_root, run_id, language)
    preprocessing = load_preprocessing_manifest(data_root, run_id, language)
    tokenizer_hash = sha256_file(tokenizer_model_path(data_root, run_id, language))
    split_results = {}
    for split in splits:
        split_results[split] = pack_split(
            data_root,
            run_id,
            language,
            split,
            processor,
            preprocessing,
            tokenizer_hash,
            progress=progress,
        )
    meta = {
        "schema_version": 1,
        "run_id": run_id,
        "language": language,
        "tokenizer_path": str(tokenizer_model_path(data_root, run_id, language).relative_to(data_root)),
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_vocab_size": int(processor.vocab_size()),
        "pad_id": int(processor.pad_id()),
        "unk_id": int(processor.unk_id()),
        "bos_id": int(processor.bos_id()),
        "eos_id": int(processor.eos_id()),
        "preprocessing_manifest_sha256": preprocessing["_manifest_sha256"],
        "splits": split_results,
    }
    atomic_write_json(packed_meta_path(data_root, run_id, language), meta)
    return meta


def load_packed_meta(data_root: Path, run_id: str, language: str) -> dict[str, Any]:
    path = packed_meta_path(data_root, run_id, language)
    if not path.is_file():
        raise LMError(f"missing packed-token metadata: {path}; run pack_tokens.py first")
    return read_json(path)


class PackedTokens:
    """Memory-mapped packed token stream with next-token windows."""

    def __init__(self, path: Path, seq_len: int) -> None:
        if not path.is_file():
            raise LMError(f"missing packed tokens: {path}")
        self.path = path
        self.seq_len = seq_len
        mapped = np.memmap(path, dtype=np.uint16, mode="r")
        try:
            self.data = mapped.copy()
        except MemoryError:
            self.data = mapped
        if self.data.size < seq_len + 1:
            raise LMError(f"{path} has {self.data.size} tokens; need at least seq_len+1={seq_len + 1}")

    def __len__(self) -> int:
        return int(self.data.size)

    @property
    def n_windows(self) -> int:
        return max(int(self.data.size) - self.seq_len, 0)

    def window(self, start: int) -> tuple[np.ndarray, np.ndarray]:
        end = start + self.seq_len
        tokens = np.asarray(self.data[start : end + 1], dtype=np.int64)
        return tokens[:-1], tokens[1:]

    def batch(self, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        starts = np.asarray(starts, dtype=np.int64)
        offsets = starts[:, None] + np.arange(self.seq_len + 1, dtype=np.int64)
        windows = np.asarray(self.data[offsets], dtype=np.int64)
        return windows[:, :-1], windows[:, 1:]


def materialize_split(
    data_root: Path,
    run_id: str,
    language: str,
    split: str,
    seq_len: int,
    local_root: Path | None = None,
    progress=emit,
) -> PackedTokens:
    source = packed_bin_path(data_root, run_id, language, split)
    if not source.is_file():
        raise LMError(f"missing packed {split} split: {source}; run pack_tokens.py first")
    work = local_work_root(run_id, language, local_root)
    if Path("/content").is_dir():
        source = copy_to_local_ssd(source, work / f"{split}.bin", progress=progress)
    return PackedTokens(source, seq_len)


def special_ids_from_processor(processor: Encoder) -> dict[str, int]:
    return {
        "pad_id": int(processor.pad_id()),
        "unk_id": int(processor.unk_id()),
        "bos_id": int(processor.bos_id()),
        "eos_id": int(processor.eos_id()),
        "vocab_size": int(processor.vocab_size()),
    }


def config_with_tokenizer(
    language: str,
    data_root: Path,
    run_id: str,
    root: Path | None = None,
    *,
    no_positions: bool = False,
) -> tuple[ModelConfig, Encoder, dict[str, Any]]:
    processor, tokenizer_manifest = load_tokenizer(data_root, run_id, language)
    ids = special_ids_from_processor(processor)
    config = load_language_config(
        language, vocab_size=ids["vocab_size"], root=root, no_positions=no_positions
    )
    config.pad_id = ids["pad_id"]
    config.unk_id = ids["unk_id"]
    config.bos_id = ids["bos_id"]
    config.eos_id = ids["eos_id"]
    return config, processor, tokenizer_manifest
