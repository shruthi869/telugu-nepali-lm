# Telugu & Nepali Language Models from Scratch

Two independent decoder-only Transformer language models, one for **Telugu**
and one for **Nepali**, built end to end: web-scale corpus collection and
cleaning, SentencePiece tokenizers, and a GPT-style model written in plain
PyTorch with no `nn.Transformer*` and no HuggingFace model classes. Everything
was designed to run on free Colab or Kaggle T4 GPUs, which can disconnect at
any time. Every stage is resumable from durable checkpoints.

The two models share an architecture but **not** data, tokenizer, vocabulary,
or weights.

## Results

| | Telugu | Nepali |
|---|---:|---:|
| Clean corpus (train split, whitespace tokens) | 1.03B | 0.995B |
| Documents removed as exact / near duplicates | 15 / 619 | 49,749 / 354 |
| Tokenizer: BPE vocab / fertility (subwords per word) | 48K / 1.72 | 48K / 1.23 |
| Unknown-token rate (byte fallback) | 0.0% | 0.0% |
| Model parameters (weight-tied) | 25.58M | 25.58M |
| Packed training tokens (1 epoch) | 540M | 520M |
| **Validation perplexity** | **145.3** | **103.9** |
| Validation bits per byte | 0.581 | 0.488 |

The **positional-embedding ablation** (Nepali) retrained the model from
scratch without learned absolute positions. Perplexity rose from 103.9 to 114.6
and bits per byte from 0.488 to 0.498. Causal masking alone still lets the model
infer a coarse sense of position.

The measured metrics, evaluation output, training logs and figures behind
these numbers are in [`report/`](report/).

## Pipeline

```text
 sources.json ─► pull.py ─► preprocess.py ─► train_tokenizer.py ─► pack_tokens.py ─► train_lm.py ─► eval_lm.py
  (approved      (resumable    (extract, filter,   (SentencePiece      (uint16 token     (GPT-style      (PPL, BPB,
   sources)       collector)    dedup, split)       BPE per language)   memmaps)          decoder)        attention)
```

### 1. Data collection (`lma_pipeline/pulling/`)
- A standard-library HTTP collector with a **SQLite task ledger**. Interrupted
  downloads resume from durable `.part` files that are checkpointed every
  16 MiB.
- It rate-limits each host, honours `robots.txt` and path allow-lists, and
  retries transient errors (including `Retry-After`) with backoff. It never
  bypasses access controls.
- Sources include AI4Bharat Sangraha, HPLT v2, Wikipedia dumps, Wikisource,
  Global Voices and Press Information Bureau releases. Each source is allowed
  only if it is approved in [`configs/sources.json`](configs/sources.json) with
  a dated policy snapshot.

### 2. Cleaning & deduplication (`lma_pipeline/`)
- **Normalization:** NFC normalization and PII redaction (emails, phone
  numbers, ID-like numbers), keeping the zero-width joiners that Indic scripts
  rely on.
- **Rule-based filters:** text is rejected if it is too short, has a low ratio
  of native script, has encoding damage, is URL- or symbol-heavy, or is
  repeated boilerplate.
- **Language ID:** fastText `lid.176` keeps only text identified as the target
  language with confidence ≥ 0.70.
- **Exact dedup:** removes documents with identical content hashes.
- **Near dedup:** 128-value densified one-permutation **MinHash** with 16×8
  **LSH** banding over 5-word shingles. Each candidate pair is confirmed with
  the exact Jaccard similarity (≥ 0.80) before a document is removed.
  - Signatures are generated in parallel into disk-backed partitions on the
    local SSD and incrementally checkpointed to Google Drive.
- **Splits:** deterministic, hash-based 98/1/1 train/validation/test splits.
  Deduplication runs over the whole corpus before splitting, so no duplicate
  can leak across splits.
- **Manifest:** each language gets one, with source hashes, the language-ID
  model hash, rejection counts and token counts.

### 3. Tokenization (`lma_tokenizer/`)
- Each language gets its own SentencePiece **BPE** tokenizer, trained on a
  deterministic 25M-word sample with byte fallback, so there are zero unknown
  tokens.
- The vocabulary size is chosen from {16K, 24K, 32K, 48K} by measuring
  fertility (subwords per word) on held-out text.

### 4. Model & training (`lma_lm/`)
- **Architecture:** a pre-norm GPT-style decoder with 4 layers, d_model 384,
  6 heads, d_ff 1536, GELU and a context of 128. Token and position embeddings
  are learned, and the output head is tied to the input embeddings. Every
  operation in the forward pass is explicit.
- **Data loading:** tokens are packed into `uint16` memory-mapped files, so
  training streams from local disk instead of the slow Drive mount.
- **Optimization:** AdamW with linear warmup and cosine decay, gradient
  clipping, and mixed precision. One epoch per language on a single T4.
- **Crash-safe checkpoints:** model, optimizer, scheduler, grad-scaler and RNG
  states are all saved, so a disconnect resumes exactly from `latest.pt`.
- **Evaluation:** packed-window perplexity, tokenizer-independent bits per
  byte, sampled generations at several temperatures, causal-mask leak checks,
  and attention entropy and distance analysis per head.

## Repository layout

```text
.
├── pull.py, preprocess.py          # data collection / cleaning entry points
├── train_tokenizer.py              # SentencePiece training
├── pack_tokens.py, train_lm.py, eval_lm.py
├── lma_pipeline/                   # collection, extraction, filters, dedup, splits
├── lma_tokenizer/                  # tokenizer sampling, training, metrics
├── lma_lm/                         # model, data loading, training, evaluation
├── configs/                        # source registry + policy snapshots
├── tel/, nep/                      # per-language tokenizer and model configs
├── report/                         # metrics JSON, figures, training logs, reports
└── tests/                          # unit, crash-recovery and smoke tests
```

## Running it

Python 3.10+, on Linux, macOS or Google Colab.

```bash
pip install -r requirements.txt        # PyTorch/numpy come from the Colab runtime
DATA=/content/drive/MyDrive/lma_data

python3 pull.py --data-root $DATA                                   # collect raw data
python3 preprocess.py --data-root $DATA --download-model            # fetch fastText LID once
python3 -u preprocess.py --data-root $DATA --run-id draft-v1 --workers 2
python3 -u train_tokenizer.py --data-root $DATA --run-id draft-v1 --language all

python -u train_lm.py --smoke                                       # end-to-end check on toy data
python -u pack_tokens.py --data-root $DATA --run-id draft-v1 --language tel
python -u train_lm.py    --data-root $DATA --run-id draft-v1 --language tel
python -u eval_lm.py     --data-root $DATA --run-id draft-v1 --language tel
```

Repeat the last three commands with `--language nep`. If a run is interrupted,
rerun the same command and it continues from the last checkpoint. Add
`--status` to `pull.py`, `preprocess.py`, `pack_tokens.py` or `train_lm.py` to see its progress. Raw data, packed tokens and model
checkpoints are kept outside Git.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests run offline. They cover:
- **Collector:** a local HTTP fixture server simulates throttling, `503`s and
  truncated downloads.
- **Crash recovery:** faults injected during interrupted signature generation,
  shards written before their ledger commit, and simulated runtime resets.
- **Tokenizer:** gating checks and deterministic training samples.
- **Model:** causal-mask leakage, weight tying and parameter counts,
  checkpoint round-trips, and an end-to-end smoke run on toy data.
