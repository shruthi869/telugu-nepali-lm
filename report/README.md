# Results and evidence

This folder holds the measured outputs behind every number in the top-level
README, plus the scripts that regenerate them.

| File | What it contains |
|---|---|
| `phase1_metrics.json` | Corpus and tokenizer metrics exported from the preprocessing manifests: dedup counts, split sizes, tokenizer fertility and vocabulary |
| `phase1_exact_source_tokens.json` | A full scan of the training split attributing SentencePiece tokens to each source |
| `phase2_metrics.json` | Model architecture and parameter counts, plus the combined evaluation metrics |
| `phase2_tel_eval.json`, `phase2_nep_eval.json` | Validation loss, perplexity, bits per byte, sampled generations and attention statistics |
| `phase2_nep_nopos_eval.json` | The same evaluation for the Nepali model trained without positional embeddings |
| `logs/` | Training logs, one JSON line per step (loss, learning rate, throughput) |
| `figures/` | The causal-mask check, parameter breakdown, Telugu vs Nepali comparison, training curve and attention entropy/distance plots |

## Regenerating

Corpus and tokenizer metrics from the Drive-backed manifests (no rerun of the
pipeline):

```bash
python3 collect_phase1_metrics.py --data-root /content/drive/MyDrive/lma_data --run-id draft-v1 --output phase1_metrics.json
```

The manifest's `whitespace_tokens` value counts Unicode `\w+` matches, not
SentencePiece tokens. To count the actual tokenizer training tokens, run this
resumable, shard-level scan:

```bash
python3 count_training_tokens.py --data-root /content/drive/MyDrive/lma_data --run-id draft-v1 --language all
```

Model metrics and figures:

```bash
python3 collect_phase2_metrics.py --data-root ../artifacts/extracted --run-id draft-v1 --output phase2_metrics.json
python3 render_phase2_figures.py     # architecture figures (no checkpoint needed)
python3 render_phase2_results.py     # result figures from the eval JSON (no torch needed)
```
