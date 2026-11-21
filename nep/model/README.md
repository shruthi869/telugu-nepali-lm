The Nepali model uses the shared `lma_lm` implementation with [`configs/model.json`](configs/model.json).
Checkpoints and packed tokens stay on Drive under `models/<run-id>/nep/` and `tokenized/<run-id>/nep/`.
The no-position ablation writes `models/<run-id>/nep/nopos/` from [`configs/model_nopos.json`](configs/model_nopos.json).
