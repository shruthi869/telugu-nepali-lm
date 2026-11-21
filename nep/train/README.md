Train the Nepali model:

```bash
python -u train_lm.py --data-root /content/drive/MyDrive/lma_data --run-id draft-v1 --language nep
```

Optional no-position ablation (does not overwrite the standard checkpoint):

```bash
python -u train_lm.py --data-root /content/drive/MyDrive/lma_data --run-id draft-v1 --language nep --no-positions
```
