# Version 2: Longformer (Workstream B)

Goal: fine-tune `allenai/longformer-base-4096` on `data/train.csv` (Colab GPU), predict on `data/test.csv`.

Notes
- If you hit GPU out-of-memory: batch size 1-2 + gradient accumulation, or cap max length at 2048.
- Do not commit checkpoints (git-ignored). Share weights via Drive if needed.

Checklist
- [ ] Tokenisation (max_length 4096 or documented cap)
- [ ] Fine-tune with HuggingFace Trainer
- [ ] Save `results/longformer.csv`
- [ ] Predict on `generalization_set/` -> `results/longformer_generalization.csv`
- [ ] Note training time, GPU used, hyperparameters (for the report)
