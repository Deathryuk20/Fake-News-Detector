# Fake News Detection Agent: Classical ML vs Longformer vs Prompted LLM

AI course project. The system reads a full-length news article and outputs **Real / Fake**
(with a confidence score and, for the LLM version, a written explanation).

We build the same classifier three ways and compare them on identical data:

| Version | Approach | Training? | Folder |
|---|---|---|---|
| 1 | TF-IDF + Logistic Regression / SVM / Random Forest | Yes (lightweight) | `classical_ml/` |
| 2 | Fine-tuned Longformer (reads up to 4,096 tokens) | Yes (fine-tune, GPU) | `longformer/` |
| 3 | Pretrained LLM via LangChain (zero-shot / few-shot / chain-of-thought) | No | `llm_prompting/` |

All three are evaluated on (a) the shared held-out test split and (b) an independently
collected set of recent articles (`generalization_set/`) to measure cross-source generalisation.

## Repo layout

```
data/                  raw Kaggle files (not committed) + shared train.csv / test.csv
classical_ml/          Version 1
longformer/            Version 2
llm_prompting/         Version 3
generalization_set/    hand-collected articles from outside the Kaggle data
evaluation/            reads every model predictions, builds the comparison table
results/               prediction CSVs from each model (one file per model)
report/                synopsis, literature review, Responsible AI section, final report
```

## Setup

```bash
pip install -r requirements.txt
```

1. Download "Fake and Real News Dataset" from Kaggle (Fake.csv, True.csv) into `data/raw/`.
2. Create the shared split (run ONCE, then never change it):
   ```bash
   python 01_inspect_and_split.py --fake data/raw/Fake.csv --true data/raw/True.csv
   ```
3. Each teammate trains on `data/train.csv` and predicts on `data/test.csv`.

## Rules everyone follows

- **Never modify `data/train.csv` or `data/test.csv`.** The comparison is only valid if all models use the same split.
- Every model saves predictions to `results/<model_name>.csv` with exactly these columns:
  `article_id,true_label,predicted_label,confidence`  (label: 0 = Fake, 1 = Real)
- Predict on the generalisation set too, saved as `results/<model_name>_generalization.csv`.
- Write your own code and be able to explain every line (viva).

## Team and ownership

| Workstream | Owner |
|---|---|
| A. Data + classical ML | _name_ |
| B. Longformer | _name_ |
| C. LLM prompting | _name_ |
| D. Generalisation set + report | _name_ |

## Status

- [ ] Shared split created
- [ ] Version 1 predictions in `results/`
- [ ] Version 2 predictions in `results/`
- [ ] Version 3 predictions in `results/`
- [ ] Generalisation set collected and labelled
- [ ] Evaluation table built
- [ ] Report + demo
