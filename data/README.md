# data/

- `raw/`  : put Kaggle `Fake.csv` and `True.csv` here (git-ignored, too large).
- `train.csv`, `test.csv` : the SHARED split, produced by `01_inspect_and_split.py`. Commit these. Never edit them.
- `inspection_report.txt` : dataset numbers (duplicates, Reuters leak, article lengths) to cite in the report.

Columns: `article_id, text, label`  (0 = Fake, 1 = Real)
