# generalization_set/ (Workstream D)

Hand-collected articles from OUTSIDE the Kaggle dataset, published after its time frame.

Target: 15-20 real + 15-20 fake. File: `generalization_set.csv`

Columns: `article_id, text, label, source, url, date, why_this_label`

Rules
- Real: established outlets with editorial standards (BBC, Reuters, AP, ...).
- Fake: sources with documented reputations, OR claims debunked by a fact-checker (cite it in `why_this_label`).
- Keep articles full length (this is a full-article system).
- Same label convention as train/test: 0 = Fake, 1 = Real.
