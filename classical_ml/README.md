# Version 1: Classical ML (Workstream A)

Goal: clean text -> TF-IDF -> train Logistic Regression, SVM, Random Forest on `data/train.csv`.

Checklist
- [ ] Text cleaning function
- [ ] TF-IDF fitted on TRAIN ONLY (never on test)
- [ ] Train LR, SVM, RF
- [ ] Save `results/logreg.csv`, `results/svm.csv`, `results/rf.csv`
- [ ] Top words per class for LR (interpretability, useful in the viva)
- [ ] Predict on `generalization_set/` -> `results/<model>_generalization.csv`
