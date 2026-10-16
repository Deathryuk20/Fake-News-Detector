# evaluation/

Reads every `results/*.csv` and produces one comparison table:
Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices,
for both the shared test set and the generalisation set.

Inputs follow one format: `article_id,true_label,predicted_label,confidence`.
Whoever finishes their workstream first owns this.
