"""
Step 1 (Workstream A): inspect the dataset and create the SHARED train/test split.

Run once, from the repo root:
    python 01_inspect_and_split.py --fake data/raw/Fake.csv --true data/raw/True.csv

Outputs (commit these so teammates use the exact same split):
    data/train.csv, data/test.csv   columns: article_id, text, label   (0 = Fake, 1 = Real)
    data/inspection_report.txt      numbers you will cite in the report

IMPORTANT: do not re-run with different settings after teammates start.
The split must stay identical for everyone, or the final comparison is invalid.
"""

import argparse
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", required=True)
    ap.add_argument("--true", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--strip-reuters", action="store_true",
                    help="remove the '(Reuters)' dateline leak from Real articles")
    args = ap.parse_args()

    fake = pd.read_csv(args.fake)
    true = pd.read_csv(args.true)
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=== BASIC SHAPE ===")
    log(f"Fake articles: {len(fake)}   Real articles: {len(true)}   Total: {len(df)}")
    log(f"Columns: {list(df.columns)}")

    # Combine title + body into one text field
    df["text"] = df.get("title", "").fillna("") + " " + df["text"].fillna("")
    df["text"] = df["text"].str.strip()

    log("\n=== PROBLEMS TO CHECK ===")
    empty = (df["text"].str.len() < 20).sum()
    log(f"Near-empty articles (<20 chars): {empty}")

    dupes = df.duplicated(subset="text").sum()
    log(f"Exact duplicate articles: {dupes}")

    # Duplicates across labels would be especially bad (same text, both labels)
    cross = df.groupby("text")["label"].nunique()
    log(f"Texts appearing under BOTH labels: {(cross > 1).sum()}")

    log("\n=== LEAKAGE CHECK (why 99% accuracy can be misleading) ===")
    reuters_real = true["text"].fillna("").str.contains(r"\(Reuters\)", regex=True).mean()
    reuters_fake = fake["text"].fillna("").str.contains(r"\(Reuters\)", regex=True).mean()
    log(f"Real articles containing '(Reuters)': {reuters_real:.1%}")
    log(f"Fake articles containing '(Reuters)': {reuters_fake:.1%}")
    log("If the first number is high and the second near 0, a model can score well")
    log("just by spotting 'Reuters' -- report this in your limitations section.")

    if "subject" in df.columns:
        log("\nSubject distribution by label (topics may also leak the label):")
        log(str(pd.crosstab(df["subject"], df["label"])))

    log("\n=== LENGTH (words) ===")
    df["words"] = df["text"].str.split().str.len()
    log(str(df.groupby("label")["words"].describe().round(0)))
    over = (df["words"] > 400).mean()
    log(f"Articles over ~400 words (roughly BERT's 512-token ceiling): {over:.1%}")

    # Clean-up before splitting
    df = df[df["text"].str.len() >= 20]
    df = df.drop_duplicates(subset="text", keep="first")
    # drop any text that still appears under both labels (ambiguous)
    ambiguous = df.groupby("text")["label"].transform("nunique") > 1
    df = df[~ambiguous]

    if args.strip_reuters:
        df["text"] = df["text"].apply(
            lambda t: re.sub(r"^.{0,60}\(Reuters\)\s*-?\s*", "", t))
        log("\nStripped '(Reuters)' datelines from text.")

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["article_id"] = df.index

    train, test = train_test_split(
        df[["article_id", "text", "label"]],
        test_size=TEST_SIZE, random_state=SEED, stratify=df["label"])

    os.makedirs(args.out, exist_ok=True)
    train.to_csv(os.path.join(args.out, "train.csv"), index=False)
    test.to_csv(os.path.join(args.out, "test.csv"), index=False)

    log("\n=== SPLIT SAVED ===")
    log(f"After cleaning: {len(df)} articles")
    log(f"train: {len(train)}   test: {len(test)}   (seed={SEED}, stratified)")
    log(f"train label balance: {train['label'].value_counts(normalize=True).round(3).to_dict()}")
    log(f"test  label balance: {test['label'].value_counts(normalize=True).round(3).to_dict()}")

    with open(os.path.join(args.out, "inspection_report.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
