# Version 3: Prompted LLM via LangChain (Workstream C)

Goal: no training. Prompt a pretrained LLM to classify AND explain, on `data/test.csv`.

Prompt strategies to compare
1. Zero-shot
2. Few-shot (examples taken from TRAIN only, never test)
3. Chain-of-thought

Rules
- API keys go in a local `.env` file (git-ignored). NEVER commit keys.
- Force a parseable output format (e.g. JSON: label, confidence, explanation).
- Save the explanation text too; it is the extra value of this version.

Checklist
- [ ] LangChain pipeline + output parser
- [ ] Run each strategy on test set -> `results/llm_zeroshot.csv`, `llm_fewshot.csv`, `llm_cot.csv`
- [ ] Run on `generalization_set/`
- [ ] Log cost / number of API calls
