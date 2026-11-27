"""
Creates blind labeling sheets (no weak label, no model prediction shown) so the team can
hand-label a random sample and measure how reliable the keyword labels and the model are.
120 patents: 20 are shared by all 3 labelers (to measure agreement), 100 are split between them.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "labeling"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "processed" / "patents_final.csv")
sample = df.sample(120, random_state=7).reset_index(drop=True)
sample["abstract_start"] = sample["abstract"].fillna("").str.slice(0, 700)
cols = ["patent_id", "title", "abstract_start"]

shared, rest = sample.iloc[:20], sample.iloc[20:]
for n in range(3):
    part = pd.concat([shared, rest.iloc[n::3]])[cols].copy()
    part["my_domain"] = ""
    part.to_csv(OUT / f"label_sheet_{n + 1}.csv", index=False, encoding="utf-8-sig")
    print(f"label_sheet_{n + 1}.csv: {len(part)} patents")

sample[["patent_id"]].to_csv(OUT / "sample_ids.csv", index=False)
print(f"\nSaved to {OUT}")