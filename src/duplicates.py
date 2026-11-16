import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
df = pd.read_csv(PROC / "patents_labeled.csv")
emb = np.load(PROC / "embeddings.npy")

THRESHOLD = 0.90
pairs = []
for start in range(0, len(emb), 1000):
    sims = emb[start:start + 1000] @ emb.T
    rows, cols = np.where(sims >= THRESHOLD)
    for r, c in zip(rows, cols):
        i = start + r
        if i < c:
            pairs.append((i, c, float(sims[r, c])))

out = pd.DataFrame(pairs, columns=["idx_a", "idx_b", "similarity"])
for col in ["patent_id", "title", "assignee"]:
    out[f"{col}_a"] = df.loc[out["idx_a"], col].values
    out[f"{col}_b"] = df.loc[out["idx_b"], col].values

out["same_assignee"] = out["assignee_a"] == out["assignee_b"]
out["type"] = np.where(out["similarity"] >= 0.98, "Same invention (likely patent family)",
              np.where(out["similarity"] >= 0.95, "Near-duplicate", "Partial overlap"))
out = out.sort_values("similarity", ascending=False)
out.to_csv(PROC / "duplicate_pairs.csv", index=False)

print(out["type"].value_counts())
print("\nSame assignee:", out["same_assignee"].mean().round(2))
print("\nCross-assignee overlaps (the interesting ones):")
cross = out[~out["same_assignee"]]
print(cross[["similarity", "title_a", "title_b", "assignee_a", "assignee_b"]].head(8).to_string())