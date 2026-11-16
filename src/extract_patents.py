import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "patent_analysis_data.csv"
OUT = ROOT / "data" / "processed" / "patents_clean.csv"
COLS = ["Patent ID", "Title", "Assignee", "Priority Date", "Publication Date", "Abstract"]

kept = []
for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=2000):
    chunk = chunk.dropna(subset=["Title", "Abstract"])
    chunk = chunk[~chunk["Abstract"].str.contains("Abstract not found", na=False)]
    chunk = chunk[chunk["Abstract"].str.len() > 100]
    chunk = chunk[chunk["Title"].str.isascii() & chunk["Abstract"].str.isascii()]
    kept.append(chunk)

df = pd.concat(kept, ignore_index=True)
df.columns = ["patent_id", "title", "assignee", "priority_date", "publication_date", "abstract"]
df["year"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.year
df = df.drop_duplicates(subset="patent_id")

df.to_csv(OUT, index=False)
print(f"Saved {len(df)} patents to {OUT}")
print(df.head())
print(df["year"].value_counts().sort_index())