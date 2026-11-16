import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
df = pd.read_csv(PROC / "patents_clustered.csv")

SUFFIX = r"\b(inc|llc|ltd|limited|co|corp|corporation|company|gmbh|ag|sa|plc|lp|bv|nv)\b\.?"
ACADEMIC = ["universit", "institute", "college", "regents", "school",
            "hospital", "foundation", "academy", "research council"]

def clean_assignees(raw):
    s = str(raw).lower()
    s = re.sub(SUFFIX, "", s)
    s = re.sub(r"[^\w\s,\-]", " ", s)
    parts = [re.sub(r"\s+", " ", p).strip() for p in s.split(",")]
    return [p for p in parts if p]

df["assignee_list"] = df["assignee"].apply(clean_assignees)
df["assignee_clean"] = df["assignee_list"].apply(lambda l: "; ".join(l))
df["primary_assignee"] = df["assignee_list"].apply(lambda l: l[0] if l else "unknown")
df["is_academic"] = df["assignee"].fillna("").str.lower().apply(
    lambda s: any(k in s for k in ACADEMIC))
df = df.drop(columns=["assignee_list"])
df.to_csv(PROC / "patents_final.csv", index=False)

print("Top 10 assignees:")
print(df["primary_assignee"].value_counts().head(10))
print("\nAcademic share:", df["is_academic"].mean().round(3))
print(df.groupby("year")["is_academic"].mean().tail(8).round(3))