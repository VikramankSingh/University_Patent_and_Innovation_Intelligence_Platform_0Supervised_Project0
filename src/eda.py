"""
Exploratory Data Analysis (EDA)
Covers: missing values, data-cleaning funnel, class imbalance, distributions,
outliers, correlations, relationships with the target, and trends.
Figures + summary are saved to reports/eda/.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "patent_analysis_data.csv"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["Patent ID", "Title", "Assignee", "Priority Date", "Publication Date", "Abstract"]
summary = []


def say(text=""):
    print(text)
    summary.append(text)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def main():
    # ============ 1. RAW DATA: missing values + cleaning funnel ============
    say("=== 1. Raw data quality (scanning the raw CSV, takes a few minutes) ===")
    n_raw = 0
    missing = {c: 0 for c in COLS}
    stages = {"raw rows": 0, "has title + abstract": 0, "abstract is real (not 'not found')": 0,
              "abstract > 100 chars": 0, "English / ASCII text": 0}
    for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=5000):
        n_raw += len(chunk)
        for c in COLS:
            missing[c] += int(chunk[c].isna().sum())
        not_found = chunk["Abstract"].str.contains("Abstract not found", na=False)
        missing["Abstract"] += int(not_found.sum())
        stages["raw rows"] += len(chunk)
        c1 = chunk.dropna(subset=["Title", "Abstract"])
        stages["has title + abstract"] += len(c1)
        c2 = c1[~c1["Abstract"].str.contains("Abstract not found", na=False)]
        stages["abstract is real (not 'not found')"] += len(c2)
        c3 = c2[c2["Abstract"].str.len() > 100]
        stages["abstract > 100 chars"] += len(c3)
        c4 = c3[c3["Title"].str.isascii() & c3["Abstract"].str.isascii()]
        stages["English / ASCII text"] += len(c4)

    df = pd.read_csv(PROC / "patents_final.csv")
    stages["after removing duplicate IDs (final)"] = len(df)

    say(f"Raw rows: {n_raw:,}")
    miss_pct = {c: 100 * v / n_raw for c, v in missing.items()}
    for c, p in miss_pct.items():
        say(f"  missing {c}: {p:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(miss_pct.keys(), miss_pct.values(), color="#4C78A8")
    ax.set_ylabel("% missing")
    ax.set_title("Missing values in the raw data (Abstract includes 'Abstract not found')")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "01_missing_values.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    names, vals = list(stages.keys()), list(stages.values())
    ax.barh(names[::-1], vals[::-1], color="#59A14F")
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:,}", va="center")
    ax.set_title("Data cleaning funnel")
    ax.set_xlim(0, max(vals) * 1.15)
    save(fig, "02_cleaning_funnel.png")
    say(f"Cleaning funnel: {n_raw:,} raw -> {len(df):,} patents kept "
        f"({100 * len(df) / n_raw:.1f}%)")

    # ============ 2. Class imbalance ============
    say("\n=== 2. Class balance (weak labels from title keywords) ===")
    counts = df["domain"].value_counts()
    for k, v in counts.items():
        say(f"  {k:34s} {v:6d}  ({100 * v / len(df):.1f}%)")
    trainable = counts.drop("Other", errors="ignore")
    say(f"Imbalance ratio (largest/smallest trainable class): "
        f"{trainable.max() / trainable.min():.1f}x")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#E15759")
    ax.set_title("Patents per domain (class imbalance)")
    ax.set_xlabel("Number of patents")
    save(fig, "03_class_balance.png")

    # ============ 3. Distributions + outliers ============
    say("\n=== 3. Text length distributions and outliers ===")
    df["abstract_words"] = df["abstract"].fillna("").str.split().str.len()
    df["title_words"] = df["title"].fillna("").str.split().str.len()
    q1, q3 = df["abstract_words"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    n_out = int((df["abstract_words"] > upper).sum())
    say(f"Abstract length (words): mean={df['abstract_words'].mean():.0f}, "
        f"median={df['abstract_words'].median():.0f}, max={df['abstract_words'].max()}")
    say(f"Outliers above Q3+1.5*IQR ({upper:.0f} words): {n_out} "
        f"({100 * n_out / len(df):.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    clip = df["abstract_words"].quantile(0.99)
    axes[0].hist(df["abstract_words"].clip(upper=clip), bins=50, color="#4C78A8")
    axes[0].set_title("Abstract length (words, clipped at 99th percentile)")
    axes[0].set_xlabel("words")
    axes[1].hist(df["title_words"], bins=range(0, int(df["title_words"].max()) + 2),
                 color="#F28E2B")
    axes[1].set_title("Title length (words)")
    axes[1].set_xlabel("words")
    save(fig, "04_text_length_distributions.png")

    order = counts.index.tolist()
    data = [df.loc[df["domain"] == d, "abstract_words"].clip(upper=clip) for d in order]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_title("Abstract length by domain")
    ax.set_ylabel("words")
    save(fig, "05_abstract_length_by_domain.png")

    # ============ 4. Trends ============
    say("\n=== 4. Patents over time ===")
    yearly = df.dropna(subset=["year"]).groupby(df["year"].astype("Int64")).size()
    say(f"Years: {int(yearly.index.min())}-{int(yearly.index.max())}; "
        f"peak year {int(yearly.idxmax())} with {int(yearly.max()):,} patents")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(yearly.index.astype(int), yearly.values, color="#76B7B2")
    ax.set_title("Patents per publication year")
    ax.set_xlabel("year")
    save(fig, "06_patents_per_year.png")

    # ============ 5. Assignees ============
    say("\n=== 5. Assignee concentration ===")
    top = df["primary_assignee"].value_counts().head(15)
    say(f"Distinct primary assignees: {df['primary_assignee'].nunique():,}")
    say(f"Top 15 assignees hold {100 * top.sum() / len(df):.1f}% of patents; "
        f"#1 = {top.index[0]} ({100 * top.iloc[0] / len(df):.1f}%)")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#B07AA1")
    ax.set_title("Top 15 assignees (cleaned names)")
    save(fig, "07_top_assignees.png")

    # ============ 6. Correlations ============
    say("\n=== 6. Correlations between numeric features (Spearman) ===")
    num = pd.DataFrame({
        "abstract_words": df["abstract_words"],
        "title_words": df["title_words"],
        "year": df["year"],
        "n_assignees": df["assignee_clean"].fillna("").str.count("; ") + 1,
        "is_academic": df["is_academic"].astype(int),
    })
    corr = num.corr(method="spearman")
    say(corr.round(2).to_string())
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Spearman correlation of numeric features")
    save(fig, "08_correlations.png")

    # ============ 7. Text vs target: distinctive words per domain ============
    say("\n=== 7. Most distinctive abstract words per domain (TF-IDF) ===")
    d = df[df["domain"] != "Other"]
    vec = TfidfVectorizer(stop_words="english", min_df=5, max_features=20000)
    X = vec.fit_transform(d["abstract"].fillna(""))
    terms = np.array(vec.get_feature_names_out())
    overall = np.asarray(X.mean(axis=0)).ravel()
    rows = []
    for dom in sorted(d["domain"].unique()):
        m = np.asarray(X[(d["domain"] == dom).values].mean(axis=0)).ravel()
        words = ", ".join(terms[(m - overall).argsort()[-8:][::-1]])
        rows.append({"domain": dom, "top_terms": words})
        say(f"  {dom}: {words}")
    pd.DataFrame(rows).to_csv(OUT / "top_terms_by_domain.csv", index=False)

    # ============ 8. Duplicates ============
    say("\n=== 8. Duplicates ===")
    pairs = pd.read_csv(PROC / "duplicate_pairs.csv")
    say(f"Overlap pairs (similarity >= 0.90): {len(pairs):,}")
    say(pairs["type"].value_counts().to_string())

    (OUT / "eda_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nSaved figures and eda_summary.txt to {OUT}")


if __name__ == "__main__":
    main()