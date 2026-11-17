"""
Module 4: Technology Classification - full supervised ML experiment.

What this script does (maps to the course rubric):
  1. Leakage-safe split: duplicate / near-duplicate patents are grouped so a patent
     "family" never lands in both train and test.
  2. Baseline + multiple models (Naive Bayes, Logistic Regression, Linear SVM,
     Random Forest, Gradient Boosting, embedding-based Logistic Regression).
  3. Cross-validation on the training set only (model selection never touches test).
  4. Hyperparameter tuning (RandomizedSearchCV) on the TF-IDF + Logistic Regression pipeline.
  5. One final evaluation on the held-out test set + error analysis.
  6. Saves the best text pipeline for the Streamlit app.
"""
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (StratifiedGroupKFold, RandomizedSearchCV,
                                     cross_validate, train_test_split)
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
SEED = 42


# ---------------------------------------------------------------- helpers
def build_groups(df, pairs):
    """Group patents that are duplicates (embedding similarity >= 0.90 or same title)."""
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in zip(pairs["idx_a"], pairs["idx_b"]):
        union(int(a), int(b))

    norm = (df["title"].fillna("").str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
    first = {}
    for i, t in enumerate(norm):
        if t in first:
            union(i, first[t])
        else:
            first[t] = i
    return np.array([find(i) for i in range(len(df))])


def tfidf(max_features=20000, ngram=(1, 2), min_df=3):
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram, min_df=min_df,
                           stop_words="english", sublinear_tf=True)


def get_models():
    """name -> (estimator, input type). 'text' = abstracts, 'emb' = sentence embeddings."""
    return {
        "Baseline (majority class)": (DummyClassifier(strategy="most_frequent"), "text"),
        "TF-IDF + Naive Bayes": (Pipeline([("tfidf", tfidf()), ("clf", ComplementNB())]), "text"),
        "TF-IDF + Logistic Regression": (Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]), "text"),
        "TF-IDF + Linear SVM": (Pipeline([
            ("tfidf", tfidf()),
            ("clf", LinearSVC(class_weight="balanced"))]), "text"),
        "TF-IDF + Random Forest": (Pipeline([
            ("tfidf", tfidf(max_features=10000)),
            ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED))]), "text"),
        "TF-IDF + SVD + Gradient Boosting": (Pipeline([
            ("tfidf", tfidf(max_features=10000)),
            ("svd", TruncatedSVD(n_components=150, random_state=SEED)),
            ("clf", HistGradientBoostingClassifier(max_iter=150, class_weight="balanced",
                                                   random_state=SEED))]), "text"),
        "Embeddings + Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"), "emb"),
    }


def main():
    # ------------------------------------------------------------ load
    df_all = pd.read_csv(PROC / "patents_final.csv")
    emb_all = np.load(PROC / "embeddings_abstract.npy")
    pairs = pd.read_csv(PROC / "duplicate_pairs.csv")

    groups_all = build_groups(df_all, pairs)
    keep = (df_all["domain"] != "Other").values
    df = df_all[keep].reset_index(drop=True)
    emb = emb_all[keep]
    groups = groups_all[keep]
    X_text = df["abstract"].fillna("").astype(object)
    y = df["domain"].to_numpy(dtype=object)

    print(f"Patents used (excluding 'Other'): {len(df)}")
    print(f"Duplicate groups: {len(np.unique(groups))} "
          f"({len(df) - len(np.unique(groups))} patents are repeats of another patent)\n")

    # ------------------------------------------------- leakage-safe split
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(outer.split(X_text, y, groups))
    assert not set(groups[train_idx]) & set(groups[test_idx]), "group leakage!"
    Xt_tr, Xt_te = X_text.iloc[train_idx], X_text.iloc[test_idx]
    Xe_tr, Xe_te = emb[train_idx], emb[test_idx]
    y_tr, y_te, g_tr = y[train_idx], y[test_idx], groups[train_idx]
    print(f"Train: {len(train_idx)}  |  Test (held out): {len(test_idx)}  "
          f"| no duplicate group appears in both\n")

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    scoring = {"f1": "f1_macro", "acc": "accuracy"}

    # ------------------------------------------ compare models with CV
    rows, fitted = [], {}
    for name, (model, kind) in get_models().items():
        t0 = time.time()
        Xtr = Xt_tr if kind == "text" else Xe_tr
        Xte = Xt_te if kind == "text" else Xe_te
        cvres = cross_validate(model, Xtr, y_tr, groups=g_tr, cv=cv, scoring=scoring)
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte)
        rows.append({
            "model": name, "input": kind,
            "cv_macro_f1": cvres["test_f1"].mean(), "cv_f1_std": cvres["test_f1"].std(),
            "cv_accuracy": cvres["test_acc"].mean(),
            "test_macro_f1": f1_score(y_te, pred, average="macro"),
            "test_accuracy": accuracy_score(y_te, pred),
            "seconds": round(time.time() - t0, 1),
        })
        fitted[name] = (model, kind)
        print(f"{name:38s} CV macro-F1={rows[-1]['cv_macro_f1']:.3f} "
              f"(+/-{rows[-1]['cv_f1_std']:.3f})  [{rows[-1]['seconds']}s]")

    # ----------------------------------------- hyperparameter tuning
    print("\nTuning TF-IDF + Logistic Regression (RandomizedSearchCV)...")
    t0 = time.time()
    search = RandomizedSearchCV(
        Pipeline([("tfidf", tfidf()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        param_distributions={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [2, 3, 5],
            "tfidf__max_features": [10000, 20000, 40000],
            "clf__C": loguniform(0.1, 30),
        },
        n_iter=10, cv=cv, scoring="f1_macro", n_jobs=-1, random_state=SEED, refit=True)
    search.fit(Xt_tr, y_tr, groups=g_tr)
    best = search.best_estimator_
    best_idx = search.best_index_
    pred = best.predict(Xt_te)
    rows.append({
        "model": "TF-IDF + Logistic Regression (tuned)", "input": "text",
        "cv_macro_f1": search.best_score_,
        "cv_f1_std": search.cv_results_["std_test_score"][best_idx],
        "cv_accuracy": np.nan,
        "test_macro_f1": f1_score(y_te, pred, average="macro"),
        "test_accuracy": accuracy_score(y_te, pred),
        "seconds": round(time.time() - t0, 1),
    })
    fitted["TF-IDF + Logistic Regression (tuned)"] = (best, "text")
    print(f"Best params: {search.best_params_}")

    # ------------------------------------------------ comparison table
    results = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    results.to_csv(REPORTS / "model_comparison.csv", index=False)
    print("\n=== Model comparison (selected by CV on TRAIN; test shown for reporting) ===")
    print(results.round(3).to_string(index=False))

    # ---------------------- pick best text pipeline (for the app) by CV
    text_rows = results[results["input"] == "text"]
    best_name = text_rows.iloc[0]["model"]
    best_model = fitted[best_name][0]
    print(f"\nBest text-based model (by CV): {best_name}")
    if results.iloc[0]["input"] == "emb":
        print("Note: the embedding model scored higher on CV; the text pipeline is saved "
              "because the app needs a text-in/label-out model.")

    # ---------------------------------------- leakage demonstration
    lr_cfg = Pipeline([("tfidf", tfidf()),
                       ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    a, b, ya, yb = train_test_split(X_text, y, test_size=0.2, random_state=SEED, stratify=y)
    naive_f1 = f1_score(yb, lr_cfg.fit(a, ya).predict(b), average="macro")
    group_f1 = results.loc[results["model"] == "TF-IDF + Logistic Regression",
                           "test_macro_f1"].iloc[0]
    print(f"\nLeakage check (same model): naive random split macro-F1={naive_f1:.3f} "
          f"vs duplicate-safe split macro-F1={group_f1:.3f}")

    # ------------------------------------------ final test evaluation
    pred = best_model.predict(Xt_te)
    labels = sorted(np.unique(y))
    report = classification_report(y_te, pred, digits=3)
    print("\n=== FINAL TEST REPORT ===")
    print(report)
    (REPORTS / "classification_report.txt").write_text(
        f"Model: {best_name}\n\n{report}", encoding="utf-8")

    cm = confusion_matrix(y_te, pred, labels=labels)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (row-normalised) - {best_name}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------- error analysis
    print("Most common confusions (actual -> predicted):")
    pairs_err = []
    for i, a_ in enumerate(labels):
        for j, p_ in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs_err.append((cm[i, j], a_, p_))
    for n, a_, p_ in sorted(pairs_err, reverse=True)[:8]:
        print(f"  {n:4d}  {a_}  ->  {p_}")

    if hasattr(best_model, "predict_proba"):
        conf = best_model.predict_proba(Xt_te).max(axis=1)
    else:
        scores = best_model.decision_function(Xt_te)
        conf = scores.max(axis=1)
    wrong = pd.DataFrame({
        "title": df.iloc[test_idx]["title"].values,
        "abstract_start": Xt_te.str[:300].values,
        "actual": y_te, "predicted": pred, "confidence": conf,
    })
    wrong = wrong[wrong["actual"] != wrong["predicted"]].sort_values("confidence", ascending=False)
    wrong.to_csv(REPORTS / "misclassified_examples.csv", index=False)
    print(f"\n{len(wrong)} misclassified test patents. Most confident mistakes "
          f"(often label noise, not model failure):")
    for _, r in wrong.head(8).iterrows():
        print(f"  [{r['actual']} -> {r['predicted']}] {r['title']}")

    # -------------------------------------------------------- save
    joblib.dump(best_model, PROC / "classifier_pipeline.joblib")
    print(f"\nSaved {PROC / 'classifier_pipeline.joblib'}")
    print(f"Saved reports to {REPORTS}")


if __name__ == "__main__":
    main()