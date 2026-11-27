"""
Clustering justification: why K = 12?
Sweeps K, reports elbow / silhouette / Davies-Bouldin, agreement with the domain labels,
compares KMeans with Agglomerative clustering, and checks stability across random seeds.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_mutual_info_score, adjusted_rand_score)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
CHOSEN_K = 12
SEED = 42


def main():
    emb = np.load(PROC / "embeddings.npy")
    df = pd.read_csv(PROC / "patents_final.csv")
    domain = df["domain"].to_numpy(dtype=object)
    labelled = domain != "Other"
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(emb), size=min(6000, len(emb)), replace=False)

    # ---------------- 1. sweep K ----------------
    rows = []
    for k in range(4, 21):
        lab = KMeans(n_clusters=k, random_state=SEED, n_init=5).fit(emb)
        l = lab.labels_
        rows.append({
            "k": k, "inertia": lab.inertia_,
            "silhouette": silhouette_score(emb[sample], l[sample], metric="cosine"),
            "davies_bouldin": davies_bouldin_score(emb, l),
            "agreement_with_domains": adjusted_mutual_info_score(domain[labelled], l[labelled]),
        })
        print(f"K={k:2d}  silhouette={rows[-1]['silhouette']:.3f}  "
              f"DB={rows[-1]['davies_bouldin']:.2f}  AMI={rows[-1]['agreement_with_domains']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "clustering_k_selection.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("inertia", "Inertia (elbow: lower is tighter)"),
              ("silhouette", "Silhouette (higher is better)"),
              ("davies_bouldin", "Davies-Bouldin (lower is better)"),
              ("agreement_with_domains", "Agreement with domain labels (AMI)")]
    for a, (col, title) in zip(ax.ravel(), panels):
        a.plot(res["k"], res[col], marker="o")
        a.axvline(CHOSEN_K, color="red", linestyle="--", label=f"K={CHOSEN_K}")
        a.set_title(title)
        a.set_xlabel("K")
        a.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "clustering_k_selection.png", dpi=150)
    plt.close(fig)

    # ---------------- 2. KMeans vs Agglomerative (on the sample) ----------------
    km = KMeans(n_clusters=CHOSEN_K, random_state=SEED, n_init=10).fit(emb)
    km_s = km.labels_[sample]
    ag_s = AgglomerativeClustering(n_clusters=CHOSEN_K, linkage="ward").fit_predict(emb[sample])
    dom_s = domain[sample]
    lab_s = dom_s != "Other"
    comp = []
    for name, l in [("KMeans", km_s), ("Agglomerative (ward)", ag_s)]:
        comp.append({
            "method": name,
            "silhouette": silhouette_score(emb[sample], l, metric="cosine"),
            "davies_bouldin": davies_bouldin_score(emb[sample], l),
            "agreement_with_domains": adjusted_mutual_info_score(dom_s[lab_s], l[lab_s]),
        })
    comp = pd.DataFrame(comp)
    comp.to_csv(REPORTS / "clustering_method_comparison.csv", index=False)
    print(f"\nMethod comparison at K={CHOSEN_K} (6,000-patent sample):")
    print(comp.round(3).to_string(index=False))

    # ---------------- 3. stability across seeds ----------------
    runs = [KMeans(n_clusters=CHOSEN_K, random_state=s, n_init=5).fit_predict(emb)
            for s in range(5)]
    aris = [adjusted_rand_score(runs[i], runs[j]) for i in range(5) for j in range(i + 1, 5)]
    print(f"\nStability across 5 random seeds at K={CHOSEN_K}: mean ARI = {np.mean(aris):.3f}")
    print(f"\nSaved results to {REPORTS}")


if __name__ == "__main__":
    main()