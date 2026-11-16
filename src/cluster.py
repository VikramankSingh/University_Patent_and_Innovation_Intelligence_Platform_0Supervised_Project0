import numpy as np
import pandas as pd
import umap
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
df = pd.read_csv(PROC / "patents_labeled.csv")
emb = np.load(PROC / "embeddings.npy")

K = 12
df["cluster"] = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(emb)

# 2D map for visualization
coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=42).fit_transform(emb)
df["umap_x"], df["umap_y"] = coords[:, 0], coords[:, 1]

# Name each cluster by its most distinctive terms
vec = TfidfVectorizer(stop_words="english", max_features=20000, min_df=5)
X = vec.fit_transform(df["abstract"].fillna(""))
terms = np.array(vec.get_feature_names_out())
overall = np.asarray(X.mean(axis=0)).ravel()
names = {}
for c in range(K):
    mean = np.asarray(X[(df["cluster"] == c).values].mean(axis=0)).ravel()
    names[c] = ", ".join(terms[(mean - overall).argsort()[-5:][::-1]])
df["cluster_name"] = df["cluster"].map(names)

df.to_csv(PROC / "patents_clustered.csv", index=False)

print(df.groupby(["cluster", "cluster_name"]).size().to_string())
print("\nSaved patents_clustered.csv")