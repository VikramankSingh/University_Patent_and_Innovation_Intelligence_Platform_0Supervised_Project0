import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "patents_labeled.csv")

texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
np.save(ROOT / "data" / "processed" / "embeddings.npy", emb)
print("Embeddings shape:", emb.shape)

# quick sanity check: most similar patents to patent #0
sims = emb @ emb[0]
top = np.argsort(-sims)[1:6]
print("\nPatent 0:", df.loc[0, "title"])
for i in top:
    print(f"  {sims[i]:.2f}  {df.loc[i, 'title']}")