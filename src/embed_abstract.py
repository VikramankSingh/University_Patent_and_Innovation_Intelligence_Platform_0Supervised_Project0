import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

df = pd.read_csv(PROC / "patents_final.csv")
texts = df["abstract"].fillna("").astype(str).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
np.save(PROC / "embeddings_abstract.npy", emb)
print("Saved", emb.shape)