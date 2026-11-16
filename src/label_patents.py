import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "patents_clean.csv")

# Order matters: the first matching domain wins
RULES = {
    "Medical Imaging": ["image", "imaging", "radiolog", "mri", "tomograph", "x-ray", "ultrasound", "pathology", "segmentation"],
    "Drug Discovery & Genomics": ["drug", "genom", "gene", "protein", "molecul", "compound", "pharma", "biomarker", "sequenc", "peptide"],
    "Clinical Decision Support": ["diagnos", "predict", "risk", "clinical", "disease", "prognos", "treatment", "therapy", "disorder", "pneumonia", "infarction", "epilep", "aneurysm", "brain", "cancer", "tumor", "cardi", "referral"],
    "Patient Monitoring & Wearables": ["monitor", "wearable", "sensor", "vital", "glucose", "ecg", "heart", "sleep", "physiolog", "patient", "movement", "body", "fitness"],
    "Health Records & Operations": ["record", "ehr", "electronic health", "scheduling", "claim", "insurance", "workflow", "medical", "health", "diet", "nutrition", "hospital"],
    "Autonomous Systems & Mobility": ["vehicle", "autonomous", "driving", "road", "accident", "drone", "robot", "rescue", "location", "positioning", "transport"],
    "Networking & Hardware": ["networking", "wireless", "v2x", "memory", "controller", "interposer", "transmission", "electronic device", "circuit", "processor", "stage"],
    "Security & Privacy": ["privacy", "secur", "blockchain", "biometric", "encrypt", "identity", "authentication", "immutable", "trusted", "identifier"],
    "Language & Knowledge Systems": ["question", "language", "knowledge graph", "corpus", "corpuses", "speech", "spoken", "semantic", "text"],
    "General ML & Data Methods": ["data", "learning", "model", "classif", "feature", "artificial intelligence", "neural", "training", "dataset", "convolution", "kalman", "instances", "video"],
}

def label(title):
    t = str(title).lower()
    for domain, words in RULES.items():
        if any(w in t for w in words):
            return domain
    return "Other"

df["domain"] = df["title"].apply(label)
print(df["domain"].value_counts())
df.to_csv(ROOT / "data" / "processed" / "patents_labeled.csv", index=False)