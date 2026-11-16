import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "patents_clean.csv")

RULES = {
    "Medical Imaging": ["image", "imaging", "radiolog", "mri", "tomograph", "x-ray", "ultrasound", "pathology", "segmentation"],
    "Drug Discovery & Genomics": ["drug", "genom", "gene", "protein", "molecul", "compound", "pharma", "biomarker", "sequenc"],
    "Patient Monitoring & Wearables": ["monitor", "wearable", "sensor", "vital", "glucose", "ecg", "heart", "sleep", "physiolog"],
    "Health Records & Operations": ["record", "ehr", "electronic health", "scheduling", "claim", "insurance", "workflow", "natural language", "text"],
    "Clinical Decision Support": ["diagnos", "predict", "risk", "clinical", "disease", "prognos", "treatment", "therapy"],
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