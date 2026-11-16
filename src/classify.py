import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "patents_labeled.csv")

# Train only on the 10 real domains (leave "Other" out)
df = df[df["domain"] != "Other"].copy()
X, y = df["abstract"], df["domain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=3)
Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
}

best_name, best_f1, best_model = None, -1, None
for name, m in models.items():
    m.fit(Xtr, y_train)
    pred = m.predict(Xte)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"\n=== {name} === accuracy={acc:.3f}  macro-F1={f1:.3f}")
    if f1 > best_f1:
        best_name, best_f1, best_model = name, f1, m

print(f"\nBest model: {best_name}")
pred = best_model.predict(Xte)
print(classification_report(y_test, pred))
labels = sorted(y.unique())
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels))

joblib.dump(best_model, ROOT / "data" / "processed" / "classifier.joblib")
joblib.dump(vec, ROOT / "data" / "processed" / "vectorizer.joblib")
print("\nSaved classifier + vectorizer")