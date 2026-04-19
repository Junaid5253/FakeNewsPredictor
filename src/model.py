from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create folders if not exist
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load Data
df = pd.read_csv(DATA_DIR / "clean_data.csv")

X = df['content']
y = df['label']

# Split FIRST 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    "logreg": LogisticRegression(max_iter=1000),
    "nb": MultinomialNB(),
    "svm": LinearSVC()
}

results = []

# Train & Evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1_score": f1
    })

    # Save model
    joblib.dump(model, MODEL_DIR / f"{name}.pkl")

# Save vectorizer
joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")

# Save metrics (for notebook)
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)

print("\n✅ Training complete. Models & results saved.")