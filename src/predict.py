import joblib
from pathlib import Path
from src.preprocessing import clean_text
import numpy as np

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ---------------- Load Vectorizer ----------------
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")

# ---------------- Load Models ----------------
MODELS = {
    "svm": joblib.load(MODEL_DIR / "svm.pkl"),
    "logreg": joblib.load(MODEL_DIR / "logreg.pkl"),
    "nb": joblib.load(MODEL_DIR / "nb.pkl")
}

# ---------------- Prediction Function ----------------
def predict_news(text, model_name="svm"):
    """
    Predict whether news is Fake or Real
    """

    # Select model
    model = MODELS.get(model_name, MODELS["svm"])

    # Preprocess
    processed_text = clean_text(text)

    # Vectorize
    text_vec = vectorizer.transform([processed_text])

    # Prediction
    prediction = model.predict(text_vec)[0]

    # Confidence (handled differently per model)
    if model_name == "svm":
        confidence = model.decision_function(text_vec)[0]
    else:
        confidence = model.predict_proba(text_vec)[0].max()


    prob = 1 / (1 + np.exp(-confidence))
    label = "Real News" if prediction == 1 else "Fake News"

    return label, prob

if __name__ == "__main__":
    print("Fake News Detector")

    text = input("Enter news text: ")
    model_choice = input("Choose model (svm / logreg / nb): ").strip().lower()

    result, prob = predict_news(text, model_choice)

    print(f"\nPrediction: {result}")
    print(f"Confidence: {prob*100:.2f}%")