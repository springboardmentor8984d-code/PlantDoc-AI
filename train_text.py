import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(r"C:\plantdiseaseai\datasets\text1.csv")

# ---------------------------------------------------------
# Boost PLANT_NAME and lowercase everything
# ---------------------------------------------------------
df["full_text"] = (
    df["plant_name"] + " " +
    df["plant_name"] + " " +     # boosts plant importance
    df["symptoms"]
).str.lower()

X = df["full_text"]
y = df["disease"]

# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# TF-IDF Vectorizer (ngrams boost accuracy)
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

# ---------------------------------------------------------
# Optimized Logistic Regression
# ---------------------------------------------------------
model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    C=2.0,
    solver="liblinear"
)

model.fit(X_train_vec, y_train)

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
X_test_vec = vectorizer.transform(X_test)
acc = model.score(X_test_vec, y_test)
print("Accuracy:", acc)

# ---------------------------------------------------------
# Save Model + Vectorizer
# ---------------------------------------------------------
os.makedirs(r"C:\plantdiseaseai\saved_models", exist_ok=True)
joblib.dump(model, r"C:\plantdiseaseai\saved_models\text_model.pkl")
joblib.dump(vectorizer, r"C:\plantdiseaseai\saved_models\text_vectorizer.pkl")

print("🔥 Text model trained successfully with boosted plant matching!")
