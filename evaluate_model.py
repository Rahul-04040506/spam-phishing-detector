import pandas as pd
import pickle
import re
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset (change filename if needed)
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only first two columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert labels to numeric
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Load saved vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# MODEL PERFORMANCE METRICS
# -----------------------------
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMODEL PERFORMANCE METRICS")
print("-" * 40)
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

# -----------------------------
# RESPONSE TIME ANALYSIS
# -----------------------------
sample_message = "Verify your bank account immediately by clicking http://fakebank.com"

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text

# Measure preprocessing
t1 = time.time()
cleaned = preprocess(sample_message)
t2 = time.time()

# Measure vectorization
vector = vectorizer.transform([cleaned])
t3 = time.time()

# Measure prediction
prediction = model.predict(vector)
probability = model.predict_proba(vector)
t4 = time.time()

# Total time
total_time = t4 - t1

print("\nRESPONSE TIME ANALYSIS")
print("-" * 40)
print(f"Text Preprocessing : {t2 - t1:.6f} seconds")
print(f"TF-IDF Vectorization: {t3 - t2:.6f} seconds")
print(f"Model Prediction   : {t4 - t3:.6f} seconds")
print(f"Total Response Time: {total_time:.6f} seconds")