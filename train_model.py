import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("combined_dataset.csv")

# Keep correct columns
data = data[['target', 'text']]
data.columns = ['Category', 'Message']

# Remove duplicates
data.drop_duplicates(inplace=True)

# Convert labels (ham=0, spam=1)
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Remove any missing values
data.dropna(inplace=True)

# Split data
X = data['Message']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  TF-IDF 
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model 
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Accuracy
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# TESTING
test_messages = [
    "Your account has been compromised, click here to secure it",
    "Congratulations! You have won a reward",
    "Please verify your bank account immediately",
    "Limited time offer just for you",
    "Hey bro, let's meet tomorrow",
    "Your subscription will expire soon, renew now",
    "Click this link to update your details",
    "Are you coming to class today?"
]

print("\n--- TESTING ---")
for msg in test_messages:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    result = "Spam" if pred == 1 else "Not Spam"
    confidence = round(max(proba) * 100, 2)

    print(f"\nMessage: {msg}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence}%")