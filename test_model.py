import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# 🔥 Realistic test messages (no obvious spam words)
messages = [
    "Congratulations, your number has been selected for a reward",
    "Please confirm your details to continue using services",
    "Your account needs attention, kindly check immediately",
    "Hey, are we still on for dinner tonight?",
    "I have sent the documents, please review them",
    "Limited time deal available exclusively for you",
    "Click the link to continue your session securely",
    "Reminder: your subscription will expire soon",
    "Let's catch up tomorrow at college",
    "Your transaction could not be processed, retry now"
]

print("\n--- TESTING WITH REALISTIC MESSAGES ---")

for msg in messages:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    result = "Spam" if pred == 1 else "Not Spam"
    confidence = round(max(proba) * 100, 2)

    print(f"\nMessage: {msg}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence}%")