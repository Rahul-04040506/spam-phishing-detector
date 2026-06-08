from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load model + vectorizer
model = pickle.load(open("model.pkl", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

#  Simple phishing detection
def detect_phishing(message):
    suspicious_words = ["login", "verify", "bank", "account", "update", "urgent"]

    has_url = "http" in message.lower() or "www" in message.lower()
    found_words = [w for w in suspicious_words if w in message.lower()]

    if has_url and found_words:
        return "⚠️ Phishing Detected", found_words
    elif has_url:
        return "⚠️ Suspicious Link", []
    else:
        return "Safe", []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['news']

    # Spam prediction
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    spam_result = "🚨 Spam" if pred == 1 else "✅ Not Spam"
    confidence = round(max(proba) * 100, 2)

    # Phishing check
    phishing_status, words = detect_phishing(message)

    return render_template(
        'index.html',
        label=spam_result,
        confidence=confidence,
        phishing=phishing_status,
        words=words,
        news=message
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how')
def how():
    return render_template('how_it_works.html')

if __name__ == "__main__":
    app.run(debug=True)