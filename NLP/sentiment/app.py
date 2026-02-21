import re
import os
import json
import pickle
import numpy as np
import nltk
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --------------------------------------------------
# INITIAL SETUP
# --------------------------------------------------

app = Flask(__name__)

# Ensure stopwords exist
nltk.download('stopwords')

# Initialize NLP tools
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --------------------------------------------------
# LOAD MODEL & VECTORIZER
# --------------------------------------------------

print("Loading model and vectorizer...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("sentiment_model.pkl not found")

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError("vectorizer.pkl not found")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!")

# --------------------------------------------------
# DATA STORAGE (Persistent History)
# --------------------------------------------------

DATA_FILE = os.path.join(BASE_DIR, "analysis_history.json")

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)


def load_history():
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_to_history(entry):
    history = load_history()
    history.append(entry)
    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=4)


# --------------------------------------------------
# TEXT PREPROCESSING
# MUST MATCH TRAINING
# --------------------------------------------------

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)


# --------------------------------------------------
# TOPIC EXTRACTION
# --------------------------------------------------

def extract_topics(text):
    keywords = [
        'delivery', 'customer service', 'product',
        'quality', 'price', 'packaging',
        'shipping', 'return', 'mobile app'
    ]

    found = [k.title() for k in keywords if k in text.lower()]
    return found if found else ['General']


# --------------------------------------------------
# NPS CALCULATION
# --------------------------------------------------

def calculate_nps(sentiment):
    if sentiment == "Positive":
        return "Promoter"
    elif sentiment == "Neutral":
        return "Passive"
    else:
        return "Detractor"


# --------------------------------------------------
# STATISTICS CALCULATION
# --------------------------------------------------

def calculate_stats():
    history = load_history()

    if not history:
        return {
            'total_reviews': 0,
            'positive_count': 0,
            'neutral_count': 0,
            'negative_count': 0,
            'positive_pct': 0,
            'neutral_pct': 0,
            'negative_pct': 0,
            'nps_score': 0,
            'recent_reviews': []
        }

    total = len(history)
    positive = sum(1 for x in history if x['sentiment'] == 'Positive')
    neutral = sum(1 for x in history if x['sentiment'] == 'Neutral')
    negative = sum(1 for x in history if x['sentiment'] == 'Negative')

    promoters = sum(1 for x in history if x['nps'] == 'Promoter')
    detractors = sum(1 for x in history if x['nps'] == 'Detractor')

    return {
        'total_reviews': total,
        'positive_count': positive,
        'neutral_count': neutral,
        'negative_count': negative,
        'positive_pct': round((positive/total)*100, 2),
        'neutral_pct': round((neutral/total)*100, 2),
        'negative_pct': round((negative/total)*100, 2),
        'nps_score': round(((promoters - detractors)/total)*100, 1),
        'recent_reviews': list(reversed(history[-5:]))
    }


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route('/')
def home():
    stats = calculate_stats()
    return render_template('dashboard.html', stats=stats)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.form.get('text')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        processed_text = preprocess_text(text)
        vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized)[0]

        # Confidence
        try:
            proba = model.predict_proba(vectorized)[0]
            confidence = round(np.max(proba) * 100, 2)
        except:
            confidence = 85.0

        # Label mapping (Adjust if needed)
        if prediction == 2:
            sentiment = "Positive"
            emotion = "Happy"
        elif prediction == 1:
            sentiment = "Neutral"
            emotion = "Neutral"
        else:
            sentiment = "Negative"
            emotion = "Unhappy"

        topics = extract_topics(text)
        nps = calculate_nps(sentiment)

        result = {
            'text': text,
            'sentiment': sentiment,
            'emotion': emotion,
            'confidence': confidence,
            'topics': topics,
            'nps': nps,
            'intent': 'Praise, Recommend' if sentiment == 'Positive' else 'Complaint',
            'timestamp': datetime.now().strftime('%b %d, %Y')
        }

        save_to_history(result)

        if request.is_json:
            return jsonify(result), 200

        return redirect('/')

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return {
        "status": "OK",
        "model_loaded": True,
        "vectorizer_loaded": True,
        "total_records": len(load_history())
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)