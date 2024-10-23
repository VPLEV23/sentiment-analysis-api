from flask import Flask, request, jsonify
import joblib
import re
import os

app = Flask(__name__)

# Load the saved model and vectorizer
def load_model_and_vectorizer():
    model_path = os.path.join('models', 'logistic_regression_model.joblib')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.joblib')

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

# Preprocess input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    input_text = data['text']
    clean_input = clean_text(input_text)

    model, vectorizer = load_model_and_vectorizer()

    transformed_input = vectorizer.transform([clean_input])
    prediction = model.predict(transformed_input)

    sentiment = 'Positive' if prediction == 2 else 'Negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
