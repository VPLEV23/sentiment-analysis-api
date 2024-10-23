import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re

# Load the saved model and vectorizer
def load_model_and_vectorizer():
    model_path = os.path.join('models', 'logistic_regression_model.joblib')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.joblib')

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

def predict_sentiment(input_text):
    model, vectorizer = load_model_and_vectorizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    clean_input = clean_text(input_text)

    transformed_input = vectorizer.transform([clean_input])

    # Predict sentiment (1 = Negative, 2 = Positive)
    prediction = model.predict(transformed_input)

    return 'Positive' if prediction == 2 else 'Negative'

if __name__ == '__main__':
    input_text = input("Enter a product review: ")
    sentiment = predict_sentiment(input_text)
    print(f"The predicted sentiment is: {sentiment}")
