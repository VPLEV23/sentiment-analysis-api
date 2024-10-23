import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os

# Load preprocessed data
def load_data():
    train_data_path = os.path.join('data', 'processed', 'train_processed.csv')
    test_data_path = os.path.join('data', 'processed', 'test_processed.csv')
    
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    return train_df, test_df

# Train and log Logistic Regression model in MLflow
def train_and_log_logistic_regression():
    with mlflow.start_run(run_name="Logistic Regression"):
        train_df, test_df = load_data()

        train_df['clean_text'].fillna('', inplace=True)
        test_df['clean_text'].fillna('', inplace=True)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(train_df['clean_text'])
        y_train = train_df['sentiment']
        X_test = vectorizer.transform(test_df['clean_text'])
        y_test = test_df['sentiment']

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f'Logistic Regression Validation Accuracy: {val_accuracy}')
        print(f'Logistic Regression Validation Classification Report: \n{classification_report(y_val, y_val_pred)}')

        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f'Logistic Regression Test Accuracy: {test_accuracy}')
        print(f'Logistic Regression Test Classification Report: \n{classification_report(y_test, y_test_pred)}')

        if not os.path.exists('models'):
            os.makedirs('models')

        model_output_path = os.path.join('models', 'logistic_regression_model.joblib')
        vectorizer_output_path = os.path.join('models', 'tfidf_vectorizer.joblib')

        joblib.dump(model, model_output_path)
        joblib.dump(vectorizer, vectorizer_output_path)

        mlflow.log_artifact(model_output_path)
        mlflow.log_artifact(vectorizer_output_path)
        print(f'Model and vectorizer saved for Logistic Regression. Artifacts logged in MLflow.')

if __name__ == '__main__':
    mlflow.set_experiment("Sentiment Analysis Models")

    print("Training Logistic Regression...")
    train_and_log_logistic_regression()
