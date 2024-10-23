import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Preprocessing the dataset
def preprocess_dataset(input_path, output_path):
    print(f'Loading dataset from {input_path}')
    data = pd.read_csv(input_path, header=None, names=['sentiment', 'title', 'review'])
    data['title'] = data['title'].fillna('')
    data['text'] = data['title'] + ' ' + data['review']
    print('Cleaning text...')
    data['clean_text'] = data['text'].apply(clean_text)
    data = data[['clean_text', 'sentiment']]
    print(f'Saving preprocessed dataset to {output_path}')
    data.to_csv(output_path, index=False)


# Main function to preprocess both training and test sets
if __name__ == '__main__':
    train_input_path = os.path.join('data', 'raw', 'train.csv')
    test_input_path = os.path.join('data', 'raw', 'test.csv')
    train_output_path = os.path.join('data', 'processed', 'train_processed.csv')
    test_output_path = os.path.join('data', 'processed', 'test_processed.csv')

    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    preprocess_dataset(train_input_path, train_output_path)
    preprocess_dataset(test_input_path, test_output_path)