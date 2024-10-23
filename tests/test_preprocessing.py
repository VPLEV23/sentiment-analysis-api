import pytest
import pandas as pd
from src.preprocessing import clean_text, preprocess_dataset
import os

sample_data = {
    'sentiment': [1, 2],
    'title': ['Amazing Product', 'Terrible Experience'],
    'review': ['I love this product, it works great!', 'This product is awful. It broke immediately.']
}

sample_df = pd.DataFrame(sample_data)

def test_clean_text():
    sample_text = "I love this product! It works GREAT."
    expected_output = "love product work great"  # Adjust to reflect lemmatization
    assert clean_text(sample_text) == expected_output

def test_preprocess_dataset(tmpdir):
    # Create temporary input and output file paths
    input_path = os.path.join(tmpdir, 'sample_input.csv')
    output_path = os.path.join(tmpdir, 'sample_output.csv')
    
    # Save the sample dataframe to the input path
    sample_df.to_csv(input_path, index=False, header=False)
    
    # Run the preprocessing function
    preprocess_dataset(input_path, output_path)
    
    # Load the preprocessed data
    processed_data = pd.read_csv(output_path)
    
    # Test that the processed data has the correct number of columns
    assert processed_data.shape[1] == 2  # clean_text and sentiment
    
    # Test that the text was cleaned correctly
    assert processed_data['clean_text'][0] == "amazing product love product work great"  # Updated for lemmatization
    assert processed_data['clean_text'][1] == "terrible experience product awful broke immediately"
    
    # Test that the sentiment column is unchanged
    assert processed_data['sentiment'].tolist() == [1, 2]


