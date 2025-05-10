import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse
import os

def preprocess_data(input_path, output_path):
    """Предобработка данных"""
    df = pd.read_csv(input_path)
    df = df.dropna()

    categories = df.select_dtypes(include=('object')).columns
    for col in categories:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    preprocess_data(args.input, args.output)