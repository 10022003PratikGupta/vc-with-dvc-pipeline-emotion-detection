import numpy as np
import pandas as pd
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import logging

logger = logging.getLogger()
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

farmatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(farmatter)
logger.addHandler(console_handler)

def safe_params(data_path: str) -> int:
    try:
        with open(data_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['feature_engineering']['max_features']
    except FileNotFoundError:
        logger.error(f"Error: {data_path} file nahi mili.")
        sys.exit(1)
    except (KeyError, TypeError):
        logger.error("Error: params.yaml mein 'feature_engineering' ya 'max_features' missing hai.")
        sys.exit(1)

def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        # index=False use karein taaki csv mein extra column na jude
        train_df.to_csv(os.path.join(data_path, "train_tfidf.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_tfidf.csv"), index=False)
        logger.info(f"BoW features successfully save ho gaye: {data_path}")
    except Exception as e:
        logger.error(f"Data save karne mein error: {e}")
        sys.exit(1)

def main():
    try:
        # 1. Load Parameters
        max_features = safe_params('params.yaml')

        # 2. Load Processed Data
        train_path = './data/processed/train_processed.csv'
        test_path = './data/processed/test_processed.csv'

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Processed files missing hain. Pehle preprocessing script run karein.")

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        # 3. Column Validation (Very Important for your 'label' error)
        for col in ['text', 'label']:
            if col not in train_data.columns:
                raise KeyError(f"DataFrame mein '{col}' column nahi mila. Available: {list(train_data.columns)}")

        # 4. Handle NaN & Types
        train_data['text'] = train_data['text'].fillna("").astype(str)   
        test_data['text'] = test_data['text'].fillna("").astype(str)

        x_train, y_train = train_data['text'].values, train_data['label'].values
        x_test, y_test = test_data['text'].values, test_data['label'].values

        # 5. Vectorization
        logger.info(f"TfidfVectorizer apply ho raha hai (max_features={max_features})...")
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)

        # 6. Create Resulting DataFrames
        train_df = pd.DataFrame(x_train_tfidf.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(x_test_tfidf.toarray())
        test_df['label'] = y_test

        # 7. Save
        data_path = os.path.join("data", "features")
        save_data(data_path, train_df, test_df)

    except FileNotFoundError as e:
        logger.info(f"File Error: {e}")
    except KeyError as e:
        logger.info(f"Column Error: {e}")
    except Exception as e:
        logger.info(f"Unexpected Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
