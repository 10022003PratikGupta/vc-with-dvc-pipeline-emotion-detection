import numpy as np
import pandas as pd
import nltk
import re
import string
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import logging

logger = logging.getLogger('data_preprocessing')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


console_handler.setFormatter(formatter)

logger.addFilter(console_handler)



# --- Preprocessing Functions ---
def lowercase(text): return text.lower()
def remove_urls(text): return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
def remove_html_tags(text): return re.sub(r'<.*?>', '', text)
def remove_punctuation(text): return text.translate(str.maketrans('', '', string.punctuation))
def remove_numbers(text): return re.sub(r'\d+', '', text)
def remove_extra_spaces(text): return re.sub(r'\s+', ' ', text).strip()
def tokenization(text): return text.split()

def stopwords_removal(tokens):
    try:
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word not in stop_words]
    except Exception: return tokens

def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def join_tokens(tokens): return " ".join(tokens)

def normalize_text(text):
    try:
        if pd.isna(text): return "" # Handle null values
        text = lowercase(str(text))
        text = remove_urls(text)
        text = remove_html_tags(text)
        text = remove_punctuation(text)
        text = remove_numbers(text) 
        text = remove_extra_spaces(text)
        tokens = tokenization(text)
        tokens = stopwords_removal(tokens)
        tokens = lemmatization(tokens) 
        tokens = stemming(tokens) 
        return join_tokens(tokens)
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        return str(text)

def save_data(data_path: str, train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        # index=False zaroori hai warna extra 'Unnamed: 0' column ban jayega
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    except Exception as e:
        logger.error(f"Saving error: {e}")
        sys.exit(1)

def main() -> None:
    try:
        # 1. Load Data
        if not os.path.exists('./data/raw/train.csv'):
            raise FileNotFoundError("Raw data files nahi mili. Pehle ingestion step run karein.")
            
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        # 2. Check Column (Jo KeyError aapko pehle aa rahi thi)
        if 'text' not in train_data.columns:
            raise KeyError(f"'text' column missing hai. Available columns: {list(train_data.columns)}")

        # 3. NLTK Setup
        logger.info("NLTK resources download ho rahi hain...")
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        # 4. Transform
        logger.info("Preprocessing start ho raha hai...")
        train_data['text'] = train_data['text'].apply(normalize_text)
        test_data['text'] = test_data['text'].apply(normalize_text)
         
        # 5. Save
        data_path = os.path.join("data", "processed")
        save_data(data_path, train_data, test_data)
        logger.info("Success: Processed data save ho gaya!")

    except FileNotFoundError as e:
        logger.error(f"File Error: {e}")
    except KeyError as e:
        logger.error(f"Column Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


