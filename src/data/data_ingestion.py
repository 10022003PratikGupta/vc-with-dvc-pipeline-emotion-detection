import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import sys
import logging
# import configure
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel("DEBUG")

farmatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(farmatter)
file_handler.setFormatter(farmatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


file_handler.setFormatter(farmatter)

def load_params(data_path: str) -> float: 
    try:
        with open(data_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['data_ingestion']['test_size']
    
    except FileNotFoundError:
        logger.error(f"Error: {data_path} file nahi mili.")
        sys.exit(1)
    except KeyError:
        logger.error("Error: 'test_size' key params.yaml mein missing hai.")
        sys.exit(1)

def read_data(url: str) -> pd.DataFrame:
    try:
        # Pehle headers check karte hain
        df_cols = pd.read_csv(url, nrows=0).columns.tolist()
        required_cols = ['text', 'label']
        
        # Check if columns exist
        for col in required_cols:
            if col not in df_cols:
                raise KeyError(f"CSV mein '{col}' column nahi hai. Available: {df_cols}")
        
        df = pd.read_csv(url, usecols=required_cols, nrows=10000)
        return df
    except FileNotFoundError:
        logger.error(f"Error: CSV file is path par nahi hai: {url}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Read Data Error: {e}")
        sys.exit(1)

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.info(f"Data successfully save ho gaya: {data_path}")
    except Exception as e:
        logger.error(f"Save Data Error: {e}")
        sys.exit(1)

def main() -> None:
    # 1. Load Params
    test_size = load_params('params.yaml')

    # 2. Read Data
    url = r'C:\Users\Pratik\OneDrive\Desktop\text.csv'
    df = read_data(url)
    
    # 3. Split Data
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        data_path = os.path.join("data", "raw")
        
        # 4. Save Data
        save_data(data_path, train_data, test_data)
    except ValueError as e:
        print(f"Split Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
