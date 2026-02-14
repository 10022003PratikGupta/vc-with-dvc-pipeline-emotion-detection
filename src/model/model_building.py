import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.ensemble import HistGradientBoostingClassifier
import yaml
import logging

logger = logging.getLogger()
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

farmatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(farmatter)

logger.addHandler(console_handler)

def load_params(data_path: str) -> dict:
    try:
        with open(data_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['model_building']
    except FileNotFoundError:
        logging.error(f"Error: {data_path} file nahi mili.")
        sys.exit(1)
    except (KeyError, TypeError):
        logging.error("Error: params.yaml mein 'model_building' section missing hai.")
        sys.exit(1)

def save_model(clf, data_path: str) -> None:
    try:
        # Pickle file save karna
        with open(data_path, 'wb') as file:
            pickle.dump(clf, file)
        logging.error(f"Model successfully save ho gaya: {data_path}")
    except Exception as e:
        logging.error(f"Model save karne mein error: {e}")
        sys.exit(1)

def main() -> None:
    try:
        # 1. Load Data
        train_path = './data/features/train_bow.csv'
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Feature file nahi mili: {train_path}")

        train_data = pd.read_csv(train_path)

        # 2. Extract Features and Target
        # iloc use karte waqt check karein ki data khali toh nahi
        if train_data.empty:
            raise ValueError("Train data khali hai. Training possible nahi hai.")

        x_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        
        # 3. Load Parameters
        params = load_params('params.yaml')
        
        # 4. Model Training
        logging.error("Model training start ho rahi hai...")
        # HistGradientBoostingClassifier NaN values handle kar leta hai, 
        # lekin hum check kar rahe hain taaki params sahi hon.
        clf = HistGradientBoostingClassifier(
            max_iter=params.get('max_iter', 100), 
            learning_rate=params.get('learning_rate', 0.1), 
            max_depth=params.get('max_depth', None)
        )
        
        clf.fit(x_train, y_train)
        
        # 5. Save Model
        save_model(clf, 'models/model.pkl')

    except FileNotFoundError as e:
        logging.error(f"File Error: {e}")
    except ValueError as e:
        logging.error(f"Data Error: {e}")
    except KeyError as e:
        logging.error(f"Parameter Error: params.yaml mein zaroori keys nahi hain: {e}")
    except Exception as e:
        logging.error(f"Unexpected Training Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


