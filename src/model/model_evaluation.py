import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os
import sys
import logging
logger = logging.getLogger()
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

farmatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(farmatter)

logger.addHandler(console_handler)
def main():
    try:
        # 1. Load Data check
        test_path = './data/features/test_bow.csv'
        model_path = 'models/model.pkl'

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data nahi mila: {test_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model nahi mila: {model_path}")

        test_data = pd.read_csv(test_path)
        
        if test_data.empty:
            raise ValueError("Test data khali hai!")

        x_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        # 2. Load Model
        with open(model_path, 'rb') as file:
            clf = pickle.load(file)

        # 3. Predictions
        logging.info("Predictions generate ho rahi hain...")
        y_pred = clf.predict(x_test)

        # 4. Metrics Calculation
        logging.info("Classification Report:")
        cr = classification_report(y_test, y_pred, output_dict=True) # Dict for JSON
        cr_text = classification_report(y_test, y_pred) # For printing
        print(cr_text)

        cm = confusion_matrix(y_test, y_pred)

        # 5. Visualizations
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.savefig('confusion_matrix.png') # Plot save karna behtar hai pipeline mein
            plt.show()
        except Exception as e:
            logging.error(f"Plotting Error: {e} (Plotting skip ho rahi hai)")

        # 6. Save Metrics to JSON
        metrics_dict = {
            "classification_report": cr,
            "confusion_matrix": cm.tolist() # Numpy array ko list mein convert karna zaroori hai
        }

        with open('reports/metrics.json', 'w') as file:
            json.dump(metrics_dict, file, indent=4)
        
        logging.info("Metrics successfully save ho gaye: metrics.json")

    except FileNotFoundError as e:
        logging.info(f"File Error: {e}")
    except ValueError as e:
        logging.info(f"Data Error: {e}")
    except Exception as e:
        logging.info(f"Unexpected Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
