# Create debug_recommender.py
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("debug")

BASE_DIR = Path(".")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

def debug_system():
    print("=== Debugging EvoDoc System ===\n")
    
    # Check if directories exist
    print("1. Checking directories:")
    print(f"   PROCESSED_DIR exists: {PROCESSED_DIR.exists()}")
    print(f"   MODELS_DIR exists: {MODELS_DIR.exists()}")
    
    # Check processed data files
    print("\n2. Checking processed data files:")
    required_files = ["X_columns.npy", "best_model.pkl", "filtered_label_encoder.pkl"]
    
    for file in required_files:
        if file.endswith('.npy'):
            file_path = PROCESSED_DIR / file
        else:
            file_path = MODELS_DIR / file
            
        exists = file_path.exists()
        print(f"   {file}: {'✓' if exists else '✗'}")
        
        if exists:
            try:
                if file.endswith('.npy'):
                    data = np.load(file_path)
                    print(f"      Shape/Length: {len(data) if hasattr(data, '__len__') else 'scalar'}")
                elif file.endswith('.pkl'):
                    data = joblib.load(file_path)
                    print(f"      Type: {type(data).__name__}")
            except Exception as e:
                print(f"      Error loading: {e}")
    
    # Check alternative files
    print("\n3. Checking alternative files:")
    alt_files = ["label_encoder.pkl", "logreg_model.pkl"]
    
    for file in alt_files:
        file_path = MODELS_DIR / file if file.endswith('_model.pkl') else PROCESSED_DIR / file
        exists = file_path.exists()
        print(f"   {file}: {'✓' if exists else '✗'}")

if __name__ == "__main__":
    debug_system()
