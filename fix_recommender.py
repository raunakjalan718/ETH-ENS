import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(".")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

def fix_label_encoder():
    print("Fixing label encoder...")
    
    # Load the original label encoder
    original_encoder = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
    
    # Save it as filtered encoder (they're the same in this case)
    joblib.dump(original_encoder, MODELS_DIR / "filtered_label_encoder.pkl")
    
    print("✓ Label encoder fixed")

if __name__ == "__main__":
    fix_label_encoder()

