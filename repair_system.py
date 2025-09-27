import joblib
import numpy as np
from pathlib import Path

def repair_system():
    print("=== Repairing EvoDoc System ===")
    
    BASE_DIR = Path(".")
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    
    # Ensure directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check and fix best model
    best_model_path = MODELS_DIR / "best_model.pkl"
    logreg_model_path = MODELS_DIR / "logreg_model.pkl"
    
    if not best_model_path.exists() and logreg_model_path.exists():
        print("Copying Logistic Regression as best model...")
        logreg_model = joblib.load(logreg_model_path)
        joblib.dump(logreg_model, best_model_path)
        print("✓ Best model created")
    
    # Check label encoder
    filtered_encoder_path = MODELS_DIR / "filtered_label_encoder.pkl"
    regular_encoder_path = PROCESSED_DIR / "label_encoder.pkl"
    
    if not filtered_encoder_path.exists() and regular_encoder_path.exists():
        print("Copying label encoder...")
        encoder = joblib.load(regular_encoder_path)
        joblib.dump(encoder, filtered_encoder_path)
        print("✓ Label encoder copied")
    
    # Check symptom columns
    x_columns_path = PROCESSED_DIR / "X_columns.npy"
    symptom_columns_path = PROCESSED_DIR / "symptom_columns.npy"
    
    if not x_columns_path.exists() and symptom_columns_path.exists():
        print("Copying symptom columns...")
        columns = np.load(symptom_columns_path)
        np.save(x_columns_path, columns)
        print("✓ Symptom columns copied")
    
    print("✓ System repair complete")

if __name__ == "__main__":
    repair_system()
