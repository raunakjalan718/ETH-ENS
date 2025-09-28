import numpy as np
import joblib
from pathlib import Path

class CombinedModel:
    def __init__(self, models):
        self.models = models
    
    def predict_proba(self, X):
        predictions = []
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            predictions.append(pred)
            print(f"Got prediction from {name} model")
        return np.mean(predictions, axis=0)

def combine_models():
    print("="*60)
    print("STEP 6: COMBINE MODELS")
    print("="*60)
    
    models_dir = Path("./models")
    
    models = {}
    
    if (models_dir / "model1.pkl").exists():
        models['respiratory'] = joblib.load(models_dir / "model1.pkl")
        print("Loaded model1 (respiratory)")
    
    if (models_dir / "model2.pkl").exists():
        models['heart'] = joblib.load(models_dir / "model2.pkl")
        print("Loaded model2 (heart)")
    
    if len(models) < 2:
        print("Need at least 2 models to combine")
        return
    
    combined = CombinedModel(models)
    
    joblib.dump(combined, models_dir / "combined.pkl")
    print(f"Combined {len(models)} models")
    print("Saved as combined.pkl")

if __name__ == "__main__":
    combine_models()
