import joblib
import json
from pathlib import Path

MODELS_DIR = Path("./models")

def update_primary_model():
    print("=== Updating Primary Model to Logistic Regression ===")
    
    # Load the logistic regression model
    logreg_model = joblib.load(MODELS_DIR / "logreg_model.pkl")
    
    # Save it as the best model
    joblib.dump(logreg_model, MODELS_DIR / "best_model.pkl")
    
    # Load logistic regression results
    with open(MODELS_DIR / "logreg_results.json", "r") as f:
        logreg_results = json.load(f)
    
    # Update training summary
    summary = {
        "primary_model": "logreg",
        "primary_model_name": "Logistic Regression",
        "primary_accuracy": logreg_results["test_accuracy"],
        "primary_f1": logreg_results["test_f1"],
        "primary_top3": logreg_results["top3_accuracy"],
        "primary_top5": logreg_results["top5_accuracy"],
        "model_details": {
            "algorithm": "Logistic Regression",
            "solver": "saga",
            "regularization": "L2",
            "classes": 721,
            "features": 377
        },
        "performance_summary": {
            "excellent_top5_accuracy": "97.52% - Outstanding for medical diagnosis",
            "strong_accuracy": "85.48% - Very good for 721-class problem",
            "balanced_performance": "85.06% F1-score shows good precision-recall balance"
        }
    }
    
    with open(MODELS_DIR / "primary_model_info.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Logistic Regression set as primary model")
    print(f"✓ Accuracy: {logreg_results['test_accuracy']:.4f}")
    print(f"✓ Top-5 Accuracy: {logreg_results['top5_accuracy']:.4f}")
    print("✓ Model configuration saved")

if __name__ == "__main__":
    update_primary_model()
