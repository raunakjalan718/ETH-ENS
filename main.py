import sys
import logging
from pathlib import Path
from data_handler import DataHandler
from model_trainer import ModelTrainer
from recommender import TreatmentRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("evodoc.main")

BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def main():
    print("===== EvoDoc Medical Treatment Recommendation System =====")
    print("AI-powered medical recommendations using real medical datasets")
    print("Completely dynamic - adapts to any medical dataset structure")
    print()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Available modes:")
        print("1. setup       - Auto-discover and process medical datasets")
        print("2. train-all   - Train all ML models (XGBoost, Random Forest, Logistic Regression)")
        print("3. train-xgb   - Train only XGBoost model")
        print("4. train-rf    - Train only Random Forest model")
        print("5. train-lr    - Train only Logistic Regression model")
        print("6. evaluate    - Evaluate model performance")
        print("7. recommend   - Interactive treatment recommendations")
        print("8. all         - Run complete pipeline")
        print()
        mode = input("Select mode (1-8 or name): ").strip()
        
        mode_map = {
            "1": "setup",
            "2": "train-all",
            "3": "train-xgb", 
            "4": "train-rf",
            "5": "train-lr",
            "6": "evaluate",
            "7": "recommend",
            "8": "all"
        }
        mode = mode_map.get(mode, mode)
    
    try:
        if mode == "setup":
            setup_data()
        elif mode == "train-all":
            train_all_models()
        elif mode == "train-xgb":
            train_single_model("xgb")
        elif mode == "train-rf":
            train_single_model("rf")
        elif mode == "train-lr":
            train_single_model("lr")
        elif mode == "evaluate":
            evaluate_models()
        elif mode == "recommend":
            run_recommendations()
        elif mode == "all":
            run_complete_pipeline()
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)
            
    except Exception as e:
        log.error(f"Error in {mode} mode: {e}")
        sys.exit(1)

def setup_data():
    print("\n=== Step 1: Dynamic Data Discovery & Setup ===")
    handler = DataHandler()
    success = handler.process_all_data()
    
    if success:
        print("✓ Data setup complete!")
        print("  - Auto-discovered and processed medical datasets")
        print("  - All data structures adapted to your specific dataset")
    else:
        print("✗ Data setup failed!")
        print("Please ensure you have CSV files in the data/raw directory")
        sys.exit(1)

def train_all_models():
    print("\n=== Training All Models ===")
    
    if not Path("./data/processed/X.npy").exists():
        print("Data not found. Running setup first...")
        setup_data()
    
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    if results:
        print("\n=== Training Results Summary ===")
        for name, result in results.items():
            print(f"{name.upper()}: Accuracy = {result['test_accuracy']:.4f}, "
                  f"F1 = {result['test_f1']:.4f}, Top-3 = {result['top3_accuracy']:.4f}, "
                  f"Time = {result['training_time']:.2f}s")
        
        print("✓ All models training complete!")
        print(f"  - Best model: {trainer.best_model_name.upper()} with {trainer.best_score:.4f} accuracy")
    else:
        print("✗ Model training failed!")
        sys.exit(1)

def train_single_model(model_type):
    model_names = {"xgb": "XGBoost", "rf": "Random Forest", "lr": "Logistic Regression"}
    print(f"\n=== Training {model_names[model_type]} Model ===")
    
    if not Path("./data/processed/X.npy").exists():
        print("Data not found. Running setup first...")
        setup_data()
    
    trainer = ModelTrainer()
    
    if not trainer.prepare_data():
        print("✗ Data preparation failed!")
        sys.exit(1)
    
    try:
        if model_type == "xgb":
            result = trainer.train_xgboost()
        elif model_type == "rf":
            result = trainer.train_random_forest()
        elif model_type == "lr":
            result = trainer.train_logistic_regression()
        
        print(f"\n=== {model_names[model_type]} Training Results ===")
        print(f"Cross-validation: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"Test F1-Score: {result['test_f1']:.4f}")
        print(f"Top-3 Accuracy: {result['top3_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {result['top5_accuracy']:.4f}")
        print(f"Training Time: {result['training_time']:.2f} seconds")
        
        print(f"✓ {model_names[model_type]} training complete!")
        print(f"  - Model saved as {model_type}_model.pkl")
        print(f"  - Results saved as {model_type}_results.json")
        
    except Exception as e:
        print(f"✗ {model_names[model_type]} training failed: {e}")
        sys.exit(1)

def evaluate_models():
    print("\n=== Model Evaluation ===")
    
    available_models = []
    model_files = {
        "best": MODELS_DIR / "best_model.pkl",
        "xgb": MODELS_DIR / "xgb_model.pkl",
        "rf": MODELS_DIR / "rf_model.pkl",
        "logreg": MODELS_DIR / "logreg_model.pkl"
    }
    
    for name, path in model_files.items():
        if path.exists():
            available_models.append(name)
    
    if not available_models:
        print("No trained models found. Running training first...")
        train_all_models()
        available_models = ["best"]
    
    trainer = ModelTrainer()
    
    print(f"Available models: {', '.join(available_models)}")
    
    for model_name in available_models:
        try:
            metrics = trainer.evaluate_model(model_name)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    print("✓ Model evaluation complete!")

def run_recommendations():
    print("\n=== Treatment Recommendations ===")
    
    if not Path("./models/best_model.pkl").exists():
        print("Models not found. Running training first...")
        train_all_models()
    
    recommender = TreatmentRecommender()
    recommender.run_interactive_session()

def run_complete_pipeline():
    print("\n=== Running Complete EvoDoc Pipeline ===")
    
    print("\n[1/4] Auto-discovering and processing medical data...")
    setup_data()
    
    print("\n[2/4] Training all ML models...")
    train_all_models()
    
    print("\n[3/4] Evaluating model performance...")
    evaluate_models()
    
    print("\n[4/4] System ready for intelligent recommendations!")
    print("✓ Complete pipeline finished successfully!")
    print("\nYour EvoDoc system is now ready!")
    print("\nRun 'python main.py recommend' to start making recommendations")

if __name__ == "__main__":
    main()
