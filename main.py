import sys
import logging
from pathlib import Path
from data_handler import DataHandler
from model_trainer import ModelTrainer
from recommender import TreatmentRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("evodoc.main")

def main():
    print("===== EvoDoc Medical Treatment Recommendation System =====")
    print("AI-powered medical recommendations using real medical datasets")
    print("Completely dynamic - adapts to any medical dataset structure")
    print()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Available modes:")
        print("1. setup    - Auto-discover and process medical datasets")
        print("2. train    - Train ML models on discovered data")
        print("3. evaluate - Evaluate model performance")
        print("4. recommend - Interactive treatment recommendations")
        print("5. all      - Run complete pipeline")
        print()
        mode = input("Select mode (1-5 or name): ").strip()
        
        mode_map = {
            "1": "setup",
            "2": "train", 
            "3": "evaluate",
            "4": "recommend",
            "5": "all"
        }
        mode = mode_map.get(mode, mode)
    
    try:
        if mode == "setup":
            setup_data()
        elif mode == "train":
            train_models()
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

def train_models():
    print("\n=== Step 2: Adaptive Model Training ===")
    
    if not Path("./data/processed/X.npy").exists():
        print("Data not found. Running setup first...")
        setup_data()
    
    trainer = ModelTrainer()
    results = trainer.train_models()
    
    if results:
        print("\n=== Training Results ===")
        for name, result in results.items():
            print(f"{name.upper()}: Accuracy = {result['test_accuracy']:.3f}, "
                  f"F1 = {result['test_f1']:.3f}, Top-3 = {result['top3_accuracy']:.3f}, "
                  f"Top-5 = {result['top5_accuracy']:.3f}")
        
        print("✓ Model training complete!")
        print(f"  - Best model: {trainer.best_model_name.upper()} with {trainer.best_score:.3f} accuracy")
        print("  - Models automatically adapted to your dataset size and complexity")
    else:
        print("✗ Model training failed!")
        sys.exit(1)

def evaluate_models():
    print("\n=== Step 3: Comprehensive Model Evaluation ===")
    
    if not Path("./models/best_model.pkl").exists():
        print("Models not found. Running training first...")
        train_models()
    
    trainer = ModelTrainer()
    metrics = trainer.evaluate_model()
    
    print("\n=== Detailed Evaluation Results ===")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top3']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top5']:.4f}")
    
    print("✓ Model evaluation complete!")
    return metrics

def run_recommendations():
    print("\n=== Step 4: Intelligent Treatment Recommendations ===")
    
    if not Path("./models/best_model.pkl").exists():
        print("Models not found. Running training first...")
        train_models()
    
    recommender = TreatmentRecommender()
    recommender.run_interactive_session()

def run_complete_pipeline():
    print("\n=== Running Complete EvoDoc Pipeline ===")
    
    print("\n[1/4] Auto-discovering and processing medical data...")
    setup_data()
    
    print("\n[2/4] Training adaptive ML models...")
    train_models()
    
    print("\n[3/4] Evaluating model performance...")
    evaluate_models()
    
    print("\n[4/4] System ready for intelligent recommendations!")
    print("✓ Complete pipeline finished successfully!")
    print("\nYour EvoDoc system is now ready with:")
    print("  - Dynamically processed medical data from your datasets")
    print("  - ML models automatically adapted to your data characteristics")
    print("  - AI-powered medical insights via KimiAI integration")
    print("  - Zero hardcoded values - completely adaptive system")
    print("\nRun 'python main.py recommend' to start making recommendations")

if __name__ == "__main__":
    main()
