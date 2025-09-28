import sys
import logging
from pathlib import Path
from data_handler import DataHandler
from model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("evodoc.main")

def main():
    print("===== EvoDoc ENS Demo =====")
    print("Manual Incremental Training for Hackathon Demo")
    print("No Automatic Processes - Full Manual Control")
    print()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Available modes:")
        print("1. setup       - Process your medical dataset")
        print("2. demo        - ENS demo (MAIN FEATURE)")
        print("3. quick-train - Quick model training for baseline")
        print()
        mode = input("Select mode (1-3 or name): ").strip()
        
        mode_map = {"1": "setup", "2": "demo", "3": "quick-train"}
        mode = mode_map.get(mode, mode)
    
    try:
        if mode == "setup":
            setup_data()
        elif mode == "demo":
            run_step_by_step_demo()
        elif mode == "quick-train":
            quick_train()
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)
            
    except Exception as e:
        log.error(f"Error in {mode} mode: {e}")
        sys.exit(1)

def setup_data():
    print("\n=== Data Setup ===")
    handler = DataHandler()
    success = handler.process_all_data()
    
    if success:
        print("Data setup complete!")
        print("You can now run: python main.py demo")
    else:
        print("Data setup failed!")

def run_step_by_step_demo():
    print("\n" + "="*70)
    print("ENS DEMO - HACKATHON PRESENTATION")
    print("="*70)
    print("Full manual control for demonstration purposes")
    print("Each step can be executed individually")
    print("="*70)
    
    try:
        from hackathon_demo import StepByStepDemo
        demo = StepByStepDemo()
        demo.run_demo()
    except Exception as e:
        print(f"Demo error: {e}")

def quick_train():
    print("\n=== Quick Training for Baseline ===")
    
    if not Path("./data/processed/X.npy").exists():
        print("Data not found. Running setup first...")
        setup_data()
    
    trainer = ModelTrainer()
    
    if not trainer.prepare_data():
        print("Data preparation failed!")
        sys.exit(1)
    
    try:
        result = trainer.train_logistic_regression()
        print(f"Quick training complete! Accuracy: {result['test_accuracy']:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
