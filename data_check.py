import pandas as pd
from pathlib import Path

def check_dataset():
    print("="*60)
    print("DATASET CHECKER")
    print("="*60)
    
    possible_paths = [
        "./data/raw/Training.csv",
        "./data/raw/Symptom2Disease.csv",
        "./data/raw/symptom_disease.csv"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"Found: {path}")
            df = pd.read_csv(path)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First few rows:")
            print(df.head())
            break
    else:
        print("No dataset found")

if __name__ == "__main__":
    check_dataset()
