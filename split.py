import pandas as pd
import numpy as np
from pathlib import Path
import json

def split_dataset():
    print("="*60)
    print("STEP 1: SPLIT DATASET")
    print("="*60)

    data_dir = Path("./data")
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    possible_paths = [
        "./data/raw/Training.csv",
        "./data/raw/Symptom2Disease.csv",
        "./data/raw/symptom_disease.csv"
    ]

    dataset_path = None
    for path in possible_paths:
        if Path(path).exists():
            dataset_path = path
            break

    if not dataset_path:
        print("No dataset found in data/raw/")
        return

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    target_col = None
    possible_targets = ['prognosis', 'Disease', 'disease', 'label', 'target']

    for col in df.columns:
        col_lower = col.lower()
        if any(target in col_lower for target in possible_targets):
            target_col = col
            break

    if target_col is None:
        print("Available columns:", df.columns.tolist())
        print("Could not find disease column automatically.")
        print("Please specify which column contains the diseases:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        
        try:
            choice = int(input("Enter column number: "))
            target_col = df.columns[choice]
        except:
            print("Invalid choice. Using last column as target.")
            target_col = df.columns[-1]

    print(f"Using target column: {target_col}")

    diseases = df[target_col].unique()
    print(f"Total diseases: {len(diseases)}")
    print(f"Sample diseases: {diseases[:10]}")

    disease_groups = {
        'respiratory': ['pneumonia', 'bronchitis', 'asthma', 'cold', 'flu', 'cough'],
        'heart': ['hypertension', 'heart', 'cardiac', 'blood pressure'],
        'diabetes': ['diabetes', 'obesity', 'thyroid'],
        'brain': ['migraine', 'headache', 'seizure', 'stroke']
    }

    print("\nSplitting into groups...")

    for group_name, keywords in disease_groups.items():
        group_diseases = []
        for disease in diseases:
            if any(keyword in disease.lower() for keyword in keywords):
                group_diseases.append(disease)
        
        if group_diseases:
            group_data = df[df[target_col].isin(group_diseases)]
            file_path = splits_dir / f"{group_name}.csv"
            group_data.to_csv(file_path, index=False)
            print(f"Created {group_name}: {len(group_data)} samples, {len(group_diseases)} diseases")
        else:
            print(f"No diseases found for {group_name} group")

    all_grouped_diseases = []
    for keywords in disease_groups.values():
        for disease in diseases:
            if any(keyword in disease.lower() for keyword in keywords):
                all_grouped_diseases.append(disease)

    remaining_diseases = [d for d in diseases if d not in all_grouped_diseases]
    if remaining_diseases:
        remaining_data = df[df[target_col].isin(remaining_diseases)]
        file_path = splits_dir / "other.csv"
        remaining_data.to_csv(file_path, index=False)
        print(f"Created other: {len(remaining_data)} samples, {len(remaining_diseases)} diseases")

    print("Dataset splitting complete!")

    split_info = {
        'original_dataset': dataset_path,
        'target_column': target_col,
        'total_samples': len(df),
        'total_diseases': len(diseases),
        'groups_created': list(disease_groups.keys()) + (['other'] if remaining_diseases else [])
    }

    with open(splits_dir / "info.json", 'w') as f:
        json.dump(split_info, f, indent=2)

if __name__ == "__main__":
    split_dataset()