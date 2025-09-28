import pandas as pd
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import joblib
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("evodoc.trainer")

def train_second_group():
    print("="*60)
    print("STEP 5: ENS-POWERED INCREMENTAL TRAINING - SECOND MODEL")
    print("="*60)
    print("Demonstrating ENS benefit: No retraining of existing models needed!")
    
    available_datasets = ['heart', 'diabetes', 'brain', 'other']
    
    print("Available datasets from original split:")
    for i, dataset in enumerate(available_datasets, 1):
        dataset_file = Path(f"./data/splits/{dataset}.csv")
        if dataset_file.exists():
            df = pd.read_csv(dataset_file)
            diseases = df['diseases'].unique()
            print(f"{i}. {dataset}: {len(df)} samples, {len(diseases)} diseases")
            print(f"   Sample diseases: {diseases[:3]}")
    
    dataset_name = input(f"\nWhich dataset to train on? ({'/'.join(available_datasets)}): ").strip()
    
    if dataset_name not in available_datasets:
        print(f"Invalid choice. Using 'heart' as default.")
        dataset_name = 'heart'
    
    data_dir = Path("./data")
    models_dir = Path("./models")
    
    with open(data_dir / "ens.json", 'r') as f:
        registry = json.load(f)
    
    counter = registry['counter']
    entities = registry['entities']
    
    data_file = data_dir / "splits" / f"{dataset_name}.csv"
    if not data_file.exists():
        print(f"{dataset_name} data not found")
        return
    
    dataset = pd.read_csv(data_file)
    print(f"\nLoaded {dataset_name} dataset: {len(dataset)} samples")
    
    target_col = 'diseases'
    diseases = dataset[target_col].unique()
    print(f"{dataset_name.title()} diseases: {diseases}")
    
    print(f"\nADDING NEW {dataset_name.upper()} DISEASES TO ENS BLOCKCHAIN:")
    print("-" * 80)
    new_count = 0
    
    icd_prefixes = {
        'heart': 'I',
        'diabetes': 'E', 
        'brain': 'G',
        'other': 'Z'
    }
    
    icd_prefix = icd_prefixes.get(dataset_name, 'X')
    
    for disease in diseases:
        ens_name = f"{disease.lower().replace(' ', '_')}.evodoc"
        if ens_name not in entities:
            counter += 1
            stable_hex = f"0x{counter:06X}"
            entities[ens_name] = {
                'type': 'disease',
                'name': disease,
                'code': stable_hex,
                'id': counter,
                'icd10': f"{icd_prefix}{counter:02d}",
                'blockchain_address': stable_hex,
                'ens_domain': ens_name,
                'category': dataset_name
            }
            print(f"NEW: {ens_name} -> {stable_hex} (ICD-10: {icd_prefix}{counter:02d})")
            new_count += 1
        else:
            existing_entity = entities[ens_name]
            print(f"EXISTS: {ens_name} -> {existing_entity['code']}")
    
    print(f"\nENS INCREMENTAL UPDATE COMPLETE:")
    print(f"   Added {new_count} new {dataset_name} diseases to blockchain")
    print(f"   Total ENS entities now: {counter}")
    print(f"   Existing respiratory model unchanged!")
    
    registry['counter'] = counter
    registry['entities'] = entities
    
    with open(data_dir / "ens.json", 'w') as f:
        json.dump(registry, f, indent=2)
    
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    
    disease_map = {}
    for ens_name, entity in entities.items():
        if entity['type'] == 'disease':
            disease_map[entity['name']] = entity['id']
    
    y_encoded = y.map(disease_map)
    
    class_counts = Counter(y_encoded)
    rare_classes = [cls for cls, count in class_counts.items() if count < 5]
    
    if rare_classes:
        log.info(f"Removing {len(rare_classes)} rare classes with < 5 samples")
        mask = ~y_encoded.isin(rare_classes)
        X_filtered = X[mask]
        y_filtered = y_encoded[mask]
        
        unique_labels = y_filtered.unique()
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        y_remapped = y_filtered.map(label_mapping)
        
        log.info(f"After filtering: X={X_filtered.shape}, y={len(y_remapped)}, classes={len(unique_labels)}")
    else:
        X_filtered = X
        y_remapped = y_encoded
        unique_labels = y_encoded.unique()
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_remapped, test_size=0.2, random_state=42, stratify=y_remapped
        )
    except ValueError as e:
        log.warning(f"Stratified split failed: {e}")
        log.info("Using random split instead")
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_remapped, test_size=0.2, random_state=42
        )
    
    log.info("Training Logistic Regression model...")
    
    n_classes = len(unique_labels)
    n_samples = X_train.shape[0]
    
    params = {
        "max_iter": 200,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": 1,
        "solver": "saga",
        "C": 1.0,
        "tol": 1e-3
    }
    
    if n_samples > 100000:
        params.update({
            "max_iter": 100,
            "tol": 1e-2,
            "C": 0.1
        })
        log.info("Using optimized settings for large dataset")
    
    model = LogisticRegression(**params)
    
    start_time = time.time()
    
    log.info("Starting model training...")
    model.fit(X_train, y_train)
    log.info("Model training completed")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    training_time = time.time() - start_time
    
    log.info(f"Logistic Regression: Test Acc={accuracy:.4f}, F1={f1:.4f}, Time={training_time:.2f}s")
    
    print(f"\n{dataset_name.title()} model accuracy: {accuracy:.4f}")
    
    joblib.dump(model, models_dir / "model2.pkl")
    
    print("Model saved as model2.pkl")

if __name__ == "__main__":
    train_second_group()