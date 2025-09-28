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
log = logging.getLogger("evodoc.ens")

def train_first_group():
    print("="*80)
    print("STEP 3: ENS-POWERED INCREMENTAL TRAINING")
    print("="*80)
    print("Training first medical AI model with ENS blockchain naming")
    print("No traditional database IDs - only ENS .evodoc addresses!")
    print("="*80)
    
    data_dir = Path("./data")
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "ens.json", 'r') as f:
        registry = json.load(f)
    
    counter = registry['counter']
    entities = registry['entities']
    
    data_file = data_dir / "splits" / "respiratory.csv"
    if not data_file.exists():
        print("Respiratory data not found")
        return
    
    dataset = pd.read_csv(data_file)
    print(f"Loaded respiratory dataset: {len(dataset)} samples")
    
    target_col = 'diseases'
    diseases = dataset[target_col].unique()
    symptoms = [col for col in dataset.columns if col != target_col]
    
    print(f"Diseases to register: {len(diseases)}")
    print(f"Symptoms to register: {len(symptoms)}")
    
    print("\nREGISTERING MEDICAL ENTITIES IN ENS BLOCKCHAIN:")
    print("-" * 80)
    
    disease_ens_names = []
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
                'icd10': f"J{counter:02d}",
                'blockchain_address': stable_hex,
                'ens_domain': ens_name,
                'category': 'respiratory'
            }
            disease_ens_names.append(ens_name)
            print(f"NEW: {ens_name} -> {stable_hex} (ICD-10: J{counter:02d})")
    
    symptom_ens_names = []
    for symptom in symptoms:
        ens_name = f"{symptom.lower().replace(' ', '_')}.evodoc"
        if ens_name not in entities:
            counter += 1
            stable_hex = f"0x{counter:06X}"
            entities[ens_name] = {
                'type': 'symptom',
                'name': symptom,
                'code': stable_hex,
                'id': counter,
                'blockchain_address': stable_hex,
                'ens_domain': ens_name
            }
            symptom_ens_names.append(ens_name)
    
    registry['counter'] = counter
    registry['entities'] = entities
    
    with open(data_dir / "ens.json", 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\nENS REGISTRATION COMPLETE:")
    print(f"   Total entities in blockchain: {counter}")
    print(f"   New diseases registered: {len(disease_ens_names)}")
    print(f"   Total symptoms registered: {len(symptom_ens_names)}")
    
    print(f"\nTRAINING AI MODEL WITH ENS STABLE CODES:")
    print("-" * 80)
    
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    
    ens_disease_map = {}
    for ens_name, entity in entities.items():
        if entity['type'] == 'disease':
            ens_disease_map[entity['name']] = entity['id']
    
    y_encoded = y.map(ens_disease_map)
    
    print(f"Using ENS blockchain IDs instead of traditional ML labels")
    print(f"   Example: '{diseases[0]}' -> ENS ID {ens_disease_map[diseases[0]]}")
    
    class_counts = Counter(y_encoded)
    rare_classes = [cls for cls, count in class_counts.items() if count < 5]
    
    if rare_classes:
        log.info(f"Filtering {len(rare_classes)} rare ENS entities with < 5 samples")
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
    
    print(f"\nMODEL RESULTS:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Training Time: {training_time:.2f}s")
    
    joblib.dump(model, models_dir / "model1.pkl")

    print(f"\nModel saved as model1.pkl with ENS integration")

if __name__ == "__main__":
    train_first_group()