import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

log = logging.getLogger("evodoc.ens")

class MedicalENSRegistry:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.registry_file = self.data_dir / "medical_ens.json"
        self.counter = 0
        self.entities = {}
        self.load_registry()
    
    def load_registry(self):
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.counter = data.get('counter', 0)
                    self.entities = data.get('entities', {})
                log.info(f"ENS Registry loaded: {len(self.entities)} entities")
            else:
                self.initialize_registry()
        except Exception as e:
            log.warning(f"Registry load failed: {e}")
            self.initialize_registry()
    
    def save_registry(self):
        self.data_dir.mkdir(exist_ok=True)
        registry_data = {
            'counter': self.counter,
            'entities': self.entities,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def initialize_registry(self):
        self.counter = 0
        self.entities = {}
        self.save_registry()
    
    def register_disease(self, disease_name: str, icd10: str = None, snomed: str = None) -> str:
        ens_name = f"{disease_name.lower().replace(' ', '_')}.evodoc"
        
        if ens_name in self.entities:
            return ens_name
        
        self.counter += 1
        stable_code = f"0x{self.counter:06X}"
        
        self.entities[ens_name] = {
            'type': 'disease',
            'canonical_name': disease_name,
            'stable_code': stable_code,
            'numeric_id': self.counter,
            'icd10': icd10 or f"J{self.counter:02d}",
            'snomed': snomed or f"{6142000 + self.counter}",
            'created_at': datetime.now().isoformat()
        }
        
        self.save_registry()
        log.info(f"Registered disease: {ens_name} -> {stable_code}")
        return ens_name
    
    def register_symptom(self, symptom_name: str) -> str:
        ens_name = f"{symptom_name.lower().replace(' ', '_')}.evodoc"
        
        if ens_name in self.entities:
            return ens_name
        
        self.counter += 1
        stable_code = f"0x{self.counter:06X}"
        
        self.entities[ens_name] = {
            'type': 'symptom',
            'canonical_name': symptom_name,
            'stable_code': stable_code,
            'numeric_id': self.counter,
            'created_at': datetime.now().isoformat()
        }
        
        self.save_registry()
        log.info(f"Registered symptom: {ens_name} -> {stable_code}")
        return ens_name
    
    def get_disease_mapping(self) -> Dict[str, int]:
        disease_map = {}
        for ens_name, entity in self.entities.items():
            if entity['type'] == 'disease':
                disease_map[entity['canonical_name']] = entity['numeric_id']
        return disease_map
    
    def get_symptom_mapping(self) -> Dict[str, int]:
        symptom_map = {}
        for ens_name, entity in self.entities.items():
            if entity['type'] == 'symptom':
                symptom_map[entity['canonical_name']] = entity['numeric_id']
        return symptom_map
    
    def resolve_ens(self, ens_name: str) -> Optional[Dict]:
        return self.entities.get(ens_name)
    
    def display_registry_table(self):
        print("\n" + "="*100)
        print("MEDICAL ENS REGISTRY - VISUAL DEMONSTRATION")
        print("="*100)
        
        diseases = [(k, v) for k, v in self.entities.items() if v['type'] == 'disease']
        symptoms = [(k, v) for k, v in self.entities.items() if v['type'] == 'symptom']
        
        if diseases:
            print(f"\nDISEASES REGISTERED ({len(diseases)} total):")
            print("-" * 100)
            print(f"{'ENS Name':<35} {'Stable Code':<12} {'Canonical Name':<25} {'ICD-10':<8} {'SNOMED':<10}")
            print("-" * 100)
            for ens_name, entity in diseases:
                print(f"{ens_name:<35} {entity['stable_code']:<12} {entity['canonical_name']:<25} {entity['icd10']:<8} {entity['snomed']:<10}")
        
        if symptoms:
            print(f"\nSYMPTOMS REGISTERED ({len(symptoms)} total):")
            print("-" * 100)
            print(f"{'ENS Name':<35} {'Stable Code':<12} {'Canonical Name':<25}")
            print("-" * 100)
            for ens_name, entity in symptoms:
                print(f"{ens_name:<35} {entity['stable_code']:<12} {entity['canonical_name']:<25}")
        
        if not diseases and not symptoms:
            print("No entities registered yet. Start by splitting and training on datasets.")

class DatasetManager:
    def __init__(self):
        self.data_dir = Path("./data")
        self.splits_dir = self.data_dir / "splits"
        self.splits_dir.mkdir(exist_ok=True)
        self.available_datasets = []
    
    def discover_main_dataset(self):
        possible_paths = [
            "./data/raw/Training.csv",
            "./data/raw/Symptom2Disease.csv",
            "./data/raw/symptom_disease.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"Found main dataset: {path}")
                return path
        
        print("No main dataset found. Please ensure your CSV is in data/raw/")
        return None
    
    def split_dataset_manual(self, dataset_path: str):
        print(f"\nAnalyzing dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        target_col = None
        for col in ['prognosis', 'Disease', 'disease']:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col:
            print("No disease column found")
            return
        
        diseases = df[target_col].unique()
        print(f"Found {len(diseases)} unique diseases")
        print("Sample diseases:", diseases[:10])
        
        disease_groups = {
            'respiratory': [],
            'cardiovascular': [],
            'metabolic': [],
            'neurological': []
        }
        
        keywords = {
            'respiratory': ['pneumonia', 'bronchitis', 'asthma', 'cold', 'flu', 'cough'],
            'cardiovascular': ['hypertension', 'heart', 'cardiac', 'blood pressure'],
            'metabolic': ['diabetes', 'obesity', 'thyroid'],
            'neurological': ['migraine', 'headache', 'seizure', 'stroke']
        }
        
        for disease in diseases:
            disease_lower = disease.lower()
            assigned = False
            
            for group, group_keywords in keywords.items():
                if any(keyword in disease_lower for keyword in group_keywords):
                    disease_groups[group].append(disease)
                    assigned = True
                    break
            
            if not assigned:
                disease_groups['respiratory'].append(disease)
        
        for group_name, group_diseases in disease_groups.items():
            if group_diseases:
                group_data = df[df[target_col].isin(group_diseases)]
                file_path = self.splits_dir / f"{group_name}_data.csv"
                group_data.to_csv(file_path, index=False)
                
                print(f"Created {group_name} dataset: {len(group_data)} samples, {len(group_diseases)} diseases")
                self.available_datasets.append(group_name)
        
        print(f"\nDataset splitting complete!")
        print(f"Available datasets: {', '.join(self.available_datasets)}")
    
    def show_available_datasets(self):
        datasets = list(self.splits_dir.glob("*.csv"))
        self.available_datasets = [f.stem.replace('_data', '') for f in datasets]
        
        if not self.available_datasets:
            print("No split datasets found. Run dataset splitting first.")
            return
        
        print(f"\nAvailable datasets for training:")
        for i, dataset in enumerate(self.available_datasets, 1):
            file_path = self.splits_dir / f"{dataset}_data.csv"
            df = pd.read_csv(file_path)
            print(f"{i}. {dataset}: {len(df)} samples")

class StepByStepTrainer:
    def __init__(self):
        self.ens_registry = MedicalENSRegistry()
        self.dataset_manager = DatasetManager()
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        self.trained_models = {}
    
    def train_on_dataset(self, dataset_name: str):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib
        
        data_file = self.dataset_manager.splits_dir / f"{dataset_name}_data.csv"
        if not data_file.exists():
            print(f"Dataset {dataset_name} not found")
            return None
        
        print(f"\nStep-by-step training on {dataset_name} dataset:")
        print("="*60)
        
        dataset = pd.read_csv(data_file)
        print(f"1. Loaded dataset: {len(dataset)} samples")
        
        target_col = None
        for col in ['prognosis', 'Disease', 'disease']:
            if col in dataset.columns:
                target_col = col
                break
        
        diseases = dataset[target_col].unique()
        symptoms = [col for col in dataset.columns if col != target_col]
        
        print(f"2. Found {len(diseases)} diseases and {len(symptoms)} symptoms")
        
        print("3. Registering entities in ENS registry...")
        entities_before = len(self.ens_registry.entities)
        
        for disease in diseases:
            self.ens_registry.register_disease(disease)
        
        for symptom in symptoms:
            self.ens_registry.register_symptom(symptom)
        
        entities_after = len(self.ens_registry.entities)
        print(f"   Registered {entities_after - entities_before} new entities")
        
        print("4. Preparing training data with ENS mapping...")
        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]
        
        disease_mapping = self.ens_registry.get_disease_mapping()
        y_encoded = y.map(disease_mapping)
        
        print(f"5. Splitting data for training...")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        print("6. Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("7. Evaluating model performance...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Model accuracy: {accuracy:.4f}")
        
        print("8. Saving trained model...")
        model_file = self.models_dir / f"{dataset_name}_model.pkl"
        joblib.dump(model, model_file)
        
        self.trained_models[dataset_name] = {
            'model': model,
            'accuracy': accuracy,
            'diseases': len(diseases),
            'samples': len(dataset)
        }
        
        print(f"   Model saved: {model_file}")
        print(f"Training on {dataset_name} complete!")
        
        return model, accuracy
    
    def show_training_progress(self):
        print("\n" + "="*80)
        print("TRAINING PROGRESS OVERVIEW")
        print("="*80)
        
        if not self.trained_models:
            print("No models trained yet.")
            return
        
        print(f"{'Dataset':<15} {'Accuracy':<10} {'Diseases':<10} {'Samples':<10} {'Status':<10}")
        print("-" * 80)
        
        for dataset_name, info in self.trained_models.items():
            print(f"{dataset_name:<15} {info['accuracy']:<10.4f} {info['diseases']:<10} {info['samples']:<10} {'Trained':<10}")
        
        total_diseases = sum(info['diseases'] for info in self.trained_models.values())
        total_samples = sum(info['samples'] for info in self.trained_models.values())
        avg_accuracy = np.mean([info['accuracy'] for info in self.trained_models.values()])
        
        print("-" * 80)
        print(f"{'TOTAL':<15} {avg_accuracy:<10.4f} {total_diseases:<10} {total_samples:<10} {len(self.trained_models)} models")
    
    def combine_models_step_by_step(self):
        if len(self.trained_models) < 2:
            print("Need at least 2 trained models to combine")
            return
        
        print("\n" + "="*60)
        print("STEP-BY-STEP MODEL COMBINATION")
        print("="*60)
        
        print("1. Loading individual trained models...")
        import joblib
        
        loaded_models = {}
        for dataset_name in self.trained_models.keys():
            model_file = self.models_dir / f"{dataset_name}_model.pkl"
            if model_file.exists():
                loaded_models[dataset_name] = joblib.load(model_file)
                print(f"   Loaded {dataset_name} model")
        
        print(f"2. Creating combined model with {len(loaded_models)} individual models...")
        
        combined_model = CombinedENSModel(loaded_models, self.ens_registry)
        
        print("3. Saving combined model...")
        combined_file = self.models_dir / "combined_ens_model.pkl"
        joblib.dump(combined_model, combined_file)
        
        print(f"4. Combined model saved: {combined_file}")
        print("Model combination complete!")
        
        return combined_model

class CombinedENSModel:
    def __init__(self, models: Dict, ens_registry: MedicalENSRegistry):
        self.models = models
        self.ens_registry = ens_registry
        self.model_names = list(models.keys())
    
    def predict(self, X):
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict_proba(X)
                predictions.append(pred)
            except Exception as e:
                print(f"Error with {model_name} model: {e}")
        
        if not predictions:
            return np.array([])
        
        combined_pred = np.mean(predictions, axis=0)
        return np.argmax(combined_pred, axis=1)
    
    def predict_proba(self, X):
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict_proba(X)
                predictions.append(pred)
            except Exception as e:
                print(f"Error with {model_name} model: {e}")
        
        if not predictions:
            return np.array([])
        
        return np.mean(predictions, axis=0)