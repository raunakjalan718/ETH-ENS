import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("evodoc.trainer")

BASE_DIR = Path(".")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.label_encoder = None
        
    def load_data(self):
        try:
            X = np.load(PROCESSED_DIR / "X.npy")
            y = np.load(PROCESSED_DIR / "y.npy")
            symptom_columns = np.load(PROCESSED_DIR / "X_columns.npy")
            label_encoder = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
            
            log.info(f"Loaded data: X={X.shape}, y={y.shape}")
            log.info(f"Classes: {len(label_encoder.classes_)}, Features: {len(symptom_columns)}")
            
            return X, y, symptom_columns, label_encoder
        except FileNotFoundError as e:
            log.error(f"Data files not found: {e}")
            log.error("Please run data processing first")
            return None, None, None, None
    
    def filter_rare_classes(self, X, y, min_samples=2):
        """Remove classes with fewer than min_samples"""
        class_counts = Counter(y)
        
        rare_classes = [cls for cls, count in class_counts.items() if count < min_samples]
        
        if rare_classes:
            log.info(f"Removing {len(rare_classes)} rare classes with < {min_samples} samples")
            
            mask = ~np.isin(y, rare_classes)
            X_filtered = X[mask]
            y_filtered = y[mask]
            
            unique_labels = np.unique(y_filtered)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            y_remapped = np.array([label_mapping[label] for label in y_filtered])
            
            log.info(f"After filtering: X={X_filtered.shape}, y={y_remapped.shape}, classes={len(unique_labels)}")
            
            return X_filtered, y_remapped, unique_labels
        
        return X, y, np.unique(y)
    
    def get_model_configs(self, n_classes, n_features):
        base_configs = {
            "xgb": {
                "model": XGBClassifier,
                "params": {
                    "n_estimators": min(100, max(50, n_features // 5)),
                    "max_depth": min(6, max(3, int(np.log2(n_features)))),
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "eval_metric": 'mlogloss',
                    "use_label_encoder": False,
                    "verbosity": 0
                }
            },
            "rf": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": min(100, max(50, n_features // 10)),
                    "max_depth": min(10, max(5, int(np.sqrt(n_features)))),
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": 0
                }
            },
            "logreg": {
                "model": LogisticRegression,
                "params": {
                    "max_iter": min(1000, max(200, n_classes * 2)),
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": 0
                }
            }
        }
        
        if n_classes > 100:
            base_configs["logreg"]["params"]["solver"] = "saga"
            base_configs["logreg"]["params"]["C"] = 0.1
        
        return base_configs
    
    def train_models(self):
        X, y, symptom_columns, label_encoder = self.load_data()
        
        if X is None:
            return None
            
        self.label_encoder = label_encoder
        
        X_filtered, y_filtered, unique_labels = self.filter_rare_classes(X, y, min_samples=2)
        
        n_classes = len(unique_labels)
        n_features = X_filtered.shape[1]
        
        log.info(f"Training with {n_classes} classes and {n_features} features")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
        except ValueError as e:
            log.warning(f"Stratified split failed: {e}")
            log.info("Using random split instead")
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42
            )
        
        models_config = self.get_model_configs(n_classes, n_features)
        results = {}
        
        for name, config in models_config.items():
            log.info(f"Training {name.upper()} model...")
            
            try:
                model = config["model"](**config["params"])
                
                try:
                    cv_folds = min(3, len(np.unique(y_train)))
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=cv_folds, scoring='accuracy'
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as cv_error:
                    log.warning(f"Cross-validation failed for {name}: {cv_error}")
                    cv_mean, cv_std = 0.0, 0.0
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                top3_acc = self.top_k_accuracy(y_test, y_proba, k=min(3, n_classes))
                top5_acc = self.top_k_accuracy(y_test, y_proba, k=min(5, n_classes))
                
                results[name] = {
                    'model': model,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'test_accuracy': accuracy,
                    'test_f1': f1,
                    'top3_accuracy': top3_acc,
                    'top5_accuracy': top5_acc
                }
                
                log.info(f"{name.upper()}: CV={cv_mean:.4f}±{cv_std:.4f}, "
                        f"Test Acc={accuracy:.4f}, F1={f1:.4f}, Top3={top3_acc:.4f}, Top5={top5_acc:.4f}")
                
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                log.error(f"Error training {name}: {e}")
                continue
        
        if not results:
            log.error("No models were successfully trained")
            return None
        
        self.models = results
        
        for name, result in results.items():
            model_path = MODELS_DIR / f"{name}_model.pkl"
            joblib.dump(result['model'], model_path)
            log.info(f"Saved {name} model to {model_path}")
        
        joblib.dump(self.best_model, MODELS_DIR / "best_model.pkl")
        joblib.dump(self.label_encoder, MODELS_DIR / "label_encoder.pkl")
        
        filtered_label_encoder = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
        filtered_classes = [filtered_label_encoder.classes_[i] for i in unique_labels]
        
        class FilteredLabelEncoder:
            def __init__(self, original_encoder, filtered_classes, label_mapping):
                self.original_encoder = original_encoder
                self.classes_ = np.array(filtered_classes)
                self.label_mapping = label_mapping
            
            def inverse_transform(self, y):
                return self.classes_[y]
        
        label_mapping = {new_label: old_label for new_label, old_label in enumerate(unique_labels)}
        filtered_encoder = FilteredLabelEncoder(label_encoder, filtered_classes, label_mapping)
        
        joblib.dump(filtered_encoder, MODELS_DIR / "filtered_label_encoder.pkl")
        
        log.info(f"Best model: {self.best_model_name.upper()} with accuracy {self.best_score:.4f}")
        
        return results
    
    def top_k_accuracy(self, y_true, y_proba, k=3):
        if k > y_proba.shape[1]:
            k = y_proba.shape[1]
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    
    def evaluate_model(self, model_name="best"):
        if model_name == "best":
            model = joblib.load(MODELS_DIR / "best_model.pkl")
        else:
            model = joblib.load(MODELS_DIR / f"{model_name}_model.pkl")
        
        X, y, symptom_columns, label_encoder = self.load_data()
        
        X_filtered, y_filtered, unique_labels = self.filter_rare_classes(X, y, min_samples=2)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42
            )
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        n_classes = len(unique_labels)
        
        print("\n=== Model Evaluation Report ===")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Test F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Top-3 Accuracy: {self.top_k_accuracy(y_test, y_proba, k=min(3, n_classes)):.4f}")
        print(f"Top-5 Accuracy: {self.top_k_accuracy(y_test, y_proba, k=min(5, n_classes)):.4f}")
        
        print(f"\nDataset Statistics:")
        print(f"Total samples (after filtering): {len(X_filtered)}")
        print(f"Features: {len(symptom_columns)}")
        print(f"Classes (after filtering): {n_classes}")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'top3': self.top_k_accuracy(y_test, y_proba, k=min(3, n_classes)),
            'top5': self.top_k_accuracy(y_test, y_proba, k=min(5, n_classes))
        }
