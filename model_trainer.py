import numpy as np
import pandas as pd
import joblib
import logging
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from collections import Counter
import time

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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.unique_labels = None
        
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
    
    def filter_rare_classes(self, X, y, min_samples=5):
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
    
    def prepare_data(self):
        X, y, symptom_columns, label_encoder = self.load_data()
        
        if X is None:
            return False
            
        self.label_encoder = label_encoder
        X_filtered, y_filtered, unique_labels = self.filter_rare_classes(X, y, min_samples=5)
        self.unique_labels = unique_labels
        
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
        except ValueError as e:
            log.warning(f"Stratified split failed: {e}")
            log.info("Using random split instead")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42
            )
        
        log.info(f"Data prepared: Train={self.X_train.shape}, Test={self.X_test.shape}")
        return True
    
    def train_xgboost(self, save_model=True):
        log.info("Training XGBoost model...")
        
        n_classes = len(self.unique_labels)
        n_features = self.X_train.shape[1]
        
        params = {
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
        
        model = XGBClassifier(**params)
        start_time = time.time()
        
        try:
            cv_folds = min(3, len(np.unique(self.y_train)))
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_folds, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as cv_error:
            log.warning(f"Cross-validation failed: {cv_error}")
            cv_mean, cv_std = 0.0, 0.0
        
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        top3_acc = self.top_k_accuracy(self.y_test, y_proba, k=min(3, n_classes))
        top5_acc = self.top_k_accuracy(self.y_test, y_proba, k=min(5, n_classes))
        training_time = time.time() - start_time
        
        results = {
            'model': model,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': accuracy,
            'test_f1': f1,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'training_time': training_time,
            'params': params
        }
        
        log.info(f"XGBoost: CV={cv_mean:.4f}±{cv_std:.4f}, Test Acc={accuracy:.4f}, "
                f"F1={f1:.4f}, Top3={top3_acc:.4f}, Top5={top5_acc:.4f}, Time={training_time:.2f}s")
        
        if save_model:
            joblib.dump(model, MODELS_DIR / "xgb_model.pkl")
            with open(MODELS_DIR / "xgb_results.json", "w") as f:
                json.dump({k: v for k, v in results.items() if k != 'model'}, f, indent=2)
            log.info("XGBoost model saved")
        
        return results
    
    def train_random_forest(self, save_model=True):
        log.info("Training Random Forest model...")
        
        n_classes = len(self.unique_labels)
        n_features = self.X_train.shape[1]
        
        params = {
            "n_estimators": min(100, max(50, n_features // 10)),
            "max_depth": min(10, max(5, int(np.sqrt(n_features)))),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0
        }
        
        model = RandomForestClassifier(**params)
        start_time = time.time()
        
        try:
            cv_folds = min(3, len(np.unique(self.y_train)))
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_folds, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as cv_error:
            log.warning(f"Cross-validation failed: {cv_error}")
            cv_mean, cv_std = 0.0, 0.0
        
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        top3_acc = self.top_k_accuracy(self.y_test, y_proba, k=min(3, n_classes))
        top5_acc = self.top_k_accuracy(self.y_test, y_proba, k=min(5, n_classes))
        training_time = time.time() - start_time
        
        results = {
            'model': model,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': accuracy,
            'test_f1': f1,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'training_time': training_time,
            'params': params
        }
        
        log.info(f"Random Forest: CV={cv_mean:.4f}±{cv_std:.4f}, Test Acc={accuracy:.4f}, "
                f"F1={f1:.4f}, Top3={top3_acc:.4f}, Top5={top5_acc:.4f}, Time={training_time:.2f}s")
        
        if save_model:
            joblib.dump(model, MODELS_DIR / "rf_model.pkl")
            with open(MODELS_DIR / "rf_results.json", "w") as f:
                json.dump({k: v for k, v in results.items() if k != 'model'}, f, indent=2)
            log.info("Random Forest model saved")
        
        return results
    
    def train_logistic_regression(self, save_model=True):
        log.info("Training Logistic Regression model...")
        
        n_classes = len(self.unique_labels)
        n_samples = self.X_train.shape[0]
        
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
        
        if n_samples > 150000:
            log.info("Skipping cross-validation for large dataset")
            cv_mean, cv_std = 0.0, 0.0
        else:
            try:
                cv_folds = min(3, len(np.unique(self.y_train)))
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_folds, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as cv_error:
                log.warning(f"Cross-validation failed: {cv_error}")
                cv_mean, cv_std = 0.0, 0.0
        
        log.info("Starting model training...")
        model.fit(self.X_train, self.y_train)
        log.info("Model training completed")
        
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        top3_acc = self.top_k_accuracy(self.y_test, y_proba, k=min(3, n_classes))
        top5_acc = self.top_k_accuracy(self.y_test, y_proba, k=min(5, n_classes))
        training_time = time.time() - start_time
        
        results = {
            'model': model,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': accuracy,
            'test_f1': f1,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'training_time': training_time,
            'params': params
        }
        
        log.info(f"Logistic Regression: CV={cv_mean:.4f}±{cv_std:.4f}, Test Acc={accuracy:.4f}, "
                f"F1={f1:.4f}, Top3={top3_acc:.4f}, Top5={top5_acc:.4f}, Time={training_time:.2f}s")
        
        if save_model:
            joblib.dump(model, MODELS_DIR / "logreg_model.pkl")
            with open(MODELS_DIR / "logreg_results.json", "w") as f:
                json.dump({k: v for k, v in results.items() if k != 'model'}, f, indent=2)
            log.info("Logistic Regression model saved")
        
        return results
    
    def train_all_models(self):
        if not self.prepare_data():
            return None
        
        results = {}
        
        try:
            results['xgb'] = self.train_xgboost()
        except Exception as e:
            log.error(f"XGBoost training failed: {e}")
        
        try:
            results['rf'] = self.train_random_forest()
        except Exception as e:
            log.error(f"Random Forest training failed: {e}")
        
        try:
            results['logreg'] = self.train_logistic_regression()
        except Exception as e:
            log.error(f"Logistic Regression training failed: {e}")
        
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
            best_model = results[best_model_name]['model']
            best_score = results[best_model_name]['test_accuracy']
            
            joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
            
            filtered_label_encoder = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
            filtered_classes = [filtered_label_encoder.classes_[i] for i in self.unique_labels]
            
            class FilteredLabelEncoder:
                def __init__(self, original_encoder, filtered_classes, label_mapping):
                    self.original_encoder = original_encoder
                    self.classes_ = np.array(filtered_classes)
                    self.label_mapping = label_mapping
                
                def inverse_transform(self, y):
                    return self.classes_[y]
            
            label_mapping = {new_label: old_label for new_label, old_label in enumerate(self.unique_labels)}
            filtered_encoder = FilteredLabelEncoder(self.label_encoder, filtered_classes, label_mapping)
            
            joblib.dump(filtered_encoder, MODELS_DIR / "filtered_label_encoder.pkl")
            
            summary = {
                'best_model': best_model_name,
                'best_accuracy': best_score,
                'all_results': {k: {key: val for key, val in v.items() if key != 'model'} 
                               for k, v in results.items()}
            }
            
            with open(MODELS_DIR / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            log.info(f"Best model: {best_model_name.upper()} with accuracy {best_score:.4f}")
            
            self.models = results
            self.best_model = best_model
            self.best_score = best_score
            self.best_model_name = best_model_name
        
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
        
        if self.X_test is None:
            self.prepare_data()
        
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        
        n_classes = len(self.unique_labels)
        
        print(f"\n=== {model_name.upper()} Model Evaluation Report ===")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Test F1-Score: {f1_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"Top-3 Accuracy: {self.top_k_accuracy(self.y_test, y_proba, k=min(3, n_classes)):.4f}")
        print(f"Top-5 Accuracy: {self.top_k_accuracy(self.y_test, y_proba, k=min(5, n_classes)):.4f}")
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred, average='weighted'),
            'top3': self.top_k_accuracy(self.y_test, y_proba, k=min(3, n_classes)),
            'top5': self.top_k_accuracy(self.y_test, y_proba, k=min(5, n_classes))
        }