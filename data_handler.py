import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("evodoc.data")

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    dir_path.mkdir(exist_ok=True)

class DataHandler:
    def __init__(self):
        self.symptom_columns = []
        self.label_encoder = LabelEncoder()
        
    def discover_datasets(self):
        log.info("Discovering available datasets...")
        
        csv_files = list(RAW_DIR.rglob("*.csv"))
        if not csv_files:
            csv_files = list(Path("../data/raw").rglob("*.csv"))
        
        discovered_files = {}
        
        for csv_file in csv_files:
            try:
                df_sample = pd.read_csv(csv_file, nrows=5)
                columns = [col.lower().strip() for col in df_sample.columns]
                
                file_info = {
                    "path": csv_file,
                    "columns": df_sample.columns.tolist(),
                    "shape": f"~{len(df_sample)} rows (sample)",
                    "potential_target": None,
                    "potential_features": []
                }
                
                target_keywords = ["disease", "diseases", "prognosis", "diagnosis", "target", "label", "class"]
                
                for col in columns:
                    if any(keyword in col for keyword in target_keywords):
                        file_info["potential_target"] = df_sample.columns[columns.index(col)]
                        break
                
                numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
                if file_info["potential_target"] and file_info["potential_target"] in numeric_cols:
                    numeric_cols.remove(file_info["potential_target"])
                file_info["potential_features"] = numeric_cols
                
                discovered_files[csv_file.name] = file_info
                log.info(f"Discovered: {csv_file.name} - Target: {file_info['potential_target']}, Features: {len(file_info['potential_features'])}")
                
            except Exception as e:
                log.warning(f"Could not analyze {csv_file}: {e}")
        
        return discovered_files
    
    def auto_select_dataset(self, discovered_files):
        if not discovered_files:
            log.error("No CSV files found")
            return None
            
        best_file = None
        best_score = 0
        
        for filename, info in discovered_files.items():
            score = 0
            
            if info["potential_target"]:
                score += 10
            
            score += len(info["potential_features"])
            
            symptom_keywords = ["symptom", "disease", "medical"]
            if any(keyword in filename.lower() for keyword in symptom_keywords):
                score += 5
            
            if score > best_score:
                best_score = score
                best_file = info
        
        if best_file:
            log.info(f"Auto-selected dataset: {best_file['path'].name}")
            return best_file
        
        return None
    
    def load_and_analyze_dataset(self, file_info):
        log.info(f"Loading dataset: {file_info['path']}")
        
        try:
            df = pd.read_csv(file_info['path'])
            log.info(f"Loaded dataset with shape {df.shape}")
            
            if file_info["potential_target"]:
                target_col = file_info["potential_target"]
            else:
                target_candidates = []
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.8:
                        target_candidates.append((col, df[col].nunique()))
                
                if target_candidates:
                    target_col = min(target_candidates, key=lambda x: x[1])[0]
                    log.info(f"Auto-detected target column: {target_col}")
                else:
                    log.error("Could not identify target column")
                    return None, None, None
            
            if target_col not in df.columns:
                log.error(f"Target column '{target_col}' not found in dataset")
                return None, None, None
            
            y = df[target_col].values
            X = df.drop(columns=[target_col])
            
            if X.select_dtypes(include=[np.number]).shape[1] == 0:
                log.error("No numeric features found in dataset")
                return None, None, None
            
            X = X.select_dtypes(include=[np.number])
            
            self.symptom_columns = X.columns.tolist()
            
            unique_diseases = np.unique(y)
            log.info(f"Found {len(unique_diseases)} unique diseases")
            log.info(f"Sample diseases: {list(unique_diseases[:10])}")
            
            y_encoded = self.label_encoder.fit_transform(y)
            
            return X.values, y_encoded, self.symptom_columns
            
        except Exception as e:
            log.error(f"Error processing dataset: {e}")
            return None, None, None
    
    def save_processed_data(self, X, y, symptom_columns):
        log.info("Saving processed data...")
        
        np.save(PROCESSED_DIR / "X.npy", X)
        np.save(PROCESSED_DIR / "y.npy", y)
        np.save(PROCESSED_DIR / "X_columns.npy", np.array(symptom_columns))
        joblib.dump(self.label_encoder, PROCESSED_DIR / "label_encoder.pkl")
        
        log.info(f"Processed data saved: X={X.shape}, y={len(y)}, features={len(symptom_columns)}")
    
    def process_all_data(self):
        log.info("Starting complete data processing pipeline...")
        
        discovered_files = self.discover_datasets()
        
        if not discovered_files:
            log.error("No datasets found. Please place CSV files in data/raw directory.")
            return False
        
        selected_file = self.auto_select_dataset(discovered_files)
        
        if not selected_file:
            log.error("Could not auto-select a suitable dataset.")
            return False
        
        X, y, symptom_columns = self.load_and_analyze_dataset(selected_file)
        
        if X is None:
            log.error("Failed to process dataset.")
            return False
        
        self.save_processed_data(X, y, symptom_columns)
        
        log.info("Data processing pipeline completed successfully!")
        return True
