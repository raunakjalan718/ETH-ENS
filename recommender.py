import numpy as np
import pandas as pd
import joblib
import logging
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("evodoc.recommend")

BASE_DIR = Path(".")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

KIMI_API_KEY = "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"
KIMI_API_URL = "https://openrouter.ai/api/v1/chat/completions"

class TreatmentRecommender:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.symptom_columns = None
        self.load_models_and_data()
    
    def load_models_and_data(self):
        try:
            self.model = joblib.load(MODELS_DIR / "best_model.pkl")
            
            try:
                self.label_encoder = joblib.load(MODELS_DIR / "filtered_label_encoder.pkl")
            except FileNotFoundError:
                self.label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
            
            self.symptom_columns = np.load(PROCESSED_DIR / "X_columns.npy").tolist()
            
            log.info("Models and data loaded successfully")
            log.info(f"Available symptoms: {len(self.symptom_columns)}")
            log.info(f"Available diseases: {len(self.label_encoder.classes_)}")
            
        except Exception as e:
            log.error(f"Error loading models/data: {e}")
            raise
    
    def build_feature_vector(self, symptoms):
        feature_vector = np.zeros(len(self.symptom_columns))
        
        matched_symptoms = []
        for symptom in symptoms:
            symptom_normalized = symptom.lower().strip()
            
            if symptom_normalized in self.symptom_columns:
                idx = self.symptom_columns.index(symptom_normalized)
                feature_vector[idx] = 1
                matched_symptoms.append(symptom_normalized)
            else:
                for i, col_symptom in enumerate(self.symptom_columns):
                    if (symptom_normalized in col_symptom.lower() or 
                        col_symptom.lower() in symptom_normalized or
                        any(word in col_symptom.lower() for word in symptom_normalized.split() if len(word) > 2)):
                        feature_vector[i] = 1
                        matched_symptoms.append(col_symptom)
                        break
        
        log.info(f"Matched symptoms: {matched_symptoms}")
        return feature_vector.reshape(1, -1)
    
    def predict_diseases(self, symptoms, top_k=5):
        feature_vector = self.build_feature_vector(symptoms)
        
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            predictions.append({
                'disease': disease,
                'confidence': confidence
            })
        
        return predictions
    
    def get_kimi_insights(self, patient_data, disease_predictions):
        try:
            symptoms_text = ", ".join(patient_data["symptoms"])
            diseases_text = ", ".join([f"{p['disease']} ({p['confidence']:.0%})" for p in disease_predictions[:3]])
            
            prompt = f"""
            As a medical AI assistant, analyze this patient case:
            
            Patient: {patient_data['name']}, Age {patient_data['age']}, Gender {patient_data['gender']}
            Symptoms: {symptoms_text}
            Allergies: {', '.join(patient_data['allergies']) if patient_data['allergies'] else 'None'}
            Current medications: {', '.join(patient_data['current_medications']) if patient_data['current_medications'] else 'None'}
            
            Top predicted conditions: {diseases_text}
            
            Provide:
            1. Brief assessment of the most likely diagnosis
            2. General treatment recommendations
            3. Important safety considerations
            4. When to seek immediate medical attention
            5. General care recommendations
            
            Keep response concise and medically accurate.
            """
            
            headers = {
                "Authorization": f"Bearer {KIMI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "anthropic/claude-3-sonnet",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            
            response = requests.post(KIMI_API_URL, headers=headers, json=data, timeout=30)
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "AI insights not available at this time."
                
        except Exception as e:
            log.error(f"Error getting AI insights: {e}")
            return "AI insights not available at this time."
    
    def get_patient_input(self):
        print("\n=== EvoDoc Treatment Recommendation System ===")
        print("Please provide patient information:")
        
        name = input("Patient name: ").strip()
        age = input("Age: ").strip()
        gender = input("Gender (M/F/O): ").strip()
        
        print(f"\nEnter symptoms (comma-separated):")
        print(f"Sample symptoms: {', '.join(self.symptom_columns[:15])}...")
        symptoms_input = input("Symptoms: ").strip()
        symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
        
        allergies_input = input("Allergies (comma-separated, or press Enter if none): ").strip()
        allergies = [a.strip() for a in allergies_input.split(",") if a.strip()]
        
        current_meds_input = input("Current medications (comma-separated, or press Enter if none): ").strip()
        current_medications = [m.strip() for m in current_meds_input.split(",") if m.strip()]
        
        return {
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "allergies": allergies,
            "current_medications": current_medications
        }
    
    def display_recommendations(self, patient_data, disease_predictions, ai_insights):
        print(f"\n=== Medical Analysis for {patient_data['name']} ===")
        
        print("\nTop Disease Predictions:")
        for i, pred in enumerate(disease_predictions, 1):
            print(f"{i}. {pred['disease']} (Confidence: {pred['confidence']:.1%})")
        
        print("\n=== AI Medical Insights ===")
        print(ai_insights)
        
        print("\n⚠️  IMPORTANT: This is an AI-generated recommendation. Always consult with a healthcare professional before making any medical decisions.")
    
    def run_interactive_session(self):
        while True:
            try:
                patient_data = self.get_patient_input()
                disease_predictions = self.predict_diseases(patient_data["symptoms"])
                ai_insights = self.get_kimi_insights(patient_data, disease_predictions)
                self.display_recommendations(patient_data, disease_predictions, ai_insights)
                
                continue_input = input("\nWould you like to analyze another patient? (y/n): ").strip().lower()
                if continue_input != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
