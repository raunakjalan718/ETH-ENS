# EvoDoc – Decentralized AI Healthcare with ENS
**ENS-powered medical AI that enables incremental training without retraining existing models**

EvoDoc solves healthcare AI's biggest problems: fragmented data, opaque models, and non-portable patient identities. By using ENS as a naming layer for medical entities, we create the first truly interoperable medical AI system where diseases, symptoms, medicines, patients, and models all have human-readable blockchain identities.

---

## The Problem

Traditional healthcare AI systems cannot communicate with each other. When you add new disease categories, you must retrain everything from scratch. Patient data stays locked in institutional silos. There's no way to verify which model made a prediction or what data it was trained on.

---

## Our Solution

EvoDoc reimagines ENS as `.evodoc` domains that name every medical entity:

- **flu.evodoc** maps to Influenza with ICD-10 codes and stable hex identifiers  
- **fever.evodoc** represents symptoms with consistent IDs across all models  
- **ravi123.evodoc** creates portable patient profiles that work everywhere  
- **model.v1.evodoc** provides verifiable AI with accuracy metrics and dataset provenance  

The breakthrough is **incremental training**: train a respiratory model, then add a cardiovascular model later without touching the first one. ENS stable identities make this possible.

---

## What Makes This Special

- **ENS Identity Layer**  
  Every medical entity gets a permanent, human-readable name with stable hex codes. No more arbitrary database IDs that break when systems change.  

- **Incremental Training**  
  Add new disease categories by training specialized models that combine seamlessly. The respiratory model trained last month still works when you add cardiology today.  

- **Verifiable AI**  
  Each prediction links to the exact model version, accuracy metrics, and dataset hash. Judges can verify that a diagnosis came from a specific, auditable model.  

- **Portable Patients**  
  Patient profiles follow ENS identities across hospitals, apps, and countries. Your medical history isn't trapped in one system.  

---

## Technical Approach

We use Logistic Regression on **377 binary symptom features**, achieving **85% accuracy** on disease prediction.  
The dataset contains **200,000+ medical samples** split into respiratory, cardiovascular, diabetes, neurological, and other categories.  

The ENS registry replaces traditional label encoders with **stable identities**.  
When training the second model, we only add new diseases to the registry while preserving existing hex codes.  
This enables model combination **without conflicts or retraining**.  

---

## Project Structure

The codebase follows a **step-by-step demo flow** where each script performs one clear function.  
The ENS registry persists as **JSON locally**, with a roadmap for on-chain deployment using **IPFS** for model artifacts.  

---

## Performance Results

- Created **400+ ENS entities** including 25 diseases and 377 symptoms.  
- Models achieved strong **baseline performance**.  
- Demonstrated **zero retraining** when adding new categories.  
- Successfully combined predictions from specialized models using **ENS ID alignment**.  

---

## Future Vision

- **Phase 1**: Deploy `.evodoc` domains on-chain with IPFS storage.  
- **Phase 2**: Integrate medical standards like **RxNorm** and **SIDER**.  
- **Phase 3**: Enable **federated learning** where hospitals share ENS-mapped updates without exposing patient data.  
- **Phase 4**: Support **multilingual ENS** for global healthcare interoperability.  

---

## Why This Matters for ENS

This isn’t ENS as an afterthought. ENS is the **core technology** that enables:  
- Incremental machine learning  
- Verifiable AI provenance  
- Portable medical identities  

We demonstrate that ENS can power the **next generation of decentralized healthcare infrastructure**.  

---

## Quick Start

Install dependencies and run the demo:

git clone https://github.com/raunakjalan718/ETH-ENS.git
cd evodoc-ens
python -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib

Place your medical dataset CSV in data/raw/ then run:

python split.py      # Split dataset by disease groups
python setup.py      # Initialize ENS registry  
python train1.py     # Train first model (respiratory)
python train2.py     # Train second model (your choice: heart/diabetes/brain)
python combine.py    # Combine without retraining first model
python test.py       # Interactive patient demo
