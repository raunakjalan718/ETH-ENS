import json
from pathlib import Path

def show_ens():
    print("="*100)
    print("EVODOC ENS BLOCKCHAIN MEDICAL REGISTRY")
    print("="*100)
    print("Decentralized naming for medical AI - replacing 0x addresses with human-readable names")
    print("="*100)
    
    with open("./data/ens.json", 'r') as f:
        registry = json.load(f)
    
    entities = registry['entities']
    
    diseases = [(k, v) for k, v in entities.items() if v['type'] == 'disease']
    symptoms = [(k, v) for k, v in entities.items() if v['type'] == 'symptom']
    
    if diseases:
        print(f"\nDISEASE ENS DOMAINS ({len(diseases)} registered):")
        print("=" * 100)
        print(f"{'ENS Domain':<40} {'Blockchain Code':<15} {'Medical Name':<25} {'ICD-10':<8}")
        print("=" * 100)
        
        for ens_name, entity in diseases:
            print(f"{ens_name:<40} {entity['code']:<15} {entity['name']:<25} {entity.get('icd10', 'N/A'):<8}")
    
    if symptoms:
        print(f"\nSYMPTOM ENS DOMAINS ({len(symptoms)} registered):")
        print("=" * 100)
        print(f"{'ENS Domain':<40} {'Blockchain Code':<15} {'Symptom Name':<25}")
        print("=" * 100)
        
        for ens_name, entity in symptoms[:15]:
            print(f"{ens_name:<40} {entity['code']:<15} {entity['name']:<25}")
        
        if len(symptoms) > 15:
            print(f"... and {len(symptoms) - 15} more symptom ENS domains")
    
    print(f"\nENS BLOCKCHAIN REGISTRY STATISTICS:")
    print("=" * 100)
    print(f"Total ENS entities: {len(entities)}")
    print(f"Disease domains: {len(diseases)}")
    print(f"Symptom domains: {len(symptoms)}")
    print(f"Registry counter: {registry['counter']}")
    
    print(f"\nENS INNOVATION HIGHLIGHTS:")
    print("=" * 100)
    print("HUMAN-READABLE NAMES: flu.evodoc instead of 0x000001")
    print("BLOCKCHAIN STABILITY: Permanent hex codes for ML reproducibility")
    print("MEDICAL INTEROPERABILITY: ICD-10 codes integrated")
    print("DECENTRALIZED REGISTRY: No central database required")
    print("PORTABLE PROFILES: Patient data follows ENS identity")
    print("INCREMENTAL TRAINING: Add new models without retraining existing ones")
    
    print(f"\nHACKATHON DEMO VALUE:")
    print("=" * 100)
    print("ENS replaces traditional ML label encoding")
    print("Each medical entity gets a permanent blockchain address")
    print("Human-readable names improve AI transparency")
    print("Enables decentralized medical AI networks")
    print("Solves ML reproducibility with stable naming")

if __name__ == "__main__":
    show_ens()