import json
from pathlib import Path
from datetime import datetime

def setup_ens():
    print("="*60)
    print("STEP 2: SETUP ENS REGISTRY")
    print("="*60)
    
    data_dir = Path("./data")
    registry_file = data_dir / "ens.json"
    
    registry_data = {
        'counter': 0,
        'entities': {},
        'created_at': datetime.now().isoformat()
    }
    
    with open(registry_file, 'w') as f:
        json.dump(registry_data, f, indent=2)
    
    print("ENS registry initialized")
    print("Registry will store:")
    print("- Disease names: flu.evodoc")
    print("- Stable codes: 0x000001")
    print("- Medical codes: ICD-10, SNOMED")

if __name__ == "__main__":
    setup_ens()
