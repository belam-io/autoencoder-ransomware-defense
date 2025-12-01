import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
import json

n = 50000

def random_response(status_code):
    """Generates a response payload based on the simple integer status_code."""
    if status_code == 0:
        error_code = "200.001.00"
        error_message = "Request Processed Successfully"
    else:
        # Use common failure/pending codes for non-zero status
        error_code = random.choice(["401.003.01", "500.100.02", "400.000.01"])
        error_message = random.choice([
            "Error Occurred - Invalid Access Token - ",
            "Internal Service Failure",
            "Transaction Failed"
        ])
        
    return {
        "requestId": str(uuid.uuid4()).replace("-", "")[:30],
        "errorCode": error_code,
        "errorMessage": error_message
    }

rows = []
base_time = datetime.now() - timedelta(days=30)

# Define integer status codes for the core transaction status
CORE_STATUS_CODES = [0, 0, 0, 10, 11, 20] # 0 = Success, others are various failures/pending

for _ in range(n):
    ts = base_time + timedelta(seconds=random.randint(0, 30*24*3600))
    
    # 🌟 FIX: Use simple integer status code for the main column
    txn_status_code = random.choice(CORE_STATUS_CODES)
    
    rows.append({
        "TxnID": str(uuid.uuid4()),
        "ShortCode": random.choice(["600111", "600222", "600333", "600444"]),
        "Amount": round(random.uniform(10, 50000), 2),
        "Msisdn": "2547" + str(random.randint(10000000, 99999999)),
        
        # 🌟 FIX: Corrected typo to match required column name
        "BillRefNumber": random.choice(["INV001", "INV002", "ACC998", "ACC777", "REF555"]),
        
        # 🌟 FIX: Used integer status code
        "StatusCode": txn_status_code, 
        
        # Response contains the detailed, complex error codes (like 200.001.00)
        "Response": json.dumps(random_response(txn_status_code)),
        "timestamp": ts.isoformat()
    })

df = pd.DataFrame(rows)
file_path = "data/kafka_mpesa_dataset.csv"

# Ensure the data directory exists before saving
import os
os.makedirs(os.path.dirname(file_path), exist_ok=True)

df.to_csv(file_path, index=False)

print(f"Successfully generated {n} rows to: {file_path}")
print("\n🔥 NEXT STEP REQUIRED:")
print("Ensure you have created the empty file 'autoencoder/__init__.py' and then re-run Docker Compose.")