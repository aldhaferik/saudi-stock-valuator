from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimizer import ValuationOptimizer
import uvicorn
import os
import sys
import traceback
import numpy as np  # Added to detect numpy types

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str

# --- THE SANITIZER FUNCTION ---
# This converts all NumPy types (which break JSON) into standard Python types
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    else:
        return obj

@app.get("/")
def home():
    return {"status": "Alive", "message": "Send POST request to /analyze with ticker"}

@app.post("/analyze")
def analyze_stock(request: StockRequest):
    print(f"📥 Received request for ticker: {request.ticker}")
    
    try:
        # 1. Initialize Optimizer
        optimizer = ValuationOptimizer()
        
        # 2. Run Strategy
        print(f"   -> Running find_optimal_strategy for {request.ticker}...")
        raw_result = optimizer.find_optimal_strategy(request.ticker)
        
        # 3. Check for internal errors
        if "error" in raw_result:
            print(f"   ⚠️ Optimizer returned error: {raw_result['error']}")
            return raw_result
            
        # 4. SANITIZE THE DATA (Crucial Fix)
        # We wrap the result in our cleaner function before returning
        clean_result = convert_numpy(raw_result)
        
        print("   ✅ Success! Data sanitized and returning.")
        return clean_result
        
    except Exception as e:
        # Capture the FULL error traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        
        print(f"🔥 CRITICAL SERVER CRASH: {error_msg}")
        print(tb)
        
        raise HTTPException(status_code=500, detail=f"Server Crash: {error_msg}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
