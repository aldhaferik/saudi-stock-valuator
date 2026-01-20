from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimizer import ValuationOptimizer
import uvicorn
import os
import sys
import traceback
import numpy as np
import pandas as pd
import math

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str

# --- THE NAN-PROOF CLEANER ---
def clean_for_json(obj):
    # 1. Handle Pandas DataFrames & Series
    # We convert them to dicts, then RECURSE to clean the contents (catch NaNs)
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        # Ensure dates in index are strings
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime('%Y-%m-%d')
        return clean_for_json(df.to_dict()) # <--- Recurse here!
        
    if isinstance(obj, pd.Series):
        return clean_for_json(obj.to_dict()) # <--- Recurse here!

    # 2. Handle NumPy Integers/Floats
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
        
    # 3. Handle NumPy Arrays
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
        
    # 4. Handle Lists and Dicts (Recursion)
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
        
    # 5. Handle Standalone NaNs / Infinities (The Crash Fix)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
            
    # Handle Pandas NA/Null types
    if pd.isna(obj):
        return None
        
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
            
        # 4. DEEP CLEAN & SANITIZE
        print("   -> Sanitizing data (Removing NaNs/NumPy types)...")
        clean_result = clean_for_json(raw_result)
        
        print("   ✅ Success! Data sanitized and returning.")
        return clean_result
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"🔥 CRITICAL SERVER CRASH: {error_msg}")
        print(tb)
        raise HTTPException(status_code=500, detail=f"Server Crash: {error_msg}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
