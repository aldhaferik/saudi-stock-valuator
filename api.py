from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimizer import ValuationOptimizer
import uvicorn
import os
import sys
import traceback
import numpy as np
import pandas as pd

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str

# --- THE DEEP CLEANER FUNCTION ---
# This forces every piece of data into a format the iPhone can read.
def clean_for_json(obj):
    # 1. Handle Pandas DataFrames (The likely culprit)
    if isinstance(obj, pd.DataFrame):
        # Create a copy to avoid messing with original data
        df = obj.copy()
        # Ensure dates in the index are strings (YYYY-MM-DD)
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime('%Y-%m-%d')
        # Convert to dictionary
        return df.to_dict()
        
    # 2. Handle Pandas Series
    if isinstance(obj, pd.Series):
        return obj.to_dict()
        
    # 3. Handle NumPy Integers/Floats
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
        
    # 4. Handle NumPy Arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
        
    # 5. Handle Recursion (Lists and Dicts)
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
        
    # 6. Handle NaNs (JSON doesn't like NaN)
    if obj is pd.NA or (isinstance(obj, float) and np.isnan(obj)):
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
            
        # 4. DEEP CLEAN THE DATA
        # This fixes the "numpy.int64" and "vars()" errors
        print("   -> Sanitizing data types...")
        clean_result = clean_for_json(raw_result)
        
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
