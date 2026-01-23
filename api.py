from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimizer import ValuationOptimizer
import uvicorn
import os
import math
import numpy as np
import pandas as pd

app = FastAPI()

# --- THE VIP LIST ---
# Add any codes you want here.
ALID_CODES = {
    "NEW-PASSWORD-HERE": "VIP User",
    "FAMILY-ONLY": "Guest"
}

class StockRequest(BaseModel):
    ticker: str
    access_code: str  

# --- DATA CLEANER ---
def clean_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime('%Y-%m-%d')
        return clean_for_json(df.to_dict())
    if isinstance(obj, pd.Series):
        return clean_for_json(obj.to_dict())
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val): return None
        return val
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return None
    if pd.isna(obj): return None
    return obj

@app.post("/analyze")
def analyze_stock(request: StockRequest):
    # 1. CHECK THE CODE (The Bouncer)
    if request.access_code not in VALID_CODES:
        print(f"⛔ Intruder detected! Tried code: {request.access_code}")
        # Return a special "403 Forbidden" error
        raise HTTPException(status_code=403, detail="Invalid Access Code. Please contact Khaled for access.")

    print(f"✅ Access Granted to: {VALID_CODES[request.access_code]}")
    
    # 2. PROCEED AS NORMAL
    try:
        optimizer = ValuationOptimizer()
        raw_result = optimizer.find_optimal_strategy(request.ticker)
        
        if "error" in raw_result:
            return raw_result
            
        data = clean_for_json(raw_result)
        
        # Calculate Verdict (Same as before)
        meta = data.get("full_data", {}).get("meta", {})
        name = meta.get("shortName") or meta.get("longName") or request.ticker
        strategies = data.get("strategies", {}).get("solver", {})
        history = data.get("history", [])
        fair_value = 0.0
        current_price = meta.get("currentPrice", 0.0)
        
        if history and strategies:
            latest_preds = history[-1].get("predictions", {})
            for model, detail in strategies.items():
                w = detail.get("weight", 0.0)
                p = latest_preds.get(model, 0.0)
                if p: fair_value += w * p
        
        upside = 0.0
        verdict = "Neutral"
        if current_price > 0 and fair_value > 0:
            upside = ((fair_value - current_price) / current_price) * 100
            if upside >= 5: verdict = "Undervalued"
            elif upside <= -5: verdict = "Overvalued"
            else: verdict = "Fairly Valued"

        data["valuation_summary"] = {
            "company_name": name,
            "fair_value": fair_value,
            "current_price": current_price,
            "verdict": verdict,
            "upside_percent": upside
        }
        
        return data
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
