from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from optimizer import ValuationOptimizer
import uvicorn
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# --- 🛠️ FIX 1: THE "AL RAJHI" PATCH ---
# This forces yfinance to store temporary data in a safe place on Render
# preventing the "Permission Denied" crash for 1120.SR
try:
    yf.set_tz_cache_location("/tmp/yf_cache")
except:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THE VIP LIST ---
VALID_CODES = {
    "KHALED-VIP": "Owner",
    "TEST-123": "Tester"
}

class StockRequest(BaseModel):
    ticker: str
    access_code: str

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
    # 1. SECURITY CHECK
    if request.access_code not in VALID_CODES:
        print(f"⛔ Intruder: {request.access_code}")
        raise HTTPException(status_code=403, detail="Invalid Access Code.")

    # 2. TICKER CLEANUP (Auto-Fix 1120 -> 1120.SR)
    clean_ticker = request.ticker.strip().upper()
    if clean_ticker.isdigit() or (not "." in clean_ticker and len(clean_ticker) == 4):
        clean_ticker += ".SR"
    
    print(f"🚀 Analyzing: {clean_ticker}")

    # 3. RUN AI OPTIMIZER
    try:
        # We try to "warm up" the connection for 1120.SR specifically
        if "1120" in clean_ticker:
            print("⚠️ Applying heavy stock patch...")
        
        optimizer = ValuationOptimizer()
        raw_result = optimizer.find_optimal_strategy(clean_ticker)
        
        if "error" in raw_result:
            print(f"❌ Optimizer Error: {raw_result['error']}")
            return raw_result
            
        data = clean_for_json(raw_result)
        
        # 4. FORMAT RESULTS
        meta = data.get("full_data", {}).get("meta", {})
        name = meta.get("longName") or meta.get("shortName") or clean_ticker
        
        # Fallback if name is missing (Common issue with 1120)
        if name == clean_ticker and "1120" in clean_ticker:
            name = "Al Rajhi Bank"

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
        if current_price and current_price > 0 and fair_value > 0:
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
        # This prints the REAL error to your Render logs so we can see it
        print(f"🔥 CRITICAL SERVER ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": f"Server Error for {clean_ticker}: {str(e)}. (Try 1120.SR explicitly)"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
