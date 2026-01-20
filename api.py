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

# --- DATA CLEANER (Keeps JSON Safe) ---
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
    try:
        optimizer = ValuationOptimizer()
        raw_result = optimizer.find_optimal_strategy(request.ticker)
        
        if "error" in raw_result:
            return raw_result
            
        data = clean_for_json(raw_result)
        
        # --- NEW: CALCULATE VERDICT & NAME ---
        # 1. Get Company Name
        meta = data.get("full_data", {}).get("meta", {})
        # Try different fields where the name might hide
        name = meta.get("shortName") or meta.get("longName") or meta.get("address1") or request.ticker
        
        # 2. Calculate Fair Value
        # We take the weights (AI Strategy) and the latest predictions
        strategies = data.get("strategies", {}).get("solver", {})
        history = data.get("history", [])
        
        fair_value = 0.0
        current_price = meta.get("currentPrice", 0.0)
        
        if history and strategies:
            latest_predictions = history[-1].get("predictions", {})
            
            # Sum up: (Weight * Prediction) for each model
            for model, detail in strategies.items():
                weight = detail.get("weight", 0.0)
                pred_price = latest_predictions.get(model, 0.0)
                if pred_price:
                    fair_value += weight * pred_price
        
        # 3. Determine Verdict
        upside = 0.0
        verdict = "Neutral"
        if current_price > 0 and fair_value > 0:
            upside = ((fair_value - current_price) / current_price) * 100
            if upside >= 5: verdict = "Undervalued"
            elif upside <= -5: verdict = "Overvalued"
            else: verdict = "Fairly Valued"

        # 4. Inject into response
        data["valuation_summary"] = {
            "company_name": name,
            "fair_value": fair_value,
            "current_price": current_price,
            "verdict": verdict,
            "upside_percent": upside
        }
        
        return data
        
    except Exception as e:
        error_msg = str(e)
        print(f"CRITICAL ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
