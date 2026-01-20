# File: api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimizer import ValuationOptimizer
import uvicorn
import os

app = FastAPI()

# 1. Define the Input Format (What the iPhone sends)
class StockRequest(BaseModel):
    ticker: str

# 2. Define the Endpoint
@app.post("/analyze")
def analyze_stock(request: StockRequest):
    try:
        # Initialize your existing logic
        optimizer = ValuationOptimizer()
        
        # Run the same math you used in Streamlit
        result = optimizer.find_optimal_strategy(request.ticker)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result['error'])
            
        # Return the raw data (The iPhone will design the charts)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Entry point for the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
