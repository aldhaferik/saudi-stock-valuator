from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import random
from bs4 import BeautifulSoup
from datetime import datetime

app = FastAPI()

# 1. ALLOW YOUR WEBSITE TO CONNECT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. VIP CODES
VALID_CODES = {
    "KHALED-VIP": "Owner",
    "TEST-123": "Tester"
}

class StockRequest(BaseModel):
    ticker: str
    access_code: str

# 3. THE ENGINE (Previously optimizer.py)
class DataFetcher:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36"
        ]

    def fetch(self, ticker):
        clean_ticker = ticker
        if ticker.replace('.','').isdigit() and not ticker.endswith('.SR'):
            clean_ticker = f"{ticker}.SR"
            
        print(f"🚀 Analyzing: {clean_ticker}")

        # SOURCE 1: YAHOO FINANCE
        try:
            print("🔹 Trying Yahoo...")
            session = requests.Session()
            session.headers.update({"User-Agent": random.choice(self.user_agents)})
            stock = yf.Ticker(clean_ticker, session=session)
            hist = stock.history(period="1mo")
            if not hist.empty:
                return {"history": stock.history(period="5y"), "info": stock.info, "source": "Yahoo"}
        except: pass

        # SOURCE 2: WEB SCRAPER (Google Finance)
        try:
            print("🔸 Trying Scraper...")
            symbol = ticker.split(".")[0]
            url = f"https://www.google.com/finance/quote/{symbol}:TADAWUL"
            r = requests.get(url, headers={"User-Agent": random.choice(self.user_agents)}, timeout=5)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                price_div = soup.find("div", {"class": "YMlKec fxKbKc"})
                if price_div:
                    price = float(price_div.text.replace("SAR", "").replace(",", "").strip())
                    # Generate simple history for the chart
                    dates = pd.date_range(end=datetime.now(), periods=30)
                    hist = pd.DataFrame({"Close": [price]*30, "Open": [price]*30}, index=dates)
                    info = {"longName": f"Saudi Market: {symbol}", "currentPrice": price, "trailingEps": price/18, "bookValue": price/2.5, "trailingPE": 18}
                    return {"history": hist, "info": info, "source": "WebScraper"}
        except: pass

        return None

# 4. THE API ENDPOINT
@app.post("/analyze")
def analyze_stock(request: StockRequest):
    if request.access_code not in VALID_CODES:
        raise HTTPException(status_code=403, detail="Invalid Access Code")

    fetcher = DataFetcher()
    data = fetcher.fetch(request.ticker)
    
    if not data:
        return {"error": "Could not retrieve data from Yahoo or Scraper."}

    # CALCULATE VALUATION
    hist = data["history"]
    info = data["info"]
    current_price = hist["Close"].iloc[-1]
    eps = info.get("trailingEps") or 0
    
    # Simple Valuation Logic
    fair_value = (eps * 18.0) if eps > 0 else (current_price * 1.05)
    
    verdict = "Fairly Valued"
    if fair_value > current_price * 1.05: verdict = "Undervalued"
    if fair_value < current_price * 0.95: verdict = "Overvalued"

    # CLEAN DATA FOR JSON (Handle NaNs)
    def clean(val):
        if isinstance(val, float) and (pd.isna(val) or np.isinf(val)): return 0
        return val

    return {
        "valuation_summary": {
            "company_name": info.get("longName", request.ticker),
            "fair_value": clean(fair_value),
            "current_price": clean(current_price),
            "verdict": verdict,
            "upside_percent": clean(((fair_value - current_price)/current_price)*100)
        },
        "source_used": data["source"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
