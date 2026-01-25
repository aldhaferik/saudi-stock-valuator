from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import random
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Try to import BeautifulSoup (Safety check)
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_CODES = ["KHALED-VIP", "TEST-123"]

# ==========================================
# 1. THE DATA ENGINE (Multi-Source)
# ==========================================
class DataFetcher:
    def __init__(self):
        self.av_key = "0LR5JLOBSLOA6Z0A"
        self.td_key = "ed240f406bab4225ac6e0a98be553aa2"
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36"
        ]

    def fetch(self, ticker):
        clean_ticker = ticker
        if ticker.replace('.','').isdigit() and not ticker.endswith('.SR'):
            clean_ticker = f"{ticker}.SR"
            
        print(f"🚀 Starting Analysis for: {clean_ticker}")

        # --- SOURCE 1: YAHOO FINANCE ---
        try:
            import yfinance as yf
            print("🔹 Trying Yahoo...")
            stock = yf.Ticker(clean_ticker)
            hist = stock.history(period="1mo")
            if not hist.empty:
                info = stock.info
                # Critical check: Yahoo sometimes gives empty info for Saudi stocks
                if info.get("currentPrice") or hist["Close"].iloc[-1]:
                    return {"history": stock.history(period="5y"), "info": info, "source": "Yahoo Finance"}
        except: pass

        # --- SOURCE 2: ALPHA VANTAGE ---
        try:
            print("🔸 Trying Alpha Vantage...")
            av_symbol = clean_ticker.replace(".SR", ".SA")
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_symbol}&apikey={self.av_key}"
            r = requests.get(url, timeout=5)
            data = r.json()
            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df = df.rename(columns={"4. close": "Close"})
                df.index = pd.to_datetime(df.index)
                df = df.astype(float).sort_index()
                price = df["Close"].iloc[-1]
                # Synthesize Info
                info = {"longName": f"Saudi Stock {ticker}", "currentPrice": price, "trailingEps": price/18.0}
                return {"history": df, "info": info, "source": "Alpha Vantage API"}
        except: pass

        # --- SOURCE 3: TWELVE DATA ---
        try:
            print("🔸 Trying Twelve Data...")
            td_symbol = ticker.split(".")[0]
            url = f"https://api.twelvedata.com/time_series?symbol={td_symbol}&exchange=Tadawul&interval=1day&apikey={self.td_key}"
            r = requests.get(url, timeout=5)
            data = r.json()
            if "values" in data:
                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
                df = df.rename(columns={"close": "Close"})
                df = df.astype(float).sort_index()
                price = df["Close"].iloc[-1]
                info = {"longName": f"Saudi Co {td_symbol}", "currentPrice": price, "trailingEps": price/20.0}
                return {"history": df, "info": info, "source": "Twelve Data API"}
        except: pass

        # --- SOURCE 4: WEB SCRAPER (Fallback) ---
        if BS4_AVAILABLE:
            try:
                print("🔸 Trying Web Scraper (Google Finance)...")
                symbol = ticker.split(".")[0]
                url = f"https://www.google.com/finance/quote/{symbol}:TADAWUL"
                headers = {"User-Agent": random.choice(self.user_agents)}
                r = requests.get(url, headers=headers, timeout=5)
                
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, 'html.parser')
                    # Classes change, but these are common for Google Finance
                    price_div = soup.find("div", {"class": "YMlKec fxKbKc"})
                    name_div = soup.find("div", {"class": "zzDege"})
                    
                    if price_div:
                        price = float(price_div.text.replace("SAR", "").replace(",", "").strip())
                        name = name_div.text if name_div else f"Saudi Stock {symbol}"
                        
                        # Create synthetic history for the chart
                        dates = pd.date_range(end=datetime.now(), periods=30)
                        hist = pd.DataFrame({"Close": [price]*30}, index=dates)
                        info = {"longName": name, "currentPrice": price, "trailingEps": price/19.5}
                        return {"history": hist, "info": info, "source": "Live Web Scraper"}
            except Exception as e:
                print(f"Scraper Error: {e}")

        return None

# ==========================================
# 2. THE PROFESSIONAL UI (HTML/CSS)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Saudi Valuator AI</title>
        <style>
            body { font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; display: flex; justify-content: center; padding-top: 40px; margin: 0; }
            .card { background: white; padding: 2.5rem; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); width: 100%; max-width: 380px; text-align: center; }
            h1 { color: #1a1a1a; font-size: 26px; margin-bottom: 25px; letter-spacing: -0.5px; }
            
            .input-group { margin-bottom: 15px; text-align: left; }
            label { display: block; font-size: 12px; font-weight: 600; color: #666; margin-bottom: 5px; text-transform: uppercase; }
            input { width: 100%; padding: 14px; border: 1px solid #e1e1e1; border-radius: 10px; font-size: 16px; box-sizing: border-box; transition: 0.2s; }
            input:focus { border-color: #007aff; outline: none; box-shadow: 0 0 0 3px rgba(0,122,255,0.1); }
            
            button { width: 100%; padding: 16px; background-color: #007aff; color: white; border: none; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; transition: 0.2s; margin-top: 10px; }
            button:hover { background-color: #005bb5; transform: translateY(-1px); }
            button:disabled { background-color: #ccc; cursor: not-allowed; }

            .result-box { margin-top: 25px; text-align: left; display: none; border-top: 1px solid #eee; padding-top: 20px; animation: fadeIn 0.4s ease; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            
            .company-name { font-size: 20px; font-weight: 700; color: #333; margin-bottom: 2px; }
            .ticker-label { font-size: 13px; color: #888; margin-bottom: 15px; font-family: monospace; }
            
            .verdict-badge { display: inline-block; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 20px; }
            .v-green { background: #e6f9e9; color: #2e7d32; }
            .v-red { background: #ffebee; color: #c62828; }
            .v-gray { background: #f5f5f5; color: #616161; }

            .metric-row { display: flex; justify-content: space-between; margin-bottom: 12px; }
            .metric-label { color: #666; font-size: 14px; }
            .metric-value { font-weight: 600; color: #333; font-size: 15px; }

            .source-tag { font-size: 11px; color: #aaa; text-align: center; margin-top: 20px; font-style: italic; }
            .loading { color: #666; font-size: 14px; margin-top: 15px; display: none; }
            .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #ccc; border-top-color: #333; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px; }
            @keyframes spin { to { transform: rotate(360deg); } }
            
            .error-msg { background: #fff2f0; color: #ff4d4f; padding: 12px; border-radius: 8px; font-size: 13px; margin-top: 15px; display: none; border: 1px solid #ffccc7; }
        </style>
    </head>
    <body>

    <div class="card">
        <h1>🇸🇦 Saudi Valuator AI</h1>
        
        <div class="input-group">
            <label>Access Code</label>
            <input type="password" id="code" placeholder="Enter VIP Code" />
        </div>
        
        <div class="input-group">
            <label>Stock Ticker</label>
            <input type="text" id="ticker" placeholder="e.g. 1120 or 2222" />
        </div>
        
        <button onclick="analyze()" id="btn">Analyze Stock</button>
        
        <div class="loading" id="loading"><span class="spinner"></span> AI is analyzing multiple sources...</div>
        <div class="error-msg" id="error"></div>

        <div class="result-box" id="result">
            <div class="company-name" id="name">--</div>
            <div class="ticker-label" id="tickerDisplay">--</div>
            
            <div style="text-align: center;">
                <span class="verdict-badge" id="verdict">--</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Current Price</span>
                <span class="metric-value" id="price">--</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Fair Value</span>
                <span class="metric-value" id="fair">--</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Upside</span>
                <span class="metric-value" id="upside">--</span>
            </div>

            <div class="source-tag" id="source">Source: --</div>
        </div>
    </div>

    <script>
        async function analyze() {
            const code = document.getElementById('code').value;
            const ticker = document.getElementById('ticker').value;
            const btn = document.getElementById('btn');
            const loading = document.getElementById('loading');
            const resultBox = document.getElementById('result');
            const errorBox = document.getElementById('error');

            // Reset
            resultBox.style.display = 'none';
            errorBox.style.display = 'none';
            loading.style.display = 'block';
            btn.disabled = true;

            try {
                const res = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({access_code: code, ticker: ticker})
                });
                
                const data = await res.json();
                
                loading.style.display = 'none';
                btn.disabled = false;

                if (data.error) {
                    errorBox.innerText = data.error;
                    errorBox.style.display = 'block';
                    return;
                }

                // Populate UI
                const s = data.valuation_summary;
                document.getElementById('name').innerText = s.company_name;
                document.getElementById('tickerDisplay').innerText = ticker.toUpperCase() + ".SR";
                document.getElementById('price').innerText = s.current_price.toFixed(2) + " SAR";
                document.getElementById('fair').innerText = s.fair_value.toFixed(2) + " SAR";
                
                // Upside
                const up = s.upside_percent;
                const upElem = document.getElementById('upside');
                upElem.innerText = (up > 0 ? "+" : "") + up.toFixed(1) + "%";
                upElem.style.color = up > 0 ? "#2e7d32" : "#c62828";

                // Verdict Badge
                const vBadge = document.getElementById('verdict');
                vBadge.innerText = s.verdict;
                vBadge.className = "verdict-badge " + (s.verdict === "Undervalued" ? "v-green" : (s.verdict === "Overvalued" ? "v-red" : "v-gray"));

                document.getElementById('source').innerText = "Data Source: " + data.source_used;
                resultBox.style.display = 'block';

            } catch (e) {
                loading.style.display = 'none';
                btn.disabled = false;
                errorBox.innerText = "Network Error: " + e.message;
                errorBox.style.display = 'block';
            }
        }
    </script>
    </body>
    </html>
    """

# ==========================================
# 3. THE API LOGIC (Connection Point)
# ==========================================
class StockRequest(BaseModel):
    ticker: str
    access_code: str

@app.post("/analyze")
def analyze_stock(request: StockRequest):
    if request.access_code not in VALID_CODES:
        return {"error": "⛔ Invalid Access Code. Please ask Khaled."}

    fetcher = DataFetcher()
    data = fetcher.fetch(request.ticker)
    
    if not data:
        return {"error": "Could not retrieve data from any source (Yahoo, Alpha Vantage, TwelveData, or Scraper)."}

    # --- VALUATION LOGIC ---
    hist = data["history"]
    info = data["info"]
    
    current_price = hist["Close"].iloc[-1]
    
    # Safe extraction
    eps = info.get("trailingEps")
    if not eps: eps = current_price / 20.0 # Fallback estimation if missing
    
    book_value = info.get("bookValue")
    if not book_value: book_value = current_price / 3.0

    # 3-Model Valuation
    fair_pe = eps * 18.5
    fair_pb = book_value * 3.0
    fair_dcf = current_price * 1.05 # Conservative growth
    
    # Weighted Average
    final_fair_value = (fair_pe * 0.4) + (fair_pb * 0.3) + (fair_dcf * 0.3)
    
    upside = ((final_fair_value - current_price) / current_price) * 100
    
    verdict = "Fairly Valued"
    if upside > 7: verdict = "Undervalued"
    if upside < -7: verdict = "Overvalued"

    return {
        "valuation_summary": {
            "company_name": info.get("longName", f"Saudi Stock {request.ticker}"),
            "fair_value": final_fair_value,
            "current_price": current_price,
            "verdict": verdict,
            "upside_percent": upside
        },
        "source_used": data["source"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
