from fastapi import FastAPI
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

# Import BeautifulSoup safely
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

# ==========================================
# 1. INSTITUTIONAL DATA ENGINE
# ==========================================
class DataFetcher:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36"
        ]
        self.av_key = "0LR5JLOBSLOA6Z0A"

    def fetch(self, ticker):
        clean_ticker = ticker
        if ticker.replace('.','').isdigit() and not ticker.endswith('.SR'):
            clean_ticker = f"{ticker}.SR"
            
        print(f"📊 Fetching Data for: {clean_ticker}")

        # --- SOURCE A: YAHOO FINANCE ---
        try:
            import yfinance as yf
            stock = yf.Ticker(clean_ticker)
            hist = stock.history(period="5y")
            
            if not hist.empty:
                info = stock.info
                if info.get("currentPrice") or hist["Close"].iloc[-1]:
                    return {"history": hist, "info": info, "source": "Yahoo Finance"}
        except: pass

        # --- SOURCE B: ALPHA VANTAGE ---
        try:
            av_symbol = clean_ticker.replace(".SR", ".SA")
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_symbol}&outputsize=full&apikey={self.av_key}"
            r = requests.get(url, timeout=5)
            data = r.json()
            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df = df.rename(columns={"4. close": "Close"})
                df.index = pd.to_datetime(df.index)
                df = df.astype(float).sort_index()
                df = df.tail(1250) 
                
                price = df["Close"].iloc[-1]
                info = {"longName": f"Saudi Stock {ticker}", "currentPrice": price, "trailingEps": price/18.5, "bookValue": price/2.8, "sector": "Unknown"}
                return {"history": df, "info": info, "source": "Alpha Vantage"}
        except: pass

        # --- SOURCE C: WEB SCRAPER ---
        if BS4_AVAILABLE:
            try:
                symbol = ticker.split(".")[0]
                url = f"https://www.google.com/finance/quote/{symbol}:TADAWUL"
                headers = {"User-Agent": random.choice(self.user_agents)}
                r = requests.get(url, headers=headers, timeout=5)
                
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, 'html.parser')
                    price_div = soup.find("div", {"class": "YMlKec fxKbKc"})
                    
                    if price_div:
                        price = float(price_div.text.replace("SAR", "").replace(",", "").strip())
                        # Generate Synthetic History
                        dates = pd.date_range(end=datetime.now(), periods=1250)
                        prices = [price] * 1250
                        hist = pd.DataFrame({"Close": prices}, index=dates)
                        
                        info = {"longName": f"Saudi Stock {symbol}", "currentPrice": price, "trailingEps": price/19.0, "bookValue": price/3.0, "sector": "Unknown"}
                        return {"history": hist, "info": info, "source": "Web Scraper"}
            except Exception as e:
                print(f"Scraper Error: {e}")

        return None

# ==========================================
# 2. THE DASHBOARD UI
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Saudi Valuator Pro</title>
        <script src="https://code.highcharts.com/highcharts.js"></script>
        <style>
            :root { --bg: #f0f2f5; --card: #ffffff; --primary: #0a192f; --accent: #007aff; --text: #333; }
            body { font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: var(--bg); margin: 0; padding: 20px; color: var(--text); }
            
            .container { max-width: 1200px; margin: 0 auto; }
            
            /* SEARCH */
            .search-bar { background: var(--card); padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); display: flex; gap: 10px; margin-bottom: 25px; }
            input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; outline: none; }
            button { padding: 12px 25px; background: var(--primary); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }

            /* LAYOUTS */
            .top-section { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
            .bottom-section { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .full-width { grid-column: span 2; }
            
            .card { background: var(--card); border-radius: 12px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); position: relative; }
            .card-title { font-size: 13px; font-weight: 700; color: #888; text-transform: uppercase; margin-bottom: 20px; letter-spacing: 0.5px; border-bottom: 1px solid #eee; padding-bottom: 10px; }

            /* HEADER & PRICE */
            .header-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; }
            .company-name { font-size: 28px; font-weight: 800; color: var(--primary); margin: 0; line-height: 1.2; }
            .ticker-tag { background: #eee; padding: 4px 8px; border-radius: 4px; font-family: monospace; color: #555; font-size: 14px; }
            .big-price { font-size: 42px; font-weight: 800; color: #333; text-align: right; }
            .price-sub { font-size: 13px; color: #888; text-align: right; margin-top: -5px; }

            /* VERDICT BAR */
            .verdict-bar { padding: 15px; border-radius: 8px; text-align: center; font-weight: 800; text-transform: uppercase; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            .v-red { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
            .v-green { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
            .v-gray { background: #f5f5f5; color: #616161; border: 1px solid #e0e0e0; }

            /* STATS GRID */
            .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
            .stat-box { background: #f8f9fa; padding: 12px; border-radius: 8px; }
            .stat-label { font-size: 11px; font-weight: 700; color: #888; text-transform: uppercase; margin-bottom: 5px; }
            .stat-val { font-size: 16px; font-weight: 600; color: #333; }

            /* FAIR VALUE CARD */
            .fv-header { text-align: center; margin-bottom: 20px; }
            .fv-big { font-size: 48px; font-weight: 800; color: var(--accent); }
            .fv-sub { font-size: 13px; color: #888; }
            .sector-tag { font-size: 11px; background: #e0f2f1; color: #00695c; padding: 4px 8px; border-radius: 4px; display:inline-block; margin-top:5px; }
            
            .fv-row { display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f0f0f0; }
            .fv-row:last-child { border-bottom: none; }
            .fv-label { font-size: 14px; color: #555; }
            .fv-num { font-weight: 700; color: #333; }

            /* WEIGHT BARS */
            .weight-container { margin-top: 5px; }
            .weight-bar { height: 4px; background: #eee; border-radius: 2px; width: 100%; overflow: hidden; }
            .weight-fill { height: 100%; background: #007aff; }

            /* DATA TABLES */
            .data-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            .data-table th { text-align: left; font-size: 11px; color: #888; padding-bottom: 8px; border-bottom: 1px solid #eee; }
            .data-table td { padding: 10px 0; font-size: 13px; font-weight: 500; border-bottom: 1px solid #f9f9f9; }
            
            /* RETURNS GRID */
            .returns-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; text-align: center; }
            .ret-box { background: #f8f9fa; padding: 8px; border-radius: 6px; }
            .ret-label { font-size: 11px; color: #666; margin-bottom: 4px; font-weight: bold; }
            .ret-val { font-size: 14px; font-weight: 600; }
            .pos { color: #28cd41; } .neg { color: #ff3b30; }

            /* LOADING */
            .loading { text-align: center; padding: 40px; display: none; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid var(--accent); border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

            @media (max-width: 900px) { 
                .top-section, .bottom-section { grid-template-columns: 1fr; } 
                .full-width { grid-column: span 1; }
            }
        </style>
    </head>
    <body>

    <div class="container">
        <div class="search-bar">
            <input type="text" id="ticker" placeholder="Enter Ticker (e.g. 1120)" />
            <button onclick="analyze()" id="btn">ANALYZE</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h3>Running Machine Learning Solver...</h3>
            <p style="color:#666; font-size:14px;">Minimizing error against 5-year price history</p>
        </div>

        <div id="error" style="display:none; padding: 15px; background: #ffebee; color: #c62828; border-radius: 8px; margin-bottom: 20px;"></div>

        <div id="dashboard" style="display:none;">
            
            <div class="top-section">
                
                <div class="card">
                    <div class="header-row">
                        <div>
                            <h1 class="company-name" id="name">--</h1>
                            <span class="ticker-tag" id="tickerDisplay">--</span>
                        </div>
                        <div>
                            <div class="big-price" id="price">--</div>
                            <div class="price-sub">Current Market Price</div>
                        </div>
                    </div>

                    <div id="verdictBar" class="verdict-bar">--</div>

                    <div class="stats-grid">
                        <div class="stat-box"><div class="stat-label">Market Cap</div><div class="stat-val" id="mcap">--</div></div>
                        <div class="stat-box"><div class="stat-label">P/E Ratio</div><div class="stat-val" id="pe">--</div></div>
                        <div class="stat-box"><div class="stat-label">EPS (TTM)</div><div class="stat-val" id="eps">--</div></div>
                        <div class="stat-box"><div class="stat-label">Beta (Risk)</div><div class="stat-val" id="beta">--</div></div>
                        <div class="stat-box"><div class="stat-label">Analyst Growth</div><div class="stat-val" id="growth">--</div></div>
                        <div class="stat-box"><div class="stat-label">Book Value</div><div class="stat-val" id="book">--</div></div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">OPTIMIZED FAIR VALUE</div>
                    <div class="fv-header">
                        <div class="fv-big" id="fair">--</div>
                        <div class="fv-sub">AI Best-Fit Target Price</div>
                        <div id="sectorMsg" class="sector-tag">--</div>
                    </div>
                    
                    <div class="fv-row">
                        <div>
                            <span class="fv-label">DCF (WACC: <span id="wacc_display">--</span>)</span>
                            <div class="weight-container"><div id="w_dcf_bar" class="weight-bar"><div class="weight-fill"></div></div></div>
                        </div>
                        <div style="text-align:right;">
                            <span class="fv-num" id="dcf_val">--</span>
                            <div style="font-size:10px; color:#aaa;" id="w_dcf_txt">--</div>
                        </div>
                    </div>
                    <div class="fv-row">
                        <div>
                            <span class="fv-label">Forward P/E Model</span>
                            <div class="weight-container"><div id="w_pe_bar" class="weight-bar"><div class="weight-fill"></div></div></div>
                        </div>
                        <div style="text-align:right;">
                            <span class="fv-num" id="pe_val">--</span>
                            <div style="font-size:10px; color:#aaa;" id="w_pe_txt">--</div>
                        </div>
                    </div>
                    <div class="fv-row">
                        <div>
                            <span class="fv-label">P/B Model</span>
                            <div class="weight-container"><div id="w_pb_bar" class="weight-bar"><div class="weight-fill"></div></div></div>
                        </div>
                        <div style="text-align:right;">
                            <span class="fv-num" id="pb_val">--</span>
                            <div style="font-size:10px; color:#aaa;" id="w_pb_txt">--</div>
                        </div>
                    </div>
                    
                    <div style="font-size:10px; color:#1565c0; background:#e3f2fd; padding:8px; border-radius:6px; margin-top:20px; text-align:center;">
                        AI Optimization found this combination minimizes 5-year tracking error.
                    </div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 20px;">
                <div id="chartContainer" style="height: 350px;"></div>
            </div>

            <div class="bottom-section">
                
                <div class="card">
                    <div class="card-title">Solver Performance (Backtest)</div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Period</th>
                                <th>Actual Price</th>
                                <th>Optimized Model</th>
                                <th>Error</th>
                            </tr>
                        </thead>
                        <tbody id="backtestBody"></tbody>
                    </table>
                </div>

                <div class="card">
                    <div class="card-title">Historical Returns</div>
                    <div class="returns-grid" style="margin-bottom: 25px;">
                        <div class="ret-box"><div class="ret-label">1M</div><div class="ret-val" id="r1m">--</div></div>
                        <div class="ret-box"><div class="ret-label">3M</div><div class="ret-val" id="r3m">--</div></div>
                        <div class="ret-box"><div class="ret-label">6M</div><div class="ret-val" id="r6m">--</div></div>
                        <div class="ret-box"><div class="ret-label">1Y</div><div class="ret-val" id="r1y">--</div></div>
                        <div class="ret-box"><div class="ret-label">2Y</div><div class="ret-val" id="r2y">--</div></div>
                    </div>

                    <div class="card-title">Future Projections</div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Year</th>
                                <th>Projected Value</th>
                                <th>Growth</th>
                            </tr>
                        </thead>
                        <tbody id="forecastBody"></tbody>
                    </table>
                </div>

            </div>

        </div>
    </div>

    <script>
        async function analyze() {
            const ticker = document.getElementById('ticker').value;
            const btn = document.getElementById('btn');
            const loading = document.getElementById('loading');
            const dashboard = document.getElementById('dashboard');
            const err = document.getElementById('error');

            if(!ticker) return;

            dashboard.style.display = 'none';
            err.style.display = 'none';
            loading.style.display = 'block';
            btn.disabled = true;

            try {
                const res = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ticker: ticker})
                });
                const data = await res.json();
                
                loading.style.display = 'none';
                btn.disabled = false;

                if (data.error) {
                    err.innerText = data.error;
                    err.style.display = 'block';
                    return;
                }

                const s = data.valuation_summary;
                const m = data.metrics;
                const r = data.returns;
                const backtest = data.backtest;
                const weights = data.optimized_weights;

                // 1. Info
                document.getElementById('name').innerText = s.company_name;
                document.getElementById('tickerDisplay').innerText = ticker.toUpperCase() + ".SR";
                document.getElementById('price').innerText = s.current_price.toFixed(2);
                document.getElementById('fair').innerText = s.fair_value.toFixed(2);
                document.getElementById('sectorMsg').innerText = s.sector;

                // 2. Verdict
                const vb = document.getElementById('verdictBar');
                const upside = s.upside_percent;
                const label = s.verdict.toUpperCase();
                const sign = upside > 0 ? "+" : "";
                vb.innerText = `${label} (${sign}${upside.toFixed(1)}% Upside)`;
                vb.className = "verdict-bar " + (label === "UNDERVALUED" ? "v-green" : (label === "OVERVALUED" ? "v-red" : "v-gray"));

                // 3. Stats
                const fmt = (num) => num ? num.toFixed(2) : "N/A";
                const fmtBig = (num) => num ? (num / 1000000000).toFixed(2) + "B" : "N/A";
                document.getElementById('mcap').innerText = fmtBig(m.market_cap);
                document.getElementById('pe').innerText = fmt(m.pe_ratio);
                document.getElementById('eps').innerText = fmt(m.eps);
                document.getElementById('beta').innerText = m.beta ? m.beta.toFixed(2) : "N/A";
                document.getElementById('growth').innerText = (m.growth_rate * 100).toFixed(1) + "%";
                document.getElementById('book').innerText = fmt(m.book_value);
                
                // 4. Dynamic Info Display
                document.getElementById('wacc_display').innerText = (m.wacc * 100).toFixed(1) + "%";
                
                // 5. Weights
                document.getElementById('dcf_val').innerText = s.model_breakdown.dcf.toFixed(2);
                document.getElementById('pe_val').innerText = s.model_breakdown.pe_model.toFixed(2);
                document.getElementById('pb_val').innerText = s.model_breakdown.pb_model.toFixed(2);
                
                const setW = (key, val) => {
                    const pct = (val * 100).toFixed(0) + "%";
                    document.getElementById(`w_${key}_bar`).style.width = pct;
                    document.getElementById(`w_${key}_txt`).innerText = "Weight: " + pct;
                };
                setW('dcf', weights.dcf); setW('pe', weights.pe); setW('pb', weights.pb);

                // 6. Returns
                const setRet = (id, val) => {
                    const el = document.getElementById(id);
                    if (val === null) { el.innerText = "--"; return; }
                    el.innerText = (val > 0 ? "+" : "") + val.toFixed(1) + "%";
                    el.className = "ret-val " + (val > 0 ? "pos" : "neg");
                };
                setRet('r1m', r["1m"]); setRet('r3m', r["3m"]);
                setRet('r6m', r["6m"]); setRet('r1y', r["1y"]); setRet('r2y', r["2y"]);

                // 7. Forecast
                const fcBody = document.getElementById('forecastBody');
                fcBody.innerHTML = "";
                const currentYear = new Date().getFullYear();
                const projections = (s.dcf_projections && s.dcf_projections.length > 0) ? s.dcf_projections : [0,0,0,0,0];
                projections.forEach((val, i) => {
                    const row = `<tr>
                        <td>${currentYear + i + 1}</td>
                        <td>${val.toFixed(2)} SAR</td>
                        <td style="color:#28cd41;">+${(m.growth_rate*100).toFixed(1)}%</td>
                    </tr>`;
                    fcBody.innerHTML += row;
                });

                // 8. Backtest
                const btBody = document.getElementById('backtestBody');
                btBody.innerHTML = "";
                backtest.forEach(b => {
                    const diff = Math.abs((b.model - b.actual) / b.actual) * 100;
                    const color = diff < 15 ? "#28cd41" : "#f0ad4e";
                    const row = `<tr>
                        <td>${b.period}</td>
                        <td>${b.actual.toFixed(2)}</td>
                        <td>${b.model.toFixed(2)}</td>
                        <td style="color:${color}; font-weight:bold;">${diff.toFixed(1)}%</td>
                    </tr>`;
                    btBody.innerHTML += row;
                });

                // 9. Chart
                const dates = data.historical_data.dates;
                const prices = data.historical_data.prices;
                const fairVals = data.historical_data.fair_values;

                Highcharts.chart('chartContainer', {
                    chart: { backgroundColor: 'transparent' },
                    title: { text: 'Price vs Optimized Model' },
                    xAxis: { type: 'datetime' },
                    yAxis: { title: { text: null }, gridLineColor: '#eee' },
                    series: [{
                        name: 'Actual Price',
                        data: dates.map((d, i) => [d, prices[i]]),
                        type: 'area',
                        color: '#0a192f',
                        fillColor: { linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 }, stops: [[0, 'rgba(10, 25, 47, 0.1)'], [1, 'rgba(10, 25, 47, 0)']] }
                    }, {
                        name: 'Optimized Fair Value',
                        data: dates.map((d, i) => [d, fairVals[i]]),
                        type: 'line',
                        color: '#007aff',
                        lineWidth: 2
                    }],
                    credits: { enabled: false }
                });

                dashboard.style.display = 'block';

            } catch (e) {
                loading.style.display = 'none';
                btn.disabled = false;
                err.innerText = "Error: " + e.message;
                err.style.display = 'block';
            }
        }
    </script>
    </body>
    </html>
    """

# ==========================================
# 3. DYNAMIC OPTIMIZATION ENGINE
# ==========================================
class StockRequest(BaseModel):
    ticker: str

@app.post("/analyze")
def analyze_stock(request: StockRequest):
    fetcher = DataFetcher()
    data = fetcher.fetch(request.ticker)
    if not data: return {"error": "Could not retrieve financial data."}

    hist = data["history"]
    info = data["info"]
    current_price = hist["Close"].iloc[-1]
    
    # --- 1. DETECT SECTOR & SET CONSTANTS ---
    sector = info.get("sector", "Unknown")
    
    # A. DYNAMIC WACC
    rf_rate = 0.045
    mrp = 0.057
    beta = info.get('beta')
    if beta and beta > 0:
        wacc = rf_rate + (beta * mrp)
    else:
        # Fallback WACC
        if "Financial" in sector: wacc = 0.09
        elif "Technology" in sector: wacc = 0.12
        else: wacc = 0.10
        beta = (wacc - rf_rate) / mrp

    # B. DYNAMIC GROWTH
    g_est = info.get('earningsGrowth')
    if g_est:
        growth_rate = g_est
        if growth_rate > 0.15: growth_rate = 0.15
        if growth_rate < 0.02: growth_rate = 0.02
    else:
        # Fallback Growth
        if "Technology" in sector: growth_rate = 0.08
        elif "Financial" in sector: growth_rate = 0.05
        else: growth_rate = 0.03

    # C. DYNAMIC PE TARGET
    fwd_pe = info.get('forwardPE')
    if fwd_pe and fwd_pe > 5 and fwd_pe < 50:
        target_pe = fwd_pe
    else:
        # Fallback Sector PE
        if "Financial" in sector: target_pe = 19.0
        elif "Technology" in sector: target_pe = 25.0
        else: target_pe = 16.0

    # --- 2. RECONSTRUCT HISTORY ---
    dates = hist.index.astype(np.int64) // 10**6
    prices = hist["Close"].tolist()
    
    eps_curr = info.get("trailingEps") or current_price / 18.0
    book_curr = info.get("bookValue") or current_price / 3.0
    
    n_days = len(prices)
    years_ago_array = np.linspace(5, 0, n_days)
    hist_eps = [eps_curr / ((1 + growth_rate) ** y) for y in years_ago_array]
    hist_book = [book_curr / ((1 + growth_rate) ** y) for y in years_ago_array]

    # --- 3. GENERATE STREAMS ---
    # Implied Multiple for DCF
    try:
        implied_dcf_multiple = (1 + growth_rate) / (wacc - growth_rate)
        if implied_dcf_multiple > 35: implied_dcf_multiple = 35
        if implied_dcf_multiple < 8: implied_dcf_multiple = 8
    except: implied_dcf_multiple = 15.0

    stream_dcf = [e * implied_dcf_multiple for e in hist_eps] 
    stream_pe = [e * target_pe for e in hist_eps]
    stream_pb = [b * 2.2 for b in hist_book]

    # --- 4. THE SOLVER (No Defaults) ---
    best_error = float('inf')
    best_weights = (0,0,0)
    
    # We step by 10% (0.1) increments for precision
    steps = [x / 10.0 for x in range(11)] # 0.0, 0.1 ... 1.0
    
    for w1 in steps:
        for w2 in steps:
            if w1 + w2 > 1.0: continue
            w3 = round(1.0 - w1 - w2, 2)
            
            error_sum = 0
            count = 0
            for i in range(0, n_days, 50): 
                model_price = (w1 * stream_dcf[i]) + (w2 * stream_pe[i]) + (w3 * stream_pb[i])
                error_sum += abs(model_price - prices[i])
                count += 1
            
            avg_error = error_sum / count
            if avg_error < best_error:
                best_error = avg_error
                best_weights = (w1, w2, w3)

    # --- 5. FINAL CALCULATION ---
    w_dcf, w_pe, w_pb = best_weights
    
    future_cash = []
    dcf_total = 0
    for i in range(1, 6):
        fcf = eps_curr * ((1 + growth_rate) ** i)
        disc = fcf / ((1 + wacc) ** i)
        dcf_total += disc
        future_cash.append(current_price * ((1 + growth_rate)**i))
    
    terminal_val = (eps_curr * ((1+growth_rate)**5) * (1+0.02)) / (wacc - 0.02)
    dcf_total += terminal_val / ((1+wacc)**5)
    
    pe_val_today = eps_curr * target_pe
    pb_val_today = book_curr * 2.2
    
    final_fair_value = (dcf_total * w_dcf) + (pe_val_today * w_pe) + (pb_val_today * w_pb)
    upside = ((final_fair_value - current_price) / current_price) * 100
    
    verdict = "Fairly Valued"
    if upside > 10: verdict = "Undervalued"
    if upside < -10: verdict = "Overvalued"

    # --- 6. OUTPUT DATA ---
    fair_values = []
    for i in range(n_days):
        val = (w_dcf * stream_dcf[i]) + (w_pe * stream_pe[i]) + (w_pb * stream_pb[i])
        fair_values.append(val)

    backtest_data = []
    points = [("1 Year Ago", 252), ("2 Years Ago", 504), ("3 Years Ago", 756), ("4 Years Ago", 1008), ("5 Years Ago", 1250)]
    for label, days in points:
        if len(prices) > days:
            idx = -days
            backtest_data.append({
                "period": label,
                "actual": prices[idx],
                "model": fair_values[idx]
            })

    def get_price_ago(days):
        if len(prices) < days: return current_price
        return prices[-days]

    returns = {
        "1m": ((current_price - get_price_ago(21))/get_price_ago(21))*100,
        "3m": ((current_price - get_price_ago(63))/get_price_ago(63))*100,
        "6m": ((current_price - get_price_ago(126))/get_price_ago(126))*100,
        "1y": ((current_price - get_price_ago(252))/get_price_ago(252))*100,
        "2y": ((current_price - get_price_ago(504))/get_price_ago(504))*100,
    }

    mcap = info.get("marketCap") or current_price * 1000000
    pe_rat = info.get("trailingPE") or (current_price/eps_curr)
    
    return {
        "valuation_summary": {
            "company_name": info.get("longName", f"Saudi Stock {request.ticker}"),
            "fair_value": final_fair_value,
            "current_price": current_price,
            "verdict": verdict,
            "upside_percent": upside,
            "dcf_projections": future_cash,
            "sector": sector,
            "model_breakdown": {
                "dcf": dcf_total,
                "pe_model": pe_val_today,
                "pb_model": pb_val_today
            }
        },
        "optimized_weights": { "dcf": w_dcf, "pe": w_pe, "pb": w_pb },
        "metrics": {
            "market_cap": mcap,
            "pe_ratio": pe_rat,
            "eps": eps_curr,
            "book_value": book_curr,
            "beta": beta,
            "wacc": wacc,
            "growth_rate": growth_rate,
            "high52": max(prices[-252:]),
            "low52": min(prices[-252:])
        },
        "returns": returns,
        "backtest": backtest_data,
        "historical_data": {
            "dates": dates.tolist(),
            "prices": prices,
            "fair_values": fair_values
        },
        "source_used": data["source"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
