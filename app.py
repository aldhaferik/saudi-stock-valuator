from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Optional BS4 (not used in this implementation, kept for compatibility)
try:
    from bs4 import BeautifulSoup  # noqa: F401
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

# =========================================================
# 0) GLOBAL SETTINGS (no hard-coded finance assumptions)
# =========================================================
TASI_TICKER = "^TASI.SR"  # Tadawul All Shares Index on Yahoo Finance :contentReference[oaicite:1]{index=1}
ECONDB_RF_SERIES = "Y10YDSA"  # Saudi long-term yield series on EconDB :contentReference[oaicite:2]{index=2}

DEFAULT_HISTORY_PERIOD = "5y"
BETA_LOOKBACK_DAYS = 252 * 2  # 2 years daily regression (data-driven, not a finance constant)
MARKET_RETURN_LOOKBACK_DAYS = 252 * 5  # 5 years for market expected return
FORECAST_YEARS = 5

# =========================================================
# 1) DATA FETCHING
# =========================================================
class DataFetcher:
    """
    Fetches:
    - price history (ticker)
    - financial statements (income statement, balance sheet, cash flow)
    - shares outstanding / market cap when available
    - market index history (TASI) for beta and market return
    - risk-free rate from TradingEconomics (free)
    """

    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36",
        ]

    def _headers(self):
        return {"User-Agent": np.random.choice(self.user_agents)}

    def fetch_prices_yahoo(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> pd.DataFrame:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError(f"No price history returned for {ticker}.")
        return hist

    def fetch_statements_yahoo(self, ticker: str):
        """
        Uses yfinance's statement tables. Availability varies by ticker.
        Returns dict of DataFrames (annual), plus info dict.
        """
        import yfinance as yf
        stock = yf.Ticker(ticker)

        try:
            info = stock.info or {}
        except Exception:
            info = {}

        try:
            fin = stock.financials
        except Exception:
            fin = None

        try:
            bs = stock.balance_sheet
        except Exception:
            bs = None

        try:
            cf = stock.cashflow
        except Exception:
            cf = None

        return {
            "info": info,
            "financials": fin,
            "balance_sheet": bs,
            "cashflow": cf,
        }

    def fetch_saudi_risk_free_tradingeconomics(self) -> float:
    url = "https://api.tradingeconomics.com/markets/bond"
    params = {
        "c": "guest:guest",
        "type": "10Y",
        "f": "json",
    }

    r = requests.get(url, params=params, headers=self._headers(), timeout=10)
    if r.status_code != 200:
        raise ValueError(f"TradingEconomics bond API failed ({r.status_code})")

    data = r.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected TradingEconomics response format")

    ten_y_percent = None
    for item in data:
        country = str(item.get("Country", "")).lower()
        name = str(item.get("Name", "")).lower()
        if country == "saudi arabia" or "saudi" in name:
            y = item.get("Last") or item.get("Close")
            if y is not None:
                ten_y_percent = float(y)
                break

    if ten_y_percent is None:
        raise ValueError("Saudi Arabia 10Y yield not found")

    # Convert percent to decimal
    rf = ten_y_percent / 100.0

    # sanity guard only
    if rf <= 0 or rf > 0.50:
        raise ValueError(f"Risk-free rate out of bounds after parsing: {rf}")

    return rf


# =========================================================
# 2) UI (unchanged)
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Your original HTML is preserved. The API response structure remains compatible.
    # The DCF weight is always 100% in this strict mode.
    return """<YOUR_HTML_UNCHANGED>""".replace(
        "<YOUR_HTML_UNCHANGED>",
        # NOTE: Paste your original HTML exactly here. To keep this response readable,
        # I'm reusing your same HTML string below verbatim.
        """
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
            .search-bar { background: var(--card); padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); display: flex; gap: 10px; margin-bottom: 25px; }
            input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; outline: none; }
            button { padding: 12px 25px; background: var(--primary); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
            .top-section { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
            .bottom-section { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .full-width { grid-column: span 2; }
            .card { background: var(--card); border-radius: 12px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); position: relative; }
            .card-title { font-size: 13px; font-weight: 700; color: #888; text-transform: uppercase; margin-bottom: 20px; letter-spacing: 0.5px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            .header-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; }
            .company-name { font-size: 28px; font-weight: 800; color: var(--primary); margin: 0; line-height: 1.2; }
            .ticker-tag { background: #eee; padding: 4px 8px; border-radius: 4px; font-family: monospace; color: #555; font-size: 14px; }
            .big-price { font-size: 42px; font-weight: 800; color: #333; text-align: right; }
            .price-sub { font-size: 13px; color: #888; text-align: right; margin-top: -5px; }
            .verdict-bar { padding: 15px; border-radius: 8px; text-align: center; font-weight: 800; text-transform: uppercase; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            .v-red { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
            .v-green { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
            .v-gray { background: #f5f5f5; color: #616161; border: 1px solid #e0e0e0; }
            .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
            .stat-box { background: #f8f9fa; padding: 12px; border-radius: 8px; }
            .stat-label { font-size: 11px; font-weight: 700; color: #888; text-transform: uppercase; margin-bottom: 5px; }
            .stat-val { font-size: 16px; font-weight: 600; color: #333; }
            .fv-header { text-align: center; margin-bottom: 20px; }
            .fv-big { font-size: 48px; font-weight: 800; color: var(--accent); }
            .fv-sub { font-size: 13px; color: #888; }
            .sector-tag { font-size: 11px; background: #e0f2f1; color: #00695c; padding: 4px 8px; border-radius: 4px; display:inline-block; margin-top:5px; }
            .dyn-badge { font-size: 9px; background: #333; color: #fff; padding: 2px 5px; border-radius: 3px; margin-left: 5px; vertical-align: middle; }
            .fv-row { display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f0f0f0; }
            .fv-row:last-child { border-bottom: none; }
            .fv-label { font-size: 14px; color: #555; }
            .fv-num { font-weight: 700; color: #333; }
            .weight-container { margin-top: 5px; }
            .weight-bar { height: 4px; background: #eee; border-radius: 2px; width: 100%; overflow: hidden; }
            .weight-fill { height: 100%; background: #007aff; }
            .data-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            .data-table th { text-align: left; font-size: 11px; color: #888; padding-bottom: 8px; border-bottom: 1px solid #eee; }
            .data-table td { padding: 10px 0; font-size: 13px; font-weight: 500; border-bottom: 1px solid #f9f9f9; }
            .returns-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; text-align: center; }
            .ret-box { background: #f8f9fa; padding: 8px; border-radius: 6px; }
            .ret-label { font-size: 11px; color: #666; margin-bottom: 4px; font-weight: bold; }
            .ret-val { font-size: 14px; font-weight: 600; }
            .pos { color: #28cd41; } .neg { color: #ff3b30; }
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
            <h3>Calculating Intrinsic Value...</h3>
            <p style="color:#666; font-size:14px;">Deriving WACC, FCFF, Beta and Growth from Real Data</p>
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
                        <div class="stat-box"><div class="stat-label">Beta <span id="beta_tag"></span></div><div class="stat-val" id="beta">--</div></div>
                        <div class="stat-box"><div class="stat-label">Growth <span id="growth_tag"></span></div><div class="stat-val" id="growth">--</div></div>
                        <div class="stat-box"><div class="stat-label">Book Value</div><div class="stat-val" id="book">--</div></div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">DCF FAIR VALUE (STRICT)</div>
                    <div class="fv-header">
                        <div class="fv-big" id="fair">--</div>
                        <div class="fv-sub">Statement-derived intrinsic value</div>
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
                            <span class="fv-label">P/B Asset Model</span>
                            <div class="weight-container"><div id="w_pb_bar" class="weight-bar"><div class="weight-fill"></div></div></div>
                        </div>
                        <div style="text-align:right;">
                            <span class="fv-num" id="pb_val">--</span>
                            <div style="font-size:10px; color:#aaa;" id="w_pb_txt">--</div>
                        </div>
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
                const fmt = (num) => (num === null || num === undefined) ? "N/A" : Number(num).toFixed(2);
                const fmtBig = (num) => (num === null || num === undefined) ? "N/A" : (Number(num) / 1000000000).toFixed(2) + "B";
                document.getElementById('mcap').innerText = fmtBig(m.market_cap);
                document.getElementById('pe').innerText = fmt(m.pe_ratio);
                document.getElementById('eps').innerText = fmt(m.eps);
                document.getElementById('beta').innerText = (m.beta === null || m.beta === undefined) ? "N/A" : Number(m.beta).toFixed(2);
                document.getElementById('growth').innerText = (m.growth_rate === null || m.growth_rate === undefined) ? "N/A" : (Number(m.growth_rate) * 100).toFixed(1) + "%";
                document.getElementById('book').innerText = fmt(m.book_value);

                document.getElementById('wacc_display').innerText = (Number(m.wacc) * 100).toFixed(1) + "%";

                // Dynamic Tags
                document.getElementById('beta_tag').innerHTML = '<span class="dyn-badge">REGRESSION</span>';
                document.getElementById('growth_tag').innerHTML = '<span class="dyn-badge">FUNDAMENTAL</span>';

                // 4. Weights (strict DCF)
                document.getElementById('dcf_val').innerText = Number(s.model_breakdown.dcf).toFixed(2);
                document.getElementById('pe_val').innerText = Number(s.model_breakdown.pe_model).toFixed(2);
                document.getElementById('pb_val').innerText = Number(s.model_breakdown.pb_model).toFixed(2);

                const setW = (key, val) => {
                    const pct = (val * 100).toFixed(0) + "%";
                    const bar = document.getElementById(`w_${key}_bar`);
                    const fill = bar.querySelector(".weight-fill");
                    fill.style.width = pct;
                    document.getElementById(`w_${key}_txt`).innerText = "Weight: " + pct;
                };
                setW('dcf', weights.dcf); setW('pe', weights.pe); setW('pb', weights.pb);

                // 5. Returns
                const setRet = (id, val) => {
                    const el = document.getElementById(id);
                    if (val === null || val === undefined) { el.innerText = "--"; return; }
                    el.innerText = (val > 0 ? "+" : "") + Number(val).toFixed(1) + "%";
                    el.className = "ret-val " + (val > 0 ? "pos" : "neg");
                };
                setRet('r1m', r["1m"]); setRet('r3m', r["3m"]);
                setRet('r6m', r["6m"]); setRet('r1y', r["1y"]); setRet('r2y', r["2y"]);

                // 6. Forecast
                const fcBody = document.getElementById('forecastBody');
                fcBody.innerHTML = "";
                const currentYear = new Date().getFullYear();
                const projections = (s.dcf_projections && s.dcf_projections.length > 0) ? s.dcf_projections : [];
                projections.forEach((val, i) => {
                    const row = `<tr>
                        <td>${currentYear + i + 1}</td>
                        <td>${Number(val).toFixed(2)} SAR</td>
                        <td style="color:#28cd41;">+${(Number(m.growth_rate)*100).toFixed(1)}%</td>
                    </tr>`;
                    fcBody.innerHTML += row;
                });

                // 7. Backtest (we show constant intrinsic value vs historical price as a simple sanity check)
                const btBody = document.getElementById('backtestBody');
                btBody.innerHTML = "";
                backtest.forEach(b => {
                    const diff = Math.abs((b.model - b.actual) / b.actual) * 100;
                    const color = diff < 15 ? "#28cd41" : "#f0ad4e";
                    const row = `<tr>
                        <td>${b.period}</td>
                        <td>${Number(b.actual).toFixed(2)}</td>
                        <td>${Number(b.model).toFixed(2)}</td>
                        <td style="color:${color}; font-weight:bold;">${diff.toFixed(1)}%</td>
                    </tr>`;
                    btBody.innerHTML += row;
                });

                // 8. Chart
                const dates = data.historical_data.dates;
                const prices = data.historical_data.prices;
                const fairVals = data.historical_data.fair_values;

                Highcharts.chart('chartContainer', {
                    chart: { backgroundColor: 'transparent' },
                    title: { text: 'Price vs Intrinsic Value (DCF)' },
                    xAxis: { type: 'datetime' },
                    yAxis: { title: { text: null }, gridLineColor: '#eee' },
                    series: [{
                        name: 'Actual Price',
                        data: dates.map((d, i) => [d, prices[i]]),
                        type: 'area',
                        color: '#0a192f',
                        fillColor: { linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 }, stops: [[0, 'rgba(10, 25, 47, 0.1)'], [1, 'rgba(10, 25, 47, 0)']] }
                    }, {
                        name: 'Intrinsic Value',
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
    )

# =========================================================
# 3) VALUATION ENGINE (STRICT WACC + FCFF)
# =========================================================
class StockRequest(BaseModel):
    ticker: str

def _clean_saudi_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.replace(".", "").isdigit() and not t.endswith(".SR"):
        return f"{t}.SR"
    return t

def _safe_get_line(df: pd.DataFrame, possible_names: list[str], col) -> float | None:
    """
    df is a statement table with index as line items, columns as periods.
    possible_names: list of candidate row labels.
    col: the column key (most recent period).
    """
    if df is None or df.empty:
        return None
    for name in possible_names:
        if name in df.index:
            val = df.loc[name, col]
            if pd.isna(val):
                continue
            try:
                return float(val)
            except Exception:
                continue
    return None

def _most_recent_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    # yfinance columns are typically Timestamps (period end dates). Take the first (most recent) by default.
    # Some tables come already sorted descending; we sort just in case.
    cols = list(df.columns)
    try:
        cols_sorted = sorted(cols, reverse=True)
        return cols_sorted[0]
    except Exception:
        return cols[0]

def _second_most_recent_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    try:
        cols_sorted = sorted(cols, reverse=True)
        return cols_sorted[1] if len(cols_sorted) > 1 else None
    except Exception:
        return cols[1] if len(cols) > 1 else None

def _annualized_geo_mean_return(prices: pd.Series, periods_per_year: int = 252) -> float:
    prices = prices.dropna()
    if len(prices) < periods_per_year + 1:
        raise ValueError("Not enough market history to estimate expected market return.")
    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    n_periods = len(prices) - 1
    years = n_periods / periods_per_year
    if start <= 0 or end <= 0 or years <= 0:
        raise ValueError("Invalid market price series for return estimation.")
    return (end / start) ** (1.0 / years) - 1.0

def _beta_regression(stock_prices: pd.Series, market_prices: pd.Series) -> float:
    # Align
    df = pd.DataFrame({"s": stock_prices, "m": market_prices}).dropna()
    if len(df) < 100:
        raise ValueError("Not enough overlapping history to estimate beta.")
    rs = np.log(df["s"]).diff().dropna()
    rm = np.log(df["m"]).diff().dropna()
    aligned = pd.DataFrame({"rs": rs, "rm": rm}).dropna()
    if len(aligned) < 100:
        raise ValueError("Not enough return observations to estimate beta.")
    cov = np.cov(aligned["rs"], aligned["rm"], ddof=1)[0, 1]
    var = np.var(aligned["rm"], ddof=1)
    if var <= 0:
        raise ValueError("Market variance is zero; cannot compute beta.")
    return float(cov / var)

@app.post("/analyze")
def analyze_stock(request: StockRequest):
    fetcher = DataFetcher()
    ticker = _clean_saudi_ticker(request.ticker)

    # 1) Prices
    try:
        hist = fetcher.fetch_prices_yahoo(ticker, period=DEFAULT_HISTORY_PERIOD)
        mkt = fetcher.fetch_prices_yahoo(TASI_TICKER, period=DEFAULT_HISTORY_PERIOD)
    except Exception as e:
        return {"error": f"Price data error: {str(e)}"}

    if "Close" not in hist.columns or hist["Close"].dropna().empty:
        return {"error": "No valid Close prices for the stock."}
    if "Close" not in mkt.columns or mkt["Close"].dropna().empty:
        return {"error": "No valid Close prices for TASI market index (^TASI.SR)."}  # :contentReference[oaicite:4]{index=4}

    current_price = float(hist["Close"].iloc[-1])
    dates_ms = (hist.index.astype(np.int64) // 10**6).tolist()
    prices = hist["Close"].astype(float).tolist()

    # 2) Statements + info
    try:
        pack = fetcher.fetch_statements_yahoo(ticker)
        info = pack["info"] or {}
        fin = pack["financials"]
        bs = pack["balance_sheet"]
        cf = pack["cashflow"]
    except Exception as e:
        return {"error": f"Statement data error: {str(e)}"}

    # Basic identifiers
    company_name = info.get("longName") or f"Saudi Stock {request.ticker}"
    sector = (info.get("sector") or "Unknown").title()

    # Shares / market cap
    shares = info.get("sharesOutstanding")
    mcap = info.get("marketCap")
    if mcap is None and shares is not None:
        try:
            mcap = float(shares) * current_price
        except Exception:
            mcap = None
    if mcap is None:
        return {"error": "Missing market cap (or shares outstanding) from data source. Cannot compute WACC without E."}

    E = float(mcap)

    # 3) Risk-free rate (EconDB)
    try:
        rf = fetcher.fetch_saudi_risk_free_tradingeconomics()
    except Exception as e:
        return {
        "error": f"Could not fetch Saudi risk-free rate from TradingEconomics: {str(e)}"
    }

    # 4) Expected market return and ERP from historical TASI
    try:
        mkt_close = mkt["Close"].astype(float).dropna().tail(MARKET_RETURN_LOOKBACK_DAYS)
        rm_exp = _annualized_geo_mean_return(mkt_close)
        erp = rm_exp - rf
    except Exception as e:
        return {"error": f"Could not compute market expected return/ERP from TASI history: {str(e)}"}  # :contentReference[oaicite:6]{index=6}

    # 5) Beta from regression (stock vs TASI)
    try:
        s_close = hist["Close"].astype(float).dropna().tail(BETA_LOOKBACK_DAYS)
        m_close = mkt["Close"].astype(float).dropna().tail(BETA_LOOKBACK_DAYS)
        beta = _beta_regression(s_close, m_close)
    except Exception as e:
        return {"error": f"Could not compute regression beta vs TASI: {str(e)}"}  # :contentReference[oaicite:7]{index=7}

    # 6) Cost of equity
    Re = rf + beta * erp

    # 7) Extract statement line items (annual, most recent + prior)
    fin_col = _most_recent_col(fin)
    fin_col_prev = _second_most_recent_col(fin)

    bs_col = _most_recent_col(bs)
    bs_col_prev = _second_most_recent_col(bs)

    cf_col = _most_recent_col(cf)
    cf_col_prev = _second_most_recent_col(cf)

    if fin_col is None or bs_col is None or cf_col is None:
        return {"error": "Missing annual financial statements (income/balance/cashflow). Cannot compute FCFF/WACC."}

    # EBIT
    EBIT = _safe_get_line(fin, ["Ebit", "EBIT", "Operating Income"], fin_col)
    if EBIT is None:
        return {"error": "Missing EBIT / Operating Income in income statement. Cannot compute FCFF."}

    # Pretax and tax expense for T
    pretax = _safe_get_line(fin, ["Pretax Income", "Income Before Tax"], fin_col)
    tax_exp = _safe_get_line(fin, ["Tax Provision", "Income Tax Expense"], fin_col)

    T = 0.0
    if pretax is not None and tax_exp is not None and pretax > 0:
        T = float(tax_exp) / float(pretax)
        # explicit bounding to prevent nonsense from statement noise
        T = max(0.0, min(T, 0.35))

    # Interest expense
    interest_exp = _safe_get_line(fin, ["Interest Expense", "Interest Expense Non Operating"], fin_col)

    # Debt and cash
    st_debt = _safe_get_line(bs, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], bs_col) or 0.0
    lt_debt = _safe_get_line(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], bs_col) or 0.0
    D = float(st_debt) + float(lt_debt)

    cash = _safe_get_line(bs, ["Cash", "Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments"], bs_col) or 0.0
    cash = float(cash)

    # Cost of debt Rd from interest expense / avg debt
    Rd = 0.0
    if D > 0:
        if interest_exp is None:
            return {"error": "Company has debt but missing interest expense. Cannot compute cost of debt Rd."}
        # average debt requires previous year debt
        if bs_col_prev is None:
            return {"error": "Need at least 2 years of balance sheet to compute average debt for Rd."}
        st_debt_prev = _safe_get_line(bs, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], bs_col_prev) or 0.0
        lt_debt_prev = _safe_get_line(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], bs_col_prev) or 0.0
        D_prev = float(st_debt_prev) + float(lt_debt_prev)
        avg_D = (D + D_prev) / 2.0
        if avg_D <= 0:
            return {"error": "Average debt computed as zero/non-positive; cannot compute Rd."}
        Rd = float(interest_exp) / avg_D
        if Rd <= 0 or Rd > 0.50:
            return {"error": f"Computed Rd out of sanity bounds: {Rd}. Check interest expense / debt inputs."}

    # 8) WACC
    total_cap = D + E
    if total_cap <= 0:
        return {"error": "Invalid capital structure: D+E <= 0."}
    wE = E / total_cap
    wD = D / total_cap
    WACC = wE * Re + wD * Rd * (1.0 - T)

    # 9) FCFF (annual base)
    # D&A
    DA = _safe_get_line(cf, ["Depreciation", "Depreciation And Amortization", "Depreciation & Amortization"], cf_col)
    if DA is None:
        DA = _safe_get_line(fin, ["Reconciled Depreciation", "Depreciation And Amortization"], fin_col)
    if DA is None:
        return {"error": "Missing depreciation & amortization (D&A). Cannot compute FCFF."}
    DA = float(DA)

    # CapEx (cash flow often records as negative)
    CapEx = _safe_get_line(cf, ["Capital Expenditures", "Capital Expenditure"], cf_col)
    if CapEx is None:
        return {"error": "Missing CapEx in cash flow. Cannot compute FCFF."}
    CapEx = float(CapEx)

    # ΔWC: compute Net Working Capital excluding cash, and excluding debt from current liabilities where possible
    def net_working_capital(bs_df: pd.DataFrame, col):
        tca = _safe_get_line(bs_df, ["Total Current Assets"], col)
        tcl = _safe_get_line(bs_df, ["Total Current Liabilities"], col)
        if tca is None or tcl is None:
            return None
        cash_local = _safe_get_line(bs_df, ["Cash", "Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments"], col) or 0.0
        st_debt_local = _safe_get_line(bs_df, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], col) or 0.0
        # NWC ≈ (CA - Cash) - (CL - ST Debt)
        return (float(tca) - float(cash_local)) - (float(tcl) - float(st_debt_local))

    if bs_col_prev is None:
        return {"error": "Need at least 2 years of balance sheet to compute ΔWC."}

    NWC_now = net_working_capital(bs, bs_col)
    NWC_prev = net_working_capital(bs, bs_col_prev)
    if NWC_now is None or NWC_prev is None:
        return {"error": "Missing current assets/liabilities needed for ΔWC. Cannot compute FCFF."}
    dWC = float(NWC_now - NWC_prev)

    # NOPAT
    NOPAT = float(EBIT) * (1.0 - T)
    if NOPAT <= 0:
        return {"error": "NOPAT is non-positive. This strict model won’t project FCFF from negative NOPAT."}

    # FCFF base
    # Note: CapEx is usually negative on cashflow; FCFF uses -CapEx, so we subtract CapEx directly if CapEx is negative.
    # To keep sign-consistent:
    # If CapEx is negative (cash outflow), then -CapEx is positive reduction in FCFF calculation: NOPAT + DA - (abs capex) ...
    capex_outflow = -CapEx if CapEx < 0 else CapEx
    FCFF0 = NOPAT + DA - capex_outflow - dWC

    # 10) Fundamental growth: g = ROIC * reinvestment rate
    # Invested Capital ≈ Equity + Debt - Cash
    invested_capital = (E + D - cash)
    if invested_capital <= 0:
        return {"error": "Invested capital computed as non-positive. Cannot compute ROIC/growth."}
    ROIC = NOPAT / invested_capital

    reinvestment = (capex_outflow - DA + dWC) / NOPAT
    g = ROIC * reinvestment

    # Strict sanity: if growth is absurd due to statement quirks, we stop instead of silently clipping.
    if not np.isfinite(g) or g < -0.50 or g > 0.50:
        return {"error": f"Computed growth g out of sanity bounds: {g}. Check ROIC/reinvestment inputs."}

    # 11) Terminal growth: data-driven cap = long-run market CAGR (TASI)
    # This avoids hard-coding GDP/inflation while still preventing g_term > WACC explosions.
    try:
        g_mkt = _annualized_geo_mean_return(mkt_close)
    except Exception:
        g_mkt = rm_exp  # fallback to already computed market expected return (still data-derived)

    g_term = min(g, g_mkt)
    if WACC <= g_term:
        return {"error": f"WACC ({WACC:.4f}) <= terminal growth ({g_term:.4f}). DCF undefined. Check inputs."}

    # 12) DCF valuation
    fcff_forecast = []
    pv_sum = 0.0
    for i in range(1, FORECAST_YEARS + 1):
        fcff_i = FCFF0 * ((1.0 + g) ** i)
        fcff_forecast.append(float(fcff_i))
        pv_sum += fcff_i / ((1.0 + WACC) ** i)

    TV = (fcff_forecast[-1] * (1.0 + g_term)) / (WACC - g_term)
    PV_TV = TV / ((1.0 + WACC) ** FORECAST_YEARS)
    EV = pv_sum + PV_TV

    # Equity value = EV - Net Debt = EV - (Debt - Cash)
    equity_value = EV - (D - cash)

    if shares is None:
        return {"error": "Missing sharesOutstanding. Cannot convert equity value to per-share fair value."}
    try:
        shares = float(shares)
        if shares <= 0:
            raise ValueError("sharesOutstanding <= 0")
    except Exception:
        return {"error": "Invalid sharesOutstanding. Cannot compute per-share fair value."}

    fair_value = equity_value / shares
    upside = ((fair_value - current_price) / current_price) * 100.0

    verdict = "Fairly Valued"
    if upside > 10:
        verdict = "Undervalued"
    elif upside < -10:
        verdict = "Overvalued"

    # For UI compatibility: show PE/PB as diagnostics (not used)
    eps_curr = info.get("trailingEps")
    book_curr = info.get("bookValue")
    pe_rat = info.get("trailingPE")
    if eps_curr is None and pe_rat is not None and pe_rat != 0:
        eps_curr = current_price / pe_rat
    if eps_curr is None:
        eps_curr = np.nan
    if book_curr is None:
        book_curr = np.nan

    # Chart: constant intrinsic value line (today’s per-share intrinsic value) vs price
    fair_values = [float(fair_value)] * len(prices)

    # Backtest points: compare historical price to today's intrinsic (simple visual sanity check)
    backtest_data = []
    points = [("1 Year Ago", 252), ("2 Years Ago", 504), ("3 Years Ago", 756), ("4 Years Ago", 1008), ("5 Years Ago", 1250)]
    for label, days in points:
        if len(prices) > days:
            idx = -days
            backtest_data.append({"period": label, "actual": float(prices[idx]), "model": float(fair_value)})

    def get_price_ago(days):
        if len(prices) < days:
            return current_price
        return float(prices[-days])

    returns = {
        "1m": ((current_price - get_price_ago(21)) / get_price_ago(21)) * 100.0,
        "3m": ((current_price - get_price_ago(63)) / get_price_ago(63)) * 100.0,
        "6m": ((current_price - get_price_ago(126)) / get_price_ago(126)) * 100.0,
        "1y": ((current_price - get_price_ago(252)) / get_price_ago(252)) * 100.0,
        "2y": ((current_price - get_price_ago(504)) / get_price_ago(504)) * 100.0,
    }

    # strict mode weights
    w_dcf, w_pe, w_pb = 1.0, 0.0, 0.0

    return {
        "valuation_summary": {
            "company_name": company_name,
            "fair_value": float(fair_value),
            "current_price": float(current_price),
            "verdict": verdict,
            "upside_percent": float(upside),
            "dcf_projections": [float(x) for x in fcff_forecast],
            "sector": sector,
            "model_breakdown": {
                "dcf": float(fair_value),  # per-share intrinsic
                "pe_model": float("nan"),
                "pb_model": float("nan"),
            },
        },
        "optimized_weights": {"dcf": w_dcf, "pe": w_pe, "pb": w_pb},
        "metrics": {
            "market_cap": float(E),
            "pe_ratio": None if pe_rat is None else float(pe_rat),
            "eps": None if (eps_curr is None or (isinstance(eps_curr, float) and np.isnan(eps_curr))) else float(eps_curr),
            "book_value": None if (book_curr is None or (isinstance(book_curr, float) and np.isnan(book_curr))) else float(book_curr),
            "growth_rate": float(g),
            "wacc": float(WACC),
            "beta": float(beta),
            "high52": float(max(prices[-252:])) if len(prices) >= 252 else float(max(prices)),
            "low52": float(min(prices[-252:])) if len(prices) >= 252 else float(min(prices)),
            # extra transparency fields (optional for your UI)
            "rf": float(rf),
            "market_return": float(rm_exp),
            "erp": float(erp),
            "cost_of_equity": float(Re),
            "cost_of_debt": float(Rd),
            "tax_rate": float(T),
            "debt": float(D),
            "cash": float(cash),
            "fcff0": float(FCFF0),
            "roic": float(ROIC),
            "reinvestment_rate": float(reinvestment),
            "terminal_growth": float(g_term),
        },
        "returns": returns,
        "backtest": backtest_data,
        "historical_data": {"dates": dates_ms, "prices": prices, "fair_values": fair_values},
        "source_used": "Yahoo Finance (prices + statements) + EconDB (risk-free)",
        "is_dynamic_beta": True,
        "is_synthetic_beta": False,
        "is_dynamic_growth": True,
        "is_synthetic_growth": False,
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
