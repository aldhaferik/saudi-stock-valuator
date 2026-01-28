# app.py
from __future__ import annotations

import os
import math
import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================================================
# 0) APP + CORS
# =========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 1) CONFIG
#    - Goal: accuracy via time-aligned fundamentals (TTM/quarterly -> daily forward-fill)
# =========================================================
DEFAULT_HISTORY_PERIOD = "5y"
TRADING_DAYS = 252

# windows
BETA_LOOKBACK_DAYS = TRADING_DAYS * 2              # ~2y
MARKET_RETURN_LOOKBACK_DAYS = TRADING_DAYS * 5     # ~5y
TRAIN_WINDOW_DAYS = TRADING_DAYS * 3               # train weights on last 3y
TEST_WINDOW_DAYS = TRADING_DAYS * 1                # test on last 1y (walk-forward)
SOLVER_SAMPLE_STEP = 5                             # sample every ~week for optimization
N_WEIGHT_SAMPLES = 6000                            # continuous-ish (Dirichlet) search

FORECAST_YEARS = 5
TASI_TICKER = "^TASI.SR"

# Backup price sources (you requested)
ALPHA_VANTAGE_KEY = "0LR5JLOBSLOA6Z0A"
TWELVE_DATA_KEY = "ed240f406bab4225ac6e0a98be553aa2"

# Risk-free source (your repo file)
RISK_FREE_XLSX_PATH = "saudi_yields.xlsx"
RISK_FREE_COLUMN_NAME = "10-Year government bond yield"

# Robustness bounds (sanity only, not a finance “assumption”)
GROWTH_MIN = -0.20
GROWTH_MAX = 0.40
WACC_MAX = 0.50

# =========================================================
# 2) JSON-SAFE SERIALIZATION
# =========================================================
def json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    return obj

# =========================================================
# 3) SMALL HTML UI (unchanged)
# =========================================================
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
        .weight-fill { height: 100%; background: #007aff; width: 0%; }

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
        <p style="color:#666; font-size:14px;">Time-aligning fundamentals (TTM) + walk-forward tuning</p>
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
                <div class="card-title">FAIR VALUE</div>
                <div class="fv-header">
                    <div class="fv-big" id="fair">--</div>
                    <div class="fv-sub">Model-weighted target</div>
                    <div id="sectorMsg" class="sector-tag">--</div>
                </div>

                <div class="fv-row">
                    <div style="width:70%">
                        <span class="fv-label">DCF (WACC: <span id="wacc_display">--</span>)</span>
                        <div class="weight-container">
                            <div class="weight-bar"><div class="weight-fill" id="w_dcf_fill"></div></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="fv-num" id="dcf_val">--</span>
                        <div style="font-size:10px; color:#aaa;" id="w_dcf_txt">--</div>
                    </div>
                </div>

                <div class="fv-row">
                    <div style="width:70%">
                        <span class="fv-label">P/E Model (TTM)</span>
                        <div class="weight-container">
                            <div class="weight-bar"><div class="weight-fill" id="w_pe_fill"></div></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="fv-num" id="pe_val">--</span>
                        <div style="font-size:10px; color:#aaa;" id="w_pe_txt">--</div>
                    </div>
                </div>

                <div class="fv-row">
                    <div style="width:70%">
                        <span class="fv-label">P/B Model</span>
                        <div class="weight-container">
                            <div class="weight-bar"><div class="weight-fill" id="w_pb_fill"></div></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="fv-num" id="pb_val">--</span>
                        <div style="font-size:10px; color:#aaa;" id="w_pb_txt">--</div>
                    </div>
                </div>

                <div class="fv-row">
                    <div style="width:70%">
                        <span class="fv-label">EV/EBITDA Model (TTM)</span>
                        <div class="weight-container">
                            <div class="weight-bar"><div class="weight-fill" id="w_ev_ebitda_fill"></div></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="fv-num" id="ev_ebitda_val">--</span>
                        <div style="font-size:10px; color:#aaa;" id="w_ev_ebitda_txt">--</div>
                    </div>
                </div>

            </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <div id="chartContainer" style="height: 350px;"></div>
        </div>

        <div class="bottom-section">
            <div class="card">
                <div class="card-title">Walk-forward Backtest</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th>Actual Price</th>
                            <th>Model</th>
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

                <div class="card-title">Future Projections (DCF)</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Year</th>
                            <th>Projected FCFF</th>
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

        const text = await res.text();
        let data;
        try { data = JSON.parse(text); }
        catch (e) { throw new Error("Non-JSON response: " + text.slice(0, 200)); }

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

        document.getElementById('name').innerText = s.company_name ?? "--";
        document.getElementById('tickerDisplay').innerText = ticker.toUpperCase() + ".SR";
        document.getElementById('price').innerText = (s.current_price ?? 0).toFixed(2);
        document.getElementById('fair').innerText = (s.fair_value ?? 0).toFixed(2);
        document.getElementById('sectorMsg').innerText = s.sector ?? "--";

        const vb = document.getElementById('verdictBar');
        const upside = s.upside_percent ?? 0;
        const label = (s.verdict ?? "Fairly Valued").toUpperCase();
        const sign = upside > 0 ? "+" : "";
        vb.innerText = `${label} (${sign}${Number(upside).toFixed(1)}% Upside)`;
        vb.className = "verdict-bar " + (label === "UNDERVALUED" ? "v-green" : (label === "OVERVALUED" ? "v-red" : "v-gray"));

        const fmt = (num) => (num === null || num === undefined) ? "N/A" : Number(num).toFixed(2);
        const fmtBig = (num) => (num === null || num === undefined) ? "N/A" : (Number(num) / 1000000000).toFixed(2) + "B";

        document.getElementById('mcap').innerText = fmtBig(m.market_cap);
        document.getElementById('pe').innerText = fmt(m.pe_ratio);
        document.getElementById('eps').innerText = fmt(m.eps);
        document.getElementById('beta').innerText = (m.beta === null || m.beta === undefined) ? "N/A" : Number(m.beta).toFixed(2);
        document.getElementById('growth').innerText = (m.growth_rate === null || m.growth_rate === undefined) ? "N/A" : (Number(m.growth_rate) * 100).toFixed(1) + "%";
        document.getElementById('book').innerText = fmt(m.book_value);

        document.getElementById('wacc_display').innerText =
            (m.wacc === null || m.wacc === undefined) ? "N/A" : (Number(m.wacc) * 100).toFixed(1) + "%";

        document.getElementById('beta_tag').innerHTML = '<span class="dyn-badge">LIVE</span>';
        document.getElementById('growth_tag').innerHTML = '<span class="dyn-badge">TTM</span>';

        document.getElementById('dcf_val').innerText = fmt(s.model_breakdown?.dcf);
        document.getElementById('pe_val').innerText = fmt(s.model_breakdown?.pe_model);
        document.getElementById('pb_val').innerText = fmt(s.model_breakdown?.pb_model);
        document.getElementById('ev_ebitda_val').innerText = fmt(s.model_breakdown?.ev_ebitda_model);

        const setW = (key, val) => {
            const pct = ((val ?? 0) * 100).toFixed(0) + "%";
            const fill = document.getElementById(`w_${key}_fill`);
            const txt = document.getElementById(`w_${key}_txt`);
            if (fill) fill.style.width = pct;
            if (txt) txt.innerText = "Weight: " + pct;
        };
        setW('dcf', weights?.dcf);
        setW('pe', weights?.pe);
        setW('pb', weights?.pb);
        setW('ev_ebitda', weights?.ev_ebitda);

        const setRet = (id, val) => {
            const el = document.getElementById(id);
            if (!el) return;
            if (val === null || val === undefined) { el.innerText = "--"; el.className = "ret-val"; return; }
            const n = Number(val);
            el.innerText = (n > 0 ? "+" : "") + n.toFixed(1) + "%";
            el.className = "ret-val " + (n > 0 ? "pos" : "neg");
        };
        setRet('r1m', r?.["1m"]);
        setRet('r3m', r?.["3m"]);
        setRet('r6m', r?.["6m"]);
        setRet('r1y', r?.["1y"]);
        setRet('r2y', r?.["2y"]);

        const fcBody = document.getElementById('forecastBody');
        if (fcBody) {
            fcBody.innerHTML = "";
            const currentYear = new Date().getFullYear();
            const projections = Array.isArray(s.dcf_projections) ? s.dcf_projections : [];
            projections.forEach((val, i) => {
                const row = `<tr>
                    <td>${currentYear + i + 1}</td>
                    <td>${Number(val).toFixed(0)} SAR</td>
                    <td style="color:#28cd41;">+${(Number(m.growth_rate || 0) * 100).toFixed(1)}%</td>
                </tr>`;
                fcBody.innerHTML += row;
            });
        }

        const btBody = document.getElementById('backtestBody');
        if (btBody) {
            btBody.innerHTML = "";
            (Array.isArray(backtest) ? backtest : []).forEach(b => {
                const actual = Number(b.actual);
                const model = Number(b.model);
                const diff = (actual && isFinite(actual)) ? Math.abs((model - actual) / actual) * 100 : 0;
                const color = diff < 15 ? "#28cd41" : "#f0ad4e";
                const row = `<tr>
                    <td>${b.period ?? ""}</td>
                    <td>${isFinite(actual) ? actual.toFixed(2) : "N/A"}</td>
                    <td>${isFinite(model) ? model.toFixed(2) : "N/A"}</td>
                    <td style="color:${color}; font-weight:bold;">${diff.toFixed(1)}%</td>
                </tr>`;
                btBody.innerHTML += row;
            });
        }

        const dates = data.historical_data?.dates || [];
        const prices = data.historical_data?.prices || [];
        const fairVals = data.historical_data?.fair_values || [];

        if (dates.length && prices.length && fairVals.length) {
            Highcharts.chart('chartContainer', {
                chart: { backgroundColor: 'transparent' },
                title: { text: 'Price vs Model (TTM fundamentals + walk-forward weights)' },
                xAxis: { type: 'datetime' },
                yAxis: { title: { text: null }, gridLineColor: '#eee' },
                series: [{
                    name: 'Actual Price',
                    data: dates.map((d, i) => [d, prices[i]]),
                    type: 'area'
                }, {
                    name: 'Model Fair Value',
                    data: dates.map((d, i) => [d, fairVals[i]]),
                    type: 'line',
                    lineWidth: 2
                }],
                credits: { enabled: false }
            });
        }

        dashboard.style.display = 'block';

    } catch (e) {
        loading.style.display = 'none';
        btn.disabled = false;
        err.innerText = "Error: " + (e?.message || e);
        err.style.display = 'block';
    }
}
</script>

</body>
</html>
"""

# =========================================================
# 4) DATA FETCHER (prices + statements)
#    - Accuracy improvements:
#      - quarterly/TTM fundamentals from report dates
#      - time-varying net debt & (best-effort) shares from statements where possible
# =========================================================
class DataFetcher:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        ]

    def _headers(self):
        return {"User-Agent": random.choice(self.user_agents)}

    @staticmethod
    def clean_saudi_ticker(ticker: str) -> str:
        t = (ticker or "").strip().upper()
        if t.replace(".", "").isdigit() and not t.endswith(".SR"):
            return f"{t}.SR"
        return t

    # ---------- Prices ----------
    def fetch_prices_yahoo(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> pd.DataFrame:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError(f"No Yahoo price history for {ticker}.")
        return hist

    def fetch_prices_twelve(self, ticker: str) -> pd.DataFrame:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": 1250,
            "apikey": TWELVE_DATA_KEY,
            "format": "JSON",
        }
        r = requests.get(url, params=params, headers=self._headers(), timeout=12)
        if r.status_code != 200:
            raise ValueError(f"Twelve Data request failed ({r.status_code}).")
        data = r.json()
        if "values" not in data or not isinstance(data["values"], list) or len(data["values"]) == 0:
            msg = data.get("message") or "No Twelve Data values."
            raise ValueError(f"Twelve Data: {msg}")
        rows = []
        for v in data["values"]:
            try:
                rows.append((pd.to_datetime(v["datetime"]), float(v["close"])))
            except Exception:
                continue
        if not rows:
            raise ValueError("Twelve Data: could not parse values.")
        df = pd.DataFrame(rows, columns=["Date", "Close"]).set_index("Date").sort_index()
        return df

    def fetch_prices_alpha_vantage(self, ticker: str) -> pd.DataFrame:
        av_symbol = ticker.replace(".SR", ".SA")
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": av_symbol,
            "outputsize": "full",
            "apikey": ALPHA_VANTAGE_KEY,
        }
        r = requests.get(url, params=params, headers=self._headers(), timeout=12)
        if r.status_code != 200:
            raise ValueError(f"Alpha Vantage request failed ({r.status_code}).")
        data = r.json()
        ts = data.get("Time Series (Daily)")
        if not ts:
            msg = data.get("Note") or data.get("Error Message") or "No daily series."
            raise ValueError(f"Alpha Vantage: {msg}")
        df = pd.DataFrame.from_dict(ts, orient="index")
        if "4. close" not in df.columns:
            raise ValueError("Alpha Vantage: missing close field.")
        df = df.rename(columns={"4. close": "Close"})
        df.index = pd.to_datetime(df.index)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_index().tail(1250)
        if df.empty:
            raise ValueError("Alpha Vantage: empty parsed dataframe.")
        return df[["Close"]]

    def fetch_prices(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> Tuple[pd.DataFrame, str]:
        try:
            return self.fetch_prices_yahoo(ticker, period=period), "Yahoo Finance"
        except Exception:
            pass
        try:
            return self.fetch_prices_twelve(ticker), "Twelve Data"
        except Exception:
            pass
        try:
            return self.fetch_prices_alpha_vantage(ticker), "Alpha Vantage"
        except Exception as e:
            raise ValueError(f"All price sources failed: {str(e)}")

    # ---------- Statements (Yahoo via yfinance) ----------
    def fetch_statements_yahoo(self, ticker: str) -> Dict[str, Any]:
        import yfinance as yf
        stock = yf.Ticker(ticker)

        try:
            info = stock.info or {}
        except Exception:
            info = {}

        def safe_attr(name: str):
            try:
                return getattr(stock, name)
            except Exception:
                return None

        # Annual
        fin_a = safe_attr("financials")
        bs_a = safe_attr("balance_sheet")
        cf_a = safe_attr("cashflow")

        # Quarterly
        fin_q = safe_attr("quarterly_financials")
        bs_q = safe_attr("quarterly_balance_sheet")
        cf_q = safe_attr("quarterly_cashflow")

        return {
            "info": info,
            "financials_annual": fin_a,
            "balance_sheet_annual": bs_a,
            "cashflow_annual": cf_a,
            "financials_quarterly": fin_q,
            "balance_sheet_quarterly": bs_q,
            "cashflow_quarterly": cf_q,
        }

    # ---------- Risk-free (Excel) ----------
    def fetch_saudi_risk_free_from_excel(self, path: str, column_name: str) -> float:
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Failed to read Excel '{path}'. Install openpyxl and ensure the file exists. Detail: {str(e)}")

        if df is None or df.empty:
            raise ValueError("Excel file is empty.")

        col = None
        for c in df.columns:
            if str(c).strip().lower() == str(column_name).strip().lower():
                col = c
                break
        if col is None:
            raise ValueError(f"Column '{column_name}' not found in Excel. Available columns: {list(df.columns)}")

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            raise ValueError(f"No numeric values found in column '{column_name}'.")

        last_val = float(s.iloc[-1])
        rf = last_val / 100.0 if last_val > 1.0 else last_val
        if not np.isfinite(rf) or rf <= 0 or rf > 0.50:
            raise ValueError(f"Risk-free out of bounds after parsing: {rf}")
        return rf

# =========================================================
# 5) STATEMENT + SERIES HELPERS (time-aligned fundamentals)
# =========================================================
def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None

def _row_lookup(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx_lower = {str(i).strip().lower(): i for i in df.index}
    for n in names:
        key = str(n).strip().lower()
        if key in idx_lower:
            return df.loc[idx_lower[key]]
    return None

def _row_contains(df: pd.DataFrame, must_contain: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    must = [m.lower() for m in must_contain]
    for idx in df.index:
        s = str(idx).lower()
        if all(m in s for m in must):
            return df.loc[idx]
    return None

def _series_from_row(df: pd.DataFrame, row_names: List[str], contains: Optional[List[str]] = None) -> Optional[pd.Series]:
    r = _row_lookup(df, row_names)
    if r is None and contains is not None:
        r = _row_contains(df, contains)
    if r is None:
        return None
    s = pd.to_numeric(r, errors="coerce")
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s

def ttm_from_quarters(q_series: pd.Series) -> pd.Series:
    # q_series indexed by report date; compute rolling sum of last 4 quarters
    s = q_series.sort_index()
    return s.rolling(4, min_periods=4).sum()

def last_value_on_or_before(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    # forward-fill to dates based on last known report date
    if series is None or series.empty:
        return pd.Series(index=dates, dtype=float)
    s = series.sort_index()
    out = pd.Series(index=dates, dtype=float)
    # reindex to union then ffill
    tmp = s.reindex(s.index.union(dates)).sort_index().ffill()
    out[:] = tmp.reindex(dates).values
    return out

def winsorize(arr: np.ndarray, p_low=0.05, p_high=0.95) -> np.ndarray:
    x = arr.copy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return arr
    lo = np.quantile(x, p_low)
    hi = np.quantile(x, p_high)
    out = np.clip(arr, lo, hi)
    return out

# =========================================================
# 6) MARKET/BETA HELPERS
# =========================================================
def annualized_geo_mean_return(prices: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    prices = prices.dropna()
    if len(prices) < periods_per_year + 1:
        raise ValueError("Not enough history to estimate market return.")
    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    n = len(prices) - 1
    years = n / periods_per_year
    if start <= 0 or end <= 0 or years <= 0:
        raise ValueError("Invalid series for return estimation.")
    return (end / start) ** (1.0 / years) - 1.0

def beta_regression(stock_prices: pd.Series, market_prices: pd.Series) -> float:
    df = pd.DataFrame({"s": stock_prices, "m": market_prices}).dropna()
    if len(df) < 120:
        raise ValueError("Not enough overlapping history for beta.")
    rs = np.log(df["s"]).diff().dropna()
    rm = np.log(df["m"]).diff().dropna()
    aligned = pd.DataFrame({"rs": rs, "rm": rm}).dropna()
    if len(aligned) < 120:
        raise ValueError("Not enough return observations for beta.")
    cov = np.cov(aligned["rs"], aligned["rm"], ddof=1)[0, 1]
    var = np.var(aligned["rm"], ddof=1)
    if var <= 0 or not np.isfinite(var):
        raise ValueError("Market variance invalid; cannot compute beta.")
    b = float(cov / var)
    if not np.isfinite(b):
        raise ValueError("Beta is not finite.")
    return b

# =========================================================
# 7) MODELS (DCF + Multiples) built from time-aligned fundamentals
# =========================================================
def dcf_per_share_from_fcff(
    fcff0: float,
    wacc: float,
    g: float,
    shares: float,
    net_debt: float,
    market_long_run_g: float,
    years: int = FORECAST_YEARS,
) -> float:
    """
    FCFF-based DCF -> Equity value -> per share.
    Terminal growth capped by market long-run CAGR (data-driven).
    """
    if shares <= 0:
        raise ValueError("shares <= 0")
    if not np.isfinite(fcff0) or fcff0 <= 0:
        raise ValueError("fcff0 must be positive and finite")
    if not np.isfinite(wacc) or wacc <= 0 or wacc > WACC_MAX:
        raise ValueError("wacc must be positive and finite")
    if not np.isfinite(g):
        raise ValueError("g not finite")

    g_term = min(g, market_long_run_g)
    if wacc <= g_term:
        raise ValueError("WACC <= terminal growth")

    pv_sum = 0.0
    last = None
    for i in range(1, years + 1):
        fcff_i = fcff0 * ((1.0 + g) ** i)
        pv_sum += fcff_i / ((1.0 + wacc) ** i)
        last = fcff_i

    tv = (last * (1.0 + g_term)) / (wacc - g_term)
    pv_tv = tv / ((1.0 + wacc) ** years)
    ev = pv_sum + pv_tv
    equity_value = ev - net_debt
    return equity_value / shares

def robust_loss_mape(y: np.ndarray, yhat: np.ndarray) -> float:
    # robust-ish MAPE: ignore invalid, avoid blow-ups
    mask = np.isfinite(y) & np.isfinite(yhat) & (y > 0)
    if mask.sum() == 0:
        return float("inf")
    yy = y[mask]
    yh = yhat[mask]
    ape = np.abs((yh - yy) / yy)
    # winsorize APE to reduce outlier domination
    ape = winsorize(ape, 0.02, 0.98)
    return float(np.mean(ape) * 100.0)

def optimize_weights_dirichlet(
    y: np.ndarray,
    X: np.ndarray,
    avail: np.ndarray,
    n_samples: int = N_WEIGHT_SAMPLES,
) -> np.ndarray:
    """
    Continuous-ish optimizer: sample many weights from Dirichlet, keep best.
    Constraints: w>=0, sum=1, and unavailable models forced to 0.
    X shape: (n_models, n_points)
    """
    n_models = X.shape[0]
    if not np.any(avail):
        raise ValueError("No models available")

    # Reduce to available subspace for sampling
    idx = np.where(avail)[0]
    k = len(idx)

    # Precompute valid point mask: need finite across chosen models and y>0
    y = y.astype(float)
    best_w_full = np.zeros(n_models, dtype=float)
    best_loss = float("inf")

    # Small deterministic seed for repeatability per request
    rnd = np.random.default_rng(42)

    # Include some corner weights (single-model)
    corner = []
    for j in idx:
        w = np.zeros(n_models, dtype=float)
        w[j] = 1.0
        corner.append(w)
    candidates = corner

    # Dirichlet draws
    alpha = np.ones(k, dtype=float)
    draws = rnd.dirichlet(alpha, size=n_samples)
    for d in draws:
        w = np.zeros(n_models, dtype=float)
        w[idx] = d
        candidates.append(w)

    # Evaluate
    for w in candidates:
        yhat = np.nansum(X.T * w, axis=1)
        loss = robust_loss_mape(y, yhat)
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_w_full = w.copy()

    if best_w_full.sum() <= 0:
        raise ValueError("Weight search failed")
    # normalize (should already sum to 1 on available models)
    best_w_full = best_w_full / best_w_full.sum()
    return best_w_full

# =========================================================
# 8) REQUEST MODEL
# =========================================================
class StockRequest(BaseModel):
    ticker: str

# =========================================================
# 9) MAIN ANALYSIS ENDPOINT (time-aligned fundamentals + walk-forward weights)
# =========================================================
@app.post("/analyze")
def analyze_stock(request: StockRequest):
    try:
        fetcher = DataFetcher()
        ticker = fetcher.clean_saudi_ticker(request.ticker)

        # ---------- Prices (stock + market index) ----------
        hist, source_stock = fetcher.fetch_prices(ticker, period=DEFAULT_HISTORY_PERIOD)
        mkt_hist, source_mkt = fetcher.fetch_prices(TASI_TICKER, period=DEFAULT_HISTORY_PERIOD)

        if hist is None or hist.empty or "Close" not in hist.columns or hist["Close"].dropna().empty:
            return JSONResponse({"error": "No valid Close prices for stock."}, status_code=200)
        if mkt_hist is None or mkt_hist.empty or "Close" not in mkt_hist.columns or mkt_hist["Close"].dropna().empty:
            return JSONResponse({"error": "No valid Close prices for market index (^TASI.SR)."}, status_code=200)

        stock_close_raw = hist["Close"].astype(float).dropna()
        mkt_close_raw = mkt_hist["Close"].astype(float).dropna()

        # Align by date intersection
        aligned_px = pd.DataFrame({"stock": stock_close_raw, "mkt": mkt_close_raw}).dropna()
        if len(aligned_px) < 300:
            return JSONResponse({"error": "Not enough overlapping price history between stock and TASI."}, status_code=200)

        stock_close = aligned_px["stock"]
        mkt_close = aligned_px["mkt"]
        dates = stock_close.index

        current_price = float(stock_close.iloc[-1])
        prices_list = stock_close.tolist()
        dates_ms = (dates.astype(np.int64) // 10**6).tolist()
        n_days = len(stock_close)

        # ---------- Statements ----------
        pack = fetcher.fetch_statements_yahoo(ticker)
        info = pack.get("info") or {}

        fin_q = pack.get("financials_quarterly")
        bs_q = pack.get("balance_sheet_quarterly")
        cf_q = pack.get("cashflow_quarterly")

        fin_a = pack.get("financials_annual")
        bs_a = pack.get("balance_sheet_annual")
        cf_a = pack.get("cashflow_annual")

        company_name = info.get("longName") or f"Saudi Stock {request.ticker}"
        sector = (info.get("sector") or "Unknown").title()

        # Shares & market cap
        shares_now = _to_float(info.get("sharesOutstanding"))
        mcap_now = _to_float(info.get("marketCap"))
        if mcap_now is None and shares_now is not None:
            mcap_now = shares_now * current_price
        if mcap_now is None or shares_now is None or shares_now <= 0:
            return JSONResponse({"error": "Missing/invalid sharesOutstanding or marketCap from statements source."}, status_code=200)

        # ---------- Risk-free (Excel) ----------
        try:
            rf = fetcher.fetch_saudi_risk_free_from_excel(RISK_FREE_XLSX_PATH, RISK_FREE_COLUMN_NAME)
            rf_method = "excel_10y"
        except Exception as e:
            return JSONResponse({"error": f"Could not fetch Saudi risk-free rate from Excel: {str(e)}"}, status_code=200)

        # ---------- Market return + ERP (data-driven from TASI) ----------
        try:
            mkt_tail = mkt_close.tail(min(MARKET_RETURN_LOOKBACK_DAYS, len(mkt_close)))
            rm_exp = annualized_geo_mean_return(mkt_tail)
            erp = rm_exp - rf
            if not np.isfinite(erp):
                raise ValueError("ERP not finite")
        except Exception as e:
            return JSONResponse({"error": f"Could not compute market return/ERP from TASI: {str(e)}"}, status_code=200)

        # ---------- Beta (regression vs TASI) ----------
        try:
            s_beta = stock_close.tail(min(BETA_LOOKBACK_DAYS, len(stock_close)))
            m_beta = mkt_close.tail(min(BETA_LOOKBACK_DAYS, len(mkt_close)))
            beta = beta_regression(s_beta, m_beta)
            beta_method = "regression_log_returns"
        except Exception as e:
            return JSONResponse({"error": f"Could not compute regression beta vs TASI: {str(e)}"}, status_code=200)

        # ---------- Cost of equity ----------
        Re = rf + beta * erp

        # ---------- Build time-aligned fundamentals (prefer quarterly -> TTM) ----------
        method_flags = {
            "rf": rf_method,
            "beta": beta_method,
            "wacc": None,
            "growth": None,
            "fcff": None,
            "fundamentals": "quarterly_ttm" if (isinstance(fin_q, pd.DataFrame) and not fin_q.empty) else "annual_fallback",
            "prices_source_stock": source_stock,
            "prices_source_market": source_mkt,
            "walk_forward": f"train={TRAIN_WINDOW_DAYS}d,test={TEST_WINDOW_DAYS}d",
        }

        # ---- Core fundamental series candidates ----
        # Net income (quarterly)
        ni_q = _series_from_row(fin_q, ["Net Income", "NetIncome"], contains=["net", "income"]) if isinstance(fin_q, pd.DataFrame) else None
        # Total stockholders equity (quarterly)
        eq_q = _series_from_row(bs_q, ["Total Stockholder Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"], contains=["total", "equity"]) if isinstance(bs_q, pd.DataFrame) else None
        # Cash from operations (quarterly)
        cfo_q = _series_from_row(cf_q, ["Total Cash From Operating Activities", "Operating Cash Flow"], contains=["operating", "cash"]) if isinstance(cf_q, pd.DataFrame) else None
        # CapEx (quarterly, often negative)
        capex_q = _series_from_row(cf_q, ["Capital Expenditures", "Capital Expenditure"], contains=["capital", "expend"]) if isinstance(cf_q, pd.DataFrame) else None
        # Depreciation (quarterly)
        da_q = _series_from_row(cf_q, ["Depreciation", "Depreciation And Amortization"], contains=["depreciation"]) if isinstance(cf_q, pd.DataFrame) else None
        # Operating income / EBIT (quarterly)
        ebit_q = _series_from_row(fin_q, ["Operating Income", "OperatingIncome", "EBIT", "Ebit"], contains=["operating", "income"]) if isinstance(fin_q, pd.DataFrame) else None

        # Debt (quarterly)
        st_debt_q = _series_from_row(bs_q, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], contains=["short", "debt"]) if isinstance(bs_q, pd.DataFrame) else None
        lt_debt_q = _series_from_row(bs_q, ["Long Term Debt", "LongTermDebt"], contains=["long", "debt"]) if isinstance(bs_q, pd.DataFrame) else None
        cash_q = _series_from_row(bs_q, ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents"], contains=["cash"]) if isinstance(bs_q, pd.DataFrame) else None

        # Shares (best-effort from statements; if unavailable we keep constant sharesOutstanding)
        shares_q = _series_from_row(bs_q, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
                                    contains=["shares"]) if isinstance(bs_q, pd.DataFrame) else None

        # If quarterly missing, fallback to annual (still time-aligned but lower resolution)
        if ni_q is None and isinstance(fin_a, pd.DataFrame) and not fin_a.empty:
            ni_q = _series_from_row(fin_a, ["Net Income", "NetIncome"], contains=["net", "income"])
        if eq_q is None and isinstance(bs_a, pd.DataFrame) and not bs_a.empty:
            eq_q = _series_from_row(bs_a, ["Total Stockholder Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"], contains=["total", "equity"])
        if cfo_q is None and isinstance(cf_a, pd.DataFrame) and not cf_a.empty:
            cfo_q = _series_from_row(cf_a, ["Total Cash From Operating Activities", "Operating Cash Flow"], contains=["operating", "cash"])
        if capex_q is None and isinstance(cf_a, pd.DataFrame) and not cf_a.empty:
            capex_q = _series_from_row(cf_a, ["Capital Expenditures", "Capital Expenditure"], contains=["capital", "expend"])
        if da_q is None and isinstance(cf_a, pd.DataFrame) and not cf_a.empty:
            da_q = _series_from_row(cf_a, ["Depreciation", "Depreciation And Amortization"], contains=["depreciation"])
        if ebit_q is None and isinstance(fin_a, pd.DataFrame) and not fin_a.empty:
            ebit_q = _series_from_row(fin_a, ["Operating Income", "OperatingIncome", "EBIT", "Ebit"], contains=["operating", "income"])
        if st_debt_q is None and isinstance(bs_a, pd.DataFrame) and not bs_a.empty:
            st_debt_q = _series_from_row(bs_a, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], contains=["short", "debt"])
        if lt_debt_q is None and isinstance(bs_a, pd.DataFrame) and not bs_a.empty:
            lt_debt_q = _series_from_row(bs_a, ["Long Term Debt", "LongTermDebt"], contains=["long", "debt"])
        if cash_q is None and isinstance(bs_a, pd.DataFrame) and not bs_a.empty:
            cash_q = _series_from_row(bs_a, ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents"], contains=["cash"])

        # ---- Effective tax rate (best-effort from annual, else clamp) ----
        T = 0.0
        try:
            pretax_a = _series_from_row(fin_a, ["Pretax Income", "Income Before Tax", "IncomeBeforeTax"], contains=["before", "tax"]) if isinstance(fin_a, pd.DataFrame) else None
            tax_a = _series_from_row(fin_a, ["Tax Provision", "Income Tax Expense", "IncomeTaxExpense"], contains=["tax"]) if isinstance(fin_a, pd.DataFrame) else None
            if pretax_a is not None and tax_a is not None and pretax_a.dropna().size > 0 and tax_a.dropna().size > 0:
                # use most recent year
                px = float(pretax_a.dropna().iloc[-1])
                tx = float(tax_a.dropna().iloc[-1])
                if np.isfinite(px) and px > 0 and np.isfinite(tx):
                    T = max(0.0, min(tx / px, 0.35))
        except Exception:
            pass

        # ---- Time-varying net debt (report-based, then forward-fill daily) ----
        # If debt/cash missing, treat as 0 but flag reliability
        st_debt_q = st_debt_q if st_debt_q is not None else pd.Series(dtype=float)
        lt_debt_q = lt_debt_q if lt_debt_q is not None else pd.Series(dtype=float)
        cash_q = cash_q if cash_q is not None else pd.Series(dtype=float)

        debt_q = (st_debt_q.fillna(0.0) + lt_debt_q.fillna(0.0)).sort_index()
        cash_q = cash_q.fillna(0.0).sort_index()
        net_debt_q = (debt_q - cash_q).sort_index()

        net_debt_daily = last_value_on_or_before(net_debt_q, dates)

        # ---- Shares time series (best-effort). Fallback: constant shares_now ----
        if shares_q is not None and shares_q.dropna().size >= 2:
            shares_daily = last_value_on_or_before(shares_q, dates)
            # sanity: if shares look insane, revert to constant
            if not np.isfinite(shares_daily.dropna().median()) or shares_daily.dropna().median() <= 0:
                shares_daily = pd.Series(index=dates, data=float(shares_now))
                method_flags["shares"] = "constant_info"
            else:
                method_flags["shares"] = "report_aligned_best_effort"
        else:
            shares_daily = pd.Series(index=dates, data=float(shares_now))
            method_flags["shares"] = "constant_info"

        # ---- EPS TTM (from net income TTM / shares) ----
        eps_ttm_daily = None
        if ni_q is not None and ni_q.dropna().size >= 4:
            ni_ttm = ttm_from_quarters(ni_q)
            ni_ttm_daily = last_value_on_or_before(ni_ttm, dates)
            eps_ttm_daily = ni_ttm_daily / shares_daily
        else:
            eps_ttm_daily = pd.Series(index=dates, dtype=float)

        # ---- BVPS (equity / shares), forward-fill ----
        if eq_q is not None and eq_q.dropna().size >= 2:
            eq_daily = last_value_on_or_before(eq_q, dates)
            bvps_daily = eq_daily / shares_daily
        else:
            bvps_daily = pd.Series(index=dates, dtype=float)

        # ---- EBITDA TTM (prefer explicit EBITDA; else EBIT + D&A) ----
        ebitda_q = _series_from_row(fin_q, ["Ebitda", "EBITDA"], contains=["ebitda"]) if isinstance(fin_q, pd.DataFrame) else None
        if ebitda_q is None and isinstance(fin_a, pd.DataFrame) and not fin_a.empty:
            ebitda_q = _series_from_row(fin_a, ["Ebitda", "EBITDA"], contains=["ebitda"])

        if ebitda_q is not None and ebitda_q.dropna().size >= 4:
            ebitda_ttm = ttm_from_quarters(ebitda_q)
            ebitda_ttm_daily = last_value_on_or_before(ebitda_ttm, dates)
            method_flags["ebitda"] = "reported_ttm"
        else:
            # fallback: (EBIT + D&A), both need to exist at least quarterly/annual (TTM where possible)
            ebitda_ttm_daily = pd.Series(index=dates, dtype=float)
            if ebit_q is not None and da_q is not None and ebit_q.dropna().size >= 2 and da_q.dropna().size >= 2:
                # If quarterly and enough points -> make TTM. If annual -> rolling(4) will require 4 years, so we’ll
                # fall back to last known annual (no TTM) by forward-filling original if too sparse.
                if ebit_q.dropna().size >= 4 and da_q.dropna().size >= 4:
                    ebit_ttm = ttm_from_quarters(ebit_q)
                    da_ttm = ttm_from_quarters(da_q)
                    ebitda_ttm_daily = last_value_on_or_before(ebit_ttm + da_ttm, dates)
                    method_flags["ebitda"] = "ebit_plus_da_ttm"
                else:
                    # annual-ish: use latest known (not TTM)
                    ebit_daily = last_value_on_or_before(ebit_q, dates)
                    da_daily = last_value_on_or_before(da_q, dates)
                    ebitda_ttm_daily = ebit_daily + da_daily
                    method_flags["ebitda"] = "ebit_plus_da_latest"
            else:
                method_flags["ebitda"] = "unavailable"

        # ---- FCFF (prefer CFO - CapEx, report-aligned; then forward-fill) ----
        fcff_q = pd.Series(dtype=float)
        if cfo_q is not None and capex_q is not None and cfo_q.dropna().size >= 2 and capex_q.dropna().size >= 2:
            # CapEx can be negative (cash outflow). Normalize to positive outflow.
            capex_out = capex_q.copy()
            capex_out = capex_out.apply(lambda x: -x if pd.notna(x) and x < 0 else x)
            fcff_q = (cfo_q - capex_out).sort_index()
            method_flags["fcff"] = "cfo_minus_capex_report_aligned"
        else:
            method_flags["fcff"] = "unavailable"

        fcff_daily = last_value_on_or_before(fcff_q, dates) if fcff_q is not None and not fcff_q.empty else pd.Series(index=dates, dtype=float)

        # ---- Growth (use report-aligned TTM fundamentals; fallback to price CAGR) ----
        g = None

        # Prefer NI TTM CAGR over last ~3 years (report-level), because it tracks business performance more than price.
        try:
            if ni_q is not None and ni_q.dropna().size >= 8:
                ni_ttm = ttm_from_quarters(ni_q).dropna().sort_index()
                # need at least 5 TTM points for a meaningful CAGR window
                if ni_ttm.size >= 5:
                    # use last 3 years if available (~12 quarters => ~9 TTM points)
                    ni_tail = ni_ttm.tail(9) if ni_ttm.size >= 9 else ni_ttm
                    start = float(ni_tail.iloc[0])
                    end = float(ni_tail.iloc[-1])
                    # approximate years from index delta
                    years = max((ni_tail.index[-1] - ni_tail.index[0]).days / 365.25, 0.5)
                    if np.isfinite(start) and np.isfinite(end) and start > 0 and end > 0 and years > 0:
                        g_try = (end / start) ** (1.0 / years) - 1.0
                        if np.isfinite(g_try):
                            g = float(max(GROWTH_MIN, min(g_try, GROWTH_MAX)))
                            method_flags["growth"] = "net_income_ttm_cagr"
        except Exception:
            pass

        # Fallback: FCFF trend (report-level), if NI not usable
        if g is None:
            try:
                if fcff_q is not None and not fcff_q.empty and fcff_q.dropna().size >= 6:
                    # Use report-level FCFF, not daily
                    f = fcff_q.dropna().sort_index()
                    f_tail = f.tail(12) if f.size >= 12 else f
                    start = float(f_tail.iloc[0])
                    end = float(f_tail.iloc[-1])
                    years = max((f_tail.index[-1] - f_tail.index[0]).days / 365.25, 0.5)
                    if np.isfinite(start) and np.isfinite(end) and start > 0 and end > 0 and years > 0:
                        g_try = (end / start) ** (1.0 / years) - 1.0
                        if np.isfinite(g_try):
                            g = float(max(GROWTH_MIN, min(g_try, GROWTH_MAX)))
                            method_flags["growth"] = "fcff_report_cagr"
            except Exception:
                pass

        # Final fallback: price CAGR (data-driven, but least “fundamental”)
        if g is None:
            try:
                years = (len(stock_close) - 1) / TRADING_DAYS
                p0 = float(stock_close.iloc[0])
                p1 = float(stock_close.iloc[-1])
                if p0 > 0 and p1 > 0 and years > 0:
                    g_try = (p1 / p0) ** (1.0 / years) - 1.0
                    if np.isfinite(g_try):
                        g = float(max(GROWTH_MIN, min(g_try, GROWTH_MAX)))
                        method_flags["growth"] = "price_cagr"
            except Exception:
                pass

        if g is None:
            return JSONResponse({"error": "Could not estimate growth from available data (TTM/FCFF/price)."}, status_code=200)

        # ---- Market long-run growth cap (data-driven) ----
        try:
            g_mkt = annualized_geo_mean_return(mkt_tail)
        except Exception:
            g_mkt = rm_exp if np.isfinite(rm_exp) else 0.03

        # ---- Cost of debt (best-effort from annual interest / avg debt) ----
        Rd = None
        try:
            interest_a = _series_from_row(fin_a, ["Interest Expense", "InterestExpense", "Interest Expense Non Operating"], contains=["interest", "expense"]) if isinstance(fin_a, pd.DataFrame) else None
            if interest_a is not None and interest_a.dropna().size >= 1 and debt_q is not None and debt_q.dropna().size >= 1:
                # debt_q may be quarterly; use last 2 debt observations to form avg debt
                d = debt_q.dropna().sort_index()
                d_tail = d.tail(2) if d.size >= 2 else d.tail(1)
                avg_debt = float(d_tail.mean()) if d_tail.size > 0 else None
                i_last = float(interest_a.dropna().iloc[-1])
                if avg_debt is not None and avg_debt > 0 and np.isfinite(i_last):
                    Rd_try = abs(i_last) / avg_debt
                    if np.isfinite(Rd_try) and 0 < Rd_try <= WACC_MAX:
                        Rd = float(Rd_try)
                        method_flags["wacc"] = "interest_over_avg_debt"
        except Exception:
            pass

        # ---- Time-varying capital weights (daily): E=price*shares, D=debt report-aligned ----
        market_cap_daily = stock_close * shares_daily
        debt_daily = last_value_on_or_before(debt_q, dates) if debt_q is not None and not debt_q.empty else pd.Series(index=dates, data=0.0)

        # ---- WACC (prefer full; else equity-only) ----
        if Rd is not None:
            total_cap_daily = (market_cap_daily + debt_daily).replace([np.inf, -np.inf], np.nan)
            wE_daily = (market_cap_daily / total_cap_daily).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            wD_daily = (debt_daily / total_cap_daily).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            wacc_daily = (wE_daily * Re + wD_daily * Rd * (1.0 - T)).clip(lower=0.0)
            method_flags["wacc"] = method_flags.get("wacc") or "wacc_full_timevarying_weights"
        else:
            wacc_daily = pd.Series(index=dates, data=float(Re))
            method_flags["wacc"] = "equity_only_Re"

        # For DCF we use the latest WACC (keeps DCF stable, avoids backtest “wiggle” from weights noise)
        WACC = float(wacc_daily.dropna().iloc[-1]) if wacc_daily.dropna().size else float(Re)
        if (not np.isfinite(WACC)) or WACC <= 0 or WACC > WACC_MAX:
            return JSONResponse({"error": f"WACC invalid after estimation: {WACC}"}, status_code=200)

        # =========================================================
        # Build MODEL SERIES (daily) using time-aligned fundamentals
        # =========================================================
        price_arr = np.array(prices_list, dtype=float)

        # --- Observed daily ratios (for target medians), with gating ---
        eps_arr = eps_ttm_daily.astype(float).values
        bvps_arr = bvps_daily.astype(float).values
        ebitda_arr = ebitda_ttm_daily.astype(float).values
        shares_arr = shares_daily.astype(float).values
        net_debt_arr = net_debt_daily.astype(float).values

        # Observed P/E history (only where EPS>0)
        pe_obs = np.full(n_days, np.nan, dtype=float)
        pe_mask = np.isfinite(eps_arr) & (eps_arr > 0) & np.isfinite(price_arr) & (price_arr > 0)
        pe_obs[pe_mask] = price_arr[pe_mask] / eps_arr[pe_mask]

        # Observed P/B history (only where BVPS>0)
        pb_obs = np.full(n_days, np.nan, dtype=float)
        pb_mask = np.isfinite(bvps_arr) & (bvps_arr > 0) & np.isfinite(price_arr) & (price_arr > 0)
        pb_obs[pb_mask] = price_arr[pb_mask] / bvps_arr[pb_mask]

        # Observed EV/EBITDA history (only where EBITDA>0)
        ev_ebitda_obs = np.full(n_days, np.nan, dtype=float)
        ev_mask = np.isfinite(ebitda_arr) & (ebitda_arr > 0) & np.isfinite(price_arr) & (price_arr > 0) & np.isfinite(shares_arr)
        if ev_mask.any():
            ev_hist = price_arr * shares_arr + net_debt_arr
            ev_ebitda_obs[ev_mask] = ev_hist[ev_mask] / ebitda_arr[ev_mask]

        # --- Define train/test split for walk-forward tuning ---
        # We tune on TRAIN window ending right before TEST window.
        if n_days < (TRAIN_WINDOW_DAYS + TEST_WINDOW_DAYS + 50):
            # not enough history: use last ~70% as train, last ~30% as test (still walk-forward-ish)
            test_start = int(n_days * 0.70)
            train_start = 0
            method_flags["walk_forward"] = f"adaptive_train=0:{test_start},test={test_start}:{n_days}"
        else:
            test_start = n_days - TEST_WINDOW_DAYS
            train_start = max(0, test_start - TRAIN_WINDOW_DAYS)

        train_idx = np.arange(train_start, test_start, dtype=int)
        test_idx = np.arange(test_start, n_days, dtype=int)

        # sample train points (weekly-ish)
        train_sample = train_idx[::SOLVER_SAMPLE_STEP] if train_idx.size else np.array([], dtype=int)

        if train_sample.size < 50:
            return JSONResponse({"error": "Not enough training points after walk-forward split."}, status_code=200)

        # --- Targets: medians computed on TRAIN window only (reduces look-ahead) ---
        def robust_median(x: np.ndarray) -> Optional[float]:
            z = x[np.isfinite(x)]
            z = z[(z > 0)]
            if z.size < 60:
                return None
            z = winsorize(z, 0.05, 0.95)
            return float(np.nanmedian(z)) if np.isfinite(np.nanmedian(z)) else None

        target_pe = robust_median(pe_obs[train_idx])
        target_pb = robust_median(pb_obs[train_idx])
        target_ev_ebitda = robust_median(ev_ebitda_obs[train_idx])

        # --- Model series (daily) ---
        stream_dcf = np.full(n_days, np.nan, dtype=float)
        stream_pe = np.full(n_days, np.nan, dtype=float)
        stream_pb = np.full(n_days, np.nan, dtype=float)
        stream_ev_ebitda = np.full(n_days, np.nan, dtype=float)

        # DCF per day using report-aligned FCFF (forward-filled daily)
        fcff_arr = fcff_daily.astype(float).values
        # FCFF can be negative; DCF requires positive base. We'll only compute where fcff>0.
        dcf_mask = np.isfinite(fcff_arr) & (fcff_arr > 0) & np.isfinite(shares_arr) & (shares_arr > 0) & np.isfinite(net_debt_arr)
        if dcf_mask.any():
            for i in np.where(dcf_mask)[0]:
                try:
                    stream_dcf[i] = dcf_per_share_from_fcff(
                        fcff0=float(fcff_arr[i]),
                        wacc=float(WACC),
                        g=float(g),
                        shares=float(shares_arr[i]),
                        net_debt=float(net_debt_arr[i]),
                        market_long_run_g=float(g_mkt),
                        years=FORECAST_YEARS,
                    )
                except Exception:
                    continue

        # P/E model (TTM EPS * target multiple)
        if target_pe is not None:
            pe_model_mask = np.isfinite(eps_arr) & (eps_arr > 0)
            stream_pe[pe_model_mask] = eps_arr[pe_model_mask] * float(target_pe)

        # P/B model (BVPS * target multiple)
        if target_pb is not None:
            pb_model_mask = np.isfinite(bvps_arr) & (bvps_arr > 0)
            stream_pb[pb_model_mask] = bvps_arr[pb_model_mask] * float(target_pb)

        # EV/EBITDA model:
        # EV_target = EBITDA * target_ev_ebitda
        # Equity = EV_target - net_debt ; per share = equity / shares
        if target_ev_ebitda is not None:
            ev_model_mask = np.isfinite(ebitda_arr) & (ebitda_arr > 0) & np.isfinite(net_debt_arr) & np.isfinite(shares_arr) & (shares_arr > 0)
            ev_target = ebitda_arr * float(target_ev_ebitda)
            eq_target = ev_target - net_debt_arr
            stream_ev_ebitda[ev_model_mask] = eq_target[ev_model_mask] / shares_arr[ev_model_mask]

        # --- Data-quality gating: model availability on TRAIN sample points ---
        X = np.vstack([stream_dcf, stream_pe, stream_pb, stream_ev_ebitda])  # (4, n_days)

        # Availability based on having enough finite points in TRAIN sample
        avail = np.array([
            np.isfinite(stream_dcf[train_sample]).sum() >= 40,
            np.isfinite(stream_pe[train_sample]).sum() >= 40,
            np.isfinite(stream_pb[train_sample]).sum() >= 40,
            np.isfinite(stream_ev_ebitda[train_sample]).sum() >= 40,
        ], dtype=bool)

        if not np.any(avail):
            return JSONResponse({"error": "No valuation models available after time-aligned fundamentals + gating (insufficient statements coverage)."}, status_code=200)

        # --- Optimize weights on TRAIN sample (walk-forward) ---
        y_train = price_arr[train_sample]
        X_train = X[:, train_sample]

        try:
            w = optimize_weights_dirichlet(y=y_train, X=X_train, avail=avail, n_samples=N_WEIGHT_SAMPLES)
        except Exception as e:
            # fallback: equal weights across available
            w = avail.astype(float)
            w = w / w.sum()
            method_flags["weights"] = f"fallback_equal_due_to_{type(e).__name__}"
        else:
            method_flags["weights"] = "dirichlet_search"

        # --- Build combined series ---
        combined = np.nansum(X.T * w, axis=1)  # length n_days

        # Optional local calibration (TRAIN-only) to correct level without leaking TEST
        # If you want “no scaling at all”, set k_train = 1.0.
        def safe_calibration(yw: np.ndarray, mw: np.ndarray) -> float:
            mask = np.isfinite(yw) & np.isfinite(mw) & (mw > 0) & (yw > 0)
            if mask.sum() < 50:
                return 1.0
            k = float(np.sum(yw[mask]) / np.sum(mw[mask]))
            if not np.isfinite(k) or k <= 0:
                return 1.0
            return k

        k_train = safe_calibration(price_arr[train_idx], combined[train_idx])
        fair_series = combined * k_train

        # Current fair value
        fair_value = float(fair_series[-1]) if np.isfinite(fair_series[-1]) else None
        if fair_value is None or (not np.isfinite(fair_value)) or fair_value <= 0:
            return JSONResponse({"error": "Could not compute a finite fair value (models produced non-finite output)."}, status_code=200)

        upside = ((fair_value - current_price) / current_price) * 100.0

        verdict = "Fairly Valued"
        if upside > 10:
            verdict = "Undervalued"
        elif upside < -10:
            verdict = "Overvalued"

        # DCF projections (FCFF forecast, using latest valid FCFF)
        dcf_projections = []
        fcff_latest = None
        try:
            fcff_latest = float(pd.Series(fcff_arr).dropna().iloc[-1])
        except Exception:
            fcff_latest = None

        if fcff_latest is not None and np.isfinite(fcff_latest) and fcff_latest > 0:
            for i in range(1, FORECAST_YEARS + 1):
                dcf_projections.append(float(fcff_latest * ((1.0 + g) ** i)))

        # Returns
        def ret_pct(days: int) -> Optional[float]:
            if n_days <= days:
                return None
            p = float(price_arr[-days])
            if not np.isfinite(p) or p == 0:
                return None
            return ((current_price - p) / p) * 100.0

        returns = {
            "1m": ret_pct(21),
            "3m": ret_pct(63),
            "6m": ret_pct(126),
            "1y": ret_pct(252),
            "2y": ret_pct(504),
        }

        # Walk-forward backtest points (in TEST window only)
        # We report a few points inside the test window to represent actual out-of-sample behavior.
        backtest_points = []
        def add_bt(label: str, idx_from_end: int):
            if idx_from_end <= 0 or idx_from_end >= n_days:
                return
            i = n_days - idx_from_end
            if i < test_start:
                return
            a = float(price_arr[i]) if np.isfinite(price_arr[i]) else None
            m_ = float(fair_series[i]) if np.isfinite(fair_series[i]) else None
            if a is None or m_ is None:
                return
            backtest_points.append({"period": label, "actual": a, "model": m_})

        add_bt("3 Months Ago (OOS)", 63)
        add_bt("6 Months Ago (OOS)", 126)
        add_bt("1 Year Ago (OOS)", 252)

        # If test window is short (adaptive), add earliest test day
        if test_start < n_days and test_start >= 0:
            a0 = float(price_arr[test_start]) if np.isfinite(price_arr[test_start]) else None
            m0 = float(fair_series[test_start]) if np.isfinite(fair_series[test_start]) else None
            if a0 is not None and m0 is not None:
                backtest_points.append({"period": "Test Start (OOS)", "actual": a0, "model": m0})

        # Current model values (calibrated)
        current_model_values = {
            "dcf": (float(stream_dcf[-1]) * k_train) if np.isfinite(stream_dcf[-1]) else None,
            "pe": (float(stream_pe[-1]) * k_train) if np.isfinite(stream_pe[-1]) else None,
            "pb": (float(stream_pb[-1]) * k_train) if np.isfinite(stream_pb[-1]) else None,
            "ev_ebitda": (float(stream_ev_ebitda[-1]) * k_train) if np.isfinite(stream_ev_ebitda[-1]) else None,
        }

        # Current observed multiples (TTM-based where possible)
        eps_now = float(eps_ttm_daily.dropna().iloc[-1]) if eps_ttm_daily.dropna().size else None
        book_now = float(bvps_daily.dropna().iloc[-1]) if bvps_daily.dropna().size else None
        pe_ratio_now = (current_price / eps_now) if (eps_now is not None and np.isfinite(eps_now) and eps_now > 0) else None
        pb_ratio_now = (current_price / book_now) if (book_now is not None and np.isfinite(book_now) and book_now > 0) else None

        # Source string
        source_used = f"{source_stock} (stock prices), {source_mkt} (market prices), Yahoo/yfinance (quarterly/annual statements), Excel (risk-free)"

        result = {
            "valuation_summary": {
                "company_name": company_name,
                "fair_value": float(fair_value),
                "current_price": float(current_price),
                "verdict": verdict,
                "upside_percent": float(upside),
                "dcf_projections": dcf_projections,
                "sector": sector,
                "model_breakdown": {
                    "dcf": current_model_values["dcf"],
                    "pe_model": current_model_values["pe"],
                    "pb_model": current_model_values["pb"],
                    "ev_ebitda_model": current_model_values["ev_ebitda"],
                    "calibration_k_train": float(k_train) if np.isfinite(k_train) else None,
                    "train_window_days": int(len(train_idx)),
                    "test_window_days": int(len(test_idx)),
                }
            },
            "optimized_weights": {
                "dcf": float(w[0]),
                "pe": float(w[1]),
                "pb": float(w[2]),
                "ev_ebitda": float(w[3]),
            },
            "metrics": {
                "market_cap": float(mcap_now),
                "pe_ratio": pe_ratio_now,
                "eps": eps_now,
                "book_value": book_now,
                "growth_rate": float(g) if np.isfinite(g) else None,
                "wacc": float(WACC) if np.isfinite(WACC) else None,
                "beta": float(beta) if np.isfinite(beta) else None,
                "high52": float(np.nanmax(price_arr[-252:])) if len(price_arr) >= 252 else float(np.nanmax(price_arr)),
                "low52": float(np.nanmin(price_arr[-252:])) if len(price_arr) >= 252 else float(np.nanmin(price_arr)),
                "rf": float(rf) if np.isfinite(rf) else None,
                "market_return": float(rm_exp) if np.isfinite(rm_exp) else None,
                "erp": float(erp) if np.isfinite(erp) else None,
                "cost_of_equity": float(Re) if np.isfinite(Re) else None,
                "cost_of_debt": None if Rd is None else float(Rd),
                "tax_rate": float(T) if np.isfinite(T) else None,
                "debt_now_report": float(debt_daily.dropna().iloc[-1]) if debt_daily.dropna().size else None,
                "cash_now_report": float(last_value_on_or_before(cash_q, dates).dropna().iloc[-1]) if (cash_q is not None and not cash_q.empty) else None,
                "net_debt_now_report": float(net_debt_daily.dropna().iloc[-1]) if net_debt_daily.dropna().size else None,
                "method_flags": method_flags,
                "multiples_targets": {
                    "target_pe_median_train": target_pe,
                    "target_pb_median_train": target_pb,
                    "target_ev_ebitda_median_train": target_ev_ebitda,
                },
            },
            "returns": returns,
            "backtest": backtest_points,
            "historical_data": {
                "dates": dates_ms,
                "prices": [float(x) for x in price_arr],
                "fair_values": [float(x) if np.isfinite(x) else None for x in fair_series],
            },
            "source_used": source_used,
            "is_dynamic_beta": True,
            "is_synthetic_beta": False,
            "is_dynamic_growth": True,
            "is_synthetic_growth": False
        }

        return JSONResponse(content=json_safe(result), status_code=200)

    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {str(e)}"}, status_code=200)

# =========================================================
# 10) RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

