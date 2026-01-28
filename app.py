# app.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime

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
# 1) CONFIG (NO hard-coded finance "assumptions" like fixed rf/mrp/sector PE)
#    - Uses: market index history (TASI) + your Excel yields file + company statements when available.
# =========================================================
DEFAULT_HISTORY_PERIOD = "5y"
TRADING_DAYS = 252
BETA_LOOKBACK_DAYS = TRADING_DAYS * 2
MARKET_RETURN_LOOKBACK_DAYS = TRADING_DAYS * 5
SOLVER_SAMPLE_STEP = 21  # ~monthly samples
FORECAST_YEARS = 5

TASI_TICKER = "^TASI.SR"

# Backup price sources (you requested)
ALPHA_VANTAGE_KEY = "0LR5JLOBSLOA6Z0A"
TWELVE_DATA_KEY = "ed240f406bab4225ac6e0a98be553aa2"

# Risk-free source (your repo file)
RISK_FREE_XLSX_PATH = "saudi_yields.xlsx"
RISK_FREE_COLUMN_NAME = "10-Year government bond yield"


# =========================================================
# 2) JSON-SAFE SERIALIZATION (prevents HTTP 500 due to NaN/Inf)
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
# 3) SMALL HTML UI (keeps the API contract; you can replace with your own UI)
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

# =========================================================
# 4) DATA FETCHER (Yahoo primary, Twelve Data / Alpha Vantage price backups)
# =========================================================
class DataFetcher:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        ]

    def _headers(self):
        return {"User-Agent": np.random.choice(self.user_agents)}

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
        # Twelve Data expects symbols; Tadawul sometimes works as "1120.SR" or "1120.SR"
        # We'll try the ticker as-is.
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": 1250,  # ~5 years trading days
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
        # values: list of dicts with "datetime","close"
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
        # Alpha Vantage symbol format for Saudi often uses .SA; you used that earlier.
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

    def fetch_prices(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> tuple[pd.DataFrame, str]:
        # Try Yahoo, then Twelve, then Alpha Vantage
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

    # ---------- Statements (Yahoo only: backups rarely provide full statements) ----------
    def fetch_statements_yahoo(self, ticker: str) -> dict:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        try:
            info = stock.info or {}
        except Exception:
            info = {}

        fin = None
        bs = None
        cf = None
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

        return {"info": info, "financials": fin, "balance_sheet": bs, "cashflow": cf}

    # ---------- Risk-free (Excel) ----------
    def fetch_saudi_risk_free_from_excel(self, path: str, column_name: str) -> float:
        # Requires openpyxl installed for .xlsx in most environments.
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Failed to read Excel '{path}'. Install openpyxl and ensure the file exists. Detail: {str(e)}")

        if df is None or df.empty:
            raise ValueError("Excel file is empty.")

        # Find column by case-insensitive match
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
# 5) STATEMENT HELPERS
# =========================================================
def most_recent_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    try:
        cols_sorted = sorted(cols, reverse=True)
        return cols_sorted[0]
    except Exception:
        return cols[0]

def second_most_recent_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    try:
        cols_sorted = sorted(cols, reverse=True)
        return cols_sorted[1] if len(cols_sorted) > 1 else None
    except Exception:
        return cols[1] if len(cols) > 1 else None

def safe_get_line(df: pd.DataFrame, possible_names: list[str], col) -> float | None:
    if df is None or df.empty or col is None:
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

def safe_get_line_contains(df: pd.DataFrame, must_contain: list[str], col) -> float | None:
    if df is None or df.empty or col is None:
        return None
    must_contain = [m.lower() for m in must_contain]
    for idx in df.index:
        s = str(idx).lower()
        ok = all(m in s for m in must_contain)
        if ok:
            val = df.loc[idx, col]
            if pd.isna(val):
                continue
            try:
                return float(val)
            except Exception:
                continue
    return None


# =========================================================
# 6) MARKET/BETA HELPERS (data-driven)
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
# 7) MODELS (DCF + Multiples)
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
    Terminal growth is capped by market long-run CAGR to avoid hard-coded GDP/inflation.
    """
    if shares <= 0:
        raise ValueError("shares <= 0")
    if not np.isfinite(fcff0) or fcff0 <= 0:
        raise ValueError("fcff0 must be positive and finite")
    if not np.isfinite(wacc) or wacc <= 0:
        raise ValueError("wacc must be positive and finite")

    if not np.isfinite(g):
        raise ValueError("g not finite")

    g_term = min(g, market_long_run_g)
    # If g_term is negative, allow it; but still must keep WACC > g_term
    if wacc <= g_term:
        raise ValueError(f"WACC ({wacc}) <= terminal growth ({g_term})")

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


def backcast_series(current_value: float, g: float, years_ago_array: np.ndarray) -> np.ndarray:
    """
    Back-cast fundamentals using a single growth rate g:
      value(t) = current_value / (1+g)^(years_ago)
    """
    if current_value is None or not np.isfinite(current_value):
        raise ValueError("current_value invalid")
    return np.array([current_value / ((1.0 + g) ** y) for y in years_ago_array], dtype=float)


def normalize_weights_simplex(ws: np.ndarray) -> np.ndarray:
    s = float(np.sum(ws))
    if s <= 0:
        raise ValueError("weight sum <= 0")
    return ws / s


def solver_best_weights(
    actual_prices: np.ndarray,
    model_matrix: np.ndarray,
    available_mask: np.ndarray,
    sample_step: int = SOLVER_SAMPLE_STEP,
    grid_step: float = 0.1,
) -> dict:
    """
    Grid-search weights on simplex for available models, minimize MAPE on sampled points.

    model_matrix shape: (n_models, n_days), aligned to actual_prices
    available_mask: bool array length n_models, True if that model is usable (finite)
    """
    if actual_prices.ndim != 1:
        raise ValueError("actual_prices must be 1D")
    n_days = actual_prices.shape[0]
    n_models = model_matrix.shape[0]

    idxs = np.arange(0, n_days, sample_step, dtype=int)
    y = actual_prices[idxs]
    if np.any(~np.isfinite(y)) or np.all(y <= 0):
        raise ValueError("actual prices invalid for solver")

    # build sampled model values
    X = model_matrix[:, idxs].copy()
    # mark unusable models
    for j in range(n_models):
        if not available_mask[j]:
            X[j, :] = np.nan

    # If none available -> fail
    if not np.any(available_mask):
        raise ValueError("No models available for solver")

    # We will solve weights for up to 4 models (DCF, PE, PB, EV/EBITDA)
    # Use grid on simplex; skip models not available by forcing weight=0.
    step = grid_step
    steps = [i * step for i in range(int(1 / step) + 1)]

    best = {"mape": float("inf"), "weights": None}

    for w0 in steps:
        for w1 in steps:
            for w2 in steps:
                w3 = 1.0 - w0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w = np.array([w0, w1, w2, w3], dtype=float)

                # enforce unavailable models weight 0
                w = w * available_mask.astype(float)
                if np.sum(w) <= 0:
                    continue
                w = normalize_weights_simplex(w)

                pred = np.nansum(X.T * w, axis=1)
                # If pred has NaN at any point, skip
                if np.any(~np.isfinite(pred)):
                    continue

                # MAPE
                denom = np.where(y == 0, np.nan, y)
                ape = np.abs((pred - y) / denom)
                mape = float(np.nanmean(ape) * 100.0)
                if not np.isfinite(mape):
                    continue
                if mape < best["mape"]:
                    best["mape"] = mape
                    best["weights"] = w.copy()

    if best["weights"] is None:
        raise ValueError("Solver could not find weights")

    return best


def calibration_k(actual_prices: np.ndarray, combined_model: np.ndarray) -> float:
    a = np.sum(actual_prices[np.isfinite(actual_prices)])
    m = np.sum(combined_model[np.isfinite(combined_model)])
    if m <= 0 or not np.isfinite(m):
        return 1.0
    k = float(a / m)
    if not np.isfinite(k) or k <= 0:
        return 1.0
    return k


# =========================================================
# 8) REQUEST MODEL
# =========================================================
class StockRequest(BaseModel):
    ticker: str


# =========================================================
# 9) MAIN ANALYSIS ENDPOINT
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

        stock_close = hist["Close"].astype(float).dropna()
        mkt_close = mkt_hist["Close"].astype(float).dropna()

        # Align by date intersection for downstream
        aligned = pd.DataFrame({"stock": stock_close, "mkt": mkt_close}).dropna()
        if len(aligned) < 200:
            return JSONResponse({"error": "Not enough overlapping price history between stock and TASI."}, status_code=200)

        stock_close = aligned["stock"]
        mkt_close = aligned["mkt"]

        # Use the aligned range for chart/backtest/solver
        current_price = float(stock_close.iloc[-1])
        prices_list = stock_close.tolist()
        dates_ms = (stock_close.index.astype(np.int64) // 10**6).tolist()
        n_days = len(prices_list)

        if n_days < 200:
            return JSONResponse({"error": "Not enough historical price points."}, status_code=200)

        # ---------- Statements ----------
        pack = fetcher.fetch_statements_yahoo(ticker)
        info = pack.get("info") or {}
        fin = pack.get("financials")
        bs = pack.get("balance_sheet")
        cf = pack.get("cashflow")

        company_name = info.get("longName") or f"Saudi Stock {request.ticker}"
        sector = (info.get("sector") or "Unknown").title()

        # Shares + market cap
        shares = info.get("sharesOutstanding")
        mcap = info.get("marketCap")
        if mcap is None and shares is not None:
            try:
                mcap = float(shares) * current_price
            except Exception:
                mcap = None

        if mcap is None:
            return JSONResponse({"error": "Missing market cap (or sharesOutstanding). Cannot compute equity value E."}, status_code=200)

        try:
            shares_f = float(shares)
            if shares_f <= 0:
                raise ValueError("sharesOutstanding <= 0")
        except Exception:
            return JSONResponse({"error": "Invalid sharesOutstanding from statements source (Yahoo)."}, status_code=200)

        E = float(mcap)

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

        # ---------- Pull required statement columns ----------
        fin_col = most_recent_col(fin)
        bs_col = most_recent_col(bs)
        cf_col = most_recent_col(cf)
        bs_col_prev = second_most_recent_col(bs)

        method_flags = {
            "rf": rf_method,
            "beta": beta_method,
            "wacc": None,
            "growth": None,
            "fcff": None,
            "prices_source_stock": source_stock,
            "prices_source_market": source_mkt,
        }

        # ---------- Debt / Cash ----------
        D = 0.0
        cash = 0.0

        st_debt = safe_get_line(bs, ["Short Long Term Debt", "Short Term Debt", "Current Debt", "ShortTermDebt"], bs_col)
        lt_debt = safe_get_line(bs, ["Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation"], bs_col)
        if st_debt is None:
            st_debt = safe_get_line_contains(bs, ["short", "debt"], bs_col)
        if lt_debt is None:
            lt_debt = safe_get_line_contains(bs, ["long", "debt"], bs_col)
        st_debt = float(st_debt) if st_debt is not None and np.isfinite(st_debt) else 0.0
        lt_debt = float(lt_debt) if lt_debt is not None and np.isfinite(lt_debt) else 0.0
        D = st_debt + lt_debt

        cash_val = safe_get_line(bs, ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents",
                                      "Cash And Cash Equivalents And Short Term Investments"], bs_col)
        if cash_val is None:
            cash_val = safe_get_line_contains(bs, ["cash"], bs_col)
        cash = float(cash_val) if cash_val is not None and np.isfinite(cash_val) else 0.0

        net_debt = D - cash

        # ---------- Taxes (effective) ----------
        pretax = safe_get_line(fin, ["Pretax Income", "Income Before Tax", "IncomeBeforeTax"], fin_col)
        if pretax is None:
            pretax = safe_get_line_contains(fin, ["before", "tax"], fin_col)
        tax_exp = safe_get_line(fin, ["Tax Provision", "Income Tax Expense", "IncomeTaxExpense"], fin_col)
        if tax_exp is None:
            tax_exp = safe_get_line_contains(fin, ["tax"], fin_col)

        T = 0.0
        if pretax is not None and tax_exp is not None and np.isfinite(pretax) and pretax > 0 and np.isfinite(tax_exp):
            T = float(tax_exp) / float(pretax)
            T = max(0.0, min(T, 0.35))

        # ---------- Cost of debt (best-effort from interest / avg debt) ----------
        Rd = None
        if D > 0:
            interest_exp = safe_get_line(fin, ["Interest Expense", "InterestExpense", "Interest Expense Non Operating"], fin_col)
            if interest_exp is None:
                interest_exp = safe_get_line_contains(fin, ["interest", "expense"], fin_col)

            interest_paid_cf = safe_get_line(cf, ["Interest Paid", "InterestPaid", "Cash Interest Paid", "CashInterestPaid"], cf_col)
            if interest_paid_cf is None:
                interest_paid_cf = safe_get_line_contains(cf, ["interest", "paid"], cf_col)

            avg_D = None
            if bs_col_prev is not None:
                st_prev = safe_get_line(bs, ["Short Long Term Debt", "Short Term Debt", "Current Debt", "ShortTermDebt"], bs_col_prev)
                lt_prev = safe_get_line(bs, ["Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation"], bs_col_prev)
                st_prev = float(st_prev) if st_prev is not None and np.isfinite(st_prev) else 0.0
                lt_prev = float(lt_prev) if lt_prev is not None and np.isfinite(lt_prev) else 0.0
                D_prev = st_prev + lt_prev
                avg_D = (D + D_prev) / 2.0 if (D + D_prev) > 0 else None

            if avg_D is not None and avg_D > 0:
                if interest_exp is not None and np.isfinite(interest_exp):
                    Rd = float(interest_exp) / float(avg_D)
                    method_flags["wacc"] = "interest_expense_over_avg_debt"
                elif interest_paid_cf is not None and np.isfinite(interest_paid_cf):
                    Rd = abs(float(interest_paid_cf)) / float(avg_D)
                    method_flags["wacc"] = "interest_paid_over_avg_debt"
            # sanity
            if Rd is not None and (not np.isfinite(Rd) or Rd <= 0 or Rd > 0.50):
                Rd = None

        # ---------- WACC ----------
        if D > 0 and Rd is not None:
            total_cap = D + E
            wE = E / total_cap
            wD = D / total_cap
            WACC = wE * Re + wD * Rd * (1.0 - T)
            method_flags["wacc"] = method_flags["wacc"] or "wacc_full"
        else:
            # If debt is unknown/interest missing, we cannot compute a reliable Rd from statements.
            # Best-effort: equity-only discount rate, flagged explicitly.
            WACC = Re
            method_flags["wacc"] = "equity_only_Re"

        # ---------- Growth (best-effort, data-driven) ----------
        # Primary: ROIC * reinvestment if we have NOPAT and invested capital, else fallback to price CAGR
        g = None

        # EBIT
        EBIT = safe_get_line(fin, [
            "Ebit", "EBIT",
            "Operating Income", "OperatingIncome",
            "Total Operating Income As Reported", "TotalOperatingIncomeAsReported",
            "Operating Profit", "OperatingProfit",
        ], fin_col)
        if EBIT is None:
            EBIT = safe_get_line_contains(fin, ["operating", "income"], fin_col)

        # D&A
        DA = safe_get_line(cf, ["Depreciation", "Depreciation And Amortization", "Depreciation & Amortization"], cf_col)
        if DA is None:
            DA = safe_get_line_contains(cf, ["depreciation"], cf_col)

        # CapEx (cash flow often negative)
        CapEx = safe_get_line(cf, ["Capital Expenditures", "Capital Expenditure"], cf_col)
        if CapEx is None:
            CapEx = safe_get_line_contains(cf, ["capital", "expend"], cf_col)

        # Working capital delta (best-effort)
        def net_working_capital(bs_df: pd.DataFrame, col):
            tca = safe_get_line(bs_df, ["Total Current Assets"], col)
            tcl = safe_get_line(bs_df, ["Total Current Liabilities"], col)
            if tca is None or tcl is None:
                return None
            cash_local = safe_get_line(bs_df, ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents",
                                              "Cash And Cash Equivalents And Short Term Investments"], col) or 0.0
            st_debt_local = safe_get_line(bs_df, ["Short Long Term Debt", "Short Term Debt", "Current Debt", "ShortTermDebt"], col) or 0.0
            return (float(tca) - float(cash_local)) - (float(tcl) - float(st_debt_local))

        dWC = None
        if bs_col_prev is not None:
            NWC_now = net_working_capital(bs, bs_col)
            NWC_prev = net_working_capital(bs, bs_col_prev)
            if NWC_now is not None and NWC_prev is not None and np.isfinite(NWC_now) and np.isfinite(NWC_prev):
                dWC = float(NWC_now - NWC_prev)

        # Build FCFF0 if possible; else fallback to CFO - CapEx (owner earnings proxy)
        fcff0 = None
        fcff_method = None

        if EBIT is not None and np.isfinite(EBIT) and DA is not None and np.isfinite(DA) and CapEx is not None and np.isfinite(CapEx) and dWC is not None and np.isfinite(dWC):
            NOPAT = float(EBIT) * (1.0 - T)
            capex_out = -float(CapEx) if float(CapEx) < 0 else float(CapEx)
            # FCFF0 = NOPAT + D&A - CapEx - dWC
            fcff0 = NOPAT + float(DA) - capex_out - float(dWC)
            if np.isfinite(fcff0) and fcff0 > 0:
                fcff_method = "ebit_da_capex_dwc"
        if fcff0 is None:
            # CFO fallback: Operating Cash Flow - CapEx
            CFO = safe_get_line(cf, ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"], cf_col)
            if CFO is None:
                CFO = safe_get_line_contains(cf, ["operating", "cash"], cf_col)
            if CFO is not None and np.isfinite(CFO) and CapEx is not None and np.isfinite(CapEx):
                capex_out = -float(CapEx) if float(CapEx) < 0 else float(CapEx)
                fcff0 = float(CFO) - capex_out
                if np.isfinite(fcff0) and fcff0 > 0:
                    fcff_method = "cfo_minus_capex"

        # Growth:
        if fcff0 is not None and fcff0 > 0 and EBIT is not None and np.isfinite(EBIT):
            # Try ROIC*reinvestment if we can estimate invested capital
            NOPAT = float(EBIT) * (1.0 - T)
            invested_capital = (E + D - cash)
            if np.isfinite(invested_capital) and invested_capital > 0 and np.isfinite(NOPAT) and NOPAT > 0 and DA is not None and CapEx is not None and dWC is not None:
                ROIC = float(NOPAT) / float(invested_capital)
                capex_out = -float(CapEx) if float(CapEx) < 0 else float(CapEx)
                reinvestment_rate = (capex_out - float(DA) + float(dWC)) / float(NOPAT)
                g_try = float(ROIC) * float(reinvestment_rate)
                if np.isfinite(g_try) and -0.20 <= g_try <= 0.40:
                    g = g_try
                    method_flags["growth"] = "roic_x_reinvestment"
        if g is None:
            # fallback: price CAGR (data-driven), capped only by sanity bounds (not "hard-coded model assumptions")
            try:
                years = (len(stock_close) - 1) / TRADING_DAYS
                p0 = float(stock_close.iloc[0])
                p1 = float(stock_close.iloc[-1])
                if p0 > 0 and p1 > 0 and years > 0:
                    g_try = (p1 / p0) ** (1.0 / years) - 1.0
                    if np.isfinite(g_try):
                        # sanity bounds only
                        g_try = max(-0.20, min(g_try, 0.40))
                        g = g_try
                        method_flags["growth"] = "price_cagr"
            except Exception:
                pass
        if g is None:
            return JSONResponse({"error": "Could not estimate growth from available data."}, status_code=200)

        method_flags["fcff"] = fcff_method or "unavailable"

        # Market long-run growth cap (data-driven)
        try:
            g_mkt = annualized_geo_mean_return(mkt_tail)
        except Exception:
            g_mkt = rm_exp if np.isfinite(rm_exp) else 0.03

        # ---------- Current fundamentals for multiples ----------
        eps_now = info.get("trailingEps")
        book_now = info.get("bookValue")

        def to_float_or_none(x):
            try:
                if x is None:
                    return None
                v = float(x)
                if not np.isfinite(v):
                    return None
                return v
            except Exception:
                return None

        eps_now = to_float_or_none(eps_now)
        book_now = to_float_or_none(book_now)

        # EBITDA from statements (best-effort)
        EBITDA = safe_get_line(fin, ["Ebitda", "EBITDA"], fin_col)
        if EBITDA is None:
            EBITDA = safe_get_line_contains(fin, ["ebitda"], fin_col)
        EBITDA = to_float_or_none(EBITDA)

        # Current observed multiples (only if inputs exist)
        pe_ratio_now = None
        if eps_now is not None and eps_now != 0:
            pe_ratio_now = current_price / eps_now

        pb_ratio_now = None
        if book_now is not None and book_now != 0:
            pb_ratio_now = current_price / book_now

        # EV/EBITDA observed now
        ev_ebitda_now = None
        if EBITDA is not None and EBITDA != 0:
            EV_now = E + D - cash
            ev_ebitda_now = EV_now / EBITDA if EBITDA != 0 else None

        # ---------- Build time series of model prices over 5y for solver ----------
        years_ago_array = np.linspace(5.0, 0.0, n_days)

        current_models = {
            "rf": rf,
            "market_return": rm_exp,
            "erp": erp,
            "beta": beta,
            "cost_of_equity": Re,
            "cost_of_debt": Rd,
            "tax_rate": T,
            "wacc": WACC,
            "growth": g,
            "method_flags": method_flags,
            "dcf": None,
            "pe": None,
            "pb": None,
            "ev_ebitda": None,
        }

        # DCF series: use fcff0 backcast if available
        stream_dcf = np.full(n_days, np.nan, dtype=float)
        if fcff0 is not None and np.isfinite(fcff0) and fcff0 > 0 and np.isfinite(WACC) and WACC > 0:
            try:
                fcff_series = backcast_series(float(fcff0), float(g), years_ago_array)
                for i in range(n_days):
                    # per-share DCF at each historical date using the backcast base FCFF
                    # (We discount forward from that date)
                    stream_dcf[i] = dcf_per_share_from_fcff(
                        fcff0=float(fcff_series[i]),
                        wacc=float(WACC),
                        g=float(g),
                        shares=float(shares_f),
                        net_debt=float(net_debt),
                        market_long_run_g=float(g_mkt),
                        years=FORECAST_YEARS,
                    )
                current_models["dcf"] = float(stream_dcf[-1]) if np.isfinite(stream_dcf[-1]) else None
            except Exception:
                pass

        # PE series: requires EPS now (we backcast EPS with g, and use target multiple = median of observed PE over history)
        stream_pe = np.full(n_days, np.nan, dtype=float)
        target_pe = None
        if eps_now is not None and eps_now > 0:
            try:
                eps_series = backcast_series(float(eps_now), float(g), years_ago_array)
                # observed PE history (using eps_series, not constant EPS)
                pe_hist = np.array(prices_list, dtype=float) / np.where(eps_series == 0, np.nan, eps_series)
                pe_hist = pe_hist[np.isfinite(pe_hist) & (pe_hist > 0)]
                if len(pe_hist) >= 60:
                    target_pe = float(np.nanmedian(pe_hist))
                if target_pe is not None and np.isfinite(target_pe) and target_pe > 0:
                    stream_pe = eps_series * target_pe
                    current_models["pe"] = float(stream_pe[-1]) if np.isfinite(stream_pe[-1]) else None
            except Exception:
                pass

        # PB series: requires book value now (backcast BV with g, target PB = median observed PB history)
        stream_pb = np.full(n_days, np.nan, dtype=float)
        target_pb = None
        if book_now is not None and book_now > 0:
            try:
                book_series = backcast_series(float(book_now), float(g), years_ago_array)
                pb_hist = np.array(prices_list, dtype=float) / np.where(book_series == 0, np.nan, book_series)
                pb_hist = pb_hist[np.isfinite(pb_hist) & (pb_hist > 0)]
                if len(pb_hist) >= 60:
                    target_pb = float(np.nanmedian(pb_hist))
                if target_pb is not None and np.isfinite(target_pb) and target_pb > 0:
                    stream_pb = book_series * target_pb
                    current_models["pb"] = float(stream_pb[-1]) if np.isfinite(stream_pb[-1]) else None
            except Exception:
                pass

        # EV/EBITDA series: requires EBITDA now. We backcast EBITDA with g and use target EV/EBITDA = median observed.
        stream_ev_ebitda = np.full(n_days, np.nan, dtype=float)
        target_ev_ebitda = None
        if EBITDA is not None and EBITDA > 0:
            try:
                ebitda_series = backcast_series(float(EBITDA), float(g), years_ago_array)
                EV_now = E + D - cash
                # Approx historical EV: assume net debt constant over time (best-effort; flagged by method)
                ev_hist = (np.array(prices_list, dtype=float) * float(shares_f)) + float(net_debt)
                ev_eb_hist = ev_hist / np.where(ebitda_series == 0, np.nan, ebitda_series)
                ev_eb_hist = ev_eb_hist[np.isfinite(ev_eb_hist) & (ev_eb_hist > 0)]
                if len(ev_eb_hist) >= 60:
                    target_ev_ebitda = float(np.nanmedian(ev_eb_hist))
                if target_ev_ebitda is not None and np.isfinite(target_ev_ebitda) and target_ev_ebitda > 0:
                    # Per-share: (EV - net_debt)/shares
                    ev_model = ebitda_series * target_ev_ebitda
                    eq_model = ev_model - float(net_debt)
                    stream_ev_ebitda = eq_model / float(shares_f)
                    current_models["ev_ebitda"] = float(stream_ev_ebitda[-1]) if np.isfinite(stream_ev_ebitda[-1]) else None
            except Exception:
                pass

        # ---------- Solver matrix ----------
        actual = np.array(prices_list, dtype=float)
        model_matrix = np.vstack([stream_dcf, stream_pe, stream_pb, stream_ev_ebitda])
        available_mask = np.array([
            np.any(np.isfinite(stream_dcf)),
            np.any(np.isfinite(stream_pe)),
            np.any(np.isfinite(stream_pb)),
            np.any(np.isfinite(stream_ev_ebitda)),
        ], dtype=bool)

        # ---------- Solve weights for best fit on history ----------
        try:
            opt = solver_best_weights(
                actual_prices=actual,
                model_matrix=model_matrix,
                available_mask=available_mask,
                sample_step=SOLVER_SAMPLE_STEP,
                grid_step=0.1,
            )
            weights_full = opt["weights"]
        except Exception as e:
            # If solver fails, fall back to "use whatever is available" equal weights
            if not np.any(available_mask):
                return JSONResponse({"error": f"No valuation models available for this ticker (statements missing). Detail: {str(e)}"}, status_code=200)
            w = available_mask.astype(float)
            w = w / np.sum(w)
            opt = {"mape": None, "weights": w}
            weights_full = w

        # ---------- Combined model series and calibration ----------
        combined = np.nansum(model_matrix.T * weights_full, axis=1)
        k = calibration_k(actual, combined)
        fair_values_full = (combined * k).tolist()

        # Current model values (at t=now), calibrated
        current_model_values = {
            "dcf": (float(stream_dcf[-1]) * k) if np.isfinite(stream_dcf[-1]) else None,
            "pe": (float(stream_pe[-1]) * k) if np.isfinite(stream_pe[-1]) else None,
            "pb": (float(stream_pb[-1]) * k) if np.isfinite(stream_pb[-1]) else None,
            "ev_ebitda": (float(stream_ev_ebitda[-1]) * k) if np.isfinite(stream_ev_ebitda[-1]) else None,
        }

        # Final fair value is calibrated combined at now
        combined_now = float(combined[-1]) if np.isfinite(combined[-1]) else None
        fair_value = (combined_now * k) if combined_now is not None else None
        if fair_value is None or not np.isfinite(fair_value) or fair_value <= 0:
            return JSONResponse({"error": "Could not compute a finite fair value (models produced non-finite output)."}, status_code=200)

        upside = ((float(fair_value) - float(current_price)) / float(current_price)) * 100.0

        verdict = "Fairly Valued"
        if upside > 10:
            verdict = "Undervalued"
        elif upside < -10:
            verdict = "Overvalued"

        # DCF projections (show per-share FCFF forecast values for transparency if FCFF available)
        dcf_projections = []
        if fcff0 is not None and np.isfinite(fcff0) and fcff0 > 0:
            for i in range(1, FORECAST_YEARS + 1):
                dcf_projections.append(float(fcff0 * ((1.0 + g) ** i)))

        # Returns
        def price_ago(days: int) -> float | None:
            if n_days <= days:
                return None
            return float(prices_list[-days])

        def ret_pct(days: int) -> float | None:
            p = price_ago(days)
            if p is None or p == 0:
                return None
            return ((float(current_price) - p) / p) * 100.0

        returns = {
            "1m": ret_pct(21),
            "3m": ret_pct(63),
            "6m": ret_pct(126),
            "1y": ret_pct(252),
            "2y": ret_pct(504),
        }

        # Backtest points (use the calibrated combined fair series)
        backtest_points = []
        for label, days in [("1 Year Ago", 252), ("2 Years Ago", 504), ("3 Years Ago", 756), ("4 Years Ago", 1008), ("5 Years Ago", 1250)]:
            if n_days > days:
                idx = -days
                backtest_points.append({
                    "period": label,
                    "actual": float(prices_list[idx]),
                    "model": float(fair_values_full[idx]) if fair_values_full[idx] is not None else None,
                })

        # Put current models in a consistent dict used by your return block
        current_models.update({
            "dcf": current_model_values["dcf"],
            "pe": current_model_values["pe"],
            "pb": current_model_values["pb"],
            "ev_ebitda": current_model_values["ev_ebitda"],
        })

        source_used = f"{source_stock} (stock prices), {source_mkt} (market prices), Yahoo (statements), Excel (risk-free)"

        # ---------- FINAL RESPONSE (JSON safe) ----------
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
                    "dcf": current_models["dcf"],
                    "pe_model": current_models["pe"],
                    "pb_model": current_models["pb"],
                    "ev_ebitda_model": current_models["ev_ebitda"],
                    "calibration_k": float(k) if np.isfinite(k) else None,
                    "backtest_mape_percent": None if opt.get("mape") is None else float(opt["mape"]),
                }
            },
            "optimized_weights": {
                "dcf": float(weights_full[0]),
                "pe": float(weights_full[1]),
                "pb": float(weights_full[2]),
                "ev_ebitda": float(weights_full[3]),
            },
            "metrics": {
                "market_cap": float(mcap),
                "pe_ratio": pe_ratio_now,
                "eps": eps_now,
                "book_value": book_now,
                "growth_rate": float(g) if np.isfinite(g) else None,
                "wacc": float(WACC) if np.isfinite(WACC) else None,
                "beta": float(beta) if np.isfinite(beta) else None,
                "high52": float(max(prices_list[-252:])) if len(prices_list) >= 252 else float(max(prices_list)),
                "low52": float(min(prices_list[-252:])) if len(prices_list) >= 252 else float(min(prices_list)),
                "rf": float(rf) if np.isfinite(rf) else None,
                "market_return": float(rm_exp) if np.isfinite(rm_exp) else None,
                "erp": float(erp) if np.isfinite(erp) else None,
                "cost_of_equity": float(Re) if np.isfinite(Re) else None,
                "cost_of_debt": None if Rd is None else float(Rd),
                "tax_rate": float(T) if np.isfinite(T) else None,
                "debt": float(D) if np.isfinite(D) else None,
                "cash": float(cash) if np.isfinite(cash) else None,
                "net_debt": float(net_debt) if np.isfinite(net_debt) else None,
                "method_flags": method_flags,
                "multiples_targets": {
                    "target_pe_median": target_pe,
                    "target_pb_median": target_pb,
                    "target_ev_ebitda_median": target_ev_ebitda,
                },
            },
            "returns": returns,
            "backtest": backtest_points,
            "historical_data": {
                "dates": dates_ms,
                "prices": [float(x) for x in prices_list],
                "fair_values": fair_values_full,
            },
            "source_used": source_used,
            "is_dynamic_beta": True,
            "is_synthetic_beta": False,
            "is_dynamic_growth": True,
            "is_synthetic_growth": False
        }

        return JSONResponse(content=json_safe(result), status_code=200)

    except Exception as e:
        # Always return JSON to prevent frontend JSON.parse failures
        return JSONResponse({"error": f"{type(e).__name__}: {str(e)}"}, status_code=200)


# =========================================================
# 10) RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
