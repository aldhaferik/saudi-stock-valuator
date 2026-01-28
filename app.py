# app.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Optional BS4 (not required)
try:
    from bs4 import BeautifulSoup  # noqa: F401
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# =========================================================
# CONFIG
# =========================================================
DEFAULT_HISTORY_PERIOD = "5y"
TASI_TICKER = "^TASI.SR"

# Backtest + estimation windows
BACKTEST_YEARS = 5
BACKTEST_STEP_TRADING_DAYS = 21  # monthly-ish
BETA_LOOKBACK_DAYS = 252 * 2     # regression beta window (2 years)
MARKET_RETURN_LOOKBACK_DAYS = 252 * 5  # market expected return window (5 years)

FORECAST_YEARS = 5

# Your API keys (you asked to add these backup sources)
ALPHAVANTAGE_API_KEY = "0LR5JLOBSLOA6Z0A"
TWELVEDATA_API_KEY = "ed240f406bab4225ac6e0a98be553aa2"

# Risk-free source (local Excel committed in your repo)
SAUDI_YIELDS_XLSX = "saudi_yields.xlsx"
SAUDI_RF_COLNAME = "10-Year government bond yield"  # column name in your excel

# =========================================================
# APP
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
# HELPERS
# =========================================================
class StockRequest(BaseModel):
    ticker: str


def _clean_saudi_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if t.replace(".", "").isdigit() and not t.endswith(".SR"):
        return f"{t}.SR"
    return t


def _to_ms_index(dt_index: pd.DatetimeIndex) -> list[int]:
    return (dt_index.astype(np.int64) // 10**6).tolist()


def _as_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _most_recent_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    try:
        return sorted(cols, reverse=True)[0]
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


def _safe_get_line(df: pd.DataFrame, possible_names: list[str], col) -> float | None:
    if df is None or df.empty or col is None:
        return None
    for name in possible_names:
        if name in df.index:
            v = df.loc[name, col]
            if pd.isna(v):
                continue
            try:
                return float(v)
            except Exception:
                continue
    return None


def _safe_get_line_contains(df: pd.DataFrame, must_contain: list[str], col) -> float | None:
    """
    Tries to find a row whose label contains all tokens in must_contain (case-insensitive).
    Example: ["operating", "income"].
    """
    if df is None or df.empty or col is None:
        return None
    tokens = [t.lower() for t in must_contain]
    for idx in df.index:
        s = str(idx).lower()
        if all(t in s for t in tokens):
            v = df.loc[idx, col]
            if pd.isna(v):
                continue
            try:
                return float(v)
            except Exception:
                continue
    return None


def _annualized_geo_mean_return(prices: pd.Series, periods_per_year: int = 252) -> float:
    prices = prices.dropna().astype(float)
    if len(prices) < periods_per_year + 2:
        raise ValueError("Not enough history to estimate annualized return.")
    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    n_periods = len(prices) - 1
    years = n_periods / periods_per_year
    if start <= 0 or end <= 0 or years <= 0:
        raise ValueError("Invalid price series for return estimation.")
    return (end / start) ** (1.0 / years) - 1.0


def _beta_regression(stock_prices: pd.Series, market_prices: pd.Series) -> float:
    df = pd.DataFrame({"s": stock_prices.astype(float), "m": market_prices.astype(float)}).dropna()
    if len(df) < 120:
        raise ValueError("Not enough overlapping history for beta.")
    rs = np.log(df["s"]).diff().dropna()
    rm = np.log(df["m"]).diff().dropna()
    aligned = pd.DataFrame({"rs": rs, "rm": rm}).dropna()
    if len(aligned) < 120:
        raise ValueError("Not enough return observations for beta.")
    cov = np.cov(aligned["rs"], aligned["rm"], ddof=1)[0, 1]
    var = np.var(aligned["rm"], ddof=1)
    if var <= 0:
        raise ValueError("Market variance non-positive.")
    return float(cov / var)


def _nearest_prior(series: pd.Series, asof: pd.Timestamp) -> float | None:
    """
    Given a Series indexed by datetime (sorted ascending), return value at the latest
    index <= asof.
    """
    if series is None or series.empty:
        return None
    s = series.dropna()
    if s.empty:
        return None
    s = s.sort_index()
    if asof < s.index[0]:
        return None
    # pandas asof works on sorted index
    try:
        v = s.asof(asof)
        return None if pd.isna(v) else float(v)
    except Exception:
        # fallback manual
        idx = s.index[s.index <= asof]
        if len(idx) == 0:
            return None
        return float(s.loc[idx[-1]])


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, np.nan, y_true)
    ape = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.nanmean(ape))


# =========================================================
# DATA FETCHER (Yahoo primary, AlphaVantage + TwelveData backups for PRICES)
# =========================================================
class DataFetcher:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36",
        ]

    def _headers(self):
        return {"User-Agent": np.random.choice(self.user_agents)}

    # -----------------------------
    # PRICES
    # -----------------------------
    def fetch_prices_yahoo(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> pd.DataFrame:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError(f"No Yahoo price history for {ticker}.")
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index)
        return hist

    def fetch_prices_alphavantage(self, ticker: str) -> pd.DataFrame:
        """
        Alpha Vantage TIME_SERIES_DAILY (close only). Returns DataFrame with Close.
        NOTE: AV uses .SA for Saudi tickers (often), but user previously used that mapping.
        We'll try both the passed ticker and the .SA variant if relevant.
        """
        symbol_try = []
        t = ticker.upper().strip()
        symbol_try.append(t.replace(".SR", ".SA"))
        symbol_try.append(t)

        last_err = None
        for sym in symbol_try:
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": sym,
                    "outputsize": "full",
                    "apikey": ALPHAVANTAGE_API_KEY,
                }
                r = requests.get(url, params=params, timeout=12)
                data = r.json()
                ts = data.get("Time Series (Daily)")
                if not ts:
                    last_err = f"No daily series for {sym}"
                    continue
                df = pd.DataFrame.from_dict(ts, orient="index")
                # keys like "4. close"
                if "4. close" not in df.columns:
                    last_err = f"AV schema missing close for {sym}"
                    continue
                df = df.rename(columns={"4. close": "Close"})
                df.index = pd.to_datetime(df.index)
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                df = df.sort_index()
                df = df.dropna(subset=["Close"])
                if df.empty:
                    last_err = f"AV close empty for {sym}"
                    continue
                return df[["Close"]]
            except Exception as e:
                last_err = str(e)
        raise ValueError(f"AlphaVantage price fetch failed: {last_err}")

    def fetch_prices_twelvedata(self, ticker: str) -> pd.DataFrame:
        """
        Twelve Data time_series endpoint.
        """
        # TwelveData expects an exchange-specific symbol sometimes. We'll try raw ticker and .SR mapping.
        symbol_try = []
        t = ticker.upper().strip()
        symbol_try.append(t.replace(".SR", ".SR"))  # keep
        symbol_try.append(t.replace(".SR", ""))     # sometimes works without suffix

        last_err = None
        for sym in symbol_try:
            try:
                url = "https://api.twelvedata.com/time_series"
                params = {
                    "symbol": sym,
                    "interval": "1day",
                    "outputsize": 5000,
                    "apikey": TWELVEDATA_API_KEY,
                    "format": "JSON",
                }
                r = requests.get(url, params=params, timeout=12)
                data = r.json()
                if data.get("status") == "error":
                    last_err = data.get("message", "Unknown TwelveData error")
                    continue
                values = data.get("values")
                if not values:
                    last_err = "No values returned"
                    continue
                df = pd.DataFrame(values)
                if "datetime" not in df.columns or "close" not in df.columns:
                    last_err = "Unexpected TwelveData schema"
                    continue
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["Close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df.set_index("datetime").sort_index()
                df = df.dropna(subset=["Close"])
                if df.empty:
                    last_err = "TwelveData close empty"
                    continue
                return df[["Close"]]
            except Exception as e:
                last_err = str(e)
        raise ValueError(f"TwelveData price fetch failed: {last_err}")

    def fetch_prices(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> tuple[pd.DataFrame, str]:
        """
        Returns (hist, source_used)
        """
        # 1) Yahoo
        try:
            return self.fetch_prices_yahoo(ticker, period=period), "Yahoo Finance"
        except Exception:
            pass
        # 2) AlphaVantage
        try:
            df = self.fetch_prices_alphavantage(ticker)
            # keep only last ~5y trading days
            df = df.tail(252 * 6)
            return df, "Alpha Vantage"
        except Exception:
            pass
        # 3) TwelveData
        df = self.fetch_prices_twelvedata(ticker)
        df = df.tail(252 * 6)
        return df, "Twelve Data"

    # -----------------------------
    # STATEMENTS (Yahoo only here)
    # -----------------------------
    def fetch_statements_yahoo(self, ticker: str):
        """
        yfinance statements. Availability varies by ticker.
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

        return {"info": info, "financials": fin, "balance_sheet": bs, "cashflow": cf}

    # -----------------------------
    # RISK-FREE (Excel)
    # -----------------------------
    def load_saudi_yields_excel(self, xlsx_path: str = SAUDI_YIELDS_XLSX) -> pd.DataFrame:
        if not os.path.exists(xlsx_path):
            raise ValueError(f"Saudi yields file not found: {xlsx_path}")
        df = pd.read_excel(xlsx_path)
        if "TIME" not in df.columns:
            raise ValueError("Excel must include a TIME column.")
        df = df.copy()
        df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        df = df.dropna(subset=["TIME"])
        df = df.sort_values("TIME")
        # convert '..' and strings to numeric
        for c in df.columns:
            if c == "TIME":
                continue
            df[c] = pd.to_numeric(df[c].replace("..", np.nan), errors="coerce")
        return df

    def build_rf_series(self, yields_df: pd.DataFrame, colname: str = SAUDI_RF_COLNAME) -> pd.Series:
        if colname not in yields_df.columns:
            raise ValueError(f"Column not found in yields Excel: {colname}")
        s = yields_df.set_index("TIME")[colname].dropna().astype(float)
        # yields in percent -> convert to decimal
        s = s / 100.0
        # sanity: keep only plausible values
        s = s[(s > 0) & (s < 0.50)]
        if s.empty:
            raise ValueError("Risk-free series is empty after cleaning.")
        return s


# =========================================================
# VALUATION CORE (compute model values at an AS-OF date)
# =========================================================
def _statement_col_asof(df: pd.DataFrame, asof: pd.Timestamp):
    """
    Pick the latest statement column whose timestamp <= asof.
    yfinance statement columns are typically period-end Timestamps.
    """
    if df is None or df.empty:
        return None
    cols = []
    for c in df.columns:
        try:
            ts = pd.to_datetime(c)
            cols.append(ts)
        except Exception:
            continue
    if not cols:
        # fallback to "most recent"
        return _most_recent_col(df)
    cols = sorted(cols)
    prior = [c for c in cols if c <= asof]
    if not prior:
        return None
    return prior[-1]


def _prev_statement_col(df: pd.DataFrame, col_ts: pd.Timestamp):
    if df is None or df.empty or col_ts is None:
        return None
    cols = []
    for c in df.columns:
        try:
            cols.append(pd.to_datetime(c))
        except Exception:
            continue
    cols = sorted(cols)
    prev = [c for c in cols if c < col_ts]
    return prev[-1] if prev else None


def _compute_targets_from_history(implied: pd.Series, asof: pd.Timestamp) -> float | None:
    """
    Target multiple = median of implied multiples up to asof (rolling-ish).
    """
    s = implied.dropna()
    if s.empty:
        return None
    s = s.sort_index()
    s = s[s.index <= asof]
    if len(s) < 30:
        return None
    return float(np.nanmedian(s.values))


def _dirichlet_weights(n: int, k: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha=np.ones(n), size=k)


def _optimize_weights(model_matrix: np.ndarray, actual: np.ndarray) -> dict:
    """
    model_matrix: shape (N, M) (N points, M models)
    actual: shape (N,)
    Weights w >= 0, sum(w)=1. Also fit a scalar calibration k for each w:
        k = argmin || actual - k*(Xw) ||^2  => k = (a·p)/(p·p)
    Objective: MAPE on calibrated predictions.
    """
    N, M = model_matrix.shape
    if N < 30:
        raise ValueError("Not enough backtest points to optimize weights.")
    if M < 2:
        raise ValueError("Need at least 2 models for combination optimization.")

    # quick prefilter: remove rows with NaN
    mask = np.isfinite(actual)
    for j in range(M):
        mask &= np.isfinite(model_matrix[:, j])
    X = model_matrix[mask]
    y = actual[mask]
    if len(y) < 30:
        raise ValueError("Not enough complete points after NaN filtering.")

    best = {"mape": float("inf"), "weights": None, "k": 1.0}

    # 1) coarse grid via Dirichlet random search
    W = _dirichlet_weights(M, k=8000, seed=7)
    for w in W:
        pred = X @ w
        denom = float(np.dot(pred, pred))
        if denom <= 0 or not np.isfinite(denom):
            continue
        k = float(np.dot(y, pred) / denom)
        pred2 = k * pred
        mape = _mape(y, pred2)
        if np.isfinite(mape) and mape < best["mape"]:
            best = {"mape": float(mape), "weights": w.copy(), "k": float(k)}

    # 2) local refinement around best with small noise
    rng = np.random.default_rng(42)
    w0 = best["weights"]
    if w0 is None:
        raise ValueError("Optimizer could not find valid weights.")

    for _ in range(4000):
        noise = rng.normal(0, 0.06, size=M)
        w = np.clip(w0 + noise, 0, None)
        s = float(w.sum())
        if s <= 0:
            continue
        w = w / s
        pred = X @ w
        denom = float(np.dot(pred, pred))
        if denom <= 0 or not np.isfinite(denom):
            continue
        k = float(np.dot(y, pred) / denom)
        pred2 = k * pred
        mape = _mape(y, pred2)
        if np.isfinite(mape) and mape < best["mape"]:
            best = {"mape": float(mape), "weights": w.copy(), "k": float(k)}

    return best


def _build_implied_multiples_series(
    prices: pd.Series,
    shares: float,
    fin: pd.DataFrame,
    bs: pd.DataFrame,
    cf: pd.DataFrame,
    debt_series: pd.Series | None,
    cash_series: pd.Series | None,
) -> dict:
    """
    Build implied multiples time series using “latest available annual statements”
    mapped to dates.

    Output:
      implied_pe: price / eps
      implied_pb: price / bvps
      implied_ev_ebitda: EV / EBITDA
    """
    if prices is None or prices.dropna().empty:
        return {}

    # Build annual EPS/BVPS/EBITDA series indexed by statement dates
    # then forward-fill daily to align with prices.

    # Statement columns as timestamps
    def _cols_as_ts(df):
        if df is None or df.empty:
            return []
        out = []
        for c in df.columns:
            try:
                out.append(pd.to_datetime(c))
            except Exception:
                pass
        return sorted(out)

    fin_cols = _cols_as_ts(fin)
    bs_cols = _cols_as_ts(bs)
    cf_cols = _cols_as_ts(cf)

    # Use intersection-ish: prefer fin cols for EPS/EBITDA, bs cols for BVPS, etc.
    eps_annual = {}
    bvps_annual = {}
    ebitda_annual = {}
    debt_annual = {}
    cash_annual = {}

    # EPS from Net Income / shares
    for c in fin_cols:
        net_income = _safe_get_line(fin, ["Net Income", "NetIncome", "Net Income Common Stockholders"], c)
        if net_income is None:
            continue
        if shares <= 0:
            continue
        eps_annual[c] = float(net_income) / float(shares)

        # EBITDA
        ebitda = _safe_get_line(fin, ["Ebitda", "EBITDA"], c)
        if ebitda is None:
            # try approximate: EBIT + D&A
            ebit = _safe_get_line(fin, ["Ebit", "EBIT", "Operating Income", "OperatingIncome"], c)
            da = None
            # D&A sometimes in cashflow for same date, but columns may not align; try nearest
            if cf is not None and not cf.empty:
                cf_c = c if c in cf.columns else _statement_col_asof(cf, c)
                da = _safe_get_line(cf, ["Depreciation", "Depreciation And Amortization"], cf_c)
            if ebit is not None and da is not None:
                ebitda = float(ebit) + float(da)
        if ebitda is not None:
            ebitda_annual[c] = float(ebitda)

    # BVPS from Total Stockholder Equity / shares
    for c in bs_cols:
        equity = _safe_get_line(bs, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Stockholders Equity"], c)
        if equity is None:
            continue
        if shares <= 0:
            continue
        bvps_annual[c] = float(equity) / float(shares)

        # debt/cash (for EV)
        st_debt = _safe_get_line(bs, ["Short Long Term Debt", "Short Term Debt", "Current Debt", "ShortTermDebt"], c) or 0.0
        lt_debt = _safe_get_line(bs, ["Long Term Debt", "LongTermDebt"], c) or 0.0
        debt_annual[c] = float(st_debt) + float(lt_debt)

        cash = _safe_get_line(
            bs,
            ["Cash", "Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments", "CashAndCashEquivalents"],
            c
        ) or 0.0
        cash_annual[c] = float(cash)

    # Convert annual dicts to series
    def _to_series(d: dict):
        if not d:
            return None
        s = pd.Series(d)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        return s

    eps_s = _to_series(eps_annual)
    bvps_s = _to_series(bvps_annual)
    ebitda_s = _to_series(ebitda_annual)
    debt_s = _to_series(debt_annual)
    cash_s = _to_series(cash_annual)

    # Align to daily prices by forward-filling last known annual value
    idx = prices.index
    out = {}

    if eps_s is not None:
        eps_daily = eps_s.reindex(idx, method="ffill")
        implied_pe = prices / eps_daily.replace(0, np.nan)
        out["implied_pe"] = implied_pe.replace([np.inf, -np.inf], np.nan)

    if bvps_s is not None:
        bvps_daily = bvps_s.reindex(idx, method="ffill")
        implied_pb = prices / bvps_daily.replace(0, np.nan)
        out["implied_pb"] = implied_pb.replace([np.inf, -np.inf], np.nan)

    if ebitda_s is not None and debt_s is not None and cash_s is not None:
        ebitda_daily = ebitda_s.reindex(idx, method="ffill")
        debt_daily = debt_s.reindex(idx, method="ffill")
        cash_daily = cash_s.reindex(idx, method="ffill")
        mcap_daily = prices * float(shares)
        ev_daily = mcap_daily + debt_daily - cash_daily
        implied_ev_ebitda = ev_daily / ebitda_daily.replace(0, np.nan)
        out["implied_ev_ebitda"] = implied_ev_ebitda.replace([np.inf, -np.inf], np.nan)

    return out


def _compute_models_asof(
    asof: pd.Timestamp,
    prices_stock: pd.Series,
    prices_mkt: pd.Series,
    rf_series: pd.Series,
    info: dict,
    fin: pd.DataFrame,
    bs: pd.DataFrame,
    cf: pd.DataFrame,
    implied_multiples: dict,
) -> dict:
    """
    Compute per-share valuations at `asof`:
      - DCF (FCFF-based, statement-driven)
      - P/E
      - P/B
      - EV/EBITDA
    Returns dict with model values and key diagnostics.
    If a model can’t be computed from available data, its value is None.
    """
    out = {
        "dcf": None,
        "pe": None,
        "pb": None,
        "ev_ebitda": None,
        "wacc": None,
        "beta": None,
        "growth": None,
        "rf": None,
        "market_return": None,
        "erp": None,
        "cost_of_equity": None,
        "cost_of_debt": None,
        "tax_rate": None,
        "method_flags": [],
    }

    # current price at asof
    if prices_stock is None or prices_stock.dropna().empty:
        return out
    if asof not in prices_stock.index:
        # nearest prior trading day
        asof_price = _nearest_prior(prices_stock, asof)
        if asof_price is None:
            return out
        P0 = float(asof_price)
    else:
        P0 = float(prices_stock.loc[asof])

    shares = info.get("sharesOutstanding")
    if shares is None:
        out["method_flags"].append("missing_sharesOutstanding")
        return out
    try:
        shares = float(shares)
        if shares <= 0:
            out["method_flags"].append("invalid_sharesOutstanding")
            return out
    except Exception:
        out["method_flags"].append("invalid_sharesOutstanding")
        return out

    # Risk-free at asof (from your Excel series)
    rf = _nearest_prior(rf_series, asof)
    if rf is None:
        out["method_flags"].append("missing_rf_at_date")
        return out
    out["rf"] = float(rf)

    # Market expected return at asof (from TASI price history window)
    mkt_slice = prices_mkt.dropna()
    mkt_slice = mkt_slice[mkt_slice.index <= asof].tail(MARKET_RETURN_LOOKBACK_DAYS)
    if len(mkt_slice) < 252 + 2:
        out["method_flags"].append("missing_market_window")
        return out
    rm_exp = _annualized_geo_mean_return(mkt_slice)
    erp = rm_exp - rf
    out["market_return"] = float(rm_exp)
    out["erp"] = float(erp)

    # Beta regression at asof
    s_slice = prices_stock.dropna()
    s_slice = s_slice[s_slice.index <= asof].tail(BETA_LOOKBACK_DAYS)
    m_slice = prices_mkt.dropna()
    m_slice = m_slice[m_slice.index <= asof].tail(BETA_LOOKBACK_DAYS)
    if len(s_slice) < 200 or len(m_slice) < 200:
        out["method_flags"].append("missing_beta_window")
        return out
    beta = _beta_regression(s_slice, m_slice)
    out["beta"] = float(beta)

    # Cost of equity
    Re = rf + beta * erp
    out["cost_of_equity"] = float(Re)

    # Statement columns as-of
    fin_col = _statement_col_asof(fin, asof)
    bs_col = _statement_col_asof(bs, asof)
    cf_col = _statement_col_asof(cf, asof)
    if fin_col is None or bs_col is None or cf_col is None:
        out["method_flags"].append("missing_statements_asof")
        # still try multiples (can be computed from annual series already)
    bs_col_prev = _prev_statement_col(bs, bs_col) if bs_col is not None else None

    # Market cap at asof
    E = P0 * shares

    # Debt + cash (for WACC and EV)
    D = None
    cash = None
    if bs_col is not None:
        st_debt = _safe_get_line(bs, ["Short Long Term Debt", "Short Term Debt", "Current Debt", "ShortTermDebt"], bs_col) or 0.0
        lt_debt = _safe_get_line(bs, ["Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation"], bs_col) or 0.0
        D = float(st_debt) + float(lt_debt)

        cash = _safe_get_line(
            bs,
            ["Cash", "Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments", "CashAndCashEquivalents"],
            bs_col
        ) or 0.0
        cash = float(cash)

    # Taxes
    T = 0.0
    if fin_col is not None:
        pretax = _safe_get_line(fin, ["Pretax Income", "Income Before Tax", "IncomeBeforeTax"], fin_col)
        tax_exp = _safe_get_line(fin, ["Tax Provision", "Income Tax Expense", "IncomeTaxExpense"], fin_col)
        if pretax is not None and tax_exp is not None and pretax > 0:
            T = float(tax_exp) / float(pretax)
            # bound only as a sanity guard
            T = max(0.0, min(T, 0.45))
    out["tax_rate"] = float(T)

    # Cost of debt Rd (try interest expense OR interest paid; if missing, DCF will be unavailable)
    Rd = None
    if D is not None and D > 0:
        interest_exp = None
        interest_paid = None
        if fin_col is not None:
            interest_exp = _safe_get_line(fin, ["Interest Expense", "InterestExpense", "Interest Expense Non Operating"], fin_col)
        if cf_col is not None:
            interest_paid = _safe_get_line(cf, ["Interest Paid", "InterestPaid", "Cash Interest Paid", "CashInterestPaid"], cf_col)

        if bs_col_prev is None:
            out["method_flags"].append("missing_prev_balance_sheet_for_Rd")
        else:
            st_debt_prev = _safe_get_line(bs, ["Short Long Term Debt", "Short Term Debt", "Current Debt", "ShortTermDebt"], bs_col_prev) or 0.0
            lt_debt_prev = _safe_get_line(bs, ["Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation"], bs_col_prev) or 0.0
            D_prev = float(st_debt_prev) + float(lt_debt_prev)
            avg_D = (D + D_prev) / 2.0
            if avg_D > 0:
                if interest_exp is not None:
                    Rd = float(interest_exp) / avg_D
                elif interest_paid is not None:
                    Rd = abs(float(interest_paid)) / avg_D

    if Rd is not None:
        out["cost_of_debt"] = float(Rd)

    # WACC only if we have a usable capital structure and Rd if debt exists
    WACC = None
    if D is None:
        out["method_flags"].append("missing_debt_cash")
    else:
        total_cap = D + E
        if total_cap > 0:
            wE = E / total_cap
            wD = D / total_cap
            if D > 0 and Rd is None:
                out["method_flags"].append("missing_Rd")
            else:
                WACC = wE * Re + wD * (0.0 if Rd is None else Rd) * (1.0 - T)
                out["wacc"] = float(WACC)

    # -----------------------------
    # MULTIPLES MODELS (data-driven target multiples)
    # -----------------------------
    # EPS, BVPS, EBITDA at asof (forward-filled annual series)
    pe_target = None
    pb_target = None
    ev_ebitda_target = None

    if implied_multiples.get("implied_pe") is not None:
        pe_target = _compute_targets_from_history(implied_multiples["implied_pe"], asof)
    if implied_multiples.get("implied_pb") is not None:
        pb_target = _compute_targets_from_history(implied_multiples["implied_pb"], asof)
    if implied_multiples.get("implied_ev_ebitda") is not None:
        ev_ebitda_target = _compute_targets_from_history(implied_multiples["implied_ev_ebitda"], asof)

    # Use the last known annual EPS/BVPS/EBITDA at asof to compute fair values from targets.
    # We recompute from statements at fin_col/bs_col to keep consistent with as-of.
    eps = None
    bvps = None
    ebitda = None

    if fin_col is not None:
        net_income = _safe_get_line(fin, ["Net Income", "NetIncome", "Net Income Common Stockholders"], fin_col)
        if net_income is not None:
            eps = float(net_income) / shares

        ebitda = _safe_get_line(fin, ["Ebitda", "EBITDA"], fin_col)
        if ebitda is None:
            ebit = _safe_get_line(fin, ["Ebit", "EBIT", "Operating Income", "OperatingIncome"], fin_col)
            da = _safe_get_line(cf, ["Depreciation", "Depreciation And Amortization"], cf_col) if cf_col is not None else None
            if ebit is not None and da is not None:
                ebitda = float(ebit) + float(da)

    if bs_col is not None:
        equity = _safe_get_line(bs, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Stockholders Equity"], bs_col)
        if equity is not None:
            bvps = float(equity) / shares

    if eps is not None and pe_target is not None and np.isfinite(pe_target) and pe_target > 0:
        out["pe"] = float(eps * pe_target)

    if bvps is not None and pb_target is not None and np.isfinite(pb_target) and pb_target > 0:
        out["pb"] = float(bvps * pb_target)

    if ebitda is not None and ev_ebitda_target is not None and D is not None and cash is not None:
        # Equity value per share from EV/EBITDA target:
        # EV* = multiple * EBITDA
        # Equity* = EV* - Debt + Cash
        ev_star = float(ev_ebitda_target) * float(ebitda)
        eq_star = ev_star - float(D) + float(cash)
        out["ev_ebitda"] = float(eq_star / shares)

    # -----------------------------
    # DCF MODEL (FCFF-based, statement-driven)
    # -----------------------------
    dcf_val = None
    g = None

    if WACC is None or (D is not None and D > 0 and Rd is None):
        out["method_flags"].append("dcf_unavailable_missing_wacc_or_rd")
    else:
        # EBIT
        EBIT = None
        if fin_col is not None:
            EBIT = _safe_get_line(fin, ["Ebit", "EBIT", "Operating Income", "OperatingIncome"], fin_col)
            if EBIT is None:
                # last chance: contains search
                EBIT = _safe_get_line_contains(fin, ["operating", "income"], fin_col)
        if EBIT is None:
            out["method_flags"].append("missing_EBIT")
        else:
            # D&A
            DA = None
            if cf_col is not None:
                DA = _safe_get_line(cf, ["Depreciation", "Depreciation And Amortization", "Depreciation & Amortization"], cf_col)
            if DA is None and fin_col is not None:
                DA = _safe_get_line(fin, ["Reconciled Depreciation", "Depreciation And Amortization"], fin_col)
            if DA is None:
                out["method_flags"].append("missing_DA")
            else:
                # CapEx (usually negative on cashflow)
                CapEx = None
                if cf_col is not None:
                    CapEx = _safe_get_line(cf, ["Capital Expenditures", "Capital Expenditure", "CapitalExpenditures"], cf_col)
                if CapEx is None:
                    out["method_flags"].append("missing_CapEx")
                else:
                    # ΔWC (best effort; if missing, we DO NOT hard-code a number; we flag and set to 0)
                    dWC = 0.0
                    if bs_col is None or bs_col_prev is None:
                        out["method_flags"].append("missing_dWC_inputs_set_0")
                    else:
                        def net_working_capital(bs_df: pd.DataFrame, col):
                            tca = _safe_get_line(bs_df, ["Total Current Assets"], col)
                            tcl = _safe_get_line(bs_df, ["Total Current Liabilities"], col)
                            if tca is None or tcl is None:
                                return None
                            cash_local = _safe_get_line(
                                bs_df,
                                ["Cash", "Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments", "CashAndCashEquivalents"],
                                col
                            ) or 0.0
                            st_debt_local = _safe_get_line(
                                bs_df,
                                ["Short Long Term Debt", "Short Term Debt", "Current Debt", "ShortTermDebt"],
                                col
                            ) or 0.0
                            return (float(tca) - float(cash_local)) - (float(tcl) - float(st_debt_local))

                        NWC_now = net_working_capital(bs, bs_col)
                        NWC_prev = net_working_capital(bs, bs_col_prev)
                        if NWC_now is None or NWC_prev is None:
                            out["method_flags"].append("missing_dWC_inputs_set_0")
                            dWC = 0.0
                        else:
                            dWC = float(NWC_now - NWC_prev)

                    NOPAT = float(EBIT) * (1.0 - float(T))

                    # FCFF0
                    capex_outflow = -float(CapEx) if float(CapEx) < 0 else float(CapEx)
                    FCFF0 = float(NOPAT) + float(DA) - float(capex_outflow) - float(dWC)

                    if not np.isfinite(FCFF0) or FCFF0 <= 0:
                        out["method_flags"].append("nonpositive_FCFF0")
                    else:
                        # Growth g = ROIC * reinvestment rate
                        invested_capital = None
                        if D is not None and cash is not None:
                            invested_capital = float(E) + float(D) - float(cash)
                        if invested_capital is None or invested_capital <= 0:
                            out["method_flags"].append("missing_invested_capital")
                        else:
                            ROIC = float(NOPAT) / invested_capital
                            reinvestment_rate = (float(capex_outflow) - float(DA) + float(dWC)) / float(NOPAT) if float(NOPAT) != 0 else np.nan
                            g = float(ROIC) * float(reinvestment_rate)
                            out["growth"] = float(g)

                            # Terminal growth: cap at long-run market return (data-derived)
                            g_term = min(float(g), float(rm_exp))
                            # Need WACC > g_term
                            if float(WACC) <= float(g_term):
                                out["method_flags"].append("wacc_leq_gterm")
                            else:
                                pv_sum = 0.0
                                fcff_last = float(FCFF0)
                                for i in range(1, FORECAST_YEARS + 1):
                                    fcff_i = float(FCFF0) * ((1.0 + float(g)) ** i)
                                    pv_sum += fcff_i / ((1.0 + float(WACC)) ** i)
                                    fcff_last = fcff_i

                                TV = (fcff_last * (1.0 + float(g_term))) / (float(WACC) - float(g_term))
                                PV_TV = TV / ((1.0 + float(WACC)) ** FORECAST_YEARS)
                                EV = pv_sum + PV_TV
                                # Equity value = EV - (Debt - Cash)
                                if D is None or cash is None:
                                    out["method_flags"].append("missing_debt_cash_for_equity_bridge")
                                else:
                                    equity_value = float(EV) - (float(D) - float(cash))
                                    dcf_val = float(equity_value / shares)
                                    out["dcf"] = float(dcf_val)

    return out


# =========================================================
# UI (keep your existing HTML)
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # This keeps your current UI intact.
    # (It will still display the 3 weights it knows about; the API will return 4.)
    # If you want, you can later add a 4th row for EV/EBITDA in the HTML.
    return open("index.html", "r", encoding="utf-8").read() if os.path.exists("index.html") else """
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>Saudi Valuator</title></head>
    <body>
      <h3>index.html not found</h3>
      <p>Put your existing dashboard HTML into an <b>index.html</b> file in the same repo as app.py.</p>
    </body></html>
    """


# =========================================================
# MAIN ENDPOINT
# =========================================================
@app.post("/analyze")
def analyze_stock(request: StockRequest):
    # Always return JSON (avoid frontend JSON.parse failures)
    try:
        fetcher = DataFetcher()
        ticker = _clean_saudi_ticker(request.ticker)

        # 1) Prices (stock + TASI) with backups
        hist_df, src_stock = fetcher.fetch_prices(ticker, period=DEFAULT_HISTORY_PERIOD)
        mkt_df, src_mkt = fetcher.fetch_prices(TASI_TICKER, period=DEFAULT_HISTORY_PERIOD)

        if "Close" not in hist_df.columns or hist_df["Close"].dropna().empty:
            return JSONResponse({"error": "No valid Close prices for the stock."}, status_code=200)
        if "Close" not in mkt_df.columns or mkt_df["Close"].dropna().empty:
            return JSONResponse({"error": "No valid Close prices for TASI market index (^TASI.SR)."}, status_code=200)

        hist_df = hist_df.copy()
        mkt_df = mkt_df.copy()
        hist_df.index = pd.to_datetime(hist_df.index)
        mkt_df.index = pd.to_datetime(mkt_df.index)

        # Use Series for convenience
        prices_stock = hist_df["Close"].astype(float).dropna()
        prices_mkt = mkt_df["Close"].astype(float).dropna()

        current_price = float(prices_stock.iloc[-1])
        dates_ms = _to_ms_index(prices_stock.index)
        prices_list = prices_stock.tolist()

        # 2) Statements + info (Yahoo)
        pack = fetcher.fetch_statements_yahoo(ticker)
        info = pack["info"] or {}
        fin = pack["financials"]
        bs = pack["balance_sheet"]
        cf = pack["cashflow"]

        company_name = info.get("longName") or f"Saudi Stock {request.ticker}"
        sector = (info.get("sector") or "Unknown").title()

        shares = info.get("sharesOutstanding")
        if shares is None:
            return JSONResponse({"error": "Missing sharesOutstanding from statements source (Yahoo)."}, status_code=200)

        try:
            shares_f = float(shares)
            if shares_f <= 0:
                raise ValueError("sharesOutstanding <= 0")
        except Exception:
            return JSONResponse({"error": "Invalid sharesOutstanding from statements source (Yahoo)."}, status_code=200)

        # 3) Risk-free series from Excel
        yields_df = fetcher.load_saudi_yields_excel(SAUDI_YIELDS_XLSX)
        rf_series = fetcher.build_rf_series(yields_df, SAUDI_RF_COLNAME)

        # 4) Build implied multiples series from history (data-driven targets)
        implied = _build_implied_multiples_series(
            prices=prices_stock,
            shares=shares_f,
            fin=fin,
            bs=bs,
            cf=cf,
            debt_series=None,
            cash_series=None,
        )

        # 5) Build backtest dates (monthly points across last 5 years)
        # Use trading-day index from stock prices.
        n = len(prices_stock)
        if n < 252 * 2:
            return JSONResponse({"error": "Not enough price history to run a 5-year backtest."}, status_code=200)

        # last index date
        end_date = prices_stock.index[-1]
        start_cut = end_date - pd.Timedelta(days=int(365.25 * BACKTEST_YEARS) + 10)
        idx = prices_stock.index[prices_stock.index >= start_cut]
        if len(idx) < 252:
            return JSONResponse({"error": "Not enough in-range price history for 5-year backtest."}, status_code=200)

        # pick step points (about monthly)
        eval_dates = list(idx[::BACKTEST_STEP_TRADING_DAYS])
        # ensure last point included
        if eval_dates[-1] != idx[-1]:
            eval_dates.append(idx[-1])

        # 6) Compute model values at each eval date and collect backtest matrix
        rows = []
        for d in eval_dates:
            models = _compute_models_asof(
                asof=pd.Timestamp(d),
                prices_stock=prices_stock,
                prices_mkt=prices_mkt,
                rf_series=rf_series,
                info=info,
                fin=fin,
                bs=bs,
                cf=cf,
                implied_multiples=implied,
            )
            rows.append({
                "date": pd.Timestamp(d),
                "actual": float(prices_stock.loc[d]),
                "dcf": models["dcf"],
                "pe": models["pe"],
                "pb": models["pb"],
                "ev_ebitda": models["ev_ebitda"],
            })

        bt = pd.DataFrame(rows).set_index("date").sort_index()

        # 7) Optimize weights on available models
        # Keep model order fixed
        model_names = ["dcf", "pe", "pb", "ev_ebitda"]
        X = bt[model_names].to_numpy(dtype=float)
        y = bt["actual"].to_numpy(dtype=float)

        # Require at least 2 usable model columns
        usable_cols = []
        for j, name in enumerate(model_names):
            col = X[:, j]
            if np.isfinite(col).sum() >= 30:
                usable_cols.append(j)

        if len(usable_cols) < 2:
            return JSONResponse(
                {"error": "Not enough computable model series to optimize (need at least 2 of DCF/PE/PB/EVEBITDA with sufficient history)."},
                status_code=200
            )

        X_use = X[:, usable_cols]
        names_use = [model_names[j] for j in usable_cols]
        opt = _optimize_weights(X_use, y)

        w_use = opt["weights"]
        k = opt["k"]

        # expand to full 4-model weights with zeros where unavailable
        weights_full = np.zeros(4, dtype=float)
        for wi, j in enumerate(usable_cols):
            weights_full[j] = float(w_use[wi])

        # 8) Compute current model values and final fair value
        current_models = _compute_models_asof(
            asof=pd.Timestamp(prices_stock.index[-1]),
            prices_stock=prices_stock,
            prices_mkt=prices_mkt,
            rf_series=rf_series,
            info=info,
            fin=fin,
            bs=bs,
            cf=cf,
            implied_multiples=implied,
        )

        model_vec_today = np.array([
            current_models["dcf"] if current_models["dcf"] is not None else np.nan,
            current_models["pe"] if current_models["pe"] is not None else np.nan,
            current_models["pb"] if current_models["pb"] is not None else np.nan,
            current_models["ev_ebitda"] if current_models["ev_ebitda"] is not None else np.nan,
        ], dtype=float)

        # If a model isn't available today but has a weight, set that weight to 0 and renormalize
        w = weights_full.copy()
        available_today = np.isfinite(model_vec_today)
        if (w[~available_today] > 0).any():
            w[~available_today] = 0.0
            s = float(w.sum())
            if s > 0:
                w = w / s
            else:
                # nothing left
                return JSONResponse({"error": "No model values could be computed for today."}, status_code=200)

        raw_combo = float(np.nansum(w * model_vec_today))
        fair_value = float(k * raw_combo)

        upside = ((fair_value - current_price) / current_price) * 100.0
        verdict = "Fairly Valued"
        if upside > 10:
            verdict = "Undervalued"
        elif upside < -10:
            verdict = "Overvalued"

        # 9) Build fair value series for chart (apply optimized weights + calibration to each date)
        # Use available columns in backtest dataframe
        fv_series = []
        for _, r in bt.iterrows():
            mv = np.array([r["dcf"], r["pe"], r["pb"], r["ev_ebitda"]], dtype=float)
            # apply same availability logic date-by-date
            ww = weights_full.copy()
            ok = np.isfinite(mv)
            ww[~ok] = 0.0
            ss = float(ww.sum())
            if ss <= 0:
                fv_series.append(np.nan)
                continue
            ww = ww / ss
            fv_series.append(float(k * np.nansum(ww * mv)))

        # For the full 5y timeline chart, we output fair values aligned to price dates.
        # We only computed fv on eval_dates; interpolate onto full index for display.
        fv_bt = pd.Series(fv_series, index=bt.index).sort_index()
        fv_full = fv_bt.reindex(prices_stock.index, method="ffill")
        fair_values_full = fv_full.astype(float).tolist()

        # 10) Backtest sample points (1y..5y) from full fair value series
        def _get_idx_ago(days: int):
            if len(prices_stock) <= days:
                return None
            return prices_stock.index[-days]

        backtest_points = []
        for label, days in [("1 Year Ago", 252), ("2 Years Ago", 504), ("3 Years Ago", 756), ("4 Years Ago", 1008), ("5 Years Ago", 1250)]:
            d = _get_idx_ago(days)
            if d is not None and d in prices_stock.index:
                backtest_points.append({
                    "period": label,
                    "actual": float(prices_stock.loc[d]),
                    "model": float(fv_full.loc[d]) if pd.notna(fv_full.loc[d]) else float("nan")
                })

        def get_price_ago(days):
            if len(prices_list) < days:
                return current_price
            return float(prices_list[-days])

        returns = {
            "1m": ((current_price - get_price_ago(21)) / get_price_ago(21)) * 100.0,
            "3m": ((current_price - get_price_ago(63)) / get_price_ago(63)) * 100.0,
            "6m": ((current_price - get_price_ago(126)) / get_price_ago(126)) * 100.0,
            "1y": ((current_price - get_price_ago(252)) / get_price_ago(252)) * 100.0,
            "2y": ((current_price - get_price_ago(504)) / get_price_ago(504)) * 100.0,
        }

        # Diagnostics for UI
        mcap = float(current_price * shares_f)
        pe_ratio_now = None
        if info.get("trailingPE") is not None:
            pe_ratio_now = _as_float(info.get("trailingPE"))
        eps_now = _as_float(info.get("trailingEps"))
        book_now = _as_float(info.get("bookValue"))

        # Provide dcf projections if DCF is available today (FCFF forecast, not price forecast).
        # We recompute quickly from today's models flags: if dcf computed, we can’t reliably output the entire FCFF stream
        # without duplicating the whole DCF internals. So we output placeholders as None unless you want me to expand it.
        dcf_projections = []

        source_used = f"Prices: {src_stock} (stock), {src_mkt} (market); Statements: Yahoo Finance; RF: local Excel ({SAUDI_YIELDS_XLSX})"

        return JSONResponse({
            "valuation_summary": {
                "company_name": company_name,
                "fair_value": float(fair_value),
                "current_price": float(current_price),
                "verdict": verdict,
                "upside_percent": float(upside),
                "dcf_projections": dcf_projections,
                "sector": sector,
                "model_breakdown": {
                    "dcf": None if current_models["dcf"] is None else float(current_models["dcf"]),
                    "pe_model": None if current_models["pe"] is None else float(current_models["pe"]),
                    "pb_model": None if current_models["pb"] is None else float(current_models["pb"]),
                    "ev_ebitda_model": None if current_models["ev_ebitda"] is None else float(current_models["ev_ebitda"]),
                    "calibration_k": float(k),
                    "backtest_mape_percent": float(opt["mape"]),
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
                "growth_rate": None if current_models["growth"] is None else float(current_models["growth"]),
                "wacc": None if current_models["wacc"] is None else float(current_models["wacc"]),
                "beta": None if current_models["beta"] is None else float(current_models["beta"]),
                "high52": float(max(prices_list[-252:])) if len(prices_list) >= 252 else float(max(prices_list)),
                "low52": float(min(prices_list[-252:])) if len(prices_list) >= 252 else float(min(prices_list)),
                "rf": None if current_models["rf"] is None else float(current_models["rf"]),
                "market_return": None if current_models["market_return"] is None else float(current_models["market_return"]),
                "erp": None if current_models["erp"] is None else float(current_models["erp"]),
                "cost_of_equity": None if current_models["cost_of_equity"] is None else float(current_models["cost_of_equity"]),
                "cost_of_debt": None if current_models["cost_of_debt"] is None else float(current_models["cost_of_debt"]),
                "tax_rate": None if current_models["tax_rate"] is None else float(current_models["tax_rate"]),
                "method_flags": current_models["method_flags"],
            },
            "returns": returns,
            "backtest": backtest_points,
            "historical_data": {
                "dates": dates_ms,
                "prices": prices_list,
                "fair_values": fair_values_full
            },
            "source_used": source_used,
            "is_dynamic_beta": True,
            "is_synthetic_beta": False,
            "is_dynamic_growth": True,
            "is_synthetic_growth": False
        }, status_code=200)

    except Exception as e:
        # Never crash with HTTP 500; always return JSON so frontend doesn’t fail JSON.parse
        return JSONResponse({"error": f"Internal error: {type(e).__name__}: {str(e)}"}, status_code=200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
