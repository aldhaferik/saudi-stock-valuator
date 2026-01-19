import streamlit as st
import yfinance as yf
import pandas as pd

class SaudiStockLoader:
    def __init__(self):
        self.suffix = ".SR"

    def fetch_full_data(self, stock_code):
        # Clean input (remove existing suffix to be safe)
        base_code = str(stock_code).replace(".SR", "").replace(".SA", "")
        
        # --- ATTEMPT 1: Standard .SR Suffix (Bulk Download Method) ---
        symbol_sr = f"{base_code}.SR"
        print(f"🕵️‍♂️ Trying {symbol_sr} via yf.download...")
        data = self._fetch_via_download(symbol_sr)
        if data: return data

        # --- ATTEMPT 2: Alternative .SA Suffix (Rare fallback) ---
        symbol_sa = f"{base_code}.SA"
        print(f"🕵️‍♂️ Trying {symbol_sa} via yf.download...")
        data = self._fetch_via_download(symbol_sa)
        if data: return data
        
        # --- ATTEMPT 3: Ticker Object Method (Last Resort) ---
        print(f"🕵️‍♂️ Trying {symbol_sr} via Ticker object...")
        data = self._fetch_via_ticker(symbol_sr)
        if data: return data

        return None

    def _fetch_via_download(self, symbol):
        try:
            # yf.download is often more robust against blocking than Ticker
            df = yf.download(symbol, period="10y", progress=False)
            
            if df.empty: return None
            
            # CLEANUP: yf.download returns MultiIndex columns sometimes. Flatten them.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Get Ticker object just for metadata (fast)
            ticker = yf.Ticker(symbol)
            
            return self._package_data(ticker.info, df, ticker.balance_sheet, ticker.income_stmt, ticker.cashflow)
        except Exception as e:
            print(f"   ❌ Download Error: {e}")
            return None

    def _fetch_via_ticker(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            prices = ticker.history(period="10y")
            if prices.empty: return None
            return self._package_data(ticker.info, prices, ticker.balance_sheet, ticker.income_stmt, ticker.cashflow)
        except Exception as e:
            print(f"   ❌ Ticker Error: {e}")
            return None

    def _package_data(self, meta, prices, bs, is_, cf):
        # 1. Timezone Strip (Critical Fix)
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        # 2. ETF Detection
        # Check QuoteType OR missing financials
        q_type = meta.get('quoteType', '').upper()
        # Some ETFs return empty dict for info, so check balance sheet too
        no_financials = bs.empty if bs is not None else True
        
        is_etf = (q_type == 'ETF') or (q_type == 'MUTUALFUND') or (no_financials)

        def sanitize(df):
            if df is None or df.empty: return df
            # Convert columns to datetime and strip timezone
            try:
                df.columns = pd.to_datetime(df.columns)
                if df.columns.tz is not None:
                    df.columns = df.columns.tz_localize(None)
            except:
                pass # Keep original columns if they aren't dates
            return df
        
        return {
            "meta": meta,
            "prices": prices,
            "is_etf": is_etf,
            "financials": {
                "balance_sheet": sanitize(bs), 
                "income_statement": sanitize(is_), 
                "cash_flow": sanitize(cf)
            }
        }

    def get_data_as_of_date(self, stock_data, valuation_date_str):
        if stock_data.get("is_etf", False): return None 

        cutoff_date = pd.to_datetime(valuation_date_str)
        
        prices = stock_data["prices"].copy()
        past_prices = prices[prices.index < cutoff_date]
        
        if past_prices.empty: return None
        simulated_current_price = past_prices['Close'].iloc[-1]

        def filter_financials(df):
            if df is None or df.empty: return df
            # Filter columns that are dates
            valid_cols = []
            for c in df.columns:
                if isinstance(c, pd.Timestamp) and c < cutoff_date:
                    valid_cols.append(c)
            return df[valid_cols]

        past_financials = {
            "balance_sheet": filter_financials(stock_data["financials"]["balance_sheet"]),
            "income_statement": filter_financials(stock_data["financials"]["income_statement"]),
            "cash_flow": filter_financials(stock_data["financials"]["cash_flow"])
        }

        return {
            "simulation_date": valuation_date_str,
            "price_at_simulation": simulated_current_price,
            "financials": past_financials
        }
