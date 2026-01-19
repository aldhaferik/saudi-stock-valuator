import streamlit as st
import yfinance as yf
import pandas as pd
import time

class SaudiStockLoader:
    def __init__(self):
        self.suffix = ".SR"

    # --- 1. ENABLE CACHING (Prevents blocking by reducing requests) ---
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_from_yahoo_cached(symbol):
        """
        Internal function to fetch data. 
        Cached for 1 hour (3600s) to prevent hitting Yahoo repeatedly.
        """
        try:
            # Clean Symbol
            clean_symbol = symbol.strip().upper()
            
            # ATTEMPT 1: Ticker Object (Rich Data)
            ticker = yf.Ticker(clean_symbol)
            
            # Fetch History (10 years)
            # auto_adjust=True fixes weird split/dividend issues
            prices = ticker.history(period="10y", auto_adjust=True)
            
            if prices.empty:
                return None
            
            # Fetch Info (Metadata)
            try:
                # Force a refresh of info to ensure it's not stale
                info = ticker.info
            except:
                info = {}
                
            return {
                "prices": prices,
                "info": info,
                "bs": ticker.balance_sheet,
                "is_": ticker.income_stmt,
                "cf": ticker.cashflow
            }
            
        except Exception as e:
            print(f"Server Error fetching {symbol}: {e}")
            return None

    def fetch_full_data(self, stock_code):
        # Handle input
        base_code = str(stock_code).replace(".SR", "").replace(".SA", "").strip()
        symbol = f"{base_code}.SR"
        
        # Call the CACHED function
        raw_data = self._fetch_from_yahoo_cached(symbol)
        
        if raw_data is None:
            # Retry with .SA suffix if .SR failed
            symbol_sa = f"{base_code}.SA"
            raw_data = self._fetch_from_yahoo_cached(symbol_sa)
            
            if raw_data is None:
                return None # Truly failed

        return self._package_data(
            raw_data["info"], 
            raw_data["prices"], 
            raw_data["bs"], 
            raw_data["is_"], 
            raw_data["cf"]
        )

    def _package_data(self, meta, prices, bs, is_, cf):
        # 1. Clean Timezone (Crucial)
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        # 2. ETF Detection
        q_type = meta.get('quoteType', '').upper()
        # Fallback: if quoteType missing, check if Balance Sheet is empty
        no_financials = bs.empty if bs is not None else True
        
        is_etf = (q_type == 'ETF') or (q_type == 'MUTUALFUND') or (no_financials)

        def sanitize(df):
            if df is None or df.empty: return df
            try:
                df.columns = pd.to_datetime(df.columns)
                if df.columns.tz is not None:
                    df.columns = df.columns.tz_localize(None)
            except:
                pass 
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
        if prices.index.tz is not None:
             prices.index = prices.index.tz_localize(None)
             
        past_prices = prices[prices.index < cutoff_date]
        
        if past_prices.empty: return None
        simulated_current_price = past_prices['Close'].iloc[-1]

        def filter_financials(df):
            if df is None or df.empty: return df
            valid_cols = []
            for c in df.columns:
                if isinstance(c, pd.Timestamp):
                    if c.tz is not None: c = c.tz_localize(None)
                    if c < cutoff_date: valid_cols.append(c)
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
