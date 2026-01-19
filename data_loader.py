import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
import random

class SaudiStockLoader:
    def __init__(self):
        self.suffix = ".SR"

    def _get_session(self):
        # Randomize User-Agent to prevent blocking
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        ]
        session = requests.Session()
        session.headers.update({
            "User-Agent": random.choice(user_agents),
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br"
        })
        return session

    def fetch_full_data(self, stock_code):
        # Clean input
        base_code = str(stock_code).replace(".SR", "").replace(".SA", "").strip()
        
        # Suffixes to try (Preferred first)
        symbols_to_try = [f"{base_code}.SR", f"{base_code}.SA"]
        
        for symbol in symbols_to_try:
            # Retry loop (Try 2 times per symbol)
            for attempt in range(2):
                try:
                    data = self._fetch_single_symbol(symbol)
                    if data: return data
                    # If failed but no error raised, try next attempt
                    time.sleep(1) # Wait 1s before retry
                except Exception as e:
                    print(f"⚠️ Attempt {attempt+1} failed for {symbol}: {e}")
                    time.sleep(1)
        
        return None

    def _fetch_single_symbol(self, symbol):
        session = self._get_session()
        ticker = yf.Ticker(symbol, session=session)
        
        # 1. Fetch History (Fastest check)
        # Note: 'max' can be heavy, '10y' is safer.
        prices = ticker.history(period="10y")
        
        if prices.empty:
            return None

        # 2. Fetch Metadata
        # We wrap this in try/except because sometimes info fails even if prices work
        try:
            info = ticker.info
        except:
            info = {}

        # 3. Clean Timezone (Crucial Fix)
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        # 4. ETF Detection
        q_type = info.get('quoteType', '').upper()
        # Some ETFs have empty 'balance_sheet', use that as backup signal
        try:
            bs = ticker.balance_sheet
            no_financials = bs.empty if bs is not None else True
        except:
            bs = pd.DataFrame()
            no_financials = True

        is_etf = (q_type == 'ETF') or (q_type == 'MUTUALFUND') or (no_financials)

        # 5. Fetch Financials (Only if not ETF)
        is_stmt = pd.DataFrame()
        cf = pd.DataFrame()
        
        if not is_etf:
            try:
                is_stmt = ticker.income_stmt
                cf = ticker.cashflow
            except:
                pass

        return self._package_data(info, prices, bs, is_stmt, cf, is_etf)

    def _package_data(self, meta, prices, bs, is_, cf, is_etf):
        def sanitize(df):
            if df is None or df.empty: return df
            # Convert columns to datetime and strip timezone
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
        # Safety check for timezone
        if prices.index.tz is not None:
             prices.index = prices.index.tz_localize(None)
             
        past_prices = prices[prices.index < cutoff_date]
        
        if past_prices.empty: return None
        simulated_current_price = past_prices['Close'].iloc[-1]

        def filter_financials(df):
            if df is None or df.empty: return df
            valid_cols = []
            for c in df.columns:
                # Ensure column is a timestamp before comparing
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
