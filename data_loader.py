import streamlit as st
import yfinance as yf
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

try:
    TWELVE_DATA_API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "YOUR_KEY_HERE")
except:
    TWELVE_DATA_API_KEY = "YOUR_KEY_HERE"

class SaudiStockLoader:
    def __init__(self, td_key=TWELVE_DATA_API_KEY):
        self.suffix = ".SR"
        self.td_key = td_key

    def fetch_full_data(self, stock_code):
        # 1. Try Yahoo Finance
        try:
            data = self._try_yahoo(stock_code)
            if data: return data
        except Exception as e: 
            print(f"   ❌ Yahoo Error: {e}")
            pass
            
        # If we reach here, no data was found
        return None

    def _try_yahoo(self, stock_code):
        clean_code = f"{stock_code}{self.suffix}" if not str(stock_code).endswith(self.suffix) else stock_code
        ticker = yf.Ticker(clean_code)
        
        # Fetch max history for trends
        prices = ticker.history(period="10y") 
        if prices.empty: return None

        # --- CRITICAL FIX: STRIP TIMEZONE AT SOURCE ---
        # This ensures the entire app works with simple dates, preventing TypeErrors
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        
        # --- ETF DETECTION LOGIC ---
        # Check 1: Explicit Quote Type
        q_type = ticker.info.get('quoteType', '').upper()
        # Check 2: Missing Financials (ETFs don't have standard balance sheets)
        no_financials = ticker.balance_sheet.empty
        
        is_etf = (q_type == 'ETF') or (q_type == 'MUTUALFUND') or (no_financials)

        return self._package_data(ticker.info, prices, ticker.balance_sheet, ticker.income_stmt, ticker.cashflow, is_etf)

    def _try_twelve_data(self, stock_code): return None
    def _try_alpha_vantage(self, stock_code): return None
    def _try_saudi_exchange_scrape(self, stock_code): return None

    def _package_data(self, meta, prices, bs, is_, cf, is_etf):
        def sanitize(df):
            if df.empty: return df
            # Convert columns to timezone-naive to avoid index errors later
            df.columns = pd.to_datetime(df.columns).tz_localize(None)
            return df
        
        return {
            "meta": meta,
            "prices": prices,
            "is_etf": is_etf, # FLAG: Tell the app this is an ETF
            "financials": {
                "balance_sheet": sanitize(bs), 
                "income_statement": sanitize(is_), 
                "cash_flow": sanitize(cf)
            }
        }

    def get_data_as_of_date(self, stock_data, valuation_date_str):
        if stock_data.get("is_etf", False):
            return None 

        cutoff_date = pd.to_datetime(valuation_date_str) # Already naive coming from string
        
        prices = stock_data["prices"].copy()
        # Double check just in case
        if prices.index.tz is not None:
             prices.index = prices.index.tz_localize(None)
             
        past_prices = prices[prices.index < cutoff_date]
        if past_prices.empty: return None
        simulated_current_price = past_prices['Close'].iloc[-1]

        def filter_financials(df):
            if df is None or df.empty: return df
            valid_cols = [c for c in df.columns if c < cutoff_date]
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
