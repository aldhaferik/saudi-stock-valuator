import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

try:
    TWELVE_DATA_API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "YOUR_KEY_HERE")
    ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "YOUR_KEY_HERE")
except:
    TWELVE_DATA_API_KEY = "YOUR_KEY_HERE"
    ALPHA_VANTAGE_API_KEY = "YOUR_KEY_HERE"

class SaudiStockLoader:
    def __init__(self, td_key=TWELVE_DATA_API_KEY, av_key=ALPHA_VANTAGE_API_KEY):
        self.suffix = ".SR"
        self.td_key = td_key
        self.av_key = av_key

    def fetch_full_data(self, stock_code):
        print(f"\n--- 🕵️‍♂️ Starting Data Hunt for Stock: {stock_code} ---")
        
        # 1. Try Yahoo Finance
        try:
            data = self._try_yahoo(stock_code)
            if data: return data
        except Exception as e: print(f"   ❌ Yahoo Error: {e}")

        # (Other sources omitted for brevity, logic remains the same)
        # If we reach here, no data was found
        return None

    def _is_valid(self, data):
        # Strict check: If Balance Sheet is missing, it's likely an ETF or bad data
        if not data: return False
        if data['financials']['balance_sheet'].empty: return False
        return True

    def _try_yahoo(self, stock_code):
        clean_code = f"{stock_code}{self.suffix}" if not str(stock_code).endswith(self.suffix) else stock_code
        ticker = yf.Ticker(clean_code)
        
        # Fetch history
        prices = ticker.history(period="10y") 
        if prices.empty: return None
        
        # CHECK: Is this an ETF?
        # ETFs usually have 'netAssets' or missing financial statements
        if ticker.balance_sheet.empty:
            # We return a special 'error' dict if it's an ETF, so the App knows why it failed
            return {"error_type": "ETF_OR_NO_DATA"}

        return self._package_data(ticker.info, prices, ticker.balance_sheet, ticker.income_stmt, ticker.cashflow)

    def _try_twelve_data(self, stock_code): return None
    def _try_alpha_vantage(self, stock_code): return None
    def _try_saudi_exchange_scrape(self, stock_code): return None

    def _package_data(self, meta, prices, bs, is_, cf):
        def sanitize(df):
            if df.empty: return df
            df.columns = pd.to_datetime(df.columns).tz_localize(None)
            return df
        
        return {
            "meta": meta,
            "prices": prices,
            "financials": {
                "balance_sheet": sanitize(bs), 
                "income_statement": sanitize(is_), 
                "cash_flow": sanitize(cf)
            }
        }

    def get_data_as_of_date(self, stock_data, valuation_date_str):
        # Handle the ETF error case early
        if "error_type" in stock_data: return None

        cutoff_date = pd.to_datetime(valuation_date_str).tz_localize(None)
        
        prices = stock_data["prices"].copy()
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
