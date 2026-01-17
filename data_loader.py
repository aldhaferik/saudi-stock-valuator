import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import os
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# --- CONFIGURATION ---
try:
    TWELVE_DATA_API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
    ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
except (FileNotFoundError, KeyError):
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
            if self._is_valid(data): 
                print("   ✅ Yahoo Finance Success!")
                return data
            print("   ⚠️ Yahoo Data Invalid (Empty or Zero Assets).")
        except Exception as e:
            print(f"   ❌ Yahoo Error: {e}")

        # 2. Try Twelve Data
        try:
            data = self._try_twelve_data(stock_code)
            if self._is_valid(data): 
                print("   ✅ Twelve Data Success!")
                return data
            print("   ⚠️ Twelve Data Invalid.")
        except Exception as e:
            print(f"   ❌ Twelve Data Error: {e}")

        # 3. Try Alpha Vantage
        try:
            data = self._try_alpha_vantage(stock_code)
            if self._is_valid(data): 
                print("   ✅ Alpha Vantage Success!")
                return data
            print("   ⚠️ Alpha Vantage Invalid.")
        except Exception as e:
            print(f"   ❌ Alpha Vantage Error: {e}")

        # 4. Try Saudi Exchange Scraper (Selenium)
        try:
            data = self._try_saudi_exchange_scrape(stock_code)
            if self._is_valid(data): 
                print("   ✅ Scraper Success!")
                return data
            print("   ⚠️ Scraper failed.")
        except Exception as e:
            print(f"   ❌ Scraper Error: {e}")

        # 5. Try Local Backup
        try:
            data = self._try_local_backup(stock_code)
            if self._is_valid(data): 
                print("   ✅ Local Backup Success!")
                return data
        except Exception as e:
            print(f"   ❌ Local Backup Error: {e}")

        print("❌ CRITICAL: All data sources failed.")
        return None

    def _is_valid(self, data):
        if not data: return False
        if data['financials']['balance_sheet'].empty: return False
        try:
            val = data['financials']['balance_sheet'].iloc[0, 0]
            if val == 0: return False
        except:
            pass
        return True

    def _try_yahoo(self, stock_code):
        clean_code = f"{stock_code}{self.suffix}" if not str(stock_code).endswith(self.suffix) else stock_code
        ticker = yf.Ticker(clean_code)
        prices = ticker.history(period="5y")
        if prices.empty: return None
        return self._package_data(ticker.info, prices, ticker.balance_sheet, ticker.income_stmt, ticker.cashflow)

    def _try_twelve_data(self, stock_code):
        # Placeholder for your Twelve Data logic
        return None 

    def _try_alpha_vantage(self, stock_code):
        # Placeholder for your Alpha Vantage logic
        return None

    def _try_saudi_exchange_scrape(self, stock_code):
        url = f"https://www.saudiexchange.sa/wps/portal/saudiexchange/hidden/company-profile-main/?companySymbol={stock_code}"
        driver = None
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            
            # Simple check to see if page loads (you can add table parsing logic here later)
            html = driver.page_source
            driver.quit()
            
            # For now, return None as we haven't written the custom table parser yet
            return None 
            
        except Exception as e:
            if driver: driver.quit()
            raise e

    def _try_local_backup(self, stock_code):
        path = "data_backup"
        clean_code = str(stock_code).replace(".SR", "")
        bs = pd.read_csv(f"{path}/{clean_code}_bs.csv", index_col=0)
        is_ = pd.read_csv(f"{path}/{clean_code}_is.csv", index_col=0)
        cf = pd.read_csv(f"{path}/{clean_code}_cf.csv", index_col=0)
        
        # Convert columns to datetime
        bs.columns = pd.to_datetime(bs.columns)
        is_.columns = pd.to_datetime(is_.columns)
        cf.columns = pd.to_datetime(cf.columns)
        
        return self._package_data({"symbol": stock_code}, pd.DataFrame(), bs, is_, cf)

    def _package_data(self, meta, prices, bs, is_, cf):
        return {
            "meta": meta,
            "prices": prices,
            "financials": {"balance_sheet": bs, "income_statement": is_, "cash_flow": cf}
        }

    def get_data_as_of_date(self, stock_data, valuation_date_str):
        cutoff_date = pd.to_datetime(valuation_date_str)
        if stock_data["prices"].empty: return None

        price_index = stock_data["prices"].index
        if price_index.tz is not None:
            cutoff_date = cutoff_date.tz_localize(price_index.tz)

        past_prices = stock_data["prices"][stock_data["prices"].index < cutoff_date]
        if past_prices.empty: return None
        simulated_current_price = past_prices['Close'].iloc[-1]

        def filter_financials(df):
            if df is None or df.empty: return df
            valid_cols = []
            for col in df.columns:
                col_dt = pd.to_datetime(col)
                if col_dt.tz is not None: col_dt = col_dt.tz_localize(None)
                naive_cutoff = cutoff_date.replace(tzinfo=None)
                if col_dt < naive_cutoff: valid_cols.append(col)
            return df[valid_cols]

        past_financials = {
            "balance_sheet": filter_financials(stock_data["financials"]["balance_sheet"]),
            "income_statement": filter_financials(stock_data["financials"]["income_statement"]),
            "cash_flow": filter_financials(stock_data["financials"]["cash_flow"])
        }

        return {
            "simulation_date": valuation_date_str,
            "price_at_simulation": simulated_current_price,
            "financials": past_financials,
            "meta": stock_data.get("meta", {}),
            "prices": stock_data.get("prices", pd.DataFrame())
        }
