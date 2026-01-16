import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import os
import time
from io import StringIO

# --- CONFIGURATION ---
try:
    TWELVE_DATA_API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
    ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
except FileNotFoundError:
# Fallback for local testing if secrets.toml isn't found    
    TWELVE_DATA_API_KEY = "ed240f406bab4225ac6e0a98be553aa2" 
    ALPHA_VANTAGE_API_KEY = "0LR5JLOBSLOA6Z0A" # <--- Paste your Key here

class SaudiStockLoader:
    def __init__(self, td_key=TWELVE_DATA_API_KEY, av_key=ALPHA_VANTAGE_API_KEY):
        self.suffix = ".SR"
        self.td_key = td_key
        self.av_key = av_key

    def fetch_full_data(self, stock_code):
        """
        Master Fetch Strategy: Yahoo -> Twelve Data -> Alpha Vantage -> Local Backup
        """
        print(f"\n--- Starting Data Hunt for Stock: {stock_code} ---")
        
        # 1. Try Yahoo Finance
        data = self._try_yahoo(stock_code)
        if data: return data

        # 2. Try Twelve Data
        data = self._try_twelve_data(stock_code)
        if data: return data

        # 3. Try Alpha Vantage
        data = self._try_alpha_vantage(stock_code)
        if data: return data

        # 4. Try Local Backup
        data = self._try_local_backup(stock_code)
        if data: return data

        print("❌ CRITICAL: All data sources failed.")
        return None

    # --- SOURCE 1: YAHOO FINANCE ---
    def _try_yahoo(self, stock_code):
        print(f"1. Attempting Yahoo Finance...")
        try:
            clean_code = f"{stock_code}{self.suffix}" if not str(stock_code).endswith(self.suffix) else stock_code
            ticker = yf.Ticker(clean_code)
            
            # Check price first (fastest check)
            prices = ticker.history(period="5y")
            if prices.empty:
                print("   -> Yahoo Price Data missing.")
                return None
            
            # Check financials
            if ticker.balance_sheet.empty:
                print("   -> Yahoo Financials missing.")
                return None

            print("   -> ✅ Success with Yahoo!")
            return self._package_data(ticker.info, prices, ticker.balance_sheet, ticker.income_stmt, ticker.cashflow)
        except Exception as e:
            print(f"   -> Yahoo Failed: {e}")
            return None

    # --- SOURCE 2: TWELVE DATA ---
    def _try_twelve_data(self, stock_code):
        print(f"2. Attempting Twelve Data...")
        if not self.td_key:
            print("   -> No API Key provided.")
            return None

        clean_code = str(stock_code).replace(".SR", "")
        base_url = "https://api.twelvedata.com"
        
        try:
            # A. Fetch Prices
            price_params = {
                "symbol": clean_code,
                "exchange": "Tadawul",
                "interval": "1day",
                "outputsize": 5000,
                "apikey": self.td_key
            }
            r = requests.get(f"{base_url}/time_series", params=price_params).json()
            
            if "values" not in r:
                print(f"   -> Twelve Data error: {r.get('message', 'Unknown')}")
                return None
                
            prices = pd.DataFrame(r["values"])
            prices['datetime'] = pd.to_datetime(prices['datetime'])
            prices.set_index('datetime', inplace=True)
            prices = prices.astype(float).sort_index()
            prices.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

            # B. Fetch Financials (Iterate endpoints)
            financials = {}
            for name in ["balance_sheet", "income_statement", "cash_flow"]:
                r_fin = requests.get(f"{base_url}/{name}", params={"symbol": clean_code, "exchange": "Tadawul", "apikey": self.td_key, "period": "annual"}).json()
                if name in r_fin:
                    df = pd.DataFrame(r_fin[name])
                    df['fiscal_date'] = pd.to_datetime(df['fiscal_date'])
                    df.set_index('fiscal_date', inplace=True)
                    financials[name] = df.T # Transpose to match Yahoo format
                else:
                    financials[name] = pd.DataFrame()

            print("   -> ✅ Success with Twelve Data!")
            return self._package_data({"symbol": stock_code}, prices, financials["balance_sheet"], financials["income_statement"], financials["cash_flow"])

        except Exception as e:
            print(f"   -> Twelve Data Failed: {e}")
            return None

    # --- SOURCE 3: ALPHA VANTAGE ---
    def _try_alpha_vantage(self, stock_code):
        print(f"3. Attempting Alpha Vantage...")
        if not self.av_key or "YOUR_KEY" in self.av_key:
            print("   -> No API Key provided.")
            return None

        # Alpha Vantage format for Saudi is usually '1120.SAR'
        av_symbol = f"{str(stock_code).replace('.SR', '')}.SAR"
        base_url = "https://www.alphavantage.co/query"

        try:
            # A. Fetch Prices
            p_params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": av_symbol,
                "outputsize": "full",
                "apikey": self.av_key
            }
            r = requests.get(base_url, params=p_params).json()
            
            # Alpha Vantage JSON parsing is tricky
            if "Time Series (Daily)" not in r:
                print(f"   -> Alpha Vantage Price error: {r.get('Note', 'Limit reached or symbol not found')}")
                return None
            
            prices = pd.DataFrame.from_dict(r["Time Series (Daily)"], orient='index')
            prices.index = pd.to_datetime(prices.index)
            prices = prices.astype(float).sort_index()
            prices.columns = ["Open", "High", "Low", "Close", "Volume"] # Rename 1. open, etc.

            # B. Fetch Financials
            financials = {}
            endpoints = {
                "BALANCE_SHEET": "balance_sheet", 
                "INCOME_STATEMENT": "income_statement", 
                "CASH_FLOW": "cash_flow"
            }
            
            for func, name in endpoints.items():
                # API Call
                f_r = requests.get(base_url, params={"function": func, "symbol": av_symbol, "apikey": self.av_key}).json()
                
                # Check for 'annualReports' key
                report_key = "annualReports"
                if report_key in f_r:
                    df = pd.DataFrame(f_r[report_key])
                    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                    df.set_index('fiscalDateEnding', inplace=True)
                    # Convert all columns to numeric, errors='coerce' turns "None" to NaN
                    df = df.apply(pd.to_numeric, errors='coerce')
                    financials[name] = df.T
                else:
                    financials[name] = pd.DataFrame()

            print("   -> ✅ Success with Alpha Vantage!")
            return self._package_data({"symbol": stock_code}, prices, financials["balance_sheet"], financials["income_statement"], financials["cash_flow"])

        except Exception as e:
            print(f"   -> Alpha Vantage Failed: {e}")
            return None

    # --- SOURCE 4: LOCAL BACKUP ---
    def _try_local_backup(self, stock_code):
        print(f"4. Attempting Local Backup...")
        path = "data_backup"
        clean_code = str(stock_code).replace(".SR", "")
        
        if not os.path.exists(f"{path}/{clean_code}_bs.csv"):
            print("   -> No local files found.")
            return None
            
        try:
            bs = pd.read_csv(f"{path}/{clean_code}_bs.csv", index_col=0)
            is_ = pd.read_csv(f"{path}/{clean_code}_is.csv", index_col=0)
            cf = pd.read_csv(f"{path}/{clean_code}_cf.csv", index_col=0)
            
            # Fix Dates
            bs.columns = pd.to_datetime(bs.columns)
            is_.columns = pd.to_datetime(is_.columns)
            cf.columns = pd.to_datetime(cf.columns)

            print("   -> ✅ Success with Local Backup!")
            # Note: We return empty prices if local, as CSV usually just has financials. 
            # You could add a price CSV too if you want.
            return self._package_data({"symbol": stock_code}, pd.DataFrame(), bs, is_, cf)
        except Exception as e:
            print(f"   -> Local Backup Read Error: {e}")
            return None

    def _package_data(self, meta, prices, bs, is_, cf):
        return {
            "meta": meta,
            "prices": prices,
            "financials": {"balance_sheet": bs, "income_statement": is_, "cash_flow": cf}
        }

    # [Paste the FIXED get_data_as_of_date method from previous steps here]
    # It is crucial for the backtesting logic.
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
            "financials": past_financials
        }

if __name__ == "__main__":
    loader = SaudiStockLoader()
    # Test with a known Saudi stock
    loader.fetch_full_data("1120")
