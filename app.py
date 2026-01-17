import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import os
from io import StringIO

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
        data = self._try_yahoo(stock_code)
        if self._is_valid(data): return data
        print("   ⚠️ Yahoo data missing or suspicious. Switching to Backup 1...")

        # 2. Try Twelve Data
        data = self._try_twelve_data(stock_code)
        if self._is_valid(data): return data
        print("   ⚠️ TwelveData missing. Switching to Backup 2...")

        # 3. Try Alpha Vantage
        data = self._try_alpha_vantage(stock_code)
        if self._is_valid(data): return data
        print("   ⚠️ AlphaVantage missing. Switching to Web Scraper...")

        # 4. Try Saudi Exchange Scraper (New!)
        data = self._try_saudi_exchange_scrape(stock_code)
        if self._is_valid(data): return data
        print("   ⚠️ Scraper failed. Checking Local Backup...")

        # 5. Try Local Backup
        data = self._try_local_backup(stock_code)
        if self._is_valid(data): return data

        print("❌ CRITICAL: All 5 data sources failed.")
        return None

    def _is_valid(self, data):
        """Checks if the data package has actual numbers."""
        if not data: return False
        if data['financials']['balance_sheet'].empty: return False
        # Check if Total Assets is 0 (common API bug)
        try:
            val = data['financials']['balance_sheet'].iloc[0, 0]
            if val == 0: return False
        except:
            pass
        return True

    # --- SOURCE 1: YAHOO ---
    def _try_yahoo(self, stock_code):
        print(f"1. Attempting Yahoo Finance...")
        try:
            clean_code = f"{stock_code}{self.suffix}" if not str(stock_code).endswith(self.suffix) else stock_code
            ticker = yf.Ticker(clean_code)
            prices = ticker.history(period="5y")
            if prices.empty or ticker.balance_sheet.empty: return None
            return self._package_data(ticker.info, prices, ticker.balance_sheet, ticker.income_stmt, ticker.cashflow)
        except: return None

    # --- SOURCE 2: TWELVE DATA ---
    def _try_twelve_data(self, stock_code):
        print(f"2. Attempting Twelve Data...")
        # ... (Same logic as previous version) ...
        # [Paste the TwelveData logic I gave you previously here]
        # For brevity, I'm skipping re-pasting the exact code block, but keep your existing logic!
        return None 

    # --- SOURCE 3: ALPHA VANTAGE ---
    def _try_alpha_vantage(self, stock_code):
        print(f"3. Attempting Alpha Vantage...")
        # ... (Same logic as previous version) ...
        # [Paste the AlphaVantage logic I gave you previously here]
        return None

    # --- SOURCE 4: SAUDI EXCHANGE SCRAPER (NEW) ---
    def _try_saudi_exchange_scrape(self, stock_code):
        print(f"4. Attempting SaudiExchange.sa Scraper...")
        
        # NOTE: Tadawul uses heavy JavaScript. Direct scraping often fails without Selenium.
        # We attempt to fetch the 'Fundamental Ratios' table which is sometimes accessible.
        
        url = f"https://www.saudiexchange.sa/wps/portal/saudiexchange/hidden/company-profile-main/?companySymbol={stock_code}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"   -> Scraper: HTTP {response.status_code}")
                return None
            
            # Use Pandas to find tables in the HTML
            dfs = pd.read_html(StringIO(response.text))
            
            # We look for tables that look like financials
            print(f"   -> Scraper found {len(dfs)} tables.")
            
            # This is highly experimental because Tadawul changes their layout
            # We return None for now because mapping dynamic HTML tables to 
            # the strict Balance Sheet format required by the Engine is risky without human verification.
            # But this confirms we CAN reach the site.
            
            # Realistically: We can grab the P/E and Beta from here if simple APIs fail.
            pass

        except Exception as e:
            print(f"   -> Scraper Failed: {e}")
        
        return None

    # --- SOURCE 5: LOCAL BACKUP ---
    def _try_local_backup(self, stock_code):
        print(f"5. Attempting Local Backup...")
        path = "data_backup"
        clean_code = str(stock_code).replace(".SR", "")
        try:
            bs = pd.read_csv(f"{path}/{clean_code}_bs.csv", index_col=0)
            is_ = pd.read_csv(f"{path}/{clean_code}_is.csv", index_col=0)
            cf = pd.read_csv(f"{path}/{clean_code}_cf.csv", index_col=0)
            
            bs.columns = pd.to_datetime(bs.columns)
            is_.columns = pd.to_datetime(is_.columns)
            cf.columns = pd.to_datetime(cf.columns)

            return self._package_data({"symbol": stock_code}, pd.DataFrame(), bs, is_, cf)
        except: return None

    def _package_data(self, meta, prices, bs, is_, cf):
        return {
            "meta": meta,
            "prices": prices,
            "financials": {"balance_sheet": bs, "income_statement": is_, "cash_flow": cf}
        }

    # [Paste the FIXED get_data_as_of_date method here]
    def get_data_as_of_date(self, stock_data, valuation_date_str):
        # ... (Same time-zone fix as before) ...
        # I can re-supply this block if you lost it.
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
