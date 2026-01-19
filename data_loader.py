import streamlit as st
import yfinance as yf
import pandas as pd
import requests

# --- STEALTH CONFIGURATION ---
# We use a custom session to trick Yahoo into thinking we are a real browser.
def get_yfinance_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    })
    return session

class SaudiStockLoader:
    def __init__(self):
        self.suffix = ".SR"

    def fetch_full_data(self, stock_code):
        # 1. Try Yahoo Finance with Stealth Session
        try:
            data = self._try_yahoo(stock_code)
            if data: return data
        except Exception as e: 
            print(f"   ❌ Yahoo Error: {e}")
            pass
            
        # If we reach here, absolutely no data was found
        return None

    def _try_yahoo(self, stock_code):
        # Ensure correct suffix
        clean_code = f"{stock_code}{self.suffix}" if not str(stock_code).endswith(self.suffix) else stock_code
        
        # USE THE STEALTH SESSION
        session = get_yfinance_session()
        ticker = yf.Ticker(clean_code, session=session)
        
        # Fetch max history
        # Note: If this returns empty, Yahoo is blocking or symbol is wrong.
        prices = ticker.history(period="10y") 
        
        if prices.empty: 
            print(f"   ⚠️ Yahoo returned empty data for {clean_code}")
            return None

        # --- CRITICAL: STRIP TIMEZONE ---
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        
        # --- ETF DETECTION ---
        q_type = ticker.info.get('quoteType', '').upper()
        no_financials = ticker.balance_sheet.empty
        
        is_etf = (q_type == 'ETF') or (q_type == 'MUTUALFUND') or (no_financials)

        return self._package_data(ticker.info, prices, ticker.balance_sheet, ticker.income_stmt, ticker.cashflow, is_etf)

    def _package_data(self, meta, prices, bs, is_, cf, is_etf):
        def sanitize(df):
            if df.empty: return df
            df.columns = pd.to_datetime(df.columns).tz_localize(None)
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
        if stock_data.get("is_etf", False):
            return None 

        cutoff_date = pd.to_datetime(valuation_date_str)
        
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
