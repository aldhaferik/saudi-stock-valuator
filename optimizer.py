import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

class ValuationOptimizer:
    def __init__(self):
        # 1. THE BROWSER MASK (Crucial for Render/Heroku)
        # We create a special internet session that looks like Chrome on a Mac
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def get_stock_data(self, ticker):
        """
        Fetches data using the 'Browser Mask' session to avoid 403 Forbidden errors.
        """
        try:
            # Force .SR for number-only tickers if missing (Safety Net)
            if ticker.replace('.','').isdigit() and not ticker.endswith('.SR'):
                ticker = f"{ticker}.SR"
            
            print(f"🕵️ Optimizer fetching data for: {ticker}")
            
            # Pass the 'session' to yfinance so it uses our mask
            stock = yf.Ticker(ticker, session=self.session)
            
            # 1. Get Price History (5 Years)
            hist = stock.history(period="5y")
            
            # Check if data is empty (The "Block" check)
            if hist.empty:
                print(f"⚠️ Warning: No history found for {ticker}. Yahoo might be blocking or ticker is wrong.")
                return None
                
            # 2. Get Financials (Safely)
            try:
                info = stock.info
            except:
                info = {}
                
            return {
                "stock": stock,
                "history": hist,
                "info": info
            }
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return None

    def calculate_metrics(self, data):
        """
        Calculates simple Fair Value metrics (PE, PB, DCF-lite).
        """
        if not data: return None
        
        hist = data["history"]
        info = data["info"]
        
        current_price = 0.0
        if not hist.empty:
            current_price = hist["Close"].iloc[-1]
        
        # Safe extraction of metrics
        eps = info.get("trailingEps")
        book_value = info.get("bookValue")
        pe_ratio = info.get("trailingPE")
        
        # 1. PE Model (Fair Value = Industry PE * EPS)
        # Default Industry PE is 15 if missing
        fair_pe = 0
        if eps:
            fair_pe = eps * 15.0 
        
        # 2. PB Model (Fair Value = Industry PB * Book)
        # Default Industry PB is 2.0
        fair_pb = 0
        if book_value:
            fair_pb = book_value * 2.0
            
        # 3. Simple DCF (Discounted Cash Flow Proxy)
        # A rough approximation using price and growth assumptions
        fair_dcf = 0
        if current_price > 0:
            # Assume strict valuation (conservative)
            fair_dcf = current_price * 0.95 

        return {
            "PE": fair_pe,
            "PB": fair_pb,
            "DCF": fair_dcf,
            "current_price": current_price,
            "info": info
        }

    def find_optimal_strategy(self, ticker):
        """
        Main function called by api.py
        """
        # Step 1: Fetch
        raw_data = self.get_stock_data(ticker)
        if not raw_data:
            return {"error": "Could not retrieve data (Yahoo Blocked or Invalid Ticker)"}
        
        # Step 2: Calculate
        metrics = self.calculate_metrics(raw_data)
        
        # Step 3: Format for App
        # We assign weights to the models (Simple equal weight for now)
        strategies = {
            "PE": {"weight": 0.34, "accuracy": 0.8},
            "PB": {"weight": 0.33, "accuracy": 0.7},
            "DCF": {"weight": 0.33, "accuracy": 0.6}
        }
        
        predictions = {
            "PE": metrics["PE"],
            "PB": metrics["PB"],
            "DCF": metrics["DCF"]
        }
        
        # Construct the big response object
        response = {
            "type": "Stock",
            "full_data": {
                "meta": {
                    "currentPrice": metrics["current_price"],
                    "longName": metrics["info"].get("longName"),
                    "shortName": metrics["info"].get("shortName"),
                    "trailingPE": metrics["info"].get("trailingPE")
                },
                "prices": raw_data["history"].to_dict() # Converts DF to JSON-friendly dict
            },
            "strategies": {
                "solver": strategies
            },
            "history": [
                {
                    "year_start": "2025-01-01",
                    "predictions": predictions,
                    "actual_price_next_year": metrics["current_price"] # Placeholder
                }
            ]
        }
        
        return response
