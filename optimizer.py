import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from data_loader import SaudiStockLoader
from valuation_engine import ValuationEngine

class ValuationOptimizer:
    def __init__(self):
        self.loader = SaudiStockLoader()
        
    def find_optimal_strategy(self, stock_code):
        print(f"\n🚀 Starting Dynamic Walk-Forward Optimization for {stock_code}...")
        
        # 1. Fetch Full Data
        full_data = self.loader.fetch_full_data(stock_code)
        if not full_data or full_data['prices'].empty:
            return {"error": "Could not retrieve sufficient data for this stock."}

        # 2. Determine Dates Dynamically
        now = datetime.now()
        current_month = now.month
        current_day = now.day
        current_year = now.year
        
        # We want to test the last 4 years on THIS specific day (e.g., April 10)
        # Generate years: [2022, 2023, 2024, 2025] (if today is 2026)
        test_years = range(current_year - 4, current_year) 
        
        history_log = []
        method_errors = {
            "DCF (Moderate)": [],
            "P/E Multiple": [],
            "P/B Multiple": [],
            "EV/EBITDA": []
        }

        # 3. The Walk-Forward Loop
        for year in test_years:
            # A. Set the "Test Date" (Same Month/Day, but past Year)
            # Handle leap years (e.g., Feb 29) by falling back to Feb 28 if needed
            try:
                test_date_obj = datetime(year, current_month, current_day)
                target_date_obj = datetime(year + 1, current_month, current_day)
            except ValueError:
                # Fallback for leap year edge cases
                test_date_obj = datetime(year, current_month, 28)
                target_date_obj = datetime(year + 1, current_month, 28)

            test_date_str = test_date_obj.strftime('%Y-%m-%d')
            target_date_str = target_date_obj.strftime('%Y-%m-%d')
            
            # B. Travel back in time
            sim_data = self.loader.get_data_as_of_date(full_data, test_date_str)
            if not sim_data or sim_data['financials']['balance_sheet'].empty:
                continue # Skip if data missing for this old year
            
            # C. Get the "Actual Price" exactly 1 year later (The Truth)
            # We look for the price ON or immediately AFTER the target date
            future_prices = full_data['prices'][full_data['prices'].index >= target_date_str]
            if future_prices.empty:
                continue 
            
            actual_future_price = future_prices['Close'].iloc[0]

            # D. Run Valuation Models (using ONLY data known at test_date)
            engine = ValuationEngine(sim_data['financials'])
            
            vals = {
                "DCF (Moderate)": engine.dcf_valuation(growth_rate=0.04),
                "P/E Multiple": engine.multiples_valuation(pe_ratio=18.0)['PE_Valuation'],
                "P/B Multiple": engine.multiples_valuation(pb_ratio=2.5)['PB_Valuation'],
                "EV/EBITDA": engine.multiples_valuation(ev_ebitda_ratio=12.0)['EBITDA_Valuation']
            }

            # E. Log Performance for this Year
            year_record = {
                "year_start": test_date_str,
                "year_end": target_date_str,
                "actual_price_next_year": actual_future_price,
                "predictions": vals
            }
            history_log.append(year_record)

            # Calculate Error
            for method, prediction in vals.items():
                if prediction > 0:
                    error = abs(prediction - actual_future_price) / actual_future_price
                    method_errors[method].append(error)

        # 4. Aggregate Scores (Average Accuracy over all years)
        if not history_log:
            # If walk-forward fails (e.g., new IPO), fall back to simple weights
            default_w = 0.25
            final_strategy = {k: {"weight": default_w, "historical_accuracy": 0.0} for k in method_errors.keys()}
            return {
                "stock": stock_code, 
                "walk_forward_history": [], 
                "strategies": {"accuracy": final_strategy, "solver": final_strategy},
                "full_data_cache": full_data,
                "error": "Not enough historical data for walk-forward validation (Stock might be too new)."
            }

        final_strategy = {}
        total_inverse_error = 0
        method_scores = {}
        
        for method, errors in method_errors.items():
            if not errors:
                avg_error = 1.0
            else:
                avg_error = np.mean(errors)
            
            accuracy_pct = max(0, 1.0 - avg_error)
            score = accuracy_pct ** 2
            method_scores[method] = {"acc": accuracy_pct, "score": score}
            total_inverse_error += score

        # Normalize
        for method, metrics in method_scores.items():
            if total_inverse_error > 0:
                weight = metrics['score'] / total_inverse_error
            else:
                weight = 0.25
            
            final_strategy[method] = {
                "weight": round(weight, 4),
                "historical_accuracy": metrics['acc'], 
            }

        return {
            "stock": stock_code,
            "walk_forward_history": history_log,
            "strategies": {
                "accuracy": final_strategy,
                "solver": final_strategy 
            },
            "full_data_cache": full_data
        }
