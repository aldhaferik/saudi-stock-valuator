import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import pandas as pd
from data_loader import SaudiStockLoader
from valuation_engine import ValuationEngine

class ValuationOptimizer:
    def __init__(self):
        self.loader = SaudiStockLoader()
        
    def find_optimal_strategy(self, stock_code):
        print(f"\n🚀 Starting Optimization for {stock_code}...")
        
        # 1. Fetch Full Data
        full_data = self.loader.fetch_full_data(stock_code)
        if not full_data:
            return {"error": "Could not retrieve data for this stock."}

        # 2. Adaptive Backtest Date (5, 4, or 3 years ago)
        years_back_options = [5, 4, 3]
        simulation_data = None
        
        for y in years_back_options:
            target_date = (datetime.now() - timedelta(days=y*365)).strftime('%Y-%m-%d')
            sim_data = self.loader.get_data_as_of_date(full_data, target_date)
            
            if sim_data and not sim_data['financials']['balance_sheet'].empty:
                simulation_data = sim_data
                break
        
        if not simulation_data:
            return {"error": "Not enough historical data (need at least 3 years)."}

        # 3. Calculate 'Past' Valuations
        engine = ValuationEngine(simulation_data['financials'])
        
        # Method Estimates (Backtest)
        dcf_mod = engine.dcf_valuation(growth_rate=0.04)
        mults = engine.multiples_valuation(pe_ratio=18.0, pb_ratio=2.5, ev_ebitda_ratio=12.0)
        
        estimates_dict = {
            "DCF (Moderate)": dcf_mod,
            "P/E Multiple": mults['PE_Valuation'],
            "P/B Multiple": mults['PB_Valuation'],
            "EV/EBITDA": mults['EBITDA_Valuation']
        }
        
        # Filter out failed methods (zeros)
        valid_methods = {k: v for k, v in estimates_dict.items() if v > 0}
        if not valid_methods:
             return {"error": "All valuation methods failed (likely negative earnings)."}

        estimate_values = np.array(list(valid_methods.values()))
        estimate_names = list(valid_methods.keys())
        
        # 4. The 'Truth'
        actual_price = simulation_data['price_at_simulation']
        
        # 5. --- NEW: Calculate Historical Accuracy per Method ---
        accuracy_report = {}
        for name, val in valid_methods.items():
            # Error % = |Predicted - Actual| / Actual
            error_margin = abs(val - actual_price) / actual_price
            accuracy_score = max(0, 1.0 - error_margin) # 1.0 is 100% accurate
            accuracy_report[name] = accuracy_score

        # 6. Run Solver
        def objective_function(weights):
            combined_val = np.sum(weights * estimate_values)
            return (combined_val - actual_price)**2

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(len(estimate_values)))
        initial_guess = [1/len(estimate_values)] * len(estimate_values)

        result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = result.x

        # 7. Package Results
        strategy = {}
        for name, weight, val in zip(estimate_names, optimal_weights, estimate_values):
            strategy[name] = {
                "weight": round(weight, 4),
                "historical_accuracy": accuracy_report[name], # <--- Added this
                "valuation_at_backtest": val
            }

        return {
            "stock": stock_code,
            "backtest_date": simulation_data['simulation_date'],
            "actual_price_then": actual_price,
            "optimized_weights": strategy,
            "full_data_cache": full_data
        }