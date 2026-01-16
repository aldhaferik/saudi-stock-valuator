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
        print(f"\n🚀 Starting Dual Optimization for {stock_code}...")
        
        # 1. Fetch Full Data
        full_data = self.loader.fetch_full_data(stock_code)
        if not full_data:
            return {"error": "Could not retrieve data for this stock."}

        # 2. Adaptive Backtest Date
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
        
        dcf_mod = engine.dcf_valuation(growth_rate=0.04)
        mults = engine.multiples_valuation(pe_ratio=18.0, pb_ratio=2.5, ev_ebitda_ratio=12.0)
        
        estimates_dict = {
            "DCF (Moderate)": dcf_mod,
            "P/E Multiple": mults['PE_Valuation'],
            "P/B Multiple": mults['PB_Valuation'],
            "EV/EBITDA": mults['EBITDA_Valuation']
        }
        
        # Filter valid methods (>0)
        valid_methods = {k: v for k, v in estimates_dict.items() if v > 0}
        if not valid_methods:
             return {"error": "All valuation methods failed (likely negative earnings)."}

        estimate_values = np.array(list(valid_methods.values()))
        estimate_names = list(valid_methods.keys())
        actual_price = simulation_data['price_at_simulation']

        # --- STRATEGY 1: The Solver (AI Optimization) ---
        # Goal: Minimize Total Error (Even if it means weird weights)
        def objective_function(weights):
            combined_val = np.sum(weights * estimate_values)
            return (combined_val - actual_price)**2

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(len(estimate_values)))
        initial_guess = [1/len(estimate_values)] * len(estimate_values)

        solver_result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        solver_strategy = {}
        for name, weight, val in zip(estimate_names, solver_result.x, estimate_values):
            # Calculate simple accuracy for display
            acc = max(0, 1.0 - (abs(val - actual_price) / actual_price))
            solver_strategy[name] = {
                "weight": round(weight, 4),
                "historical_accuracy": acc,
                "valuation_at_backtest": val
            }

        # --- STRATEGY 2: The Accuracy Score (Merit-Based) ---
        # Goal: Higher Accuracy = Higher Weight
        accuracy_strategy = {}
        total_score = 0
        accuracy_scores = []
        
        for val in estimate_values:
            raw_acc = max(0, 1.0 - (abs(val - actual_price) / actual_price))
            score = raw_acc ** 2  # Squaring to emphasize winners
            accuracy_scores.append(score)
            total_score += score
            
        for i, name in enumerate(estimate_names):
            if total_score > 0:
                weight = accuracy_scores[i] / total_score
            else:
                weight = 1.0 / len(estimate_names)
                
            raw_acc = max(0, 1.0 - (abs(estimate_values[i] - actual_price) / actual_price))
            accuracy_strategy[name] = {
                "weight": round(weight, 4),
                "historical_accuracy": raw_acc,
                "valuation_at_backtest": estimate_values[i]
            }

        return {
            "stock": stock_code,
            "backtest_date": simulation_data['simulation_date'],
            "actual_price_then": actual_price,
            "strategies": {
                "solver": solver_strategy,
                "accuracy": accuracy_strategy
            },
            "full_data_cache": full_data
        }
