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
        full_data = self.loader.fetch_full_data(stock_code)
        
        if not full_data:
            return {"error": "Could not retrieve data. Symbol might be invalid."}
            
        prices = full_data['prices']
        if prices.empty:
            return {"error": "No price history found."}

        # --- FIX: STRIP TIMEZONE IMMEDIATELY ---
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
            full_data['prices'] = prices 
            
        # --- PATH A: ETF HANDLING ---
        if full_data.get('is_etf', False):
            # Calculate Trailing ROI with DETAILS
            latest_price = prices['Close'].iloc[-1]
            latest_date = prices.index[-1]
            
            roi_metrics = {}
            periods = {
                "30 Days": 30, "60 Days": 60, "90 Days": 90, 
                "6 Months": 180, "1 Year": 365, "2 Years": 730, "5 Years": 1825
            }
            
            for label, days in periods.items():
                target_date = latest_date - timedelta(days=days)
                # Find nearest date in history (on or before target)
                past_slice = prices[prices.index <= target_date]
                
                if not past_slice.empty:
                    past_price = past_slice['Close'].iloc[-1]
                    past_date = past_slice.index[-1]
                    roi = ((latest_price - past_price) / past_price)
                    
                    # Store detailed info for the frontend
                    roi_metrics[label] = {
                        "roi": roi,
                        "ref_date": past_date.strftime('%Y-%m-%d'),
                        "ref_price": past_price
                    }
                else:
                    roi_metrics[label] = None

            return {
                "type": "ETF",
                "full_data": full_data,
                "roi_metrics": roi_metrics
            }

        # --- PATH B: STOCK VALUATION ---
        now = datetime.now()
        current_month, current_day = now.month, now.day
        test_years = range(now.year - 5, now.year) 
        
        history_log = []
        solver_training_data = [] 

        for year in test_years:
            try:
                test_date = datetime(year, current_month, current_day)
                target_date = datetime(year + 1, current_month, current_day)
            except ValueError:
                test_date = datetime(year, current_month, 28)
                target_date = datetime(year + 1, current_month, 28)

            test_date_str = test_date.strftime('%Y-%m-%d')
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            sim_data = self.loader.get_data_as_of_date(full_data, test_date_str)
            if not sim_data or sim_data['financials']['balance_sheet'].empty: continue 
            
            prices_copy = full_data['prices'].copy()
            future_prices = prices_copy[prices_copy.index >= pd.to_datetime(target_date_str)]
            
            if future_prices.empty: continue 
            actual_future_price = future_prices['Close'].iloc[0]

            engine = ValuationEngine(sim_data['financials'])
            
            debug_info = {
                "EPS": engine.get_latest_value(engine.is_, "Diluted EPS") or engine.get_latest_value(engine.is_, "Basic EPS"),
                "FCF": engine.get_latest_value(engine.cf, "Free Cash Flow"),
                "Shares": engine.get_latest_value(engine.bs, "Share Issued"),
                "BookVal": engine.get_latest_value(engine.bs, "Total Equity Gross Minority Interest")
            }
            
            vals = {
                "DCF": engine.dcf_valuation(growth_rate=0.04),
                "PE": engine.multiples_valuation(pe_ratio=18.0)['PE_Valuation'],
                "PB": engine.multiples_valuation(pb_ratio=2.5)['PB_Valuation'],
                "EV": engine.multiples_valuation(ev_ebitda_ratio=12.0)['EBITDA_Valuation']
            }

            history_log.append({
                "year_start": test_date_str,
                "actual_price_next_year": actual_future_price,
                "predictions": vals,
                "debug_inputs": debug_info
            })
            solver_training_data.append(([vals["DCF"], vals["PE"], vals["PB"], vals["EV"]], actual_future_price))

        if not history_log:
            return {"error": "Not enough historical data for backtesting."}

        method_names = ["DCF", "PE", "PB", "EV"]
        acc_scores = {name: [] for name in method_names}
        
        for record in history_log:
            actual = record['actual_price_next_year']
            for name, val in record['predictions'].items():
                if val > 0:
                    acc = max(0, 1.0 - (abs(val - actual) / actual))
                    acc_scores[name].append(acc)
                else:
                    acc_scores[name].append(0)

        final_acc_strategy = {}
        total_score = 0
        temp_scores = {}
        
        for name, acc_list in acc_scores.items():
            avg_acc = np.mean(acc_list) if acc_list else 0
            score = avg_acc ** 2 
            temp_scores[name] = score
            total_score += score
            
        for name in method_names:
            weight = temp_scores[name] / total_score if total_score > 0 else 0.25
            final_acc_strategy[name] = {"weight": weight, "accuracy": np.mean(acc_scores[name])}

        def objective_function(weights):
            total_error = 0
            for pred_vector, actual in solver_training_data:
                combined_val = np.dot(weights, pred_vector)
                total_error += (combined_val - actual) ** 2
            return total_error

        bounds_list = []
        for name in method_names:
            has_valid_data = any(x['predictions'][name] > 0 for x in history_log)
            is_accurate = final_acc_strategy[name]["accuracy"] > 0.10
            if has_valid_data and is_accurate:
                bounds_list.append((0, 1)) 
            else:
                bounds_list.append((0, 0)) 

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        if all(b == (0, 0) for b in bounds_list): bounds_list = [(0, 1)] * 4

        res = minimize(objective_function, [0.25]*4, method='SLSQP', bounds=tuple(bounds_list), constraints=constraints)
        
        final_solver_strategy = {}
        for i, name in enumerate(method_names):
            final_solver_strategy[name] = {"weight": res.x[i], "accuracy": final_acc_strategy[name]["accuracy"]}

        return {
            "type": "STOCK",
            "strategies": {"accuracy": final_acc_strategy, "solver": final_solver_strategy},
            "history": history_log,
            "full_data": full_data
        }
