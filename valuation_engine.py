import pandas as pd
import numpy as np

class ValuationEngine:
    def __init__(self, financial_data):
        """
        Args:
            financial_data (dict): The 'financials' dictionary from data_loader.
        """
        self.bs = financial_data['balance_sheet']
        self.is_ = financial_data['income_statement']
        self.cf = financial_data['cash_flow']
        
        # Extract the most recent column (Year 0 for the simulation)
        # We assume the first column is the most recent due to yfinance sorting
        self.latest_year = self.is_.columns[0]
        
    def _get_value(self, df, row_labels):
        """Helper to safely get a value from a DataFrame row (handling variations in naming)."""
        for label in row_labels:
            if label in df.index:
                return df.loc[label, self.latest_year]
        return 0.0

    def get_key_metrics(self):
        """
        Extracts specific metrics needed for valuation (EPS, EBITDA, FCF).
        Returns a dictionary of per-share metrics.
        """
        # 1. Net Income
        net_income = self._get_value(self.is_, ['Net Income', 'Net Income Common Stockholders'])
        
        # 2. Shares Outstanding (Approximation)
        # We try to get it from Balance Sheet ("Share Issued") or calculate via EPS
        shares = self._get_value(self.bs, ['Share Issued', 'Ordinary Shares Number'])
        
        if shares == 0:
            # Fallback: Net Income / Basic EPS
            basic_eps = self._get_value(self.is_, ['Basic EPS'])
            if basic_eps != 0:
                shares = net_income / basic_eps
            else:
                raise ValueError("Cannot determine Share Count. Valuation impossible.")

        # 3. Book Value
        total_equity = self._get_value(self.bs, ['Total Stockholder Equity', 'Total Equity Gross Minority Interest'])
        
        # 4. EBITDA
        # EBITDA = Operating Income + D&A
        op_income = self._get_value(self.is_, ['Operating Income', 'EBIT'])
        d_and_a = self._get_value(self.cf, ['Depreciation And Amortization'])
        ebitda = op_income + d_and_a
        
        # 5. Free Cash Flow (FCF)
        # FCF = Operating Cash Flow - Capital Expenditure
        ocf = self._get_value(self.cf, ['Operating Cash Flow'])
        capex = self._get_value(self.cf, ['Capital Expenditure']) 
        # Note: Capex is usually negative in CF statements, so we add it (OCF + (-Capex))
        # If it's positive in your data source, subtract it. Yfinance usually sends negative.
        fcf = ocf + capex 

        return {
            "shares_outstanding": shares,
            "EPS": net_income / shares,
            "BVPS": total_equity / shares,     # Book Value Per Share
            "EBITDA_PS": ebitda / shares,      # EBITDA Per Share
            "FCF_PS": fcf / shares,            # Free Cash Flow Per Share
            "net_debt_PS": 0 # Simplified. Real apps calculate (Total Debt - Cash) / Shares
        }

    def dcf_valuation(self, growth_rate=0.03, discount_rate=0.09, terminal_growth=0.02, projection_years=5):
        """
        Performs a Discounted Cash Flow valuation.
        """
        metrics = self.get_key_metrics()
        fcf_per_share = metrics['FCF_PS']
        
        if fcf_per_share <= 0:
            return 0.0 # DCF fails for negative cash flow companies

        # Project Future Cash Flows
        future_cash_flows = []
        for i in range(1, projection_years + 1):
            fcf = fcf_per_share * ((1 + growth_rate) ** i)
            discounted_fcf = fcf / ((1 + discount_rate) ** i)
            future_cash_flows.append(discounted_fcf)
            
        # Terminal Value
        last_fcf = fcf_per_share * ((1 + growth_rate) ** projection_years)
        terminal_value = (last_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        discounted_tv = terminal_value / ((1 + discount_rate) ** projection_years)
        
        fair_value = sum(future_cash_flows) + discounted_tv
        return round(fair_value, 2)

    def multiples_valuation(self, pe_ratio=None, pb_ratio=None, ev_ebitda_ratio=None):
        """
        Returns a dictionary of valuations based on input multiples.
        Example: If PE=15 and EPS=10, Price = 150.
        """
        metrics = self.get_key_metrics()
        
        valuations = {}
        
        if pe_ratio and metrics['EPS'] > 0:
            valuations['PE_Valuation'] = metrics['EPS'] * pe_ratio
        else:
            valuations['PE_Valuation'] = 0.0

        if pb_ratio and metrics['BVPS'] > 0:
            valuations['PB_Valuation'] = metrics['BVPS'] * pb_ratio
        else:
            valuations['PB_Valuation'] = 0.0
            
        if ev_ebitda_ratio and metrics['EBITDA_PS'] > 0:
            # EV = EBITDA * Multiple
            # Price = EV - Net Debt (Simplified here to just EV per share for ease)
            valuations['EBITDA_Valuation'] = metrics['EBITDA_PS'] * ev_ebitda_ratio
        else:
            valuations['EBITDA_Valuation'] = 0.0
            
        return valuations

# --- Test Block ---
if __name__ == "__main__":
    # Fake data to test the logic without calling Yahoo
    # (Matches the structure of yfinance dataframes)
    import pandas as pd
    
    # Create dummy dataframes
    data_mock = {
        'balance_sheet': pd.DataFrame({'2022-12-31': [100000, 50000]}, index=['Total Stockholder Equity', 'Share Issued']),
        'income_statement': pd.DataFrame({'2022-12-31': [10000, 2, 8000]}, index=['Net Income', 'Basic EPS', 'Operating Income']),
        'cash_flow': pd.DataFrame({'2022-12-31': [12000, -2000, 1000]}, index=['Operating Cash Flow', 'Capital Expenditure', 'Depreciation And Amortization'])
    }
    
    engine = ValuationEngine(data_mock)
    metrics = engine.get_key_metrics()
    
    print("--- Metrics ---")
    print(f"EPS: {metrics['EPS']}")
    print(f"FCF Per Share: {metrics['FCF_PS']}")
    
    print("\n--- Valuations ---")
    print(f"DCF Value (3% growth): {engine.dcf_valuation(growth_rate=0.03)} SAR")
    print(f"P/E Value (at 15x): {engine.multiples_valuation(pe_ratio=15)['PE_Valuation']} SAR")