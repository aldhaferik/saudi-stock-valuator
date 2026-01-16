import streamlit as st
import pandas as pd
from optimizer import ValuationOptimizer
import plotly.graph_objects as go

st.set_page_config(page_title="Saudi Stock Valuator AI", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    .big-metric { font-size: 26px; font-weight: bold; color: #0e1117; }
    .header-style { font-size: 18px; color: #555; font-weight: 600; margin-bottom: 10px; }
    .card { background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd; height: 100%; }
    .highlight-ai { border-left: 5px solid #ff4b4b; }
    .highlight-acc { border-left: 5px solid #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚖️ Saudi Stock Valuation: AI vs. Intuition")

with st.sidebar:
    st.header("Settings")
    stock_input = st.text_input("Stock Code", value="1120")
    run_btn = st.button("🚀 Run Dual Analysis", type="primary")

if run_btn and stock_input:
    optimizer = ValuationOptimizer()
    
    with st.spinner(f"🔍 Running dual optimization for {stock_input}..."):
        result = optimizer.find_optimal_strategy(stock_input)
        
        if "error" in result:
            st.error(result['error'])
        else:
            # --- 1. Calculate Current Values (Shared) ---
            full_data = result['full_data_cache']
            latest_price = full_data['prices']['Close'].iloc[-1]
            
            from valuation_engine import ValuationEngine
            engine = ValuationEngine(full_data['financials'])
            
            # Run models on TODAY's data
            curr_vals = {
                "DCF (Moderate)": engine.dcf_valuation(growth_rate=0.04),
                "P/E Multiple": engine.multiples_valuation(pe_ratio=18.0)['PE_Valuation'],
                "P/B Multiple": engine.multiples_valuation(pb_ratio=2.5)['PB_Valuation'],
                "EV/EBITDA": engine.multiples_valuation(ev_ebitda_ratio=12.0)['EBITDA_Valuation']
            }

            # --- 2. Calculate Both Strategies ---
            
            def calculate_result(strategy_dict):
                val = 0
                rows = []
                for method, metrics in strategy_dict.items():
                    current = curr_vals.get(method, 0)
                    w = metrics['weight']
                    val += (current * w)
                    rows.append({
                        "Method": method, 
                        "Accuracy": f"{metrics['historical_accuracy']:.1%}",
                        "Weight": f"{w:.1%}",
                        "Value": f"{current:.2f}"
                    })
                return val, rows

            ai_val, ai_rows = calculate_result(result['strategies']['solver'])
            acc_val, acc_rows = calculate_result(result['strategies']['accuracy'])
            
            # Verdicts
            ai_diff = ((ai_val - latest_price) / latest_price) * 100
            acc_diff = ((acc_val - latest_price) / latest_price) * 100

            # --- DISPLAY SECTION ---
            
            # A. Top Level Comparison
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="card">
                    <div class="header-style">Market Price</div>
                    <div class="big-metric">{latest_price:.2f} SAR</div>
                    <div style="color: #666; font-size: 14px;">Real-Time Data</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                color = "green" if ai_diff > 0 else "red"
                st.markdown(f"""
                <div class="card highlight-ai">
                    <div class="header-style">🤖 AI Solver Model</div>
                    <div class="big-metric">{ai_val:.2f} SAR</div>
                    <div style="color: {color}; font-weight:bold;">{ai_diff:+.1f}% vs Market</div>
                    <div style="font-size: 12px; color: #555;">Focus: Minimizing Total Error</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                color = "green" if acc_diff > 0 else "red"
                st.markdown(f"""
                <div class="card highlight-acc">
                    <div class="header-style">🎯 Accuracy Model</div>
                    <div class="big-metric">{acc_val:.2f} SAR</div>
                    <div style="color: {color}; font-weight:bold;">{acc_diff:+.1f}% vs Market</div>
                    <div style="font-size: 12px; color: #555;">Focus: Higher Accuracy = Higher Weight</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # B. Detailed Breakdown Tabs
            tab1, tab2 = st.tabs(["🤖 AI Solver Details", "🎯 Accuracy Model Details"])
            
            with tab1:
                st.caption("This model uses a mathematical solver to find the combination of weights that minimizes total error, even if it means giving weight to less accurate methods to balance the equation.")
                c1, c2 = st.columns([2, 1])
                c1.dataframe(pd.DataFrame(ai_rows), use_container_width=True, hide_index=True)
                
                # Chart
                fig = go.Figure(data=[go.Pie(labels=[r['Method'] for r in ai_rows], values=[float(r['Weight'].strip('%')) for r in ai_rows], hole=.4)])
                fig.update_layout(title="AI Weight Distribution", height=300, margin=dict(t=30, b=0, l=0, r=0))
                c2.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.caption("This model intuitively assigns weights based on performance. If a method was 90% accurate in the backtest, it gets a high weight. If it failed, it gets near zero.")
                c1, c2 = st.columns([2, 1])
                c1.dataframe(pd.DataFrame(acc_rows), use_container_width=True, hide_index=True)
                
                # Chart
                fig = go.Figure(data=[go.Pie(labels=[r['Method'] for r in acc_rows], values=[float(r['Weight'].strip('%')) for r in acc_rows], hole=.4)])
                fig.update_layout(title="Accuracy Weight Distribution", height=300, margin=dict(t=30, b=0, l=0, r=0))
                c2.plotly_chart(fig, use_container_width=True)
