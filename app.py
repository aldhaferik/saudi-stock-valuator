import streamlit as st
import pandas as pd
from optimizer import ValuationOptimizer
import plotly.graph_objects as go

st.set_page_config(page_title="Saudi Stock Valuator AI", page_icon="📊", layout="centered")

st.markdown("""
    <style>
    .big-metric { font-size: 32px; font-weight: bold; color: #0e1117; }
    .undervalued { color: #009933; font-weight: bold; }
    .overvalued { color: #cc0000; font-weight: bold; }
    .card { background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

st.title("🇸🇦 Saudi Stock Valuation AI")

with st.sidebar:
    st.header("Settings")
    stock_input = st.text_input("Stock Code", value="1120")
    run_btn = st.button("🚀 Run Analysis", type="primary")

if run_btn and stock_input:
    optimizer = ValuationOptimizer()
    
    with st.spinner(f"🔍 Analyzing {stock_input}..."):
        result = optimizer.find_optimal_strategy(stock_input)
        
        if "error" in result:
            st.error(result['error'])
        else:
            # --- 1. Calculate Current Values ---
            full_data = result['full_data_cache']
            latest_price = full_data['prices']['Close'].iloc[-1] # Real-time(ish) price
            
            from valuation_engine import ValuationEngine
            engine = ValuationEngine(full_data['financials'])
            
            # Run models on TODAY's data
            dcf_curr = engine.dcf_valuation(growth_rate=0.04)
            mults_curr = engine.multiples_valuation(pe_ratio=18.0, pb_ratio=2.5, ev_ebitda_ratio=12.0)
            
            curr_vals = {
                "DCF (Moderate)": dcf_curr,
                "P/E Multiple": mults_curr['PE_Valuation'],
                "P/B Multiple": mults_curr['PB_Valuation'],
                "EV/EBITDA": mults_curr['EBITDA_Valuation']
            }

            # --- 2. Calculate Weighted Fair Value ---
            fair_value = 0
            table_rows = []
            
            for method, metrics in result['optimized_weights'].items():
                val_now = curr_vals.get(method, 0)
                weight = metrics['weight']
                accuracy = metrics['historical_accuracy']
                
                fair_value += (val_now * weight)
                
                table_rows.append({
                    "Method": method,
                    "Current Value (SAR)": f"{val_now:.2f}",
                    "Historical Accuracy": f"{accuracy:.1%}",
                    "AI Weight": f"{weight:.1%}"
                })

            # --- 3. Determine Verdict (Over/Under) ---
            diff = fair_value - latest_price
            diff_pct = (diff / latest_price) * 100
            
            verdict_color = "undervalued" if diff > 0 else "overvalued"
            verdict_text = "UNDERVALUED" if diff > 0 else "OVERVALUED"
            arrow = "🔼" if diff > 0 else "🔽"

            # --- DISPLAY SECTION ---
            
            # A. Main Scorecards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="card">
                    <div style="font-size:14px; color:#666;">Current Market Price</div>
                    <div class="big-metric">{latest_price:.2f} SAR</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card">
                    <div style="font-size:14px; color:#666;">AI Fair Value</div>
                    <div class="big-metric">{fair_value:.2f} SAR</div>
                </div>
                """, unsafe_allow_html=True)

            # B. The Verdict
            st.markdown(f"""
            <div style="text-align:center; padding: 10px; font-size: 24px; border: 2px solid #eee; border-radius: 10px;">
                Verdict: <span class="{verdict_color}">{verdict_text} by {abs(diff_pct):.2f}%</span> {arrow}
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()

            # C. Detailed Breakdown Table
            st.subheader("📊 Methodology Breakdown")
            st.caption(f"Weights are optimized based on backtesting against {result['backtest_date']}.")
            
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, hide_index=True, use_container_width=True)

            # D. Visuals
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Weight Distribution
                labels = [r['Method'] for r in table_rows]
                values = [float(r['AI Weight'].strip('%')) for r in table_rows]
                fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                fig1.update_layout(title="AI Weight Allocation", height=300, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig1, use_container_width=True)

            with col_chart2:
                # Accuracy Comparison
                acc_values = [float(r['Historical Accuracy'].strip('%')) for r in table_rows]
                fig2 = go.Figure(data=[go.Bar(x=labels, y=acc_values, marker_color='#1f77b4')])
                fig2.update_layout(title="Historical Accuracy Test", yaxis_title="Accuracy %", height=300, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig2, use_container_width=True)
