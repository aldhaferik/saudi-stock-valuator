import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from optimizer import ValuationOptimizer
from valuation_engine import ValuationEngine

st.set_page_config(page_title="Saudi Stock Valuator AI", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .big-metric { font-size: 24px; font-weight: bold; color: #0e1117; }
    .card { background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd; }
    .highlight-ai { border-left: 5px solid #ff4b4b; }
    .highlight-acc { border-left: 5px solid #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

st.title("🇸🇦 Saudi Stock Valuation: AI vs. Intuition")

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
            full_data = result['full_data']
            prices = full_data.get('prices', pd.DataFrame())
            meta = full_data.get('meta', {})
            
            # --- MARKET PROFILE ---
            st.subheader(f"📊 Market Profile: {stock_input}")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.markdown(f'<div class="card"><div>Beta</div><div class="big-metric">{meta.get("beta", "N/A")}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="card"><div>P/E</div><div class="big-metric">{meta.get("trailingPE", "N/A")}</div></div>', unsafe_allow_html=True)
            
            latest_price = prices['Close'].iloc[-1] if not prices.empty else 0
            ret_1y = "N/A"
            if len(prices) > 252:
                ret = ((latest_price - prices['Close'].iloc[-252]) / prices['Close'].iloc[-252]) * 100
                ret_1y = f"{ret:.1f}%"
                
            with m3: st.markdown(f'<div class="card"><div>1Y Return</div><div class="big-metric">{ret_1y}</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="card"><div>Price</div><div class="big-metric">{latest_price:.2f} SAR</div></div>', unsafe_allow_html=True)
            
            st.divider()

            # --- CALCULATE CURRENT VALUATION ---
            engine = ValuationEngine(full_data['financials'])
            curr_vals = {
                "DCF": engine.dcf_valuation(growth_rate=0.04),
                "PE": engine.multiples_valuation(pe_ratio=18.0)['PE_Valuation'],
                "PB": engine.multiples_valuation(pb_ratio=2.5)['PB_Valuation'],
                "EV": engine.multiples_valuation(ev_ebitda_ratio=12.0)['EBITDA_Valuation']
            }

            def get_weighted_val(strategy):
                val = 0
                for m, data in strategy.items():
                    val += curr_vals[m] * data['weight']
                return val

            ai_val = get_weighted_val(result['strategies']['solver'])
            acc_val = get_weighted_val(result['strategies']['accuracy'])
            
            # --- VERDICT ---
            c1, c2 = st.columns(2)
            with c1:
                diff = ((ai_val - latest_price)/latest_price)*100
                color = "green" if diff > 0 else "red"
                st.markdown(f'<div class="card highlight-ai"><h3>🤖 AI Solver</h3><div class="big-metric">{ai_val:.2f} SAR</div><div style="color:{color}">{diff:+.1f}% vs Market</div></div>', unsafe_allow_html=True)
            with c2:
                diff = ((acc_val - latest_price)/latest_price)*100
                color = "green" if diff > 0 else "red"
                st.markdown(f'<div class="card highlight-acc"><h3>🎯 Accuracy Model</h3><div class="big-metric">{acc_val:.2f} SAR</div><div style="color:{color}">{diff:+.1f}% vs Market</div></div>', unsafe_allow_html=True)

            # --- WEIGHTS BREAKDOWN ---
            st.markdown("###")
            with st.expander("⚖️ See How The AI Decided (Weights)", expanded=True):
                rows = []
                for m in ["DCF", "PE", "PB", "EV"]:
                    rows.append({
                        "Method": m,
                        "Valuation (SAR)": f"{curr_vals[m]:.2f}",
                        "Avg Accuracy": f"{result['strategies']['accuracy'][m]['accuracy']:.1%}",
                        "🤖 Solver Weight": f"{result['strategies']['solver'][m]['weight']:.1%}",
                        "🎯 Accuracy Weight": f"{result['strategies']['accuracy'][m]['weight']:.1%}"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # --- HISTORY ---
            st.markdown("###")
            st.caption("Walk-Forward Validation History")
            hist_rows = []
            for h in result['history']:
                row = {"Date Tested": h['year_start'], "Real Price (1Yr Later)": f"{h['actual_price_next_year']:.2f}"}
                for m, v in h['predictions'].items():
                    row[m] = f"{v:.2f}"
                hist_rows.append(row)
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)
