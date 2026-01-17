import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from optimizer import ValuationOptimizer
from valuation_engine import ValuationEngine

# --- PAGE SETUP ---
st.set_page_config(page_title="Saudi Stock Valuator AI", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .big-metric { font-size: 24px; font-weight: bold; color: #0e1117; }
    .card { background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd; height: 100%; }
    .highlight-ai { border-left: 5px solid #ff4b4b; }
    .highlight-acc { border-left: 5px solid #1f77b4; }
    .header-style { font-size: 16px; color: #555; font-weight: 600; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🇸🇦 Saudi Stock Valuation: AI vs. Intuition")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    stock_input = st.text_input("Stock Code", value="1120")
    run_btn = st.button("🚀 Run Analysis", type="primary")
    st.info("Note: Price trends and metrics are based on available data from configured APIs.")

# --- MAIN LOGIC ---
if run_btn and stock_input:
    optimizer = ValuationOptimizer()
    
    with st.spinner(f"🔍 Analyzing {stock_input} (This may take a moment)..."):
        result = optimizer.find_optimal_strategy(stock_input)
        
        if "error" in result:
            st.error(result['error'])
        else:
            full_data = result['full_data']
            prices = full_data.get('prices', pd.DataFrame())
            meta = full_data.get('meta', {})
            
            # --- SECTION 1: MARKET PROFILE ---
            st.subheader(f"📊 Market Profile: {stock_input}")
            m1, m2, m3, m4 = st.columns(4)
            
            # Helper to handle missing data gracefully
            beta = meta.get("beta", "N/A")
            pe = meta.get("trailingPE", "N/A")
            if isinstance(pe, (float, int)): pe = f"{pe:.2f}"
            
            latest_price = 0.0
            ret_1y = "N/A"
            vol = "N/A"
            
            if not prices.empty:
                latest_price = prices['Close'].iloc[-1]
                vol = f"{prices['Volume'].iloc[-1]/1000:.1f}K"
                if len(prices) > 252:
                    price_ago = prices['Close'].iloc[-252]
                    ret = ((latest_price - price_ago) / price_ago) * 100
                    ret_1y = f"{ret:+.1f}%"
            
            with m1: st.markdown(f'<div class="card"><div class="header-style">Beta</div><div class="big-metric">{beta}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="card"><div class="header-style">P/E Ratio</div><div class="big-metric">{pe}</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="card"><div class="header-style">1Y Return</div><div class="big-metric" style="color:{"green" if "+" in ret_1y else "red"}">{ret_1y}</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="card"><div class="header-style">Volume</div><div class="big-metric">{vol}</div></div>', unsafe_allow_html=True)
            
            st.divider()

            # --- SECTION 2: CALCULATE VALUATIONS ---
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
            
            # --- SECTION 3: VERDICT SCORECARDS ---
            c1, c2, c3 = st.columns([1,1,1])
            
            with c1:
                st.markdown(f'<div class="card"><div class="header-style">Market Price</div><div class="big-metric">{latest_price:.2f} SAR</div></div>', unsafe_allow_html=True)

            with c2:
                diff = ((ai_val - latest_price)/latest_price)*100
                color = "green" if diff > 0 else "red"
                symbol = "🔼" if diff > 0 else "🔽"
                st.markdown(f'<div class="card highlight-ai"><div class="header-style">🤖 AI Solver Value</div><div class="big-metric">{ai_val:.2f} SAR</div><div style="color:{color}; font-weight:bold">{diff:+.1f}% {symbol}</div></div>', unsafe_allow_html=True)
                
            with c3:
                diff = ((acc_val - latest_price)/latest_price)*100
                color = "green" if diff > 0 else "red"
                symbol = "🔼" if diff > 0 else "🔽"
                st.markdown(f'<div class="card highlight-acc"><div class="header-style">🎯 Accuracy Model Value</div><div class="big-metric">{acc_val:.2f} SAR</div><div style="color:{color}; font-weight:bold">{diff:+.1f}% {symbol}</div></div>', unsafe_allow_html=True)

            # --- SECTION 4: WEIGHTS BREAKDOWN ---
            st.markdown("###")
            with st.expander("⚖️ See How The AI Decided (Weights & Logic)", expanded=True):
                rows = []
                for m in ["DCF", "PE", "PB", "EV"]:
                    rows.append({
                        "Valuation Method": m,
                        "Current Value (SAR)": f"{curr_vals[m]:.2f}",
                        "Avg Historical Accuracy": f"{result['strategies']['accuracy'][m]['accuracy']:.1%}",
                        "🤖 AI Solver Weight": f"{result['strategies']['solver'][m]['weight']:.1%}",
                        "🎯 Accuracy Weight": f"{result['strategies']['accuracy'][m]['weight']:.1%}"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # --- SECTION 5: CHARTS & HISTORY (TABS) ---
            st.markdown("###")
            tab1, tab2 = st.tabs(["📈 Price Trend (Max History)", "📅 Walk-Forward Validation Log"])
            
            with tab1:
                if not prices.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=prices.index, y=prices['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=2)))
                    fig.update_layout(
                        title=f"{stock_input} Price Trend", 
                        xaxis_title="Date", 
                        yaxis_title="Price (SAR)",
                        height=450, 
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No price history available to plot.")

            with tab2:
                st.caption("This log shows how the models performed in previous years (Backtesting).")
                hist_rows = []
                for h in result['history']:
                    row = {"Date Tested": h['year_start'], "Real Price (1Yr Later)": f"{h['actual_price_next_year']:.2f}"}
                    for m, v in h['predictions'].items():
                        row[m] = f"{v:.2f}"
                    hist_rows.append(row)
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)
