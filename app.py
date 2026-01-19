import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from optimizer import ValuationOptimizer
from valuation_engine import ValuationEngine

st.set_page_config(page_title="Saudi Stock Valuator AI", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .big-metric { font-size: 24px; font-weight: bold; color: #0e1117; }
    .card { background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd; height: 100%; }
    .highlight-ai { border-left: 5px solid #ff4b4b; }
    .highlight-acc { border-left: 5px solid #1f77b4; }
    .header-style { font-size: 16px; color: #555; font-weight: 600; margin-bottom: 5px; }
    .sub-text { font-size: 12px; color: #888; margin-top: -5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🇸🇦 Saudi Stock Valuator: AI vs. Intuition")

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
        
        # ==========================
        # MODE 1: ETF / FUND VIEW
        # ==========================
        elif result.get('type') == 'ETF':
            full_data = result['full_data']
            roi_data = result['roi_metrics']
            prices = full_data['prices']
            latest_price = prices['Close'].iloc[-1]
            
            st.success(f"🔹 **{stock_input}** identified as an ETF/Fund.")
            
            # 1. Price Header
            c1, c2 = st.columns([1, 3])
            with c1:
                 st.markdown(f'<div class="card"><div class="header-style">Current Price</div><div class="big-metric">{latest_price:.2f} SAR</div></div>', unsafe_allow_html=True)
            
            # 2. ROI Cards with DATES (The Fix)
            st.subheader("📊 Trailing Returns (ROI)")
            roi_cols = st.columns(len(roi_data))
            for i, (label, metrics) in enumerate(roi_data.items()):
                with roi_cols[i]:
                    if metrics:
                        val = metrics['roi']
                        ref_date = metrics['ref_date'] # Get the date
                        color = "green" if val > 0 else "red"
                        
                        st.markdown(f"**{label}**")
                        st.markdown(f"<span style='color:{color}; font-size:18px; font-weight:bold'>{val:.1%}</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='sub-text'>since {ref_date}</div>", unsafe_allow_html=True) # Display Date
                    else:
                        st.markdown(f"**{label}**\nN/A")

            # 3. Interactive Price Chart
            st.subheader("📈 Price Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices.index, y=prices['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=2)))
            fig.update_layout(template="plotly_white", height=450, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # 4. INSPECT MATH TABLE
            st.markdown("###")
            with st.expander("🔍 Inspect ROI Data (Verify Dates & Prices)", expanded=False):
                st.info("Check the exact dates used to calculate the returns above.")
                etf_rows = []
                for label, metrics in roi_data.items():
                    if metrics:
                        etf_rows.append({
                            "Period": label,
                            "Reference Date": metrics['ref_date'],
                            "Past Price": f"{metrics['ref_price']:.2f}",
                            "Current Price": f"{latest_price:.2f}",
                            "ROI Calculation": f"({latest_price:.2f} - {metrics['ref_price']:.2f}) / {metrics['ref_price']:.2f}"
                        })
                st.dataframe(pd.DataFrame(etf_rows), use_container_width=True)


        # ==========================
        # MODE 2: STOCK VIEW
        # ==========================
        elif result.get('type') == 'STOCK':
            full_data = result['full_data']
            prices = full_data.get('prices', pd.DataFrame())
            meta = full_data.get('meta', {})
            
            # --- MARKET PROFILE ---
            st.subheader(f"📊 Market Profile: {stock_input}")
            m1, m2, m3, m4 = st.columns(4)
            
            beta = meta.get("beta", "N/A")
            pe = meta.get("trailingPE", "N/A")
            if isinstance(pe, (float, int)): pe = f"{pe:.2f}"
            latest_price = prices['Close'].iloc[-1] if not prices.empty else 0
            
            with m1: st.markdown(f'<div class="card"><div class="header-style">Beta</div><div class="big-metric">{beta}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="card"><div class="header-style">P/E Ratio</div><div class="big-metric">{pe}</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="card"><div class="header-style">Price</div><div class="big-metric">{latest_price:.2f}</div></div>', unsafe_allow_html=True)
            
            st.divider()

            # --- VALUATION CALCULATION ---
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
            
            # --- VERDICT SCORECARDS ---
            c1, c2 = st.columns(2)
            with c1:
                diff = ((ai_val - latest_price)/latest_price)*100
                color = "green" if diff > 0 else "red"
                st.markdown(f'<div class="card highlight-ai"><div class="header-style">🤖 AI Solver Value</div><div class="big-metric">{ai_val:.2f} SAR</div><div style="color:{color}">{diff:+.1f}%</div></div>', unsafe_allow_html=True)
                
            with c2:
                diff = ((acc_val - latest_price)/latest_price)*100
                color = "green" if diff > 0 else "red"
                st.markdown(f'<div class="card highlight-acc"><div class="header-style">🎯 Accuracy Model Value</div><div class="big-metric">{acc_val:.2f} SAR</div><div style="color:{color}">{diff:+.1f}%</div></div>', unsafe_allow_html=True)

            # --- DRILL DOWN / INSPECT MATH ---
            st.markdown("### 🔍 Drill Down: Inspect the Math")
            st.info("Click the tabs below to verify the logic behind the numbers.")
            
            tab_dcf, tab_pe, tab_solver, tab_acc = st.tabs(["💵 DCF Details", "📊 P/E Details", "🤖 AI Solver Logic", "🎯 Accuracy Logic"])

            history = result['history']
            
            with tab_dcf:
                dcf_data = []
                for h in history:
                    dcf_data.append({
                        "Year": h['year_start'][:4],
                        "FCF Input": f"{h['debug_inputs']['FCF']:,.0f}",
                        "Predicted Price": f"{h['predictions']['DCF']:.2f}",
                        "Actual Price": f"{h['actual_price_next_year']:.2f}",
                        "Error": f"{abs(h['predictions']['DCF'] - h['actual_price_next_year'])/h['actual_price_next_year']:.1%}"
                    })
                st.dataframe(pd.DataFrame(dcf_data), use_container_width=True)

            with tab_pe:
                pe_data = []
                for h in history:
                    pe_data.append({
                        "Year": h['year_start'][:4],
                        "EPS Input": f"{h['debug_inputs']['EPS']:.2f}",
                        "Predicted Price": f"{h['predictions']['PE']:.2f}",
                        "Actual Price": f"{h['actual_price_next_year']:.2f}",
                        "Error": f"{abs(h['predictions']['PE'] - h['actual_price_next_year'])/h['actual_price_next_year']:.1%}"
                    })
                st.dataframe(pd.DataFrame(pe_data), use_container_width=True)

            with tab_solver:
                rows = []
                for m in ["DCF", "PE", "PB", "EV"]:
                    rows.append({
                        "Method": m,
                        "Avg Accuracy": f"{result['strategies']['accuracy'][m]['accuracy']:.1%}",
                        "AI Weight": f"{result['strategies']['solver'][m]['weight']:.1%}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            with tab_acc:
                acc_breakdown = []
                for m in ["DCF", "PE", "PB", "EV"]:
                    yearly_accs = []
                    for h in history:
                        pred = h['predictions'][m]
                        actual = h['actual_price_next_year']
                        if pred > 0:
                            acc = max(0, 1.0 - (abs(pred - actual) / actual))
                            yearly_accs.append(f"{acc:.0%}")
                        else:
                            yearly_accs.append("0%")
                    
                    row = {"Method": m, "Avg Accuracy": f"{result['strategies']['accuracy'][m]['accuracy']:.1%}"}
                    for i, yr_acc in enumerate(yearly_accs):
                        row[f"Yr {i+1}"] = yr_acc
                    acc_breakdown.append(row)
                st.dataframe(pd.DataFrame(acc_breakdown), use_container_width=True)

            # --- CHARTS ---
            st.markdown("###")
            st.caption("Price Trend History")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices.index, y=prices['Close'], mode='lines', name='Price'))
            st.plotly_chart(fig, use_container_width=True)
