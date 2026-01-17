import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from optimizer import ValuationOptimizer
from valuation_engine import ValuationEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Saudi Stock Valuator AI", page_icon="📈", layout="wide")

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .big-metric { font-size: 26px; font-weight: bold; color: #0e1117; }
    .header-style { font-size: 18px; color: #555; font-weight: 600; margin-bottom: 10px; }
    .card { background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd; height: 100%; }
    .highlight-ai { border-left: 5px solid #ff4b4b; }
    .highlight-acc { border-left: 5px solid #1f77b4; }
    .metric-label { font-size: 14px; color: #666; margin-bottom: 5px; }
    .metric-value { font-size: 22px; font-weight: bold; color: #333; }
    </style>
    """, unsafe_allow_html=True)

st.title("🇸🇦 Saudi Stock Valuation: AI vs. Intuition")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    stock_input = st.text_input("Stock Code", value="1120")
    run_btn = st.button("🚀 Run Analysis", type="primary")
    
    st.markdown("---")
    st.info("**Note on Sector Data:**\nReal-time Sector Averages are not available in free APIs. The app displays the **Stock's P/E** and **Sector Name** for your manual comparison.")

# --- MAIN APP LOGIC ---
if run_btn and stock_input:
    optimizer = ValuationOptimizer()
    
    with st.spinner(f"🔍 Analyzing {stock_input}... Fetching History & Market Data..."):
        result = optimizer.find_optimal_strategy(stock_input)
        
        if "error" in result:
            st.error(result['error'])
        else:
            full_data = result['full_data_cache']
            
            # --- 1. EXTRACT MARKET PROFILE METRICS ---
            meta = full_data.get('meta', {})
            prices_df = full_data.get('prices', pd.DataFrame())
            
            # Safely get metrics (handle missing data with defaults)
            stock_beta = meta.get('beta', 'N/A')
            stock_pe = meta.get('trailingPE', 'N/A')
            sector_name = meta.get('sector', 'Unknown Sector')
            
            # --- 2. DISPLAY: MARKET PROFILE SECTION ---
            st.subheader(f"📊 Market Profile: {stock_input} ({sector_name})")
            
            # A. Key Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.markdown(f'<div class="card"><div class="metric-label">Beta (Risk)</div><div class="metric-value">{stock_beta}</div></div>', unsafe_allow_html=True)
            
            with m2:
                # Format P/E nicely if it's a number
                pe_display = f"{stock_pe:.2f}" if isinstance(stock_pe, (int, float)) else stock_pe
                st.markdown(f'<div class="card"><div class="metric-label">Trailing P/E</div><div class="metric-value">{pe_display}</div></div>', unsafe_allow_html=True)
            
            with m3:
                # Calculate 1-Year Return
                if not prices_df.empty and len(prices_df) > 252:
                    price_1y_ago = prices_df['Close'].iloc[-252]
                    price_now = prices_df['Close'].iloc[-1]
                    ret_1y = ((price_now - price_1y_ago) / price_1y_ago) * 100
                    color = "green" if ret_1y > 0 else "red"
                    st.markdown(f'<div class="card"><div class="metric-label">1-Year Return</div><div class="metric-value" style="color:{color}">{ret_1y:.1f}%</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="card"><div class="metric-label">1-Year Return</div><div class="metric-value">N/A</div></div>', unsafe_allow_html=True)
            
            with m4:
                 # Check if volume exists
                 if not prices_df.empty and 'Volume' in prices_df.columns:
                     latest_vol = prices_df['Volume'].iloc[-1]
                     vol_display = f"{latest_vol/1000:.1f}K"
                 else:
                     vol_display = "N/A"
                 st.markdown(f'<div class="card"><div class="metric-label">Last Volume</div><div class="metric-value">{vol_display}</div></div>', unsafe_allow_html=True)

            # B. Price Trend Chart (Max History)
            if not prices_df.empty:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=prices_df.index, 
                    y=prices_df['Close'], 
                    mode='lines', 
                    name='Price', 
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_trend.update_layout(
                    title=f"📈 Price Trend (Max History)",
                    height=350,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title="Date",
                    yaxis_title="Price (SAR)",
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            st.divider()

            # --- 3. RUN VALUATION MODELS (On Today's Data) ---
            latest_price = prices_df['Close'].iloc[-1]
            engine = ValuationEngine(full_data['financials'])
            
            # Calculate individual methods
            dcf_val = engine.dcf_valuation(growth_rate=0.04)
            mults = engine.multiples_valuation(pe_ratio=18.0, pb_ratio=2.5, ev_ebitda_ratio=12.0)
            
            curr_vals = {
                "DCF (Moderate)": dcf_val,
                "P/E Multiple": mults['PE_Valuation'],
                "P/B Multiple": mults['PB_Valuation'],
                "EV/EBITDA": mults['EBITDA_Valuation']
            }

            # Helper function to calculate final weighted values
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

            # Calculate for both strategies
            ai_val, ai_rows = calculate_result(result['strategies']['solver'])
            acc_val, acc_rows = calculate_result(result['strategies']['accuracy'])
            
            # Calculate % Difference from Market
            ai_diff = ((ai_val - latest_price) / latest_price) * 100
            acc_diff = ((acc_val - latest_price) / latest_price) * 100

            # --- 4. DISPLAY VALUATION RESULTS ---
            st.subheader("🎯 Valuation Verdict")
            
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
                symbol = "🔼" if ai_diff > 0 else "🔽"
                st.markdown(f"""
                <div class="card highlight-ai">
                    <div class="header-style">🤖 AI Solver Model</div>
                    <div class="big-metric">{ai_val:.2f} SAR</div>
                    <div style="color: {color}; font-weight:bold;">{ai_diff:+.1f}% vs Market {symbol}</div>
                    <div style="font-size: 12px; color: #555; margin-top:5px;">Minimizes total mathematical error.</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                color = "green" if acc_diff > 0 else "red"
                symbol = "🔼" if acc_diff > 0 else "🔽"
                st.markdown(f"""
                <div class="card highlight-acc">
                    <div class="header-style">🎯 Accuracy Model</div>
                    <div class="big-metric">{acc_val:.2f} SAR</div>
                    <div style="color: {color}; font-weight:bold;">{acc_diff:+.1f}% vs Market {symbol}</div>
                    <div style="font-size: 12px; color: #555; margin-top:5px;">Higher accuracy = Higher weight.</div>
                </div>
                """, unsafe_allow_html=True)

           # ... (Keep all the top code from the previous app.py) ...

            # --- 5. DETAILED BREAKDOWN TABS ---
            st.markdown("###")
            tab1, tab2 = st.tabs(["📅 Walk-Forward Analysis (Year-by-Year)", "🎯 Final Weights"])
            
            with tab1:
                st.caption("This shows how each model performed in predicting the price 1 year into the future, repeated over multiple years.")
                
                # Prepare data for the timeline chart
                history = result['walk_forward_history']
                years = [rec['year_start'][:4] for rec in history]
                
                # Create a Line Chart of Accuracy Over Time
                fig_timeline = go.Figure()
                
                # Add Actual Price Line (Baseline)
                # Note: We can't easily plot "Actual" vs "Predicted" on one line because the dates shift.
                # Instead, we plot "Accuracy %" per year.
                
                methods = ["DCF (Moderate)", "P/E Multiple", "P/B Multiple", "EV/EBITDA"]
                
                for method in methods:
                    accuracies = []
                    for rec in history:
                        pred = rec['predictions'].get(method, 0)
                        actual = rec['actual_price_next_year']
                        if pred > 0:
                            # Accuracy = 1 - Error
                            acc = max(0, 1 - (abs(pred - actual) / actual))
                            accuracies.append(acc * 100)
                        else:
                            accuracies.append(0)
                    
                    fig_timeline.add_trace(go.Scatter(x=years, y=accuracies, mode='lines+markers', name=method))

                fig_timeline.update_layout(
                    title="Model Accuracy Over Time (Rolling Backtest)",
                    xaxis_title="Prediction Year",
                    yaxis_title="Accuracy % (100% = Perfect Prediction)",
                    hovermode="x unified",
                    height=400
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Show the raw data table
                st.subheader("Year-by-Year Breakdown")
                rows = []
                for rec in history:
                    row = {"Year": rec['year_start'][:4], "Real Price (1 Yr Later)": f"{rec['actual_price_next_year']:.2f}"}
                    for method, val in rec['predictions'].items():
                        row[method] = f"{val:.2f}"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            with tab2:
                st.caption("These weights are calculated by averaging the accuracy across all the years shown in the Walk-Forward test.")
                
                acc_strategy = result['strategies']['accuracy']
                # Convert to table format
                acc_rows = []
                for m, data in acc_strategy.items():
                    acc_rows.append({
                        "Method": m,
                        "Avg 4-Year Accuracy": f"{data['historical_accuracy']:.1%}",
                        "Final Weight": f"{data['weight']:.1%}"
                    })
                
                c1, c2 = st.columns([2, 1])
                c1.dataframe(pd.DataFrame(acc_rows), use_container_width=True, hide_index=True)
                
                fig = go.Figure(data=[go.Pie(labels=[r['Method'] for r in acc_rows], values=[float(r['Final Weight'].strip('%')) for r in acc_rows], hole=.4)])
                fig.update_layout(title="Final Averaged Weights", height=300, margin=dict(t=30, b=0, l=0, r=0))
                c2.plotly_chart(fig, use_container_width=True)
