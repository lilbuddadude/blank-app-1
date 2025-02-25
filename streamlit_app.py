import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="Options Scanner",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# Title and description
st.title("Options Arbitrage Scanner")
st.write("Scan for profitable covered call and cash-secured put opportunities")

# Simple mock data generator
def generate_option_data(symbol, strategy_type, min_return, max_days, safety_margin=0.01, min_otm=0.05):
    """Generate simulated option data for testing"""
    np.random.seed(hash(symbol) % 100)  # Use symbol as seed for consistent results
    
    # Stock price lookup table
    prices = {
        "AAPL": 184.25,
        "MSFT": 417.75,
        "AMZN": 178.15,
        "GOOGL": 164.32,
        "META": 487.95,
        "TSLA": 201.88,
        "NVDA": 842.32,
        "AMD": 174.49,
        "INTC": 43.15,
        "NFLX": 625.40
    }
    
    # Use lookup price or generate random
    if symbol in prices:
        stock_price = prices[symbol]
    else:
        stock_price = np.random.uniform(50, 500)
    
    # Generate random results
    results = []
    today = datetime.now()
    
    # Number of results to generate
    num_results = np.random.randint(0, 5)
    
    for _ in range(num_results):
        days_to_expiry = np.random.randint(5, max_days)
        expiry_date = (today + timedelta(days=days_to_expiry)).strftime("%Y-%m-%d")
        
        if strategy_type == "covered_call":
            # Generate ITM strike price
            strike = stock_price * np.random.uniform(0.8, 0.98)
            
            # Generate call price
            intrinsic = stock_price - strike
            time_value = stock_price * 0.01 * (days_to_expiry/30)
            call_price = intrinsic + time_value
            
            # Calculate metrics
            net_debit = stock_price * (1 + safety_margin) - call_price
            profit = strike - net_debit
            ret = profit / net_debit
            annual_ret = ret * (365 / days_to_expiry)
            
            # Only include if meets minimum return
            if annual_ret >= min_return and net_debit < strike:
                downside_protection = (stock_price - net_debit) / stock_price
                
                results.append({
                    "symbol": symbol,
                    "stock_price": stock_price,
                    "strike": strike,
                    "expiration_date": expiry_date,
                    "days_to_expiry": days_to_expiry,
                    "call_price": call_price,
                    "net_debit": net_debit,
                    "profit": profit,
                    "return": ret,
                    "annualized_return": annual_ret,
                    "implied_volatility": np.random.uniform(0.2, 0.6),
                    "downside_protection": downside_protection,
                    "volume": np.random.randint(100, 2000),
                    "open_interest": np.random.randint(500, 5000)
                })
        else:  # cash-secured put
            # Generate OTM strike price
            otm_factor = np.random.uniform(min_otm, min_otm + 0.2)
            strike = stock_price * (1 - otm_factor)
            
            # Generate put price
            put_price = stock_price * 0.01 * (days_to_expiry/30) * (1 + otm_factor)
            
            # Calculate metrics
            cash_required = strike * 100
            premium = put_price * 100
            ret = premium / cash_required
            annual_ret = ret * (365 / days_to_expiry)
            
            # Only include if meets minimum return
            if annual_ret >= min_return:
                effective_cost_basis = strike - put_price
                discount_to_current = (stock_price - effective_cost_basis) / stock_price
                
                results.append({
                    "symbol": symbol,
                    "stock_price": stock_price,
                    "strike": strike,
                    "expiration_date": expiry_date,
                    "days_to_expiry": days_to_expiry,
                    "put_price": put_price,
                    "premium": premium,
                    "cash_required": cash_required,
                    "return": ret,
                    "annualized_return": annual_ret,
                    "implied_volatility": np.random.uniform(0.2, 0.6),
                    "distance_from_current": otm_factor,
                    "effective_cost_basis": effective_cost_basis,
                    "discount_to_current": discount_to_current,
                    "volume": np.random.randint(100, 2000),
                    "open_interest": np.random.randint(500, 5000)
                })
    
    return results

# Run scan function
def run_scan(strategy, symbols, min_return, max_days, safety_margin=0.01, min_otm=0.05):
    """Run a scan for option opportunities"""
    all_results = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each symbol
    for i, symbol in enumerate(symbols):
        progress = (i + 1) / len(symbols)
        progress_bar.progress(progress)
        status_text.text(f"Scanning {symbol}... ({i+1}/{len(symbols)})")
        
        # Generate data for this symbol
        results = generate_option_data(
            symbol, 
            strategy, 
            min_return, 
            max_days, 
            safety_margin, 
            min_otm
        )
        
        all_results.extend(results)
        time.sleep(0.1)  # Small delay to simulate API call
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Convert to DataFrame
    if all_results:
        return pd.DataFrame(all_results).sort_values("annualized_return", ascending=False)
    else:
        return pd.DataFrame()

# Sidebar
with st.sidebar:
    st.header("Options Scanner")
    
    # API status info
    st.info("⚠️ Using simulated data (Schwab API pending)")
    st.write("**Schwab API Credentials**")
    st.write("✓ API Key: Vtbsc861GI4...Ve7O")
    st.write("✓ Secret: SvMJwXre...BiXr")
    
    # Strategy selector
    strategy = st.radio(
        "Strategy",
        ["Covered Calls", "Cash-Secured Puts"],
        index=0
    )
    
    st.divider()
    
    # Common settings
    st.subheader("Filter Settings")
    
    min_return = st.slider(
        "Min Annual Return",
        min_value=0.05,
        max_value=1.0,
        value=0.15,
        step=0.05,
        format="%.2f"
    )
    
    max_days = st.slider(
        "Max Days to Expiry",
        min_value=1,
        max_value=90,
        value=45
    )
    
    # Strategy-specific settings
    if strategy == "Covered Calls":
        safety_margin = st.slider(
            "Safety Margin",
            min_value=0.0,
            max_value=0.05,
            value=0.01,
            step=0.005,
            format="%.3f",
            help="Additional margin added to stock price to account for execution risk"
        )
    else:  # Cash-Secured Puts
        min_otm = st.slider(
            "Min OTM Percentage",
            min_value=0.0,
            max_value=0.3,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Minimum percentage out-of-the-money for puts"
        )
    
    # Symbol selection
    symbol_option = st.radio(
        "Symbol Selection",
        ["Common Stocks", "Custom Symbols"],
        index=0
    )
    
    if symbol_option == "Common Stocks":
        selected_symbols = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
            "TSLA", "NVDA", "AMD", "INTC", "NFLX"
        ]
    else:
        symbols_input = st.text_input(
            "Enter Symbols (comma separated)",
            value="AAPL,MSFT,AMZN,GOOGL,TSLA"
        )
        selected_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Scan button
    scan_button = st.button("Scan for Opportunities", type="primary", use_container_width=True)

# Trigger scan when button is clicked
if scan_button:
    with st.spinner("Running scan..."):
        if strategy == "Covered Calls":
            results = run_scan(
                strategy="covered_call",
                symbols=selected_symbols,
                min_return=min_return,
                max_days=max_days,
                safety_margin=safety_margin
            )
        else:  # Cash-Secured Puts
            results = run_scan(
                strategy="cash_secured_put",
                symbols=selected_symbols,
                min_return=min_return,
                max_days=max_days,
                min_otm=min_otm
            )
        
        # Save results to session state
        st.session_state.scan_results = {
            "type": "covered_call" if strategy == "Covered Calls" else "cash_secured_put",
            "data": results,
            "timestamp": datetime.now()
        }

# Display scan results
if 'scan_results' in st.session_state and st.session_state.scan_results is not None:
    results = st.session_state.scan_results.get('data')
    result_type = st.session_state.scan_results.get('type')
    timestamp = st.session_state.scan_results.get('timestamp')
    
    # Show results summary
    if not results.empty:
        st.success(f"Found {len(results)} opportunities (Scan time: {timestamp.strftime('%I:%M:%S %p')})")
        
        # Display results based on strategy
        if result_type == "covered_call":
            # Format display data
            display_data = results.copy()
            
            # Format percentages
            display_data['return'] = display_data['return'].apply(lambda x: f"{x:.2%}")
            display_data['annualized_return'] = display_data['annualized_return'].apply(lambda x: f"{x:.2%}")
            display_data['downside_protection'] = display_data['downside_protection'].apply(lambda x: f"{x:.2%}")
            display_data['implied_volatility'] = display_data['implied_volatility'].apply(lambda x: f"{x:.2%}")
            
            # Format currency values
            for col in ['stock_price', 'strike', 'call_price', 'net_debit', 'profit']:
                display_data[col] = display_data[col].apply(lambda x: f"${x:.2f}")
                
            # Display table
            st.subheader("Covered Call Opportunities")
            st.dataframe(display_data, use_container_width=True)
            
        else:  # cash_secured_put
            # Format display data
            display_data = results.copy()
            
            # Format percentages
            display_data['return'] = display_data['return'].apply(lambda x: f"{x:.2%}")
            display_data['annualized_return'] = display_data['annualized_return'].apply(lambda x: f"{x:.2%}")
            display_data['distance_from_current'] = display_data['distance_from_current'].apply(lambda x: f"{x:.2%}")
            display_data['discount_to_current'] = display_data['discount_to_current'].apply(lambda x: f"{x:.2%}")
            display_data['implied_volatility'] = display_data['implied_volatility'].apply(lambda x: f"{x:.2%}")
            
            # Format currency values
            for col in ['stock_price', 'strike', 'put_price', 'effective_cost_basis']:
                display_data[col] = display_data[col].apply(lambda x: f"${x:.2f}")
                
            display_data['premium'] = display_data['premium'].apply(lambda x: f"${x:.2f}")
            display_data['cash_required'] = display_data['cash_required'].apply(lambda x: f"${x:.2f}")
            
            # Display table
            st.subheader("Cash-Secured Put Opportunities")
            st.dataframe(display_data, use_container_width=True)
            
    else:
        st.info("No opportunities found matching your criteria. Try adjusting your filter settings.")
else:
    # Initial state - show instructions
    st.info("Use the sidebar to configure and run a scan for option opportunities.")
    
    # Show example of what the app does
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Covered Call Arbitrage")
        st.write("""
        The scanner looks for covered call opportunities where buying the stock and 
        immediately selling a call option creates an arbitrage situation (guaranteed profit).
        
        Key metrics analyzed:
        - Net debit (stock price - call premium)
        - Profit (strike price - net debit)
        - Annualized return
        - Downside protection
        """)
    
    with col2:
        st.subheader("Cash-Secured Puts")
        st.write("""
        The scanner finds attractive cash-secured put opportunities based on:
        - Put option premium relative to cash required
        - Annualized return
        - Distance out-of-the-money
        - Discount to current price if assigned
        
        This strategy is ideal for generating income or acquiring stocks at a discount.
        """)
