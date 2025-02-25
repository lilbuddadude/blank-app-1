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
def generate_option_data(symbol, strategy_type, filters):
    """Generate simulated option data for testing based on filters"""
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
        "NFLX": 625.40,
        "DIS": 114.75,
        "BA": 178.22,
        "JPM": 195.24,
        "V": 280.85,
        "MA": 465.37,
        "PFE": 27.55,
        "JNJ": 151.75,
        "WMT": 59.96,
        "HD": 353.48,
        "COST": 731.98
    }
    
    # Use lookup price or generate random
    if symbol in prices:
        base_stock_price = prices[symbol]
    else:
        base_stock_price = np.random.uniform(50, 500)
        
    # Add slight variance to stock price (±1%)
    stock_price = base_stock_price * (1 + np.random.uniform(-0.01, 0.01))
    
    # Check if the stock price meets our filter criteria
    if filters.get('min_stock_price') and stock_price < filters['min_stock_price']:
        return []
    if filters.get('max_stock_price') and stock_price > filters['max_stock_price']:
        return []
    
    # Generate random results
    results = []
    today = datetime.now()
    
    # Generate expiration dates (next few weeks/months)
    expiry_dates = []
    
    # Weekly expirations (next 8 weeks)
    for i in range(1, 9):
        # Get next Friday
        days_to_friday = (4 - today.weekday()) % 7  # 4 is Friday
        next_friday = today + timedelta(days=days_to_friday + (i-1)*7)
        expiry_dates.append((next_friday, (next_friday - today).days))
    
    # Monthly expirations (next 6 months)
    for i in range(1, 7):
        # Get expiration for next month
        next_month = today.replace(month=((today.month - 1 + i) % 12) + 1)
        if next_month.month < today.month:
            next_month = next_month.replace(year=next_month.year + 1)
        # Third Friday of the month
        day_of_week = next_month.replace(day=1).weekday()
        third_friday = next_month.replace(day=1 + ((4 - day_of_week) % 7) + 14)
        expiry_dates.append((third_friday, (third_friday - today).days))
    
    # Generate strikes around current price
    strikes = []
    # For ITM options (covered calls)
    for pct in [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]:
        strikes.append(round(stock_price * pct, 1))
    # For ATM options
    strikes.append(round(stock_price, 1))
    # For OTM options (cash-secured puts)
    for pct in [1.01, 1.02, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2, 1.25, 1.3]:
        strikes.append(round(stock_price * pct, 1))
    
    # Process each expiration date
    for expiry_date, days_to_expiry in expiry_dates:
        # Skip if doesn't meet expiration filter criteria
        if filters.get('min_days') and days_to_expiry < filters['min_days']:
            continue
        if filters.get('max_days') and days_to_expiry > filters['max_days']:
            continue
            
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        
        # Process each strike price
        for strike in strikes:
            # Skip if doesn't meet strike filter criteria
            if filters.get('min_strike') and strike < filters['min_strike']:
                continue
            if filters.get('max_strike') and strike > filters['max_strike']:
                continue
            
            # Calculate implied volatility (realistic for the stock)
            base_iv = 0.25  # Base IV of 25%
            
            # Add variance based on stock (some stocks are more volatile)
            volatile_stocks = ['TSLA', 'NVDA', 'AMD', 'ROKU', 'GME', 'AMC']
            if symbol in volatile_stocks:
                base_iv += 0.15  # Add 15% for volatile stocks
            
            # Add variance based on days to expiry (more time = more uncertainty)
            iv_time_factor = min(0.2, days_to_expiry / 365)  # Max 20% additional IV
            
            # Add variance based on strike distance from current price
            strike_distance = abs(strike - stock_price) / stock_price
            iv_strike_factor = min(0.2, strike_distance * 2)  # Max 20% additional IV
            
            # Final IV calculation
            iv = base_iv + iv_time_factor + iv_strike_factor
            iv = max(0.1, min(0.9, iv))  # Cap between 10% and 90%
            
            # Skip if IV doesn't meet filter criteria
            if filters.get('min_iv') and iv < filters['min_iv']:
                continue
            if filters.get('max_iv') and iv > filters['max_iv']:
                continue
            
            if strategy_type == "covered_call":
                # Only process strikes below current price (ITM) for covered calls
                if strike >= stock_price:
                    continue
                
                # Generate call price
                intrinsic = stock_price - strike
                time_value = stock_price * iv * np.sqrt(days_to_expiry/365) / 10
                call_price = intrinsic + time_value
                
                # Calculate metrics
                net_debit = stock_price * (1 + filters.get('safety_margin', 0)) - call_price
                profit = strike - net_debit
                ret = profit / net_debit
                annual_ret = ret * (365 / days_to_expiry)
                
                # Only include if meets minimum return
                if annual_ret >= filters.get('min_return', 0) and net_debit < strike:
                    downside_protection = (stock_price - net_debit) / stock_price
                    
                    # Skip if doesn't meet downside protection criteria
                    if filters.get('min_downside') and downside_protection < filters['min_downside']:
                        continue
                    
                    # Calculate volume and open interest (higher for near-term and near-the-money)
                    atm_factor = 1 - min(0.8, abs(strike - stock_price) / stock_price)
                    time_factor = 1 - min(0.8, days_to_expiry / 300)
                    liquidity_factor = atm_factor * time_factor
                    
                    volume = int(2000 * liquidity_factor)
                    open_interest = int(10000 * liquidity_factor)
                    
                    results.append({
                        "symbol": symbol,
                        "stock_price": stock_price,
                        "strike": strike,
                        "expiration_date": expiry_str,
                        "days_to_expiry": days_to_expiry,
                        "call_price": call_price,
                        "net_debit": net_debit,
                        "profit": profit,
                        "return": ret,
                        "annualized_return": annual_ret,
                        "implied_volatility": iv,
                        "downside_protection": downside_protection,
                        "volume": volume,
                        "open_interest": open_interest
                    })
            else:  # cash-secured put
                # Only process strikes below current price (OTM) for cash-secured puts
                if strike > stock_price:
                    continue
                
                # Calculate how far OTM
                otm_percentage = (stock_price - strike) / stock_price
                
                # Skip if doesn't meet OTM criteria
                if filters.get('min_otm') and otm_percentage < filters['min_otm']:
                    continue
                
                # Generate put price based on time value and IV
                put_price = stock_price * iv * np.sqrt(days_to_expiry/365) / 10
                
                # Calculate metrics
                cash_required = strike * 100  # Per contract
                premium = put_price * 100  # Per contract
                ret = premium / cash_required
                annual_ret = ret * (365 / days_to_expiry)
                
                # Only include if meets minimum return
                if annual_ret >= filters.get('min_return', 0):
                    effective_cost_basis = strike - put_price
                    discount_to_current = (stock_price - effective_cost_basis) / stock_price
                    
                    # Calculate volume and open interest (higher for near-term and near-the-money)
                    atm_factor = 1 - min(0.8, abs(strike - stock_price) / stock_price)
                    time_factor = 1 - min(0.8, days_to_expiry / 300)
                    liquidity_factor = atm_factor * time_factor
                    
                    volume = int(2000 * liquidity_factor)
                    open_interest = int(10000 * liquidity_factor)
                    
                    results.append({
                        "symbol": symbol,
                        "stock_price": stock_price,
                        "strike": strike,
                        "expiration_date": expiry_str,
                        "days_to_expiry": days_to_expiry,
                        "put_price": put_price,
                        "premium": premium,
                        "cash_required": cash_required,
                        "return": ret,
                        "annualized_return": annual_ret,
                        "implied_volatility": iv,
                        "distance_from_current": otm_percentage,
                        "effective_cost_basis": effective_cost_basis,
                        "discount_to_current": discount_to_current,
                        "volume": volume,
                        "open_interest": open_interest
                    })
    
    return results

# Run scan function
def run_scan(strategy, symbols, filters):
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
            filters
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
    
    # Symbol selection
    st.subheader("Symbol Selection")
    
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
        st.write(f"Selected: {', '.join(selected_symbols)}")
    else:
        symbols_input = st.text_input(
            "Enter Symbols (comma separated)",
            value="AAPL,MSFT,AMZN,GOOGL,TSLA"
        )
        selected_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Tabs for different filter categories
    filter_tabs = st.tabs(["Returns", "Stock", "Option", "IV"])
    
    with filter_tabs[0]:  # Returns tab
        st.subheader("Return Filters")
        
        min_return = st.slider(
            "Min Annual Return",
            min_value=0.05,
            max_value=1.0,
            value=0.15,
            step=0.05,
            format="%.2f"
        )
        
        if strategy == "Covered Calls":
            min_downside = st.slider(
                "Min Downside Protection",
                min_value=0.0,
                max_value=0.3,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="Minimum percentage of downside protection"
            )
            
            safety_margin = st.slider(
                "Safety Margin",
                min_value=0.0,
                max_value=0.05,
                value=0.01,
                step=0.005,
                format="%.3f",
                help="Additional margin added to stock price for safety"
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
    
    with filter_tabs[1]:  # Stock tab
        st.subheader("Stock Filters")
        
        # Stock price range
        min_stock_price, max_stock_price = st.slider(
            "Stock Price Range ($)",
            min_value=0,
            max_value=1000,
            value=(0, 1000),
            step=10
        )
    
    with filter_tabs[2]:  # Option tab
        st.subheader("Option Filters")
        
        # Strike price range
        min_strike, max_strike = st.slider(
            "Strike Price Range ($)",
            min_value=0,
            max_value=1000,
            value=(0, 1000),
            step=10
        )
        
        # Days to expiry range
        min_days, max_days = st.slider(
            "Days to Expiry Range",
            min_value=0,
            max_value=180,
            value=(0, 45),
            step=1
        )
    
    with filter_tabs[3]:  # IV tab
        st.subheader("IV Filters")
        
        # Implied Volatility range
        min_iv, max_iv = st.slider(
            "Implied Volatility Range",
            min_value=0.0,
            max_value=2.0,
            value=(0.0, 1.0),
            step=0.05,
            format="%.2f"
        )
    
    # Scan button
    scan_button = st.button("Scan for Opportunities", type="primary", use_container_width=True)

# Trigger scan when button is clicked
if scan_button:
    with st.spinner("Running scan..."):
        # Build filters dictionary
        filters = {
            'min_return': min_return,
            'min_days': min_days,
            'max_days': max_days,
            'min_stock_price': min_stock_price if min_stock_price > 0 else None,
            'max_stock_price': max_stock_price if max_stock_price < 1000 else None,
            'min_strike': min_strike if min_strike > 0 else None,
            'max_strike': max_strike if max_strike < 1000 else None,
            'min_iv': min_iv if min_iv > 0 else None,
            'max_iv': max_iv if max_iv < 1.0 else None
        }
        
        # Add strategy-specific filters
        if strategy == "Covered Calls":
            filters['safety_margin'] = safety_margin
            filters['min_downside'] = min_downside
        else:  # Cash-Secured Puts
            filters['min_otm'] = min_otm
        
        # Run the scan
        results = run_scan(
            strategy="covered_call" if strategy == "Covered Calls" else "cash_secured_put",
            symbols=selected_symbols,
            filters=filters
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
