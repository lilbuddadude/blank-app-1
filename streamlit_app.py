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
        "SMCI": 46.07,
        "NVDL": 54.05,
        "NVDX": 11.33
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
            
        expiry_str = expiry_date.strftime("%m/%d/%y")
        
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
            volatile_stocks = ['TSLA', 'NVDA', 'AMD', 'SMCI', 'NVDL', 'NVDX']
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
            
            # Convert to IV percentage (like in barchart)
            iv_percentage = iv * 100
            
            # Skip if IV doesn't meet filter criteria
            if filters.get('min_iv') and iv < filters['min_iv']:
                continue
            if filters.get('max_iv') and iv > filters['max_iv']:
                continue
            
            # Calculate delta and OTM probability
            if strategy_type == "covered_call":
                # For calls, delta decreases as strike increases
                delta = max(0.01, min(0.99, 0.99 - (strike / stock_price - 0.7) * 1.5))
                otm_prob = (1 - delta) * 100  # OTM probability as percentage
                
                # Calculate moneyness (as percentage below stock price)
                moneyness = (strike - stock_price) / stock_price * 100
                
                # Generate call price
                intrinsic = max(0, stock_price - strike)
                time_value = stock_price * iv * np.sqrt(days_to_expiry/365) / 10
                call_price = intrinsic + time_value
                
                # Calculate volume and open interest
                atm_factor = 1 - min(0.8, abs(strike - stock_price) / stock_price)
                time_factor = 1 - min(0.8, days_to_expiry / 300)
                liquidity_factor = atm_factor * time_factor
                
                volume = int(2000 * liquidity_factor) + np.random.randint(100, 1000)
                open_interest = int(10000 * liquidity_factor) + np.random.randint(100, 3000)
                
                # Calculate bid and ask
                bid_price = round(call_price * 0.95, 2)
                ask_price = round(call_price * 1.05, 2)
                
                # Calculate break-even
                break_even = stock_price - bid_price
                
                # Calculate break-even percentage
                be_percentage = (break_even - stock_price) / stock_price * 100
                
                # Calculate metrics for our strategy
                net_debit = stock_price * (1 + filters.get('safety_margin', 0)) - bid_price
                profit = strike - net_debit
                ret = profit / net_debit
                annual_ret = ret * (365 / days_to_expiry)
                annual_ret_pct = annual_ret * 100  # As percentage
                
                # Calculate potential return
                potential_return = bid_price / stock_price * 100
                
                # Only include if meets minimum return criteria
                if annual_ret >= filters.get('min_return', 0) and net_debit < strike:
                    # Calculate downside protection
                    downside_protection = (stock_price - net_debit) / stock_price
                    
                    # Skip if doesn't meet downside protection criteria
                    if filters.get('min_downside') and downside_protection < filters['min_downside']:
                        continue
                    
                    results.append({
                        "symbol": symbol,
                        "price": stock_price,
                        "exp_date": expiry_str,
                        "strike": strike,
                        "moneyness": moneyness,
                        "bid": bid_price,
                        "be_bid": break_even,
                        "be_pct": be_percentage,
                        "volume": volume,
                        "open_int": open_interest,
                        "iv_pct": iv_percentage,
                        "delta": delta,
                        "otm_prob": otm_prob,
                        "pnl_rtn": potential_return,
                        "ann_rtn": annual_ret_pct,
                        "net_debit": net_debit,
                        "profit": profit,
                        "return": ret,
                        "annualized_return": annual_ret,
                        "implied_volatility": iv,
                        "downside_protection": downside_protection,
                        "last_trade": f"{np.random.randint(1, 15)}:{np.random.randint(0, 59)} ET"
                    })
            else:  # cash-secured put
                # For puts, delta increases as strike decreases
                delta = max(0.01, min(0.99, 1.7 * (stock_price / strike - 0.7)))
                delta = 1 - delta  # Convert to put delta (negative, but we'll display absolute value)
                otm_prob = delta * 100  # OTM probability as percentage
                
                # Calculate moneyness (as percentage below stock price)
                moneyness = (strike - stock_price) / stock_price * 100
                
                # Calculate how far OTM
                otm_percentage = (stock_price - strike) / stock_price
                
                # Skip if doesn't meet OTM criteria
                if filters.get('min_otm') and otm_percentage < filters['min_otm']:
                    continue
                
                # Generate put price based on time value and IV
                intrinsic = max(0, strike - stock_price)
                time_value = stock_price * iv * np.sqrt(days_to_expiry/365) / 10
                put_price = intrinsic + time_value
                
                # Calculate volume and open interest
                atm_factor = 1 - min(0.8, abs(strike - stock_price) / stock_price)
                time_factor = 1 - min(0.8, days_to_expiry / 300)
                liquidity_factor = atm_factor * time_factor
                
                volume = int(2000 * liquidity_factor) + np.random.randint(100, 1000)
                open_interest = int(10000 * liquidity_factor) + np.random.randint(100, 3000)
                
                # Calculate bid and ask
                bid_price = round(put_price * 0.95, 2)
                ask_price = round(put_price * 1.05, 2)
                
                # Calculate break-even
                break_even = strike - bid_price
                
                # Calculate break-even percentage
                be_percentage = (break_even - stock_price) / stock_price * 100
                
                # Calculate metrics
                cash_required = strike * 100  # Per contract
                premium = bid_price * 100  # Per contract
                ret = premium / cash_required
                annual_ret = ret * (365 / days_to_expiry)
                annual_ret_pct = annual_ret * 100  # As percentage
                
                # Calculate potential return
                potential_return = bid_price / stock_price * 100
                
                # Only include if meets minimum return
                if annual_ret >= filters.get('min_return', 0):
                    effective_cost_basis = strike - bid_price
                    discount_to_current = (stock_price - effective_cost_basis) / stock_price
                    
                    results.append({
                        "symbol": symbol,
                        "price": stock_price,
                        "exp_date": expiry_str,
                        "strike": strike,
                        "moneyness": moneyness,
                        "bid": bid_price,
                        "be_bid": break_even,
                        "be_pct": be_percentage,
                        "volume": volume,
                        "open_int": open_interest,
                        "iv_pct": iv_percentage,
                        "delta": abs(delta),  # Display absolute value
                        "otm_prob": otm_prob,
                        "pnl_rtn": potential_return,
                        "ann_rtn": annual_ret_pct,
                        "premium": premium,
                        "cash_required": cash_required,
                        "return": ret,
                        "annualized_return": annual_ret,
                        "implied_volatility": iv,
                        "distance_from_current": otm_percentage,
                        "effective_cost_basis": effective_cost_basis,
                        "discount_to_current": discount_to_current,
                        "last_trade": f"{np.random.randint(1, 15)}:{np.random.randint(0, 59)} ET"
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
        return pd.DataFrame(all_results).sort_values("ann_rtn", ascending=False)
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
            "TSLA", "NVDA", "AMD", "INTC", "NFLX",
            "SMCI", "NVDL", "NVDX"
        ]
        st.write(f"Selected: {', '.join(selected_symbols)}")
    else:
        symbols_input = st.text_input(
            "Enter Symbols (comma separated)",
            value="SMCI,NVDA,NVDL,NVDX,AAPL,MSFT"
        )
        selected_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Tabs for different filter categories
    filter_tabs = st.tabs(["Returns", "Stock", "Option", "IV"])
    
    with filter_tabs[0]:  # Returns tab
        st.subheader("Return Filters")
        
        min_return = st.slider(
            "Min Annual Return",
            min_value=0.05,
            max_value=20.0,
            value=1.0,
            step=0.1,
            format="%.1f"
        ) / 100  # Convert to decimal
        
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
        
        # Common columns for both strategies (Barchart-style)
        display_columns = [
            'symbol', 'price', 'exp_date', 'strike', 'moneyness', 'bid', 
            'be_bid', 'be_pct', 'volume', 'open_int', 'iv_pct', 'delta', 
            'otm_prob', 'pnl_rtn', 'ann_rtn', 'last_trade'
        ]
        
        display_headers = [
            'Symbol', 'Price', 'Exp Date', 'Strike', 'Moneyness', 'Bid', 
            'BE (Bid)', 'BE%', 'Volume', 'Open Int', 'IV', 'Delta', 
            'OTM Prob', 'Ptnl Rtn', 'Ann Rtn', 'Last Trade'
        ]
        
        # Select only the columns we want to display
        display_data = results[display_columns].copy()
        
        # Format the data
        display_data['moneyness'] = display_data['moneyness'].apply(lambda x: f"{x:.2f}%")
        display_data['be_pct'] = display_data['be_pct'].apply(lambda x: f"{x:.2f}%")
        display_data['iv_pct'] = display_data['iv_pct'].apply(lambda x: f"{x:.2f}%")
        display_data['delta'] = display_data['delta'].apply(lambda x: f"{x:.4f}")
        display_data['otm_prob'] = display_data['otm_prob'].apply(lambda x: f"{x:.2f}%")
        display_data['pnl_rtn'] = display_data['pnl_rtn'].apply(lambda x: f"{x:.1f}%")
        display_data['ann_rtn'] = display_data['ann_rtn'].apply(lambda x: f"{x:.1f}%")
        
        # Rename columns
        display_data.columns = display_headers
        
        # Display table in a Barchart.com-like style
        st.subheader(f"{'Covered Call' if result_type == 'covered_call' else 'Cash-Secured Put'} Opportunities")
        
        # Apply custom CSS for table styling
        st.markdown("""
        <style>
        table.dataframe {
            border-collapse: collapse;
            border: none;
            font-size: 0.9em;
        }
        table.dataframe th {
            background-color: #f2f2f2;
            color: #333;
            font-weight: bold;
            text-align: center;
            padding: 8px;
            border: 1px solid #ddd;
        }
        table.dataframe td {
            text-align: right;
            padding: 8px;
            border: 1px solid #ddd;
        }
        table.dataframe tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        table.dataframe tr:hover {
            background-color: #f0f0f0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the dataframe
        st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        # Add a details section to show additional metrics
        with st.expander("Show Additional Metrics"):
            if result_type == "covered_call":
                # Format additional metrics for covered calls
                additional_data = results[['symbol', 'net_debit', 'profit', 'return', 'downside_protection']].copy()
                additional_data['return'] = additional_data['return'].apply(lambda x: f"{x:.2%}")
                additional_data['downside_protection'] = additional_data['downside_protection'].apply(lambda x: f"{x:.2%}")
                additional_data.columns = ['Symbol', 'Net Debit', 'Profit', 'Return', 'Downside Protection']
                
                st.dataframe(additional_data, use_container_width=True, hide_index=True)
            else:
                # Format additional metrics for cash-secured puts
                additional_data = results[['symbol', 'premium', 'cash_required', 'return', 'effective_cost_basis', 'discount_to_current']].copy()
                additional_data['return'] = additional_data['return'].apply(lambda x: f"{x:.2%}")
                additional_data['discount_to_current'] = additional_data['discount_to_current'].apply(lambda x: f"{x:.2%}")
                additional_data.columns = ['Symbol', 'Premium', 'Cash Required', 'Return', 'Effective Cost Basis', 'Discount to Current']
                
                st.dataframe(additional_data, use_container_width=True, hide_index=True)
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
