import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Set page config with dark theme
st.set_page_config(
    page_title="Options Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# Title and description
st.title("Options Arbitrage Scanner")
st.write("Scan for profitable covered call and cash-secured put opportunities")

# Simple function to simulate option data
def generate_mock_option_data(strategy_type="covered_call", num_rows=20):
    """Generate mock option data for testing"""
    np.random.seed(42)  # For consistent results
    
    symbols = ["SMCI", "NVDL", "NVDX", "AAPL", "MSFT", "NVDA", "AMD"]
    prices = {
        "SMCI": 46.07,
        "NVDL": 54.05,
        "NVDX": 11.33,
        "AAPL": 184.25,
        "MSFT": 417.75,
        "NVDA": 842.32,
        "AMD": 174.49
    }
    
    data = []
    
    for i in range(num_rows):
        symbol = np.random.choice(symbols)
        stock_price = prices[symbol]
        
        # Generate expiration date (next 1-2 months)
        days_to_expiry = np.random.randint(14, 60)
        expiry_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime("%m/%d/%y")
        
        # For covered calls, generate ITM strikes
        if strategy_type == "covered_call":
            strike = round(stock_price * np.random.uniform(0.85, 0.98), 2)
            moneyness = (strike - stock_price) / stock_price * 100
            
            # Option price calculation
            bid_price = round((stock_price - strike) + (stock_price * 0.03), 2)
            
            # Calculate metrics
            net_profit = (bid_price * 100) - ((100 * stock_price) - (100 * strike))
            break_even = stock_price - bid_price
            be_percentage = (break_even - stock_price) / stock_price * 100
            
            # Additional metrics
            volume = np.random.randint(500, 5000)
            open_interest = np.random.randint(1000, 20000)
            iv_pct = np.random.uniform(200, 400)
            delta = 1 - (strike / stock_price) * 1.1
            delta = max(0.1, min(0.99, delta))
            otm_prob = (1 - delta) * 100
            
            # Returns
            potential_return = (bid_price / stock_price) * 100
            annual_return = potential_return * (365 / days_to_expiry)
            
            data.append({
                "symbol": symbol,
                "price": stock_price,
                "exp_date": expiry_date,
                "strike": strike,
                "moneyness": moneyness,
                "bid": bid_price,
                "net_profit": net_profit,
                "be_bid": break_even,
                "be_pct": be_percentage,
                "volume": volume,
                "open_int": open_interest,
                "iv_pct": iv_pct,
                "delta": delta,
                "otm_prob": otm_prob,
                "pnl_rtn": potential_return,
                "ann_rtn": annual_return,
                "last_trade": f"{np.random.randint(1, 15)}:{np.random.randint(10, 59)} ET"
            })
        else:  # Cash-Secured Puts
            strike = round(stock_price * np.random.uniform(0.7, 0.95), 2)
            moneyness = (strike - stock_price) / stock_price * 100
            
            # Option price calculation
            bid_price = round(stock_price * 0.03 * (1 + abs(moneyness)/10), 2)
            
            # Calculate metrics
            net_profit = (bid_price * 100) - ((100 * strike) - (100 * stock_price))
            break_even = strike - bid_price
            be_percentage = (break_even - stock_price) / stock_price * 100
            
            # Additional metrics
            volume = np.random.randint(500, 5000)
            open_interest = np.random.randint(1000, 20000)
            iv_pct = np.random.uniform(200, 400)
            delta = (strike / stock_price) * 0.8
            delta = max(0.1, min(0.99, delta))
            otm_prob = delta * 100
            
            # Returns
            potential_return = (bid_price / stock_price) * 100
            annual_return = potential_return * (365 / days_to_expiry) * 2
            
            data.append({
                "symbol": symbol,
                "price": stock_price,
                "exp_date": expiry_date,
                "strike": strike,
                "moneyness": moneyness,
                "bid": bid_price,
                "net_profit": net_profit,
                "be_bid": break_even,
                "be_pct": be_percentage,
                "volume": volume,
                "open_int": open_interest,
                "iv_pct": iv_pct,
                "delta": delta,
                "otm_prob": otm_prob,
                "pnl_rtn": potential_return,
                "ann_rtn": annual_return,
                "last_trade": f"{np.random.randint(1, 15)}:{np.random.randint(10, 59)} ET"
            })
    
    return pd.DataFrame(data)

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
    
    # Tabs for different filter categories
    filter_tabs = st.tabs(["Return", "Price", "Option"])
    
    with filter_tabs[0]:  # Return tab
        st.subheader("Return Filters")
        
        min_annual_return = st.slider(
            "Min Annual Return (%)",
            min_value=0,
            max_value=2000,
            value=100,
            step=50
        )
    
    with filter_tabs[1]:  # Price tab
        st.subheader("Price Filters")
        
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
        
        # Days to expiry range
        min_days, max_days = st.slider(
            "Days to Expiry Range",
            min_value=0,
            max_value=180,
            value=(0, 45),
            step=1
        )
    
    # Scan button
    scan_button = st.button("Scan for Opportunities", type="primary", use_container_width=True)

# Trigger scan when button is clicked
if scan_button:
    with st.spinner("Running scan..."):
        # Generate mock data based on strategy
        results = generate_mock_option_data(
            strategy_type="covered_call" if strategy == "Covered Calls" else "cash_secured_put",
            num_rows=30
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
            'net_profit', 'be_bid', 'be_pct', 'volume', 'open_int', 
            'iv_pct', 'delta', 'otm_prob', 'pnl_rtn', 'ann_rtn', 'last_trade'
        ]
        
        # Select only the columns we want to display
        display_data = results[display_columns].copy()
        
        # Create new dataframe for display with formatted values
        formatted_df = pd.DataFrame()
        
        # Copy and format each column
        formatted_df['Symbol'] = display_data['symbol']
        formatted_df['Price'] = display_data['price'].apply(lambda x: f"${x:.2f}")
        formatted_df['Exp Date'] = display_data['exp_date']
        formatted_df['Strike'] = display_data['strike'].apply(lambda x: f"${x:.2f}")
        
        # Create colored moneyness column
        def format_moneyness(val):
            color = 'red' if val < 0 else 'green'
            return f'<span style="color:{color}">{val:.2f}%</span>'
        
        formatted_df['Moneyness'] = display_data['moneyness'].apply(format_moneyness)
        
        # Continue formatting other columns
        formatted_df['Bid'] = display_data['bid'].apply(lambda x: f"${x:.2f}")
        
        # Create colored net profit column
        def format_net_profit(val):
            color = 'red' if val < 0 else 'green'
            return f'<span style="color:{color}">${val:.2f}</span>'
        
        formatted_df['Net Profit'] = display_data['net_profit'].apply(format_net_profit)
        
        # Continue with other columns
        formatted_df['BE (Bid)'] = display_data['be_bid'].apply(lambda x: f"${x:.2f}")
        formatted_df['BE%'] = display_data['be_pct'].apply(lambda x: f"{x:.2f}%")
        formatted_df['Volume'] = display_data['volume'].apply(lambda x: f"{x:,}")
        formatted_df['Open Int'] = display_data['open_int'].apply(lambda x: f"{x:,}")
        formatted_df['IV'] = display_data['iv_pct'].apply(lambda x: f"{x:.2f}%")
        formatted_df['Delta'] = display_data['delta'].apply(lambda x: f"{x:.4f}")
        formatted_df['OTM Prob'] = display_data['otm_prob'].apply(lambda x: f"{x:.2f}%")
        formatted_df['Ptnl Rtn'] = display_data['pnl_rtn'].apply(lambda x: f"{x:.1f}%")
        formatted_df['Ann Rtn'] = display_data['ann_rtn'].apply(lambda x: f"{x:.1f}%")
        formatted_df['Last Trade'] = display_data['last_trade']
        
        # Display the table
        st.subheader(f"{'Covered Call' if result_type == 'covered_call' else 'Cash-Secured Put'} Opportunities")
        
        # Use HTML to render the formatted table with colors
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Offer download button for the data
        csv = display_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name=f"options_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv',
        )
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
