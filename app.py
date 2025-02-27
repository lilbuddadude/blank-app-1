import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sqlite3
import os
import traceback
import subprocess

# Set page config with dark theme
st.set_page_config(
    page_title="Options Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for sorting and results
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'sort_column' not in st.session_state:
    st.session_state.sort_column = None
if 'sort_ascending' not in st.session_state:
    st.session_state.sort_ascending = True
if 'db_last_updated' not in st.session_state:
    st.session_state.db_last_updated = None
if 'api_status' not in st.session_state:
    st.session_state.api_status = "Unknown"
if 'db_status_message' not in st.session_state:
    st.session_state.db_status_message = ""

# Title and description
st.title("Options Arbitrage Scanner")
st.write("Scan for profitable covered call and cash-secured put opportunities")

# Define constants
DB_PATH = 'options_data.db'

# ========== DATABASE FUNCTIONS ==========

# Setup database
def setup_database():
    """Create SQLite database and tables if they don't exist"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS options_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            price REAL,
            exp_date TEXT,
            strike REAL,
            option_type TEXT,
            bid REAL,
            ask REAL,
            volume INTEGER,
            open_interest INTEGER,
            implied_volatility REAL,
            delta REAL,
            timestamp DATETIME
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_metadata (
            id INTEGER PRIMARY KEY,
            last_updated DATETIME,
            source TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database setup error: {str(e)}")
        return False

# Load data from database
def load_from_database(option_type):
    """Load options data from SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get metadata
        metadata_df = pd.read_sql("SELECT last_updated FROM data_metadata WHERE source = ?", 
                                conn, params=(option_type,))
        
        if metadata_df.empty:
            conn.close()
            return None, None
        
        last_updated = metadata_df['last_updated'].iloc[0]
        
        # Get options data
        data_df = pd.read_sql("SELECT * FROM options_data WHERE option_type = ?", 
                             conn, params=(option_type,))
        
        conn.close()
        
        if data_df.empty:
            return None, last_updated
        
        # Process data to match our expected format
        data_df = data_df.rename(columns={
            'open_interest': 'open_int',
            'implied_volatility': 'iv_pct'
        })
        
        # Calculate additional metrics based on strategy type
        if option_type == "covered_call":
            # Calculate covered call metrics
            data_df['moneyness'] = (data_df['strike'] - data_df['price']) / data_df['price'] * 100
            data_df['net_profit'] = (data_df['bid'] * 100) - ((100 * data_df['price']) - (100 * data_df['strike']))
            data_df['be_bid'] = data_df['price'] - data_df['bid']
            data_df['be_pct'] = (data_df['be_bid'] - data_df['price']) / data_df['price'] * 100
            data_df['otm_prob'] = (1 - data_df['delta']) * 100
        else:  # cash_secured_put
            # Calculate cash-secured put metrics
            data_df['moneyness'] = (data_df['strike'] - data_df['price']) / data_df['price'] * 100
            data_df['net_profit'] = (data_df['bid'] * 100) - ((100 * data_df['strike']) - (100 * data_df['price']))
            data_df['be_bid'] = data_df['strike'] - data_df['bid']
            data_df['be_pct'] = (data_df['be_bid'] - data_df['price']) / data_df['price'] * 100
            data_df['otm_prob'] = data_df['delta'] * 100
        
        # Calculate returns
        # Add days to expiry calculation
        data_df['days_to_expiry'] = data_df['exp_date'].apply(
            lambda x: (datetime.strptime(x, "%m/%d/%y") - datetime.now()).days
        )
        data_df['days_to_expiry'] = data_df['days_to_expiry'].apply(lambda x: max(1, x))  # Ensure at least 1 day
        
        data_df['pnl_rtn'] = (data_df['bid'] / data_df['price']) * 100
        data_df['ann_rtn'] = data_df['pnl_rtn'] * (365 / data_df['days_to_expiry'])
        
        # Ensure all columns exist
        if 'last_trade' not in data_df.columns:
            data_df['last_trade'] = "N/A"
        
        # Try to parse last_updated as datetime if it's a string
        try:
            if isinstance(last_updated, str):
                last_updated = datetime.strptime(last_updated, '%Y-%m-%d %H:%M:%S.%f')
        except Exception:
            # If parsing fails, use current time
            last_updated = datetime.now()
        
        return data_df, last_updated
    except Exception as e:
        st.error(f"Error loading from database: {str(e)}")
        return None, datetime.now()

# Check if database needs initialization
def check_database_status():
    """Check if database has data and when it was last updated"""
    try:
        if not os.path.exists(DB_PATH):
            return None, "No database found"
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='data_metadata'")
        if cursor.fetchone()[0] == 0:
            conn.close()
            return None, "Database tables not initialized"
        
        cursor.execute("SELECT last_updated, source FROM data_metadata ORDER BY last_updated DESC LIMIT 1")
        result = cursor.fetchone()
        
        # Check if we have any data
        cursor.execute("SELECT COUNT(*) FROM options_data")
        data_count = cursor.fetchone()[0]
        
        conn.close()
        
        if not result:
            return None, "No metadata found"
        
        if data_count == 0:
            return None, "Database exists but contains no data"
            
        # Try to parse as datetime
        try:
            if isinstance(result[0], str):
                timestamp = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S.%f')
            else:
                timestamp = result[0]
            
            data_source = result[1] if len(result) > 1 else "Unknown"
            return timestamp, f"Database contains {data_count} records, last updated from {data_source}"
        except Exception:
            return datetime.now(), f"Database contains {data_count} records"
    except Exception as e:
        return None, f"Database error: {str(e)}"

# Function to run the scheduled fetch script
def run_refresh_script():
    """Run the scheduled_fetch.py script to refresh data"""
    try:
        result = subprocess.run(["python", "scheduled_fetch.py"], 
                               capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            st.session_state.api_status = "Success"
            return True, "Data refresh completed successfully"
        else:
            st.session_state.api_status = "Error"
            return False, f"Error: {result.stderr}"
    except Exception as e:
        st.session_state.api_status = "Error"
        return False, f"Error running refresh script: {str(e)}"

# Function to generate mock data if needed (fallback only)
def generate_mock_option_data(strategy_type="covered_call", num_rows=20):
    """Generate mock option data for testing - only used if database access fails"""
    np.random.seed(42)  # For consistent results
    
    symbols = ["SMCI", "NVDA", "AAPL", "MSFT", "AMD", "GOOGL", "META", "TSLA"]
    prices = {
        "SMCI": 46.07,
        "NVDA": 842.32,
        "AAPL": 184.25,
        "MSFT": 417.75, 
        "AMD": 174.49,
        "GOOGL": 178.35,
        "META": 515.28,
        "TSLA": 176.75
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
            bid_price = max(0.01, bid_price)  # Ensure positive
            ask_price = round(bid_price * 1.1, 2)
            
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
                "ask": ask_price,
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
                "last_trade": f"{np.random.randint(1, 15)}:{np.random.randint(10, 59)} ET",
                "days_to_expiry": days_to_expiry
            })
        else:  # Cash-Secured Puts
            strike = round(stock_price * np.random.uniform(0.7, 0.95), 2)
            moneyness = (strike - stock_price) / stock_price * 100
            
            # Option price calculation
            bid_price = round(stock_price * 0.03 * (1 + abs(moneyness)/10), 2)
            bid_price = max(0.01, bid_price)  # Ensure positive
            ask_price = round(bid_price * 1.1, 2)
            
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
                "ask": ask_price,
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
                "last_trade": f"{np.random.randint(1, 15)}:{np.random.randint(10, 59)} ET",
                "days_to_expiry": days_to_expiry
            })
    
    return pd.DataFrame(data)

# ========== APP INITIALIZATION ==========

# Setup database on app start
setup_result = setup_database()

# Get database status
last_update, status_message = check_database_status()
st.session_state.db_last_updated = last_update
st.session_state.db_status_message = status_message

# Sidebar
with st.sidebar:
    st.header("Options Scanner")
    
    # API status and database info
    if st.session_state.db_last_updated:
        last_update_str = st.session_state.db_last_updated.strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"📊 Using local database (Last updated: {last_update_str})")
        st.caption(st.session_state.db_status_message)
    else:
        st.warning("⚠️ No data in local database. Click 'Refresh Data' to fetch from API.")
        st.caption(st.session_state.db_status_message)
    
    # Refresh button
    if st.button("🔄 Refresh Data from Schwab API", use_container_width=True):
        with st.spinner("Fetching data from Schwab API..."):
            success, message = run_refresh_script()
            if success:
                st.success("✅ Data refresh completed successfully")
                # Recheck database status
                last_update, status_message = check_database_status()
                st.session_state.db_last_updated = last_update
                st.session_state.db_status_message = status_message
                # Force rerun to update the UI
                st.experimental_rerun()
            else:
                st.error(f"❌ {message}")
    
    # API status indicator
    api_status = st.session_state.api_status
    if api_status == "Success":
        st.success("✓ Schwab API Connection: Active")
    elif api_status == "Error":
        st.error("✗ Schwab API Connection: Error")
    else:
        st.info("❓ Schwab API Connection: Not tested")
    
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
        
        # Optional: Add filters for IV, delta, etc.
        max_iv = st.slider(
            "Max Implied Volatility (%)",
            min_value=0,
            max_value=1000,
            value=1000,
            step=50
        )
    
    # Scan button
    scan_button = st.button("Scan for Opportunities", type="primary", use_container_width=True)

# Trigger scan when button is clicked
if scan_button:
    with st.spinner("Running scan..."):
        strategy_type = "covered_call" if strategy == "Covered Calls" else "cash_secured_put"
        
        # First check if we have data in the database
        results, last_updated = load_from_database(strategy_type)
        
        # If no data in database, prompt user to refresh
        if results is None or results.empty:
            st.warning("No data available in database. Please click 'Refresh Data from Schwab API' in the sidebar.")
            st.session_state.scan_results = None
        else:
            # Apply price filters
            results = results[
                (results['price'] >= min_stock_price) & 
                (results['price'] <= max_stock_price)
            ]
            
            # Apply days to expiry filters
            results = results[
                (results['days_to_expiry'] >= min_days) & 
                (results['days_to_expiry'] <= max_days)
            ]
            
            # Apply return filters
            results = results[results['ann_rtn'] >= min_annual_return]
            
            # Apply IV filter if set
            if 'iv_pct' in results.columns:
                results = results[results['iv_pct'] <= max_iv]
            
            # Save results to session state
            st.session_state.scan_results = {
                "type": strategy_type,
                "data": results,
                "timestamp": datetime.now(),
                "data_as_of": last_updated if last_updated else datetime.now()
            }
            
            # Reset sorting when new scan is performed
            st.session_state.sort_column = None
            st.session_state.sort_ascending = True

# Display scan results
if 'scan_results' in st.session_state and st.session_state.scan_results is not None:
    results = st.session_state.scan_results.get('data', pd.DataFrame())
    result_type = st.session_state.scan_results.get('type', '')
    timestamp = st.session_state.scan_results.get('timestamp', datetime.now())
    data_as_of = st.session_state.scan_results.get('data_as_of')
    
    # Show results summary
    if not results.empty:
        st.success(f"Found {len(results)} opportunities (Scan time: {timestamp.strftime('%I:%M:%S %p')})")
        
        # Handle the data_as_of display with proper error checking
        if data_as_of:
            try:
                # Try to format the timestamp
                data_as_of_str = data_as_of.strftime('%Y-%m-%d %I:%M:%S %p')
                st.info(f"Data as of: {data_as_of_str}")
            except Exception:
                # If formatting fails, display generic message
                st.info("Using latest available data")
        else:
            st.info("Using freshly generated data")
        
        # Common columns for both strategies (Barchart-style)
        display_columns = [
            'symbol', 'price', 'exp_date', 'strike', 'moneyness', 'bid', 
            'net_profit', 'be_bid', 'be_pct', 'volume', 'open_int', 
            'iv_pct', 'delta', 'otm_prob', 'pnl_rtn', 'ann_rtn', 'last_trade'
        ]
        
        # Check if all columns exist in the results DataFrame
        for col in display_columns:
            if col not in results.columns:
                if col == 'last_trade':
                    results[col] = "N/A"
                else:
                    # For other missing columns, use a default value
                    results[col] = 0
        
        # Column display names mapping
        column_display_names = {
            'symbol': 'Symbol',
            'price': 'Price',
            'exp_date': 'Exp Date',
            'strike': 'Strike',
            'moneyness': 'Moneyness',
            'bid': 'Bid',
            'net_profit': 'Net Profit',
            'be_bid': 'BE (Bid)',
            'be_pct': 'BE%',
            'volume': 'Volume',
            'open_int': 'Open Int',
            'iv_pct': 'IV',
            'delta': 'Delta',
            'otm_prob': 'OTM Prob',
            'pnl_rtn': 'Ptnl Rtn',
            'ann_rtn': 'Ann Rtn',
            'last_trade': 'Last Trade'
        }
        
        # Select only the columns we want to display
        display_data = results[display_columns].copy()
        
        # Create sortable column headers
        st.subheader(f"{'Covered Call' if result_type == 'covered_call' else 'Cash-Secured Put'} Opportunities")
        
        # Add sorting controls as buttons in a horizontal layout
        cols = st.columns(len(display_columns))
        
        # Function to handle sorting when a column header is clicked
        def sort_column(col_name):
            if st.session_state.sort_column == col_name:
                # Toggle the sort direction if the same column is clicked again
                st.session_state.sort_ascending = not st.session_state.sort_ascending
            else:
                # Set the new sort column and default to ascending
                st.session_state.sort_column = col_name
                st.session_state.sort_ascending = True
            
            # Force rerun to apply sorting
            st.experimental_rerun()
        
        # Display sortable column headers
        for i, col in enumerate(display_columns):
            display_name = column_display_names[col]
            # Add a sort indicator if this column is being sorted
            if st.session_state.sort_column == col:
                sort_indicator = "▼" if st.session_state.sort_ascending else "▲"
                display_name = f"{display_name} {sort_indicator}"
            
            cols[i].button(
                display_name, 
                key=f"sort_{col}",
                on_click=sort_column,
                args=(col,),
                use_container_width=True
            )
        
        # Apply sorting if a sort column is selected
        if st.session_state.sort_column and st.session_state.sort_column in display_data.columns:
            sorted_data = display_data.sort_values(
                by=st.session_state.sort_column,
                ascending=st.session_state.sort_ascending
            )
        else:
            # Default sort by annual return (descending)
            sorted_data = display_data.sort_values(by='ann_rtn', ascending=False)
        
        # Create a new dataframe for display with formatted values
        formatted_df = pd.DataFrame()
        
        # Copy and format each column
        formatted_df['Symbol'] = sorted_data['symbol']
        formatted_df['Price'] = sorted_data['price'].apply(lambda x: f"${x:.2f}")
        formatted_df['Exp Date'] = sorted_data['exp_date']
        formatted_df['Strike'] = sorted_data['strike'].apply(lambda x: f"${x:.2f}")
        
        # Create colored moneyness column
        def format_moneyness(val):
            color = 'red' if val < 0 else 'green'
            return f'<span style="color:{color}">{val:.2f}%</span>'
        
        formatted_df['Moneyness'] = sorted_data['moneyness'].apply(format_moneyness)
        
        # Continue formatting other columns
        formatted_df['Bid'] = sorted_data['bid'].apply(lambda x: f"${x:.2f}")
        
        # Create colored net profit column
        def format_net_profit(val):
            color = 'red' if val < 0 else 'green'
            return f'<span style="color:{color}">${val:.2f}</span>'
        
        formatted_df['Net Profit'] = sorted_data['net_profit'].apply(format_net_profit)
        
        # Continue with other columns
        formatted_df['BE (Bid)'] = sorted_data['be_bid'].apply(lambda x: f"${x:.2f}")
        formatted_df['BE%'] = sorted_data['be_pct'].apply(lambda x: f"{x:.2f}%")
        formatted_df['Volume'] = sorted_data['volume'].apply(lambda x: f"{x:,}")
        formatted_df['Open Int'] = sorted_data['open_int'].apply(lambda x: f"{x:,}")
        formatted_df['IV'] = sorted_data['iv_pct'].apply(lambda x: f"{x:.2f}%")
        formatted_df['Delta'] = sorted_data['delta'].apply(lambda x: f"{x:.4f}")
        formatted_df['OTM Prob'] = sorted_data['otm_prob'].apply(lambda x: f"{x:.2f}%")
        formatted_df['Ptnl Rtn'] = sorted_data['pnl_rtn'].apply(lambda x: f"{x:.1f}%")
        formatted_df['Ann Rtn'] = sorted_data['ann_rtn'].apply(lambda x: f"{x:.1f}%")
        formatted_df['Last Trade'] = sorted_data['last_trade']
        
        # Use HTML to render the formatted table with colors
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Offer download button for the data
        try:
            csv = sorted_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name=f"options_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error creating download button: {str(e)}")
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
