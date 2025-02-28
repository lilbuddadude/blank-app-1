#!/usr/bin/env python3
"""
Options Scanner - Daily Data Fetch Script with Schwab API Integration
--------------------------------------------------------------------
This script fetches options data from Schwab API and saves it to a local SQLite database.
It should be scheduled to run once per day at market open (9:30 AM ET).

Setup instructions:
1. Schedule this script using cron (Linux/Mac) or Task Scheduler (Windows)
2. Ensure all dependencies are installed: pip install requests pandas numpy

Example cron job (runs at 9:31 AM ET weekdays):
31 9 * * 1-5 /path/to/scheduled_fetch.py

"""

import requests
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("options_fetch.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
DB_PATH = 'options_data.db'
LOG_PATH = 'options_fetch.log'

# Schwab API Credentials
API_KEY = "Vtbsc861GI48iT3JgAr8bp5Hvy5cVe7O"
API_SECRET = "SvMJwXrepRDQBiXr"

SCHWAB_AUTH_URL = "https://api.schwabapi.com/v1/oauth/token"
SCHWAB_AUTH_AUTHORIZE_URL = "https://api.schwabapi.com/v1/oauth/authorize"

# Set to True to use mock data instead of API data
USE_MOCK_DATA = False

# Watchlist of symbols to track
SYMBOLS = ["SMCI", "NVDA", "AAPL", "MSFT", "AMD", "GOOGL", "META", "TSLA"]

# Setup database
def setup_database():
    """Create SQLite database and tables if they don't exist"""
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
    logging.info("Database setup complete")

# Save data to database
def save_to_database(data, option_type):
    """Save options data to SQLite database"""
    if data.empty:
        logging.warning(f"No {option_type} data to save to database")
        return datetime.now()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear old data of this type
    cursor.execute("DELETE FROM options_data WHERE option_type = ?", (option_type,))
    
    # Insert new data
    timestamp = datetime.now()
    records_inserted = 0
    
    for _, row in data.iterrows():
        cursor.execute('''
        INSERT INTO options_data (
            symbol, price, exp_date, strike, option_type, 
            bid, ask, volume, open_interest, implied_volatility, 
            delta, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['symbol'], row['price'], row['exp_date'], row['strike'], option_type,
            row.get('bid', 0), row.get('ask', 0), row.get('volume', 0), 
            row.get('open_int', 0), row.get('iv_pct', 0), row.get('delta', 0), 
            timestamp
        ))
        records_inserted += 1
    
    # Update metadata
    cursor.execute("DELETE FROM data_metadata WHERE source = ?", (option_type,))
    cursor.execute("INSERT INTO data_metadata (last_updated, source) VALUES (?, ?)", 
                  (timestamp, option_type))
    
    conn.commit()
    conn.close()
    
    logging.info(f"Saved {records_inserted} {option_type} records to database")
    return timestamp

# Get authentication token from Schwab API
def get_schwab_auth_token():
    """Get authentication token using schwab_auth.py"""
    try:
        # Import the auth module
        import schwab_auth
        
        # Get the access token
        access_token = schwab_auth.get_tokens()
        
        if access_token:
            logging.info("Successfully obtained Schwab API access token")
            return access_token
        else:
            logging.error("Failed to get access token")
            return None
    except Exception as e:
        logging.error(f"Exception in auth token request: {str(e)}")
        return None
        
        if refresh_token:
            # Use refresh token flow
            token_data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": API_KEY,
                "client_secret": API_SECRET
            }
        else:
            # For first-time setup, you need to manually get an authorization code
            # This requires user interaction and can't be fully automated
            auth_code = input("Enter the authorization code from Schwab: ")
            
            token_data = {
                "grant_type": "authorization_code",
                "code": auth_code,
                "client_id": API_KEY,
                "client_secret": API_SECRET,
                "redirect_uri": "https://developer.schwab.com/oauth2-redirect.html"  # Must match your app's registered redirect URI
            }
        
        response = requests.post(SCHWAB_AUTH_URL, data=token_data)
        
        if response.status_code == 200:
            token_response = response.json()
            access_token = token_response.get("access_token")
            
            # Save the refresh token for future use
            if "refresh_token" in token_response:
                with open('refresh_token.txt', 'w') as f:
                    f.write(token_response["refresh_token"])
            
            logging.info("Successfully obtained Schwab API access token")
            return access_token
        else:
            logging.error(f"Auth error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Exception in auth token request: {str(e)}")
        return None

# Convert expiration date from Schwab format to app format
def format_expiration_date(exp_date_str):
    """Convert expiration date from API format to MM/DD/YY format"""
    try:
        # Schwab may use ISO format (YYYY-MM-DD)
        exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
        return exp_date.strftime("%m/%d/%y")
    except Exception:
        try:
            # Try alternative format
            exp_date = datetime.strptime(exp_date_str, "%Y%m%d")
            return exp_date.strftime("%m/%d/%y")
        except Exception:
            # Return as is if parsing fails
            return exp_date_str

# Fetch options data from Schwab API
def fetch_from_schwab_api(symbols, option_type="covered_call"):
    """
    Fetch options data from Schwab API
    
    Args:
        symbols: List of stock symbols to fetch options for
        option_type: Either "covered_call" or "cash_secured_put"
        
    Returns:
        DataFrame with options data
    """
    if USE_MOCK_DATA:
        logging.info("Using mock data (Schwab API integration disabled)")
        return generate_mock_option_data(option_type=option_type, num_rows=50)
    
    logging.info(f"Fetching {option_type} data from Schwab API for {len(symbols)} symbols")
    
    # Get authentication token
    access_token = get_schwab_auth_token()
    if not access_token:
        logging.error("Failed to get authentication token, using mock data as fallback")
        return generate_mock_option_data(option_type=option_type, num_rows=50)
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    all_options_data = []
    option_type_filter = "call" if option_type == "covered_call" else "put"
    
    # Loop through watchlist symbols
    for symbol in symbols:
        try:
            # Set parameters for options chain request
            params = {
                "symbol": symbol,
                "optionType": option_type_filter,
                # Add other relevant parameters based on Schwab's API documentation
                # For example: expiration range, strike range, etc.
            }
            
            response = requests.get(SCHWAB_OPTIONS_URL, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get underlying stock price
                underlying_price = 0
                if "underlying" in data:
                    underlying_price = float(data["underlying"].get("price", 0))
                
                # Process options data based on Schwab's response structure
                # NOTE: Adjust this parsing logic based on Schwab's actual API response structure
                if "options" in data:
                    for option in data["options"]:
                        # Extract expiration date and convert to our format
                        exp_date = format_expiration_date(option.get("expirationDate", ""))
                        strike = float(option.get("strikePrice", 0))
                        
                        # Calculate additional metrics
                        # NOTE: Some of these might be directly available from Schwab API
                        bid = float(option.get("bid", 0))
                        ask = float(option.get("ask", 0))
                        iv = float(option.get("impliedVolatility", 0)) * 100  # Convert to percentage
                        delta = float(option.get("delta", 0))
                        volume = int(option.get("volume", 0))
                        open_interest = int(option.get("openInterest", 0))
                        
                        # Format the option data
                        option_data = {
                            "symbol": symbol,
                            "price": underlying_price,
                            "exp_date": exp_date,
                            "strike": strike,
                            "bid": bid,
                            "ask": ask,
                            "volume": volume,
                            "open_int": open_interest,
                            "iv_pct": iv,
                            "delta": delta,
                        }
                        
                        all_options_data.append(option_data)
            else:
                logging.error(f"API error for {symbol}: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"Exception fetching data for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())
    
    # Convert to DataFrame
    if not all_options_data:
        logging.warning("No options data retrieved from API, using mock data as fallback")
        return generate_mock_option_data(option_type=option_type, num_rows=50)
    
    df = pd.DataFrame(all_options_data)
    
    # Calculate strategy-specific metrics
    if option_type == "covered_call":
        # Calculate covered call metrics
        df['moneyness'] = (df['strike'] - df['price']) / df['price'] * 100
        df['net_profit'] = (df['bid'] * 100) - ((100 * df['price']) - (100 * df['strike']))
        df['be_bid'] = df['price'] - df['bid']
        df['be_pct'] = (df['be_bid'] - df['price']) / df['price'] * 100
        df['otm_prob'] = (1 - df['delta']) * 100
    else:  # cash_secured_put
        # Calculate cash-secured put metrics
        df['moneyness'] = (df['strike'] - df['price']) / df['price'] * 100
        df['net_profit'] = (df['bid'] * 100) - ((100 * df['strike']) - (100 * df['price']))
        df['be_bid'] = df['strike'] - df['bid']
        df['be_pct'] = (df['be_bid'] - df['price']) / df['price'] * 100
        df['otm_prob'] = df['delta'] * 100
    
    # Calculate days to expiry
    df['days_to_expiry'] = df['exp_date'].apply(
        lambda x: (datetime.strptime(x, "%m/%d/%y") - datetime.now()).days
    )
    df['days_to_expiry'] = df['days_to_expiry'].apply(lambda x: max(1, x))  # Ensure at least 1 day
    
    # Calculate returns
    df['pnl_rtn'] = (df['bid'] / df['price']) * 100
    df['ann_rtn'] = df['pnl_rtn'] * (365 / df['days_to_expiry'])
    
    logging.info(f"Retrieved {len(df)} {option_type} options from Schwab API")
    return df

# Generate mock option data for testing
def generate_mock_option_data(strategy_type="covered_call", num_rows=20):
    """Generate mock option data for testing"""
    np.random.seed(int(time.time()))  # Use current time for some variety
    
    symbols = SYMBOLS
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
    
    for symbol in symbols:
        stock_price = prices.get(symbol, 100.0)
        
        # Generate multiple expiration dates
        for exp_days in [14, 30, 45, 60, 90]:
            expiry_date = (datetime.now() + timedelta(days=exp_days)).strftime("%m/%d/%y")
            
            # Generate multiple strikes around the current price
            if strategy_type == "covered_call":
                # For covered calls, generate multiple strikes
                strike_ranges = np.linspace(0.80, 0.99, 5)
                for strike_pct in strike_ranges:
                    strike = round(stock_price * strike_pct, 2)
                    
                    # Option price calculation
                    moneyness = (strike - stock_price) / stock_price * 100
                    bid_price = round((stock_price - strike) + (stock_price * 0.03), 2)
                    bid_price = max(0.01, bid_price)  # Ensure positive prices
                    ask_price = round(bid_price * 1.1, 2)
                    
                    # Additional metrics
                    volume = np.random.randint(500, 5000)
                    open_interest = np.random.randint(1000, 20000)
                    iv_pct = np.random.uniform(200, 400)
                    delta = 1 - (strike / stock_price) * 1.1
                    delta = max(0.1, min(0.99, delta))
                    
                    data.append({
                        "symbol": symbol,
                        "price": stock_price,
                        "exp_date": expiry_date,
                        "strike": strike,
                        "moneyness": moneyness,
                        "bid": bid_price,
                        "ask": ask_price,
                        "volume": volume,
                        "open_int": open_interest,
                        "iv_pct": iv_pct,
                        "delta": delta,
                    })
            else:  # Cash-Secured Puts
                # For cash-secured puts, generate multiple strikes
                strike_ranges = np.linspace(0.70, 0.95, 5)
                for strike_pct in strike_ranges:
                    strike = round(stock_price * strike_pct, 2)
                    
                    # Option price calculation
                    moneyness = (strike - stock_price) / stock_price * 100
                    bid_price = round(stock_price * 0.03 * (1 + abs(moneyness)/10), 2)
                    bid_price = max(0.01, bid_price)  # Ensure positive prices
                    ask_price = round(bid_price * 1.1, 2)
                    
                    # Additional metrics
                    volume = np.random.randint(500, 5000)
                    open_interest = np.random.randint(1000, 20000)
                    iv_pct = np.random.uniform(200, 400)
                    delta = (strike / stock_price) * 0.8
                    delta = max(0.1, min(0.99, delta))
                    
                    data.append({
                        "symbol": symbol,
                        "price": stock_price,
                        "exp_date": expiry_date,
                        "strike": strike,
                        "moneyness": moneyness,
                        "bid": bid_price,
                        "ask": ask_price,
                        "volume": volume,
                        "open_int": open_interest,
                        "iv_pct": iv_pct,
                        "delta": delta,
                    })
    
    return pd.DataFrame(data)

# Main function to run the data fetch
def main():
    try:
        logging.info("Starting Options Data Fetch")
        
        # Setup database
        setup_database()
        
        # Fetch covered call data
        logging.info("Fetching covered call data...")
        covered_call_data = fetch_from_schwab_api(SYMBOLS, "covered_call")
        cc_timestamp = save_to_database(covered_call_data, "covered_call")
        
        # Fetch cash-secured put data
        logging.info("Fetching cash-secured put data...")
        cash_secured_put_data = fetch_from_schwab_api(SYMBOLS, "cash_secured_put")
        csp_timestamp = save_to_database(cash_secured_put_data, "cash_secured_put")
        
        logging.info(f"Data fetch completed successfully at {datetime.now()}")
        logging.info(f"Covered Calls: {len(covered_call_data)} records")
        logging.info(f"Cash-Secured Puts: {len(cash_secured_put_data)} records")
        
        return 0
    except Exception as e:
        logging.error(f"Error in data fetch: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
