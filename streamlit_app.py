#!/usr/bin/env python3
"""
Options Scanner - Daily Data Fetch Script
-----------------------------------------
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
API_KEY = "Vtbsc861GI4...Ve7O"  # Replace with your actual API key
API_SECRET = "SvMJwXre...BiXr"  # Replace with your actual API secret

# Watchlist of symbols to track
SYMBOLS = ["SMCI", "NVDL", "NVDX", "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "TSLA"]

# Flag to use mock data (set to False when connecting to real API)
USE_MOCK_DATA = True

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
            row['bid'], row.get('ask', 0), row['volume'], row['open_int'], row['iv_pct'],
            row['delta'], timestamp
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

# Fetch options data from Schwab API
def fetch_from_schwab_api(symbols=None, option_type="covered_call"):
    """
    Fetch options data from Schwab API
    
    In a real implementation, this would make API calls to Schwab's options API.
    For now, we're using mock data.
    """
    if USE_MOCK_DATA:
        logging.info("Using mock data (Schwab API integration pending)")
        return generate_mock_option_data(option_type=option_type, num_rows=50)
    
    # This would be the real API implementation
    # Example code for when you have the real API:
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    all_options_data = []
    
    for symbol in symbols:
        try:
            response = requests.get(
                f"https://api.schwab.com/options/chains?symbol={symbol}",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                # Process the data and append to all_options_data
                # ...
            else:
                logging.error(f"API error for {symbol}: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Exception fetching data for {symbol}: {str(e)}")
    
    return pd.DataFrame(all_options_data)
    """
    
    logging.warning("Real API integration not implemented yet")
    return generate_mock_option_data(option_type=option_type, num_rows=50)

# Generate mock option data for testing
def generate_mock_option_data(strategy_type="covered_call", num_rows=20):
    """Generate mock option data for testing"""
    np.random.seed(int(time.time()))  # Use current time for some variety
    
    symbols = SYMBOLS
    prices = {
        "SMCI": 46.07,
        "NVDL": 54.05,
        "NVDX": 11.33,
        "AAPL": 184.25,
        "MSFT": 417.75,
        "NVDA": 842.32,
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
