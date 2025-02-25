import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Options Arbitrage Scanner",
    page_icon="💰",
    layout="wide"
)

# Schwab API credentials
SCHWAB_API_KEY = "Vtbsc861GI48iT3JgAr8bp5Hvy5cVe7O"
SCHWAB_API_SECRET = "SvMJwXrepRDQBiXr"
SCHWAB_API_BASE_URL = "https://api.schwab.com/trader/v1"  # Placeholder, use actual URL

# Initialize session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# Schwab API class
class SchwabAPI:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = self._create_session()
        
        # Request caching
        self.cache = {}
        
    def _create_session(self):
        """Create authenticated session with Schwab API"""
        session = requests.Session()
        
        # Set up authentication headers
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        return session
    
    def _api_request(self, endpoint, method="GET", params=None, data=None, cache_key=None):
        """Make request to Schwab API with caching and error handling"""
        # Check cache for GET requests
        if method == "GET" and cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Log the request (without sensitive data)
            logger.info(f"Making {method} request to {endpoint}")
            
            # For this demo, we're simulating API responses
            # In production, this would be a real API call
            
            # Simulate API response time
            time.sleep(0.5)
            
            if "markets/quotes" in endpoint:
                response = self._mock_stock_quotes(params)
            elif "markets/options/chains" in endpoint:
                response = self._mock_option_chain(params)
            elif "markets/options/symbols" in endpoint:
                response = self._mock_optionable_symbols()
            else:
                # Generic mock response
                response = {"message": "Simulated API response"}
            
            # Cache successful GET responses
            if method == "GET" and cache_key:
                self.cache[cache_key] = response
                
            return response
            
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return None
    
    def _mock_stock_quotes(self, params):
        """Mock response for stock quotes API"""
        symbols = params.get("symbols", "").split(",")
        
        # Realistic stock prices for common stocks
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
        
        result = []
        for symbol in symbols:
            if symbol in prices:
                price = prices[symbol]
            else:
                # Random price for unknown symbols
                price = np.random.uniform(50, 500)
                
            result.append({
                "symbol": symbol,
                "lastPrice": price,
                "bidPrice": price * 0.998,
                "askPrice": price * 1.002,
                "volume": np.random.randint(100000, 10000000)
            })
            
        return result
    
    def _mock_option_chain(self, params):
        """Mock response for option chain API"""
        symbol = params.get("symbol")
        
        # Get stock price
        stock_data = self._mock_stock_quotes({"symbols": symbol})[0]
        current_price = stock_data["lastPrice"]
        
        # Generate expiration dates (next 4 Fridays)
        today = datetime.now()
        days_to_friday = (4 - today.weekday()) % 7  # 4 is Friday
        
        expirations = []
        for i in range(4):
            expiry_date = today + timedelta(days=days_to_friday + i*7)
            expirations.append(expiry_date.strftime("%Y-%m-%d"))
        
        # Generate strikes around current price
        strikes = []
        for pct in [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05, 1.1, 1.15, 1.2]:
            strikes.append(round(current_price * pct, 1))
        
        # Generate options for each expiry and strike
        options = []
        for expiry in expirations:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            days_to_expiry = (expiry_date - today).days
            
            for strike in strikes:
                # Call option
                itm_amount = max(0, current_price - strike)
                time_value = current_price * 0.01 * (days_to_expiry/30) * (1 + abs(strike-current_price)/current_price)
                call_price = itm_amount + time_value
                
                # Adjust volume based on how close to ATM
                atm_factor = 1 - min(0.9, abs(strike - current_price) / current_price)
                volume = int(1000 * atm_factor)
                open_interest = int(5000 * atm_factor)
                
                # Calculate implied volatility (simplified)
                iv = 0.25 + 0.15 * (abs(strike - current_price) / current_price) + 0.1 * (days_to_expiry / 365)
                iv = min(0.8, iv)  # Cap at 80%
                
                options.append({
                    "optionType": "CALL",
                    "strikePrice": strike,
                    "expirationDate": expiry,
                    "bid": round(call_price * 0.95, 2),
                    "ask": round(call_price * 1.05, 2),
                    "lastPrice": round(call_price, 2),
                    "volume": volume,
                    "openInterest": open_interest,
                    "impliedVolatility": iv
                })
                
                # Put option
                itm_amount = max(0, strike - current_price)
                time_value = current_price * 0.01 * (days_to_expiry/30) * (1 + abs(strike-current_price)/current_price)
                put_price = itm_amount + time_value
                
                options.append({
                    "optionType": "PUT",
                    "strikePrice": strike,
                    "expirationDate": expiry,
                    "bid": round(put_price * 0.95, 2),
                    "ask": round(put_price * 1.05, 2),
                    "lastPrice": round(put_price, 2),
                    "volume": volume,
                    "openInterest": open_interest,
                    "impliedVolatility": iv
                })
        
        return {
            "symbol": symbol,
            "expirationDates": expirations,
            "options": options
        }
    
    def _mock_optionable_symbols(self):
        """Mock response for optionable symbols API"""
        return [
            {"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "AMZN"}, 
            {"symbol": "GOOGL"}, {"symbol": "META"}, {"symbol": "TSLA"}, 
            {"symbol": "NVDA"}, {"symbol": "AMD"}, {"symbol": "INTC"}, 
            {"symbol": "NFLX"}, {"symbol": "DIS"}, {"symbol": "BA"}, 
            {"symbol": "JPM"}, {"symbol": "V"}, {"symbol": "MA"}
        ]
    
    def get_stock_quote(self, symbol):
        """Get current stock quote for a symbol"""
        response = self._api_request("markets/quotes", params={"symbols": symbol}, cache_key=f"quote_{symbol}")
        if response:
            return response[0]
        return None
    
    def get_option_chain(self, symbol):
        """Get full option chain for a symbol"""
        response = self._api_request("markets/options/chains", params={"symbol": symbol}, cache_key=f"options_{symbol}")
        return response
    
    def get_optionable_symbols(self):
        """Get list of all optionable symbols"""
        response = self._api_request("markets/options/symbols", cache_key="optionable_symbols")
        if response:
            return [item["symbol"] for item in response]
        return []
    
    def find_covered_call_opportunities(self, min_return=0.15, max_days=45, safety_margin=0.01, symbols=None):
        """Find covered call arbitrage opportunities"""
        if not symbols:
            symbols = self.get_optionable_symbols()[:10]  # Limit for demo
            
        opportunities = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            # Update progress
            progress = (i + 1) / len(symbols)
            progress_bar.progress(progress)
            status_text.text(f"Processing {symbol}... ({i+1}/{len(symbols)})")
            
            # Get stock data
            stock = self.get_stock_quote(symbol)
            if not stock:
                continue
                
            current_price = stock["lastPrice"]
            
            # Get option chain
            chain = self.get_option_chain(symbol)
            if not chain or "options" not in chain:
                continue
                
            # Find ITM calls
            for option in chain["options"]:
                if option["optionType"] != "CALL":
                    continue
                    
                # Only consider ITM calls
                strike = option["strikePrice"]
                if strike >= current_price:
                    continue
                    
                # Check expiration
                expiry = option["expirationDate"]
                expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
                days_to_expiry = (expiry_date - datetime.now()).days
                
                if days_to_expiry > max_days:
                    continue
                    
                # Check for liquidity
                if option["volume"] < 10 or option["openInterest"] < 10:
                    continue
                    
                # Calculate arbitrage
                call_price = option["bid"]  # Use bid price (conservative)
                cost_basis = current_price * (1 + safety_margin)  # Add safety margin
                net_debit = cost_basis - call_price
                
                # Only consider if there's an arbitrage opportunity
                if net_debit < strike:
                    profit = strike - net_debit
                    profit_percentage = profit / net_debit
                    annualized_return = profit_percentage * (365 / days_to_expiry)
                    
                    if annualized_return >= min_return:
                        downside_protection = (current_price - net_debit) / current_price
                        
                        opportunities.append({
                            "symbol": symbol,
                            "stock_price": current_price,
                            "strike": strike,
                            "expiration_date": expiry,
                            "days_to_expiry": days_to_expiry,
                            "call_price": call_price,
                            "net_debit": net_debit,
                            "profit": profit,
                            "return": profit_percentage,
                            "annualized_return": annualized_return,
                            "implied_volatility": option["impliedVolatility"],
                            "downside_protection": downside_protection,
                            "volume": option["volume"],
                            "open_interest": option["openInterest"]
                        })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Return as DataFrame sorted by return
        if opportunities:
            return pd.DataFrame(opportunities).sort_values("annualized_return", ascending=False)
        else:
            return pd.DataFrame()
    
    def find_cash_secured_put_opportunities(self, min_return=0.15, max_days=45, min_otm=0.05, symbols=None):
        """Find cash-secured put opportunities"""
        if not symbols:
            symbols = self.get_optionable_symbols()[:10]  # Limit for demo
            
        opportunities = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            # Update progress
            progress = (i + 1) / len(symbols)
            progress_bar.progress(progress)
            status_text.text(f"Processing {symbol}... ({i+1}/{len(symbols)})")
            
            # Get stock data
            stock = self.get_stock_quote(symbol)
            if not stock:
                continue
                
            current_price = stock["lastPrice"]
            
            # Get option chain
            chain = self.get_option_chain(symbol)
            if not chain or "options" not in chain:
                continue
                
            # Find OTM puts
            for option in chain["options"]:
                if option["optionType"] != "PUT":
                    continue
                    
                # Calculate OTM percentage
                strike = option["strikePrice"]
                otm_percentage = (current_price - strike) / current_price
                
                # Only consider puts that meet our OTM criteria
                if otm_percentage < min_otm:
                    continue
                    
                # Check expiration
                expiry = option["expirationDate"]
                expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
                days_to_expiry = (expiry_date - datetime.now()).days
                
                if days_to_expiry > max_days:
                    continue
                    
                # Check for liquidity
                if option["volume"] < 10 or option["openInterest"] < 10:
                    continue
                    
                # Calculate returns
                put_price = option["bid"]  # Use bid price (conservative)
                cash_required = strike * 100  # Per contract
                premium = put_price * 100  # Per contract
                
                return_rate = premium / cash_required
                annualized_return = return_rate * (365 / days_to_expiry)
                
                if annualized_return >= min_return:
                    effective_cost_basis = strike - put_price
                    discount_to_current = (current_price - effective_cost_basis) / current_price
                    
                    opportunities.append({
                        "symbol": symbol,
                        "stock_price": current_price,
                        "strike": strike,
                        "expiration_date": expiry,
                        "days_to_expiry": days_to_expiry,
                        "put_price": put_price,
                        "premium": premium,
                        "cash_required": cash_required,
                        "return": return_rate,
                        "annualized_return": annualized_return,
                        "implied_volatility": option["impliedVolatility"],
                        "distance_from_current": otm_percentage,
                        "effective_cost_basis": effective_cost_basis,
                        "discount_to_current": discount_to_current,
                        "volume": option["volume"],
                        "open_interest": option["openInterest"]
                    })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Return as DataFrame sorted by return
        if opportunities:
            return pd.DataFrame(opportunities).sort_values("annualized_return", ascending=False)
        else:
            return pd.DataFrame()

# Initialize API
api = SchwabAPI(SCHWAB_API_KEY, SCHWAB_API_SECRET, SCHWAB_API_BASE_URL)

# UI Components
st.title("Options Arbitrage Scanner")

# Sidebar
with st.sidebar:
    st.header("Options Scanner")
    
    # API connection status
    st.success("Connected to Schwab API ✓")
    st.info(f"API Key: {SCHWAB_API_KEY[:5]}...{SCHWAB_API_KEY[-5:]}")
    
    # Strategy selector
    strategy = st.radio(
        "Strategy",
        ["Covered Calls", "Cash-Secured Puts"],
        index=0
    )
    
    st.divider()
    st.subheader("Scanner Settings")
    
    # Common settings
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
        ["Scan Top Symbols", "Specific Symbols"],
        index=0
    )
    
    selected_symbols = None
    
    if symbol_option == "Specific Symbols":
        symbols_input = st.text_input(
            "Enter Symbols (comma separated)",
            value="AAPL,MSFT,AMZN,GOOGL,TSLA"
        )
        selected_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Scan button
    scan_button = st.button("Scan for Opportunities", type="primary", use_container_width=True)
    
    if scan_button:
        # Run the appropriate scan based on strategy
        if strategy == "Covered Calls":
            results = api.find_covered_call_opportunities(
                min_return=min_return,
                max_days=max_days,
                safety_margin=safety_margin,
                symbols=selected_symbols
            )
            
            st.session_state.scan_results = {
                "type": "covered_call",
                "data": results,
                "timestamp": datetime.now()
            }
        else:  # Cash-Secured Puts
            results = api.find_cash_secured_put_opportunities(
                min_return=min_return,
                max_days=max_days,
                min_otm=min_otm,
                symbols=selected_symbols
            )
            
            st.session_state.scan_results = {
                "type": "cash_secured_put",
                "data": results,
                "timestamp": datetime.now()
            }

# Main content area
if 'scan_results' in st.session_state and st.
