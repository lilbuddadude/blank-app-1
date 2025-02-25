import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import json
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Options Arbitrage Scanner",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 10px 0 rgba(0,0,0,0.1);
        padding: 15px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 14px;
        color: #616161;
    }
    .highlight {
        background-color: #e6f7ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1890ff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Options Scanner Class with Schwab API Integration
class OptionsArbitrageScanner:
    def __init__(self):
        """Initialize the scanner with Schwab API credentials"""
        # Load API credentials from environment variables or Streamlit secrets
        self.api_key = os.getenv("SCHWAB_API_KEY") or st.secrets.get("SCHWAB_API_KEY", "")
        self.api_secret = os.getenv("SCHWAB_API_SECRET") or st.secrets.get("SCHWAB_API_SECRET", "")
        self.base_url = os.getenv("SCHWAB_API_BASE_URL") or st.secrets.get("SCHWAB_API_BASE_URL", "https://api.schwab.com/trader/v1")
        
        # Check if credentials are available
        if not self.api_key or not self.api_secret:
            st.warning("Schwab API credentials not found. Please configure them in settings.")
            logger.warning("API credentials not found")
            self.api_configured = False
        else:
            self.api_configured = True
            self.session = self._create_session()
            logger.info("API session created successfully")
        
        # Cache for API responses
        self.cache = {
            'options_chains': {},
            'stock_quotes': {},
            'symbols_list': None
        }
        
        # Cache timestamps
        self.cache_timestamps = {
            'options_chains': {},
            'stock_quotes': {},
            'symbols_list': None
        }
        
        # Cache expiry times (in seconds)
        self.cache_expiry = {
            'options_chains': 300,  # 5 minutes
            'stock_quotes': 60,     # 1 minute 
            'symbols_list': 3600    # 1 hour
        }
        
        # API rate limit settings
        self.rate_limit_delay = 0.2  # 200ms delay between API calls
        
    def _create_session(self):
        """Create an authenticated session with Schwab's API"""
        session = requests.Session()
        
        # Add authentication headers
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        # You may need to implement OAuth flow here depending on Schwab's requirements
        # For example:
        # token = self._get_oauth_token()
        # session.headers.update({"Authorization": f"Bearer {token}"})
        
        return session
    
    def _is_cache_valid(self, cache_type, key=None):
        """Check if a cache entry is still valid"""
        if key:
            if key not in self.cache_timestamps.get(cache_type, {}):
                return False
            timestamp = self.cache_timestamps[cache_type][key]
        else:
            if self.cache_timestamps[cache_type] is None:
                return False
            timestamp = self.cache_timestamps[cache_type]
            
        return (datetime.now() - timestamp).total_seconds() < self.cache_expiry[cache_type]
    
    def get_stock_data(self, symbols):
        """Fetch current stock data for a list of symbols with caching"""
        if not self.api_configured:
            st.error("API not configured. Please add credentials in Settings.")
            return None
            
        result = []
        symbols_to_fetch = []
        
        # Check cache for each symbol
        for symbol in symbols:
            if symbol in self.cache['stock_quotes'] and self._is_cache_valid('stock_quotes', symbol):
                result.append(self.cache['stock_quotes'][symbol])
            else:
                symbols_to_fetch.append(symbol)
        
        if not symbols_to_fetch:
            return result
        
        # Fetch data for symbols not in cache
        endpoint = f"{self.base_url}/markets/quotes"
        try:
            response = self.session.get(
                endpoint, 
                params={"symbols": ",".join(symbols_to_fetch)}
            )
            
            if response.status_code == 200:
                new_data = response.json()
                
                # Update cache with new data
                for quote in new_data:
                    symbol = quote['symbol']
                    self.cache['stock_quotes'][symbol] = quote
                    self.cache_timestamps['stock_quotes'][symbol] = datetime.now()
                    result.append(quote)
                    
                return result
            else:
                logger.error(f"Error fetching stock data: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception fetching stock data: {str(e)}")
            return None
    
    def get_option_chain(self, symbol, expiration_date=None):
        """Fetch option chain data for a given symbol with caching"""
        if not self.api_configured:
            st.error("API not configured. Please add credentials in Settings.")
            return None
            
        cache_key = f"{symbol}_{expiration_date}" if expiration_date else symbol
        
        # Check cache
        if cache_key in self.cache['options_chains'] and self._is_cache_valid('options_chains', cache_key):
            return self.cache['options_chains'][cache_key]
        
        # Fetch from API if not in cache
        endpoint = f"{self.base_url}/markets/options/chains"
        
        params = {"symbol": symbol}
        if expiration_date:
            params["expirationDate"] = expiration_date
            
        try:
            response = self.session.get(endpoint, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Update cache
                self.cache['options_chains'][cache_key] = data
                self.cache_timestamps['options_chains'][cache_key] = datetime.now()
                
                return data
            else:
                logger.error(f"Error fetching option chain for {symbol}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception fetching option chain for {symbol}: {str(e)}")
            return None
    
    def get_all_optionable_stocks(self):
        """Get a comprehensive list of all stocks with options available"""
        if not self.api_configured:
            st.error("API not configured. Please add credentials in Settings.")
            return []
            
        # Check cache
        if self.cache['symbols_list'] and self._is_cache_valid('symbols_list'):
            return self.cache['symbols_list']
        
        # Fetch from Schwab API
        endpoint = f"{self.base_url}/markets/options/symbols"
        
        try:
            response = self.session.get(endpoint)
            
            if response.status_code == 200:
                data = response.json()
                symbols = [item['symbol'] for item in data]
                
                # Update cache
                self.cache['symbols_list'] = symbols
                self.cache_timestamps['symbols_list'] = datetime.now()
                
                return symbols
            else:
                logger.error(f"Error fetching optionable stocks: {response.status_code}")
                return self._get_fallback_symbols_list()
        except Exception as e:
            logger.error(f"Exception fetching optionable stocks: {str(e)}")
            return self._get_fallback_symbols_list()
    
    def _get_fallback_symbols_list(self):
        """Provide a fallback list of optionable stocks if API fails"""
        try:
            # Example: read from a local CSV file
            if os.path.exists('optionable_stocks.csv'):
                df = pd.read_csv('optionable_stocks.csv')
                return df['symbol'].tolist()
            else:
                # Return a default list of common optionable stocks
                logger.warning("Using default symbol list - consider updating to a full list")
                return [
                    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", 
                    "INTC", "NFLX", "DIS", "BA", "JPM", "V", "MA", "PFE", "JNJ", 
                    "WMT", "HD", "COST", "SBUX", "MCD", "KO", "PEP", "CSCO", "IBM",
                    "ORCL", "CRM", "ADBE", "PYPL", "ROKU", "SQ", "SHOP", "ZM", "MRNA",
                    "BNTX", "NKE", "LULU", "TGT", "AMC", "GME", "BB", "NOK", "F", "GM",
                    "SPY", "QQQ", "IWM", "DIA"
                ]
        except Exception as e:
            logger.error(f"Error in fallback symbol list: {str(e)}")
            return []
    
    def find_covered_call_opportunities(self, min_return=0.05, max_days_to_expiry=45, 
                                       safety_margin=0.01, symbols=None, max_symbols=None):
        """
        Find ITM covered call opportunities that offer immediate arbitrage
        
        Args:
            min_return: Minimum annualized return (5% = 0.05)
            max_days_to_expiry: Maximum days to expiration to consider
            safety_margin: Additional margin added to stock price for safety
            symbols: List of specific symbols to scan (None for all available)
            max_symbols: Maximum number of symbols to process (None for all)
        
        Returns:
            DataFrame of opportunities sorted by annualized return
        """
        if not self.api_configured:
            st.error("API not configured. Please add credentials in Settings.")
            return pd.DataFrame()
            
        with st.spinner("Scanning for covered call opportunities..."):
            # Get symbols to scan
            if symbols:
                symbols_to_scan = symbols
            else:
                symbols_to_scan = self.get_all_optionable_stocks()
                
                if max_symbols and len(symbols_to_scan) > max_symbols:
                    symbols_to_scan = symbols_to_scan[:max_symbols]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Scanning {len(symbols_to_scan)} symbols for covered call opportunities")
            
            all_opportunities = []
            
            # Use parallel processing to scan multiple symbols
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_symbol = {
                    executor.submit(
                        self._process_covered_call_for_symbol, 
                        symbol, 
                        min_return, 
                        max_days_to_expiry,
                        safety_margin
                    ): symbol for symbol in symbols_to_scan
                }
                
                completed = 0
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    progress = completed / len(symbols_to_scan)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {symbol}... ({completed}/{len(symbols_to_scan)})")
                    
                    try:
                        opportunities = future.result()
                        if opportunities:
                            all_opportunities.extend(opportunities)
                            logger.info(f"Found {len(opportunities)} opportunities for {symbol}")
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            logger.info(f"Total covered call opportunities found: {len(all_opportunities)}")
            
            # Convert to DataFrame and sort
            if all_opportunities:
                df = pd.DataFrame(all_opportunities)
                return df.sort_values('annualized_return', ascending=False)
            else:
                return pd.DataFrame()
    
    def _process_covered_call_for_symbol(self, symbol, min_return=0.05, max_days_to_expiry=45, safety_margin=0.01):
        """Process covered call opportunities for a single symbol"""
        try:
            # Get current stock price
            stock_data = self.get_stock_data([symbol])
            if not stock_data or len(stock_data) == 0:
                return []
                
            current_price = stock_data[0].get('lastPrice')
            if not current_price:
                return []
            
            # Add a small delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Get option chains
            option_data = self.get_option_chain(symbol)
            if not option_data or 'expirationDates' not in option_data:
                return []
                
            opportunities = []
            
            # Process expirations within our timeframe
            for expiration in option_data['expirationDates']:
                try:
                    expiry_date = datetime.fromisoformat(expiration.replace('Z', '+00:00'))
                    days_to_expiry = (expiry_date - datetime.now()).days
                    
                    if days_to_expiry > max_days_to_expiry or days_to_expiry <= 0:
                        continue
                        
                    # Get calls for this expiration
                    calls = [opt for opt in option_data.get('options', []) 
                             if opt.get('optionType') == 'CALL' and 
                             opt.get('expirationDate') == expiration]
                    
                    # Find ITM calls with sufficient volume and open interest
                    for call in calls:
                        strike = call.get('strikePrice')
                        if not strike:
                            continue
                        
                        # Only consider ITM options
                        if strike >= current_price:
                            continue
                            
                        # Check for sufficient liquidity
                        volume = call.get('volume', 0)
                        open_interest = call.get('openInterest', 0)
                        bid_ask_spread = call.get('ask', 0) - call.get('bid', 0)
                        
                        # Skip options with low liquidity or wide spreads
                        if volume < 10 or open_interest < 10 or bid_ask_spread > 0.10 * call.get('bid', 1):
                            continue
                            
                        call_price = call.get('bid', 0)  # Use bid price to be conservative
                        
                        # Calculate potential arbitrage with safety margin
                        cost_basis = current_price * (1 + safety_margin)  # Add safety margin to stock price
                        immediate_credit = call_price
                        assignment_value = strike
                        
                        # Check if there's an immediate arbitrage opportunity
                        net_debit = cost_basis - immediate_credit
                        if net_debit < assignment_value:
                            profit = assignment_value - net_debit
                            profit_percentage = profit / net_debit
                            annualized_return = profit_percentage * (365 / days_to_expiry)
                            
                            if annualized_return >= min_return:
                                # Calculate risk metrics
                                downside_protection = (current_price - net_debit) / current_price
                                max_loss = net_debit  # Theoretical max loss if stock goes to zero
                                
                                opportunities.append({
                                    'symbol': symbol,
                                    'stock_price': current_price,
                                    'strike': strike,
                                    'expiration_date': expiry_date.strftime('%Y-%m-%d'),
                                    'days_to_expiry': days_to_expiry,
                                    'call_price': call_price,
                                    'bid_ask_spread': bid_ask_spread,
                                    'volume': volume,
                                    'open_interest': open_interest,
                                    'net_debit': net_debit,
                                    'profit': profit,
                                    'return': profit_percentage,
                                    'annualized_return': annualized_return,
                                    'implied_volatility': call.get('impliedVolatility', 0),
                                    'downside_protection': downside_protection,
                                    'max_loss': max_loss,
                                    'intrinsic_value': current_price - strike,
                                    'time_value': call_price - (current_price - strike)
                                })
                except Exception as e:
                    logger.error(f"Error processing expiration for {symbol}: {str(e)}")
                    continue
                    
            return opportunities
        except Exception as e:
            logger.error(f"Error processing covered calls for {symbol}: {str(e)}")
            return []
    
    def find_cash_secured_put_opportunities(self, min_return=0.05, max_days_to_expiry=45, 
                                           min_otm_percentage=0.05, symbols=None, max_symbols=None):
        """
        Find profitable cash-secured put opportunities
        
        Args:
            min_return: Minimum annualized return (5% = 0.05)
            max_days_to_expiry: Maximum days to expiration to consider
            min_otm_percentage: Minimum percentage OTM for puts to consider
            symbols: List of specific symbols to scan (None for all available)
            max_symbols: Maximum number of symbols to process (None for all)
        
        Returns:
            DataFrame of opportunities sorted by annualized return
        """
        if not self.api_configured:
            st.error("API not configured. Please add credentials in Settings.")
            return pd.DataFrame()
            
        with st.spinner("Scanning for cash-secured put opportunities..."):
            # Get symbols to scan
            if symbols:
                symbols_to_scan = symbols
            else:
                symbols_to_scan = self.get_all_optionable_stocks()
                
                if max_symbols and len(symbols_to_scan) > max_symbols:
                    symbols_to_scan = symbols_to_scan[:max_symbols]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Scanning {len(symbols_to_scan)} symbols for CSP opportunities")
            
            all_opportunities = []
            
            # Use parallel processing to scan multiple symbols
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_symbol = {
                    executor.submit(
                        self._process_csp_for_symbol, 
                        symbol, 
                        min_return, 
                        max_days_to_expiry,
                        min_otm_percentage
                    ): symbol for symbol in symbols_to_scan
                }
                
                completed = 0
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    progress = completed / len(symbols_to_scan)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {symbol}... ({completed}/{len(symbols_to_scan)})")
                    
                    try:
                        opportunities = future.result()
                        if opportunities:
                            all_opportunities.extend(opportunities)
                            logger.info(f"Found {len(opportunities)} opportunities for {symbol}")
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            logger.info(f"Total CSP opportunities found: {len(all_opportunities)}")
            
            # Convert to DataFrame and sort
            if all_opportunities:
                df = pd.DataFrame(all_opportunities)
                return df.sort_values('annualized_return', ascending=False)
            else:
                return pd.DataFrame()
    
    def _process_csp_for_symbol(self, symbol, min_return=0.05, max_days_to_expiry=45, min_otm_percentage=0.05):
        """Process cash-secured put opportunities for a single symbol"""
        try:
            # Get current stock price
            stock_data = self.get_stock_data([symbol])
            if not stock_data or len(stock_data) == 0:
                return []
                
            current_price = stock_data[0].get('lastPrice')
            if not current_price:
                return []
            
            # Add a small delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Get option chains
            option_data = self.get_option_chain(symbol)
            if not option_data or 'expirationDates' not in option_data:
                return []
                
            opportunities = []
            
            # Process expirations within our timeframe
            for expiration in option_data['expirationDates']:
                try:
                    expiry_date = datetime.fromisoformat(expiration.replace('Z', '+00:00'))
                    days_to_expiry = (expiry_date - datetime.now()).days
                    
                    if days_to_expiry > max_days_to_expiry or days_to_expiry <= 0:
                        continue
                        
                    # Get puts for this expiration
                    puts = [opt for opt in option_data.get('options', []) 
                           if opt.get('optionType') == 'PUT' and 
                           opt.get('expirationDate') == expiration]
                    
                    for put in puts:
                        strike = put.get('strikePrice')
                        if not strike:
                            continue
                            
                        # Calculate how far OTM this put is
                        otm_percentage = (current_price - strike) / current_price
                        
                        # Skip if it doesn't meet our OTM criteria
                        if otm_percentage < min_otm_percentage:
                            continue
                            
                        # Check for sufficient liquidity
                        volume = put.get('volume', 0)
                        open_interest = put.get('openInterest', 0)
                        bid_ask_spread = put.get('ask', 0) - put.get('bid', 0)
                        
                        # Skip options with low liquidity or wide spreads
                        if volume < 10 or open_interest < 10 or bid_ask_spread > 0.10 * put.get('bid', 1):
                            continue
                            
                        put_price = put.get('bid', 0)  # Use bid price to be conservative
                        
                        # Calculate return metrics
                        cash_required = strike * 100  # Per contract
                        premium = put_price * 100  # Per contract
                        return_rate = premium / cash_required
                        annualized_return = return_rate * (365 / days_to_expiry)
                        
                        if annualized_return >= min_return:
                            # Calculate additional metrics
                            effective_cost_basis = strike - put_price
                            discount_to_current = (current_price - effective_cost_basis) / current_price
                            
                            opportunities.append({
                                'symbol': symbol,
                                'stock_price': current_price,
                                'strike': strike,
                                'expiration_date': expiry_date.strftime('%Y-%m-%d'),
                                'days_to_expiry': days_to_expiry,
                                'put_price': put_price,
                                'bid_ask_spread': bid_ask_spread,
                                'volume': volume,
                                'open_interest': open_interest,
                                'premium': premium,
                                'cash_required': cash_required,
                                'return': return_rate,
                                'annualized_return': annualized_return,
                                'implied_volatility': put.get('impliedVolatility', 0),
                                'distance_from_current': otm_percentage,
                                'effective_cost_basis': effective_cost_basis,
                                'discount_to_current': discount_to_current
                            })
                except Exception as e:
                    logger.error(f"Error processing expiration for {symbol}: {str(e)}")
                    continue
                    
            return opportunities
        except Exception as e:
            logger.error(f"Error processing CSPs for {symbol}: {str(e)}")
            return []

# Initialize session state
if 'scanner' not in st.session_state:
    st.session_state.scanner = OptionsArbitrageScanner()
    
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Dashboard"

if 'api_configured' not in st.session_state:
    st.session_state.api_configured = st.session_state.scanner.api_configured

# Main App UI
def main():
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bull-chart.png", width=80)
        st.title("Options Scanner")
        
        # Navigation menu
        selected_tab = st.radio(
            "Navigation",
            ["Dashboard", "Covered Calls", "Cash-Secured Puts", "Watchlist", "Settings"],
            index=0
        )
        st.session_state.selected_tab = selected_tab
        
        # API status indicator
        if st.session_state.api_configured:
            st.success("API Connected", icon="✅")
        else:
            st.error("API Not Connected", icon="❌")
            st.info("Please configure API credentials in Settings tab")
        
        # Advanced settings if on a scanner page
        if selected_tab in ["Covered Calls", "Cash-Secured Puts"]:
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
            if selected_tab == "Covered Calls":
                safety_margin = st.slider(
                    "Safety Margin",
                    min_value=0.0,
                    max_value=0.05,
                    value=0.01,
                    step=0.005,
                    format="%.3f"
                )
                
                min_downside = st.slider(
                    "Min Downside Protection",
                    min_value=0.0,
                    max_value=0.3,
                    value=0.05,
                    step=0.01,
                    format="%.2f"
                )
                
                st.session_state.cc_settings = {
                    'min_return': min_return,
                    'max_days': max_days,
                    'safety_margin': safety_margin,
                    'min_downside': min_downside
                }
                
            else:  # Cash-Secured Puts
                min_otm = st.slider(
                    "Min OTM Percentage",
                    min_value=0.0,
                    max_value=0.3,
                    value=0.05,
                    step=0.01,
                    format="%.2f"
                )
                
                st.session_state.csp_settings = {
                    'min_return': min_return,
                    'max_days': max_days,
                    'min_otm': min_otm
                }
            
            # Symbol selection
            symbol_option = st.radio(
                "Symbol Selection",
                ["Scan All Symbols", "Use Watchlist", "Specific Symbols"],
                index=0
            )
            
            selected_symbols = None
            max_symbols = None
            
            if symbol_option == "Scan All Symbols":
                max_symbols = st.number_input(
                    "Max Symbols to Scan",
                    min_value=10,
                    max_value=5000,
                    value=100,
                    help="Limit the total number of symbols scanned"
                )
            elif symbol_option == "Use Watchlist":
                st.info("Using symbols from your watchlist")
                # In a real app, you would load the user's watchlist here
                selected_symbols = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA"]
            else:  # Specific Symbols
                symbols_input = st.text_input(
                    "Enter Symbols (comma separated)",
                    value="AAPL,MSFT,AMZN,GOOGL,TSLA"
                )
                selected_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            
            st.session_state.symbol_selection = {
                'option': symbol_option,
                'selected_symbols': selected_symbols,
                'max_symbols': max_symbols
            }
            
            # Scan button
            scan_button = st.button("Scan for Opportunities", type="primary", use_container_width=True)
            
            if scan_button:
                if selected_tab == "Covered Calls":
                    results = st.session_state.scanner.find_covered_call_opportunities(
                        min_return=min_return,
                        max_days_to_expiry=max_days,
                        safety_margin=safety_margin,
                        symbols=selected_symbols,
                        max_symbols=max_
