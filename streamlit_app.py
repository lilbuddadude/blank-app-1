
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (API keys, etc.)
load_dotenv()

class EnhancedOptionsArbitrageScanner:
    def __init__(self):
        # Schwab API credentials
        self.api_key = os.getenv("Vtbsc861GI48iT3JgAr8bp5Hvy5cVe7O")
        self.api_secret = os.getenv("SvMJwXrepRDQBiXr")
        self.base_url = "https://api.schwab.com/trader/v1"  # Example URL
        self.session = self._create_session()
        self.rate_limit_delay = 0.2  # 200ms delay between API calls to avoid rate limiting
        
        # Cache for API responses to reduce duplicate calls
        self.cache = {
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
        
        # Timestamps for cache entries
        self.cache_timestamps = {
            'options_chains': {},
            'stock_quotes': {},
            'symbols_list': None
        }
        
    def _create_session(self):
        """Create and authenticate a session with Schwab's API"""
        session = requests.Session()
        # Add authentication headers
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
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
        """
        Get a comprehensive list of all stocks with options available
        """
        # Check cache
        if self.cache['symbols_list'] and self._is_cache_valid('symbols_list'):
            return self.cache['symbols_list']
        
        # In a real implementation, you would pull this from Schwab API
        # Here's where you'd make the API call to get all optionable stocks
        endpoint = f"{self.base_url}/markets/options/symbols"  # Example endpoint
        
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
                # Fall back to a comprehensive list from a backup source or file
                return self._get_fallback_symbols_list()
        except Exception as e:
            logger.error(f"Exception fetching optionable stocks: {str(e)}")
            return self._get_fallback_symbols_list()
    
    def _get_fallback_symbols_list(self):
        """Provide a fallback method to get optionable stocks if API fails"""
        # This could read from a local file, a database, or a third-party API
        try:
            # Example: read from a local CSV file
            if os.path.exists('optionable_stocks.csv'):
                df = pd.read_csv('optionable_stocks.csv')
                return df['symbol'].tolist()
            else:
                # If no local file, return a default list
                # In production, you'd want a more comprehensive list
                logger.warning("Using default symbol list - consider updating to a full list")
                return [
                    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", 
                    "INTC", "NFLX", "DIS", "BA", "JPM", "V", "MA", "PFE", "JNJ", 
                    "WMT", "HD", "COST", "SBUX", "MCD", "KO", "PEP", "CSCO", "IBM",
                    "ORCL", "CRM", "ADBE", "PYPL", "ROKU", "SQ", "SHOP", "ZM", "MRNA",
                    "BNTX", "NKE", "LULU", "TGT", "AMC", "GME", "BB", "NOK", "F", "GM",
                    # Add more symbols...
                ]
        except Exception as e:
            logger.error(f"Error in fallback symbol list: {str(e)}")
            return []
    
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
    
    def find_covered_call_opportunities(self, min_return=0.05, max_days_to_expiry=45, safety_margin=0.01, max_symbols=None):
        """
        Find ITM covered call opportunities that offer immediate arbitrage using parallel processing
        
        Args:
            min_return: Minimum annualized return (5% = 0.05)
            max_days_to_expiry: Maximum days to expiration to consider
            safety_margin: Additional margin added to stock price to account for execution risk
            max_symbols: Maximum number of symbols to process (None for all)
        
        Returns:
            DataFrame of opportunities sorted by annualized return
        """
        # Get list of optionable stocks
        all_symbols = self.get_all_optionable_stocks()
        
        if max_symbols:
            symbols = all_symbols[:max_symbols]
        else:
            symbols = all_symbols
            
        logger.info(f"Scanning {len(symbols)} symbols for covered call opportunities")
        
        all_opportunities = []
        
        # Use parallel processing to scan multiple symbols simultaneously
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_covered_call_for_symbol, 
                    symbol, 
                    min_return, 
                    max_days_to_expiry,
                    safety_margin
                ): symbol for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    opportunities = future.result()
                    if opportunities:
                        all_opportunities.extend(opportunities)
                        logger.info(f"Found {len(opportunities)} opportunities for {symbol}")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
        
        logger.info(f"Total covered call opportunities found: {len(all_opportunities)}")
        
        # Convert to DataFrame and sort
        if all_opportunities:
            df = pd.DataFrame(all_opportunities)
            return df.sort_values('annualized_return', ascending=False)
        else:
            return pd.DataFrame()
    
    def find_cash_secured_put_opportunities(self, min_return=0.05, max_days_to_expiry=45, min_otm_percentage=0.05, max_symbols=None):
        """
        Find profitable cash-secured put opportunities using parallel processing
        
        Args:
            min_return: Minimum annualized return (5% = 0.05)
            max_days_to_expiry: Maximum days to expiration to consider
            min_otm_percentage: Minimum percentage OTM for puts to consider
            max_symbols: Maximum number of symbols to process (None for all)
        
        Returns:
            DataFrame of opportunities sorted by annualized return
        """
        # Get list of optionable stocks
        all_symbols = self.get_all_optionable_stocks()
        
        if max_symbols:
            symbols = all_symbols[:max_symbols]
        else:
            symbols = all_symbols
            
        logger.info(f"Scanning {len(symbols)} symbols for CSP opportunities")
        
        all_opportunities = []
        
        # Use parallel processing to scan multiple symbols simultaneously
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_csp_for_symbol, 
                    symbol, 
                    min_return, 
                    max_days_to_expiry,
                    min_otm_percentage
                ): symbol for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    opportunities = future.result()
                    if opportunities:
                        all_opportunities.extend(opportunities)
                        logger.info(f"Found {len(opportunities)} opportunities for {symbol}")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
        
        logger.info(f"Total CSP opportunities found: {len(all_opportunities)}")
        
        # Convert to DataFrame and sort
        if all_opportunities:
            df = pd.DataFrame(all_opportunities)
            return df.sort_values('annualized_return', ascending=False)
        else:
            return pd.DataFrame()


# Enhanced Web UI implementation using Streamlit
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_tags import st_tags

def run_enhanced_app():
    st.set_page_config(
        page_title="Options Arbitrage Scanner",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("Options Arbitrage Scanner")
    
    # Initialize scanner
    try:
        scanner = EnhancedOptionsArbitrageScanner()
    except Exception as e:
        st.error(f"Error initializing scanner: {str(e)}")
        st.info("Make sure you have set up your Schwab API credentials in the .env file")
        return
    
    # Sidebar controls
    st.sidebar.header("Scanner Settings")
    
    scanner_type = st.sidebar.radio(
        "Scanner Type",
        ["ITM Covered Calls", "Cash-Secured Puts"]
    )
    
    # Advanced settings in sidebar
    with st.sidebar.expander("Advanced Settings", expanded=False):
        min_return = st.slider(
            "Minimum Annualized Return",
            min_value=0.01,
            max_value=1.00,
            value=0.05,
            step=0.01,
            format="%.2f"
        )
        
        max_days = st.slider(
            "Maximum Days to Expiry",
            min_value=1,
            max_value=180,
            value=45
        )
        
        # Different additional settings based on scanner type
        if scanner_type == "ITM Covered Calls":
            safety_margin = st.slider(
                "Safety Margin for Stock Price",
                min_value=0.0,
                max_value=0.05,
                value=0.01,
                step=0.005,
                format="%.3f",
                help="Additional margin added to stock price to account for execution risk"
            )
            
            min_downside_protection = st.slider(
                "Minimum Downside Protection",
                min_value=0.0,
                max_value=0.3,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="Minimum percentage of downside protection from net debit price"
            )
        else:  # Cash-Secured Puts
            min_otm_percentage = st.slider(
                "Minimum OTM Percentage",
                min_value=0.0,
                max_value=0.3,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="Minimum percentage out-of-the-money for puts"
            )
            
            min_discount = st.slider(
                "Minimum Discount to Current Price",
                min_value=0.0,
                max_value=0.3,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="Minimum discount to current price if assigned"
            )
        
        # Liquidity filters for both scanner types
        min_volume = st.number_input(
            "Minimum Option Volume",
            min_value=0,
            max_value=1000,
            value=10,
            help="Minimum trading volume for option contracts"
        )
        
        min_open_interest = st.number_input(
            "Minimum Open Interest",
            min_value=0,
            max_value=1000,
            value=10,
            help="Minimum open interest for option contracts"
        )
        
        max_bid_ask_spread = st.slider(
            "Maximum Bid-Ask Spread",
            min_value=0.01,
            max_value=0.5,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Maximum acceptable bid-ask spread as percentage of bid price"
        )
    
    # Specific symbol filter
    st.sidebar.header("Symbol Filter")
    filter_method = st.sidebar.radio(
        "Filter Method",
        ["Scan All Available Symbols", "Specify Symbols"]
    )
    
    selected_symbols = None
    max_symbols = None
    
    if filter_method == "Specify Symbols":
        selected_symbols = st_tags(
            label="Enter Stock Symbols",
            text="Add symbols and press enter",
            value=["AAPL", "MSFT", "AMZN"],
            maxtags=50,
            key="symbols"
        )
    else:
        max_symbols = st.sidebar.number_input(
            "Maximum Symbols to Scan (0 for all)",
            min_value=0,
            max_value=5000,
            value=100,
            help="Limit the number of symbols to scan (0 means scan all)"
        )
        if max_symbols == 0:
            max_symbols = None
    
    # Main scan button
    scan_button = st.sidebar.button("Scan for Opportunities", type="primary")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Results", "Analysis", "Charts"])
    
    # Initialize session state for storing results
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = None
    
    if scan_button:
        with st.spinner("Scanning for opportunities. This may take a few minutes for a large number of symbols..."):
            try:
                if scanner_type == "ITM Covered Calls":
                    if selected_symbols:
                        # Scan specific symbols
                        opportunities = pd.DataFrame()
                        for symbol in selected_symbols:
                            symbol_opps = scanner._process_covered_call_for_symbol(
                                symbol, min_return, max_days, safety_margin
                            )
                            if symbol_opps:
                                opportunities = pd.concat([opportunities, pd.DataFrame(symbol_opps)])
                    else:
                        # Scan all or limited symbols
                        opportunities = scanner.find_covered_call_opportunities(
                            min_return=min_return,
                            max_days_to_expiry=max_days,
                            safety_margin=safety_margin,
                            max_symbols=max_symbols
                        )
                    
                    # Apply additional filters
                    if not opportunities.empty:
                        opportunities = opportunities[
                            (opportunities['volume'] >= min_volume) &
                            (opportunities['open_interest'] >= min_open_interest) &
                            (opportunities['bid_ask_spread'] <= max_bid_ask_spread * opportunities['call_price']) &
                            (opportunities['downside_protection'] >= min_downside_protection)
                        ]
                    
                    st.session_state.scan_results = {
                        'data': opportunities,
                        'type': 'covered_call'
                    }
                    
                    with tab1:
                        display_covered_call_results(opportunities)
                    
                    with tab2:
                        if not opportunities.empty:
                            display_covered_call_analysis(opportunities)
                        else:
                            st.warning("No opportunities match your criteria for analysis.")
                    
                    with tab3:
                        if not opportunities.empty:
                            display_covered_call_charts(opportunities)
                        else:
                            st.warning("No opportunities match your criteria for charting.")
                    
                else:  # Cash-Secured Puts
                    if selected_symbols:
                        # Scan specific symbols
                        opportunities = pd.DataFrame()
                        for symbol in selected_symbols:
                            symbol_opps = scanner._process_csp_for_symbol(
                                symbol, min_return, max_days, min_otm_percentage
                            )
                            if symbol_opps:
                                opportunities = pd.concat([opportunities, pd.DataFrame(symbol_opps)])
                    else:
                        # Scan all or limited symbols
                        opportunities = scanner.find_cash_secured_put_opportunities(
                            min_return=min_return,
                            max_days_to_expiry=max_days,
                            min_otm_percentage=min_otm_percentage,
                            max_symbols=max_symbols
                        )
                    
                    # Apply additional filters
                    if not opportunities.empty:
                        opportunities = opportunities[
                            (opportunities['volume'] >= min_volume) &
                            (opportunities['open_interest'] >= min_open_interest) &
                            (opportunities['bid_ask_spread'] <= max_bid_ask_spread * opportunities['put_price']) &
                            (opportunities['discount_to_current'] >= min_discount)
                        ]
                    
                    st.session_state.scan_results = {
                        'data': opportunities,
                        'type': 'cash_secured_put'
                    }
                    
                    with tab1:
                        display_csp_results(opportunities)
                    
                    with tab2:
                        if not opportunities.empty:
                            display_csp_analysis(opportunities)
                        else:
                            st.warning("No opportunities match your criteria for analysis.")
                    
                    with tab3:
                        if not opportunities.empty:
                            display_csp_charts(opportunities)
                        else:
                            st.warning("No opportunities match your criteria for charting.")
                    
            except Exception as e:
                st.error(f"Error during scan: {str(e)}")
    
    # Display stored results if they exist and no new scan was performed
    elif st.session_state.scan_results is not None:
        with tab1:
            if st.session_state.scan_results['type'] == 'covered_call':
                display_covered_call_results(st.session_state.scan_results['data'])
            else:
                display_csp_results(st.session_state.scan_results['data'])
        
    
