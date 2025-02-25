
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from streamlit_option_menu import option_menu

# Set page config
st.set_page_config(
    page_title="Options Arbitrage Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f9f9f9;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        min-width: 150px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
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
    .table-container {
        border-radius: 5px;
        box-shadow: 0 2px 10px 0 rgba(0,0,0,0.1);
        overflow: hidden;
        margin: 15px 0;
    }
    .chart-container {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 10px 0 rgba(0,0,0,0.1);
        padding: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Mock data generator for testing the UI
def generate_mock_data(strategy_type="covered_call", num_rows=50):
    np.random.seed(42)
    
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", 
               "INTC", "NFLX", "DIS", "BA", "JPM", "V", "MA", "PFE"]
    
    today = datetime.now()
    
    data = {
        'symbol': np.random.choice(symbols, num_rows),
        'stock_price': np.random.uniform(50, 500, num_rows),
        'days_to_expiry': np.random.randint(1, 45, num_rows),
    }
    
    # Add expiration dates based on days to expiry
    data['expiration_date'] = [(today + timedelta(days=int(d))).strftime('%Y-%m-%d') 
                              for d in data['days_to_expiry']]
    
    if strategy_type == "covered_call":
        # Create ITM strikes (below stock price)
        data['strike'] = data['stock_price'] * np.random.uniform(0.7, 0.95, num_rows)
        data['call_price'] = (data['stock_price'] - data['strike']) + np.random.uniform(1, 10, num_rows)
        data['bid_ask_spread'] = np.random.uniform(0.01, 0.5, num_rows)
        data['volume'] = np.random.randint(10, 1000, num_rows)
        data['open_interest'] = np.random.randint(20, 2000, num_rows)
        
        # Calculate metrics
        data['net_debit'] = data['stock_price'] - data['call_price']
        data['profit'] = data['strike'] - data['net_debit']
        data['return'] = data['profit'] / data['net_debit']
        data['annualized_return'] = data['return'] * (365 / data['days_to_expiry'])
        data['implied_volatility'] = np.random.uniform(0.2, 0.8, num_rows)
        data['downside_protection'] = (data['stock_price'] - data['net_debit']) / data['stock_price']
        data['max_loss'] = data['net_debit']
        data['intrinsic_value'] = data['stock_price'] - data['strike']
        data['time_value'] = data['call_price'] - data['intrinsic_value']
        
    else:  # cash_secured_put
        # Create OTM strikes (below stock price)
        data['strike'] = data['stock_price'] * np.random.uniform(0.7, 0.95, num_rows)
        data['put_price'] = np.random.uniform(1, 10, num_rows)
        data['bid_ask_spread'] = np.random.uniform(0.01, 0.5, num_rows)
        data['volume'] = np.random.randint(10, 1000, num_rows)
        data['open_interest'] = np.random.randint(20, 2000, num_rows)
        
        # Calculate metrics
        data['premium'] = data['put_price'] * 100  # Per contract
        data['cash_required'] = data['strike'] * 100  # Per contract
        data['return'] = data['premium'] / data['cash_required']
        data['annualized_return'] = data['return'] * (365 / data['days_to_expiry'])
        data['implied_volatility'] = np.random.uniform(0.2, 0.8, num_rows)
        data['distance_from_current'] = (data['stock_price'] - data['strike']) / data['stock_price']
        data['effective_cost_basis'] = data['strike'] - data['put_price']
        data['discount_to_current'] = (data['stock_price'] - data['effective_cost_basis']) / data['stock_price']
        
    return pd.DataFrame(data)

# Main dashboard component
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bull-chart.png", width=80)
        st.title("Options Scanner")
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Covered Calls", "Cash-Secured Puts", "Watchlist", "Settings"],
            icons=["house", "graph-up-arrow", "cash-coin", "star", "gear"],
            default_index=0,
        )
        
        # Scanner settings
        if selected in ["Covered Calls", "Cash-Secured Puts"]:
            st.subheader("Scanner Settings")
            
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
            
            if selected == "Covered Calls":
                safety_margin = st.slider(
                    "Safety Margin",
                    min_value=0.0,
                    max_value=0.05,
                    value=0.01,
                    step=0.01,
                    format="%.2f"
                )
                
                min_downside = st.slider(
                    "Min Downside Protection",
                    min_value=0.0,
                    max_value=0.2,
                    value=0.05,
                    step=0.01,
                    format="%.2f"
                )
            else:
                min_otm = st.slider(
                    "Min OTM Percentage",
                    min_value=0.0,
                    max_value=0.3,
                    value=0.05,
                    step=0.01,
                    format="%.2f"
                )
            
            min_liquidity = st.slider(
                "Min Option Volume",
                min_value=10,
                max_value=1000,
                value=50,
                step=10
            )
        
        # Set up scan button
        if selected in ["Covered Calls", "Cash-Secured Puts"]:
            scan_button = st.button("Scan Now", type="primary", use_container_width=True)
        
        # Display last scan time
        st.caption("Last scan: 2025-02-25 10:15 AM")
        
    # Main content area
    if selected == "Dashboard":
        display_dashboard()
    elif selected == "Covered Calls":
        display_covered_calls()
    elif selected == "Cash-Secured Puts":
        display_cash_secured_puts()
    elif selected == "Watchlist":
        display_watchlist()
    elif selected == "Settings":
        display_settings()

# Dashboard page
def display_dashboard():
    st.title("Options Arbitrage Dashboard")
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Total Opportunities</div>
                <div class="metric-value">72</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Avg Annual Return</div>
                <div class="metric-value">24.8%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Best Return</div>
                <div class="metric-value">42.3%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Watchlist Alerts</div>
                <div class="metric-value">3</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Strategy overview
    st.subheader("Strategy Overview")
    
    tab1, tab2 = st.tabs(["Covered Calls", "Cash-Secured Puts"])
    
    with tab1:
        # Generate mock data for covered calls
        cc_data = generate_mock_data("covered_call", 50)
        
        # Top opportunities
        st.markdown("### Top Covered Call Opportunities")
        top_cc = cc_data.sort_values('annualized_return', ascending=False).head(5)
        
        # Format data for display
        display_cc = top_cc.copy()
        display_cc['return'] = display_cc['return'].apply(lambda x: f"{x:.2%}")
        display_cc['annualized_return'] = display_cc['annualized_return'].apply(lambda x: f"{x:.2%}")
        display_cc['downside_protection'] = display_cc['downside_protection'].apply(lambda x: f"{x:.2%}")
        
        # Select and rename columns for display
        display_cc = display_cc[['symbol', 'stock_price', 'strike', 'expiration_date', 'net_debit', 'profit', 'annualized_return']]
        display_cc.columns = ['Symbol', 'Stock Price', 'Strike', 'Expiration', 'Net Debit', 'Profit', 'Annual Return']
        
        st.dataframe(display_cc, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Return by Expiration")
            
            # Group by days to expiry
            expiry_groups = cc_data.groupby('days_to_expiry')['annualized_return'].mean().reset_index()
            
            fig = px.scatter(expiry_groups, x='days_to_expiry', y='annualized_return',
                          size=expiry_groups['annualized_return'] * 100,
                          color='annualized_return',
                          color_continuous_scale='Viridis',
                          labels={'days_to_expiry': 'Days to Expiration', 
                                 'annualized_return': 'Avg. Annual Return'},
                          title="Return by Time to Expiration")
            
            fig.update_layout(
                xaxis_title="Days to Expiration",
                yaxis_title="Avg. Annual Return",
                yaxis_tickformat='.0%',
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Top Symbols by Opportunity Count")
            
            # Count opportunities by symbol
            symbol_counts = cc_data['symbol'].value_counts().reset_index()
            symbol_counts.columns = ['symbol', 'count']
            
            fig = px.bar(symbol_counts.head(10), 
                         x='symbol', 
                         y='count',
                         color='count',
                         color_continuous_scale='Viridis',
                         labels={'symbol': 'Symbol', 'count': 'Number of Opportunities'},
                         title="Symbols with Most Opportunities")
            
            fig.update_layout(
                xaxis_title="Symbol",
                yaxis_title="Opportunity Count",
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Generate mock data for cash-secured puts
        csp_data = generate_mock_data("cash_secured_put", 50)
        
        # Top opportunities
        st.markdown("### Top Cash-Secured Put Opportunities")
        top_csp = csp_data.sort_values('annualized_return', ascending=False).head(5)
        
        # Format data for display
        display_csp = top_csp.copy()
        display_csp['return'] = display_csp['return'].apply(lambda x: f"{x:.2%}")
        display_csp['annualized_return'] = display_csp['annualized_return'].apply(lambda x: f"{x:.2%}")
        display_csp['distance_from_current'] = display_csp['distance_from_current'].apply(lambda x: f"{x:.2%}")
        
        # Select and rename columns for display
        display_csp = display_csp[['symbol', 'stock_price', 'strike', 'expiration_date', 'put_price', 'annualized_return', 'distance_from_current']]
        display_csp.columns = ['Symbol', 'Stock Price', 'Strike', 'Expiration', 'Put Price', 'Annual Return', 'OTM %']
        
        st.dataframe(display_csp, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Return vs. OTM %")
            
            fig = px.scatter(csp_data, 
                             x='distance_from_current', 
                             y='annualized_return',
                             color='implied_volatility',
                             size='premium',
                             hover_name='symbol',
                             hover_data=['strike', 'expiration_date'],
                             color_continuous_scale='Viridis',
                             labels={'distance_from_current': 'OTM %', 
                                    'annualized_return': 'Annual Return',
                                    'implied_volatility': 'IV'},
                             title="Return vs. Distance OTM")
            
            fig.update_layout(
                xaxis_title="Distance OTM",
                yaxis_title="Annual Return",
                xaxis_tickformat='.0%',
                yaxis_tickformat='.0%',
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Average Return by Symbol")
            
            # Calculate average return by symbol
            symbol_returns = csp_data.groupby('symbol')['annualized_return'].mean().reset_index()
            symbol_returns = symbol_returns.sort_values('annualized_return', ascending=False).head(10)
            
            fig = px.bar(symbol_returns, 
                         x='symbol', 
                         y='annualized_return',
                         color='annualized_return',
                         color_continuous_scale='Viridis',
                         labels={'symbol': 'Symbol', 'annualized_return': 'Avg. Annual Return'},
                         title="Top Symbols by Average Return")
            
            fig.update_layout(
                xaxis_title="Symbol",
                yaxis_title="Avg. Annual Return",
                yaxis_tickformat='.0%',
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Market overview section
    st.subheader("Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Implied volatility chart
        iv_data = pd.DataFrame({
            'Date': pd.date_range(start='2025-02-01', end='2025-02-25'),
            'VIX': 15 + np.random.uniform(-2, 5, 25).cumsum()
        })
        
        fig = px.line(iv_data, x='Date', y='VIX', title="VIX (Market Volatility)")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sector performance
        sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer', 'Utilities', 'Materials']
        performance = np.random.uniform(-5, 8, len(sectors))
        
        sector_data = pd.DataFrame({
            'Sector': sectors,
            'Performance': performance
        })
        
        fig = px.bar(sector_data, 
                     x='Sector', 
                     y='Performance',
                     color='Performance',
                     color_continuous_scale='RdBu',
                     title="Sector Performance (MTD)")
        
        fig.update_layout(
            yaxis_title="% Change",
            yaxis_tickformat='.1%',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategy insights
    st.subheader("Strategy Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="highlight">
                <h4>Covered Call Strategy</h4>
                <p>The current market shows strong arbitrage opportunities in technology and healthcare sectors. 
                   Average returns have increased by 2.3% since last week, with AMD showing the highest potential returns.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="highlight">
                <h4>Cash-Secured Put Strategy</h4>
                <p>Financial sector stocks are showing attractive put premiums with modest risk profiles.
                   The 30-45 day expiration window currently offers the best risk/reward ratio based on current implied volatility levels.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Recent activity
    st.subheader("Recent Activity")
    
    activity_data = [
        {"time": "10:15 AM", "event": "Scan completed: Found 72 opportunities"},
        {"time": "09:30 AM", "event": "Market open: VIX at 16.2"},
        {"time": "08:45 AM", "event": "Added NVDA to watchlist"},
        {"time": "Yesterday", "event": "Closed AAPL covered call position (+2.4%)"},
        {"time": "Yesterday", "event": "Scanned for opportunities: Found 65 trades"}
    ]
    
    activity_df = pd.DataFrame(activity_data)
    st.dataframe(activity_df, use_container_width=True, hide_index=True)

# Covered Calls page
def display_covered_calls():
    st.title("Covered Call Opportunities")
    
    # Generate mock data
    data = generate_mock_data("covered_call", 100)
    
    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_annual = st.number_input("Min Annual Return", value=0.10, format="%.2f")
    
    with col2:
        max_days = st.number_input("Max Days to Expiry", value=45)
    
    with col3:
        min_downside = st.number_input("Min Downside Protection", value=0.05, format="%.2f")
    
    with col4:
        symbols = st.multiselect("Filter Symbols", options=sorted(data['symbol'].unique()), default=[])
    
    # Apply filters
    filtered_data = data.copy()
    
    filtered_data = filtered_data[filtered_data['annualized_return'] >= min_annual]
    filtered_data = filtered_data[filtered_data['days_to_expiry'] <= max_days]
    filtered_data = filtered_data[filtered_data['distance_from_current'] >= min_otm]
    
    if symbols:
        filtered_data = filtered_data[filtered_data['symbol'].isin(symbols)]
    
    # Sort by annual return
    filtered_data = filtered_data.sort_values('annualized_return', ascending=False)
    
    # Display results
    st.subheader(f"Found {len(filtered_data)} Opportunities")
    
    # Format data for display
    display_data = filtered_data.copy()
    
    # Format percentages
    display_data['return'] = display_data['return'].apply(lambda x: f"{x:.2%}")
    display_data['annualized_return'] = display_data['annualized_return'].apply(lambda x: f"{x:.2%}")
    display_data['distance_from_current'] = display_data['distance_from_current'].apply(lambda x: f"{x:.2%}")
    display_data['discount_to_current'] = display_data['discount_to_current'].apply(lambda x: f"{x:.2%}")
    display_data['implied_volatility'] = display_data['implied_volatility'].apply(lambda x: f"{x:.1%}")
    
    # Format currency
    display_data['stock_price'] = display_data['stock_price'].apply(lambda x: f"${x:.2f}")
    display_data['strike'] = display_data['strike'].apply(lambda x: f"${x:.2f}")
    display_data['put_price'] = display_data['put_price'].apply(lambda x: f"${x:.2f}")
    display_data['premium'] = display_data['premium'].apply(lambda x: f"${x:.2f}")
    display_data['cash_required'] = display_data['cash_required'].apply(lambda x: f"${x:.2f}")
    display_data['effective_cost_basis'] = display_data['effective_cost_basis'].apply(lambda x: f"${x:.2f}")
    
    # Select and rename columns
    columns = ['symbol', 'stock_price', 'strike', 'expiration_date', 'days_to_expiry', 
               'put_price', 'premium', 'cash_required', 'annualized_return', 
               'distance_from_current', 'discount_to_current', 'implied_volatility']
    
    column_names = ['Symbol', 'Stock Price', 'Strike', 'Expiration', 'Days',
                   'Put Price', 'Premium', 'Cash Required', 'Annual Return',
                   'OTM %', 'Discount', 'IV']
    
    display_df = display_data[columns].copy()
    display_df.columns = column_names
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Return Analysis", "Risk Metrics", "Opportunity Distribution"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Return vs Days to Expiry
            fig = px.scatter(filtered_data, 
                             x='days_to_expiry', 
                             y='annualized_return',
                             color='implied_volatility',
                             size='premium',
                             hover_name='symbol',
                             hover_data=['strike', 'expiration_date'],
                             title="Return vs. Days to Expiry")
            
            fig.update_layout(
                xaxis_title="Days to Expiry",
                yaxis_title="Annual Return",
                yaxis_tickformat='.0%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Return vs OTM Distance
            fig = px.scatter(filtered_data, 
                             x='distance_from_current', 
                             y='annualized_return',
                             color='days_to_expiry',
                             size='premium',
                             hover_name='symbol',
                             title="Return vs. OTM Distance")
            
            fig.update_layout(
                xaxis_title="OTM Distance",
                yaxis_title="Annual Return",
                xaxis_tickformat='.0%',
                yaxis_tickformat='.0%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Discount vs Return
            fig = px.scatter(filtered_data, 
                             x='discount_to_current', 
                             y='annualized_return',
                             color='days_to_expiry',
                             size='premium',
                             hover_name='symbol',
                             title="Effective Discount vs. Return")
            
            fig.update_layout(
                xaxis_title="Effective Discount to Current Price",
                yaxis_title="Annual Return",
                xaxis_tickformat='.0%',
                yaxis_tickformat='.0%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Premium vs Cash Required
            fig = px.scatter(filtered_data, 
                             x='cash_required', 
                             y='premium',
                             color='annualized_return',
                             hover_name='symbol',
                             log_x=True,
                             title="Premium vs. Cash Required")
            
            fig.update_layout(
                xaxis_title="Cash Required ($)",
                yaxis_title="Premium ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Opportunities by Symbol
            symbol_counts = filtered_data['symbol'].value_counts().reset_index()
            symbol_counts.columns = ['symbol', 'count']
            
            fig = px.bar(symbol_counts.head(15), 
                         x='symbol', 
                         y='count',
                         title="Opportunities by Symbol")
            
            fig.update_layout(
                xaxis_title="Symbol",
                yaxis_title="Number of Opportunities",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Return Heatmap by Days and OTM
            # Create bins for days and OTM
            filtered_data['days_bin'] = pd.cut(filtered_data['days_to_expiry'], 
                                              bins=[0, 7, 14, 21, 30, 45],
                                              labels=['1-7', '8-14', '15-21', '22-30', '31-45'])
            
            filtered_data['otm_bin'] = pd.cut(filtered_data['distance_from_current'], 
                                            bins=[0, 0.05, 0.1, 0.15, 0.2, 1],
                                            labels=['0-5%', '5-10%', '10-15%', '15-20%', '>20%'])
            
            # Group by bins and calculate average return
            heatmap_data = filtered_data.groupby(['days_bin', 'otm_bin'])['annualized_return'].mean().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='otm_bin', columns='days_bin', values='annualized_return')
            
            fig = px.imshow(heatmap_pivot, 
                           labels=dict(x="Days to Expiry", y="OTM Distance", color="Avg Return"),
                           x=heatmap_pivot.columns,
                           y=heatmap_pivot.index,
                           color_continuous_scale="Viridis",
                           title="Return Heatmap by Days and OTM Distance")
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)

# Watchlist page
def display_watchlist():
    st.title("Options Watchlist")
    
    # Mock watchlist data
    watchlist_data = {
        'symbol': ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN', 'GOOGL'],
        'last_price': [189.34, 415.67, 202.64, 845.92, 178.45, 482.23, 178.32, 165.82],
        'price_change': [2.45, -1.32, 5.67, 12.45, -3.21, 6.78, -0.42, 1.56],
        'percent_change': [1.31, -0.32, 2.88, 1.49, -1.77, 1.43, -0.24, 0.95],
        'option_alert': [True, False, True, True, False, False, False, False]
    }
    
    watchlist_df = pd.DataFrame(watchlist_data)
    
    # Format for display
    display_df = watchlist_df.copy()
    display_df['last_price'] = display_df['last_price'].apply(lambda x: f"${x:.2f}")
    
    # Format price change with colors
    def format_change(row):
        change = row['price_change']
        pct = row['percent_change']
        
        if change > 0:
            return f"<span style='color:green'>+${change:.2f} (+{pct:.2f}%)</span>"
        else:
            return f"<span style='color:red'>${change:.2f} ({pct:.2f}%)</span>"
    
    display_df['price_change'] = watchlist_df.apply(format_change, axis=1)
    
    # Format alerts
    def format_alert(has_alert):
        if has_alert:
            return "🔔 New Opportunity"
        else:
            return ""
    
    display_df['option_alert'] = display_df['option_alert'].apply(format_alert)
    
    # Select columns for display
    display_df = display_df[['symbol', 'last_price', 'price_change', 'option_alert']]
    display_df.columns = ['Symbol', 'Last Price', 'Change', 'Alert']
    
    # Create custom dataframe with styled HTML
    st.write(
        """
        <div class="table-container">
            <table style="width:100%">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Last Price</th>
                        <th>Change</th>
                        <th>Alert</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
        """, 
        unsafe_allow_html=True
    )
    
    for index, row in display_df.iterrows():
        st.write(
            f"""
            <tr>
                <td><strong>{row['Symbol']}</strong></td>
                <td>{row['Last Price']}</td>
                <td>{row['Change']}</td>
                <td>{row['Alert']}</td>
                <td>
                    <button style="background-color:#4CAF50; color:white; border:none; border-radius:4px; padding:2px 8px; margin-right:5px;">Scan</button>
                    <button style="background-color:#f44336; color:white; border:none; border-radius:4px; padding:2px 8px;">Remove</button>
                </td>
            </tr>
            """, 
            unsafe_allow_html=True
        )
    
    st.write(
        """
                </tbody>
            </table>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Add new symbol form
    st.subheader("Add Symbol to Watchlist")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_symbol = st.text_input("Symbol", placeholder="Enter stock symbol (e.g., AAPL)")
    
    with col2:
        st.write("<br>", unsafe_allow_html=True)
        add_button = st.button("Add to Watchlist", type="primary")
    
    # Sample alerts
    st.subheader("Recent Alerts")
    
    alerts = [
        {"time": "10:15 AM", "symbol": "TSLA", "alert": "New covered call opportunity: 24.5% annual return"},
        {"time": "09:45 AM", "symbol": "AAPL", "alert": "New cash-secured put opportunity: 18.2% annual return"},
        {"time": "Yesterday", "symbol": "NVDA", "alert": "IV spike detected (42% → 55%)"}
    ]
    
    for alert in alerts:
        st.markdown(
            f"""
            <div style="background-color:#f8f8f8; padding:10px; border-radius:5px; margin-bottom:10px;">
                <span style="color:#666; font-size:12px;">{alert['time']}</span>
                <span style="color:#1565C0; font-weight:bold; margin-left:10px;">{alert['symbol']}</span>
                <span style="margin-left:10px;">{alert['alert']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Configuration options
    st.subheader("Alert Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Alert on covered call opportunities", value=True)
        st.checkbox("Alert on cash-secured put opportunities", value=True)
        st.checkbox("Alert on IV changes", value=True)
    
    with col2:
        st.slider("Minimum annual return for alerts", 0.1, 0.5, 0.15, 0.05, format="%.2f")
        st.slider("Minimum IV change for alerts", 0.05, 0.3, 0.1, 0.05, format="%.2f")

# Settings page
def display_settings():
    st.title("Application Settings")
    
    # API settings
    st.subheader("API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input("Schwab API Key", type="password", value="•••••••••••••••••")
    
    with col2:
        api_secret = st.text_input("Schwab API Secret", type="password", value="•••••••••••••••••")
    
    # Default scanner settings
    st.subheader("Default Scanner Settings")
    
    tab1, tab2 = st.tabs(["Covered Call Defaults", "Cash-Secured Put Defaults"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Default Min Return", value=0.15, format="%.2f")
            st.number_input("Default Max Days", value=45)
            st.number_input("Default Safety Margin", value=0.01, format="%.2f")
        
        with col2:
            st.number_input("Default Min Vol", value=50)
            st.number_input("Default Min Open Interest", value=50)
            st.number_input("Default Min Downside Protection", value=0.05, format="%.2f")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Default Min Return", value=0.15, format="%.2f", key="csp_min_return")
            st.number_input("Default Max Days", value=45, key="csp_max_days")
            st.number_input("Default Min OTM %", value=0.05, format="%.2f")
        
        with col2:
            st.number_input("Default Min Vol", value=50, key="csp_min_vol")
            st.number_input("Default Min Open Interest", value=50, key="csp_min_oi")
            st.number_input("Default Min Discount", value=0.1, format="%.2f")
    
    # Auto-scan settings
    st.subheader("Auto-Scan Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Enable auto-scanning", value=True)
        st.slider("Auto-scan interval (minutes)", 5, 60, 15, 5)
    
    with col2:
        st.multiselect(
            "Auto-scan symbols", 
            options=["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "AMD"],
            default=["AAPL", "MSFT", "AMZN"]
        )
        st.radio("Scan type", options=["Watchlist Only", "All Symbols"], index=0)
    
    # UI settings
    st.subheader("Display Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Theme", options=["Light", "Dark", "System Default"], index=0)
        st.selectbox("Date Format", options=["MM/DD/YYYY", "YYYY-MM-DD", "DD/MM/YYYY"], index=1)
    
    with col2:
        st.selectbox("Default Tab", options=["Dashboard", "Covered Calls", "Cash-Secured Puts"], index=0)
        st.checkbox("Show additional metrics", value=True)
    
    # Save button
    save_button = st.button("Save Settings", type="primary")
    
    # App info
    st.markdown("""
    <div style="margin-top: 50px; color: #666; font-size: 12px;">
        <p>Options Arbitrage Scanner v1.0.0</p>
        <p>© 2025 | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()data = data.copy()
    
    filtered_data = filtered_data[filtered_data['annualized_return'] >= min_annual]
    filtered_data = filtered_data[filtered_data['days_to_expiry'] <= max_days]
    filtered_data = filtered_data[filtered_data['downside_protection'] >= min_downside]
    
    if symbols:
        filtered_data = filtered_data[filtered_data['symbol'].isin(symbols)]
    
    # Sort by annual return
    filtered_data = filtered_data.sort_values('annualized_return', ascending=False)
    
    # Display results
    st.subheader(f"Found {len(filtered_data)} Opportunities")
    
    # Format data for display
    display_data = filtered_data.copy()
    
    # Format percentages
    display_data['return'] = display_data['return'].apply(lambda x: f"{x:.2%}")
    display_data['annualized_return'] = display_data['annualized_return'].apply(lambda x: f"{x:.2%}")
    display_data['downside_protection'] = display_data['downside_protection'].apply(lambda x: f"{x:.2%}")
    display_data['implied_volatility'] = display_data['implied_volatility'].apply(lambda x: f"{x:.1%}")
    
    # Format currency
    display_data['stock_price'] = display_data['stock_price'].apply(lambda x: f"${x:.2f}")
    display_data['strike'] = display_data['strike'].apply(lambda x: f"${x:.2f}")
    display_data['call_price'] = display_data['call_price'].apply(lambda x: f"${x:.2f}")
    display_data['net_debit'] = display_data['net_debit'].apply(lambda x: f"${x:.2f}")
    display_data['profit'] = display_data['profit'].apply(lambda x: f"${x:.2f}")
    
    # Select and rename columns
    columns = ['symbol', 'stock_price', 'strike', 'expiration_date', 'days_to_expiry', 
               'call_price', 'net_debit', 'profit', 'annualized_return', 
               'downside_protection', 'implied_volatility']
    
    column_names = ['Symbol', 'Stock Price', 'Strike', 'Expiration', 'Days',
                    'Call Price', 'Net Debit', 'Profit', 'Annual Return',
                    'Downside Protection', 'IV']
    
    display_df = display_data[columns].copy()
    display_df.columns = column_names
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Return Analysis", "Risk Metrics", "Opportunity Distribution"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Return vs Days to Expiry
            fig = px.scatter(filtered_data, 
                             x='days_to_expiry', 
                             y='annualized_return',
                             color='implied_volatility',
                             size='profit',
                             hover_name='symbol',
                             hover_data=['strike', 'expiration_date'],
                             title="Return vs. Days to Expiry")
            
            fig.update_layout(
                xaxis_title="Days to Expiry",
                yaxis_title="Annual Return",
                yaxis_tickformat='.0%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Return Distribution
            fig = px.histogram(filtered_data, 
                              x='annualized_return',
                              nbins=20,
                              title="Distribution of Annual Returns")
            
            fig.update_layout(
                xaxis_title="Annual Return",
                yaxis_title="Count",
                xaxis_tickformat='.0%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Downside Protection vs Return
            fig = px.scatter(filtered_data, 
                             x='downside_protection', 
                             y='annualized_return',
                             color='days_to_expiry',
                             size='profit',
                             hover_name='symbol',
                             title="Downside Protection vs. Return")
            
            fig.update_layout(
                xaxis_title="Downside Protection",
                yaxis_title="Annual Return",
                xaxis_tickformat='.0%',
                yaxis_tickformat='.0%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time Value Chart
            fig = px.scatter(filtered_data, 
                             x='intrinsic_value', 
                             y='time_value',
                             color='days_to_expiry',
                             hover_name='symbol',
                             title="Option Premium Breakdown")
            
            fig.update_layout(
                xaxis_title="Intrinsic Value",
                yaxis_title="Time Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Opportunities by Symbol
            symbol_counts = filtered_data['symbol'].value_counts().reset_index()
            symbol_counts.columns = ['symbol', 'count']
            
            fig = px.bar(symbol_counts.head(15), 
                         x='symbol', 
                         y='count',
                         title="Opportunities by Symbol")
            
            fig.update_layout(
                xaxis_title="Symbol",
                yaxis_title="Number of Opportunities",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Return Heatmap by Days and Protection
            # Create bins for days and protection
            filtered_data['days_bin'] = pd.cut(filtered_data['days_to_expiry'], 
                                              bins=[0, 7, 14, 21, 30, 45],
                                              labels=['1-7', '8-14', '15-21', '22-30', '31-45'])
            
            filtered_data['protection_bin'] = pd.cut(filtered_data['downside_protection'], 
                                                   bins=[0, 0.05, 0.1, 0.15, 0.2, 1],
                                                   labels=['0-5%', '5-10%', '10-15%', '15-20%', '>20%'])
            
            # Group by bins and calculate average return
            heatmap_data = filtered_data.groupby(['days_bin', 'protection_bin'])['annualized_return'].mean().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='protection_bin', columns='days_bin', values='annualized_return')
            
            fig = px.imshow(heatmap_pivot, 
                           labels=dict(x="Days to Expiry", y="Downside Protection", color="Avg Return"),
                           x=heatmap_pivot.columns,
                           y=heatmap_pivot.index,
                           color_continuous_scale="Viridis",
                           title="Return Heatmap by Days and Protection")
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)

# Cash-Secured Puts page
def display_cash_secured_puts():
    st.title("Cash-Secured Put Opportunities")
    
    # Generate mock data
    data = generate_mock_data("cash_secured_put", 100)
    
    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_annual = st.number_input("Min Annual Return", value=0.10, format="%.2f")
    
    with col2:
        max_days = st.number_input("Max Days to Expiry", value=45)
    
    with col3:
        min_otm = st.number_input("Min OTM %", value=0.05, format="%.2f")
    
    with col4:
        symbols = st.multiselect("Filter Symbols", options=sorted(data['symbol'].unique()), default=[])
    
    # Apply filters
    filtered_
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
        
    
