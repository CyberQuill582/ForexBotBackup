import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_handler import DataHandler
from trading_strategy import TradingStrategy
from ml_predictor import MLPredictor
from utils import setup_logging, calculate_performance_metrics
import logging

# Setup logging
logger = setup_logging()

# Page config
st.set_page_config(page_title="Forex Trading Bot", layout="wide")

# Initialize components
data_handler = DataHandler()
trading_strategy = TradingStrategy()
ml_predictor = MLPredictor()

# Sidebar
st.sidebar.title("Trading Bot Controls")
symbol = st.sidebar.selectbox(
    "Select Trading Pair",
    ["USD/JPY", "^IXIC"]  # IXIC is NASDAQ
)

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["1d", "1h", "15m"]
)

# Main content
st.title("Forex Trading Bot Dashboard")

# Fetch and process data
try:
    df = data_handler.fetch_market_data(symbol, timeframe)
    
    # Generate signals
    signals = trading_strategy.generate_signals(df)
    
    # ML predictions
    predictions = ml_predictor.predict(df)
    
    # Combine signals and predictions
    final_signal = trading_strategy.combine_signals(signals, predictions)
    
    # Display charts
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="OHLC"
    ))
    
    # Add signals to chart
    for idx, signal in enumerate(final_signal):
        if signal == 1:  # Buy signal
            fig.add_trace(go.Scatter(
                x=[df.index[idx]],
                y=[df['Low'].iloc[idx]],
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                name='Buy Signal'
            ))
        elif signal == -1:  # Sell signal
            fig.add_trace(go.Scatter(
                x=[df.index[idx]],
                y=[df['High'].iloc[idx]],
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                name='Sell Signal'
            ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart with Signals",
        yaxis_title="Price",
        xaxis_title="Date"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    metrics = calculate_performance_metrics(df, final_signal)
    
    col1.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
    col2.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    col3.metric("Sharp Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    # Recent trades
    st.subheader("Recent Trades")
    st.dataframe(metrics['recent_trades'])
    
    # Risk management stats
    st.subheader("Risk Management")
    st.write(f"Current Drawdown: {metrics['current_drawdown']:.2f}%")
    st.write(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
except Exception as e:
    logger.error(f"Error in main app: {str(e)}")
    st.error(f"An error occurred: {str(e)}")
