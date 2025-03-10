import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from data_handler import DataHandler
from trading_strategy import TradingStrategy
from ml_predictor import MLPredictor
from backtester import Backtester
from trade_executor import TradeExecutor
from utils import setup_logging, calculate_performance_metrics, calculate_monthly_returns, calculate_max_drawdown, calculate_current_drawdown
from ai_optimizer import AIOptimizer
import logging
import os

# Setup logging
logger = setup_logging()

# Page config
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize components
data_handler = DataHandler()
trading_strategy = TradingStrategy()
ml_predictor = MLPredictor(retrain_frequency=7)  # Retrain every 7 days by default
backtester = Backtester()
trade_executor = TradeExecutor()
ai_optimizer = AIOptimizer()

# Session state initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.backtest_results = None
    st.session_state.trade_executor_initialized = False
    trade_executor.set_capital(100000)  # Set default capital
    st.session_state.trade_executor = trade_executor

# Function to apply custom CSS
def apply_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            border: 1px solid #e0e0e0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50;
            color: white;
        }
        .metric-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .chart-container {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 15px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Sidebar
st.sidebar.title("Trading Bot Controls")

sidebar_tab = st.sidebar.radio("Configuration", ["Market Data", "Strategy", "Backtest", "Live Trading"])

if sidebar_tab == "Market Data":
    st.sidebar.subheader("Market Data Settings")
    symbol = st.sidebar.selectbox(
        "Select Trading Pair",
        ["USD/JPY", "EUR/USD", "GBP/USD", "^IXIC", "^GSPC"]  # Added more options
    )

    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["1d", "1h", "15m", "5m"]
    )
    
    lookback_period = st.sidebar.slider(
        "Lookback Period",
        min_value=30,
        max_value=500,
        value=200,
        step=10
    )
    
    additional_indicators = st.sidebar.checkbox("Calculate Additional Indicators", value=True)

elif sidebar_tab == "Strategy":
    st.sidebar.subheader("Strategy Settings")
    
    st.sidebar.text("Technical Indicators Settings")
    
    macd_fast = st.sidebar.slider("MACD Fast Period", 8, 20, 12)
    macd_slow = st.sidebar.slider("MACD Slow Period", 21, 30, 26)
    macd_signal = st.sidebar.slider("MACD Signal Period", 5, 15, 9)
    
    rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)
    rsi_overbought = st.sidebar.slider("RSI Overbought Level", 70, 85, 70)
    rsi_oversold = st.sidebar.slider("RSI Oversold Level", 15, 30, 30)
    
    st.sidebar.text("Machine Learning Settings")
    
    optimize_ml = st.sidebar.checkbox("Optimize ML Hyperparameters", value=False)
    feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
    force_retrain = st.sidebar.checkbox("Force Model Retraining", value=False)
    retrain_frequency = st.sidebar.slider("Model Retraining Frequency (days)", 0, 30, 7, 
                                        help="0 means manual retraining only, otherwise retrain every N days")
    
    st.sidebar.text("Signal Combination Settings")
    
    tech_weight = st.sidebar.slider("Technical Signals Weight", 0.0, 1.0, 0.6, 0.05)
    ml_weight = 1.0 - tech_weight
    
    trend_filter = st.sidebar.checkbox("Use Trend Filter", value=True)

elif sidebar_tab == "Backtest":
    st.sidebar.subheader("Backtest Settings")
    
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=100000, step=1000)
    risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.1) / 100
    
    position_sizing = st.sidebar.selectbox(
        "Position Sizing Method",
        ["Fixed Risk", "Kelly Criterion", "Fixed Percentage"]
    )
    
    backtest_period = st.sidebar.selectbox(
        "Backtest Period",
        ["3 Months", "6 Months", "1 Year", "2 Years", "All Data"]
    )
    
    run_backtest = st.sidebar.button("Run Backtest")

elif sidebar_tab == "Live Trading":
    st.sidebar.subheader("Live Trading Settings")
    
    if not st.session_state.trade_executor_initialized:
        live_capital = st.sidebar.number_input("Trading Capital", min_value=1000, value=100000, step=1000)
        risk_per_trade_live = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.1) / 100
        max_open_positions = st.sidebar.slider("Maximum Open Positions", 1, 10, 3)
        
        initialize_trading = st.sidebar.button("Initialize Live Trading")
        
        if initialize_trading:
            trade_executor = TradeExecutor(risk_per_trade=risk_per_trade_live, max_open_positions=max_open_positions)
            trade_executor.set_capital(live_capital)
            st.session_state.trade_executor = trade_executor
            st.session_state.trade_executor_initialized = True
    else:
        st.sidebar.success("Live trading initialized")
        st.sidebar.write(f"Capital: ${st.session_state.trade_executor.capital:.2f}")
        st.sidebar.write(f"Open Positions: {len(st.session_state.trade_executor.open_positions)}")
        
        if st.sidebar.button("Close All Positions"):
            # We would need current prices, simplified here
            current_prices = {'USD/JPY': 150.0, 'EUR/USD': 1.09, 'GBP/USD': 1.27, '^IXIC': 16000, '^GSPC': 5000}
            st.session_state.trade_executor.close_all_positions(current_prices)
            st.sidebar.success("All positions closed")

# Main tabs
tabs = st.tabs(["Dashboard", "Backtest Results", "Strategy Analysis", "Market Analysis", "Live Trading"])

with tabs[0]:  # Dashboard
    st.title("Trading Bot Dashboard")
    
    # Fetch and process data
    try:
        if sidebar_tab != "Market Data":
            # Use default values if not in Market Data tab
            symbol = "USD/JPY"
            timeframe = "1d"
            lookback_period = 200
            additional_indicators = True
            
        df = data_handler.fetch_market_data(symbol, timeframe, additional_indicators)
        
        # Limit data to lookback period
        if len(df) > lookback_period:
            df = df.iloc[-lookback_period:]
        
        # Apply strategy settings if in Strategy tab
        if sidebar_tab == "Strategy":
            # Could apply custom indicator settings here
            pass
        
        # Generate signals
        signals = trading_strategy.generate_signals(df)
        
        # ML predictions
        if sidebar_tab == "Strategy":
            # Update retraining frequency
            ml_predictor.retrain_frequency = retrain_frequency
            predictions = ml_predictor.predict(df, optimize=optimize_ml, force_retrain=force_retrain)
        else:
            predictions = ml_predictor.predict(df)
        
        # Combine signals and predictions
        if sidebar_tab == "Strategy":
            final_signal = trading_strategy.combine_signals(signals, predictions, weights=(tech_weight, ml_weight))
        else:
            final_signal = trading_strategy.combine_signals(signals, predictions)
        
        # Performance metrics
        metrics = calculate_performance_metrics(df, final_signal)
        
        # Display top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Win Rate", f"{metrics['win_rate']:.2f}%", f"{metrics['win_rate'] - 50:.1f}%" if metrics['win_rate'] > 50 else f"{metrics['win_rate'] - 50:.1f}%")
        col2.metric("Profit Factor", f"{metrics['profit_factor']:.2f}", f"{metrics['profit_factor'] - 1:.2f}" if metrics['profit_factor'] > 1 else f"{metrics['profit_factor'] - 1:.2f}")
        col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", f"{metrics['sharpe_ratio'] - 1:.2f}" if metrics['sharpe_ratio'] > 1 else f"{metrics['sharpe_ratio'] - 1:.2f}")
        col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%", f"{0 - metrics['max_drawdown']:.2f}%")
        
        # Display chart
        st.subheader(f"{symbol} Price Chart with Signals")
        
        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.03, 
                          row_heights=[0.7, 0.3],
                          specs=[[{"type": "candlestick"}],
                                [{"type": "scatter"}]])
        
        # Add price candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ), row=1, col=1)
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            line=dict(color='blue', width=1),
            name="SMA 20"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            line=dict(color='orange', width=1),
            name="SMA 50"
        ), row=1, col=1)
        
        # If we have Bollinger Bands
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                line=dict(color='rgba(0,176,246,0.2)', width=1),
                name="BB Upper"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                line=dict(color='rgba(0,176,246,0.2)', width=1),
                fill='tonexty',
                name="BB Lower"
            ), row=1, col=1)
        
        # Add signals to chart
        buy_signals_x = []
        buy_signals_y = []
        sell_signals_x = []
        sell_signals_y = []
        
        for idx, signal in enumerate(final_signal):
            if signal == 1 and idx > 0:  # Buy signal
                buy_signals_x.append(df.index[idx])
                buy_signals_y.append(df['Low'].iloc[idx] * 0.99)  # Slightly below low
            elif signal == -1 and idx > 0:  # Sell signal
                sell_signals_x.append(df.index[idx])
                sell_signals_y.append(df['High'].iloc[idx] * 1.01)  # Slightly above high
        
        fig.add_trace(go.Scatter(
            x=buy_signals_x,
            y=buy_signals_y,
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='green'),
            name='Buy Signal'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=sell_signals_x,
            y=sell_signals_y,
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Sell Signal'
        ), row=1, col=1)
        
        # Add RSI indicator
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            line=dict(color='purple', width=1),
            name="RSI"
        ), row=2, col=1)
        
        # Add RSI overbought/oversold lines
        fig.add_trace(go.Scatter(
            x=df.index,
            y=[70] * len(df),
            line=dict(color='red', width=1, dash='dash'),
            name="Overbought"
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=[30] * len(df),
            line=dict(color='green', width=1, dash='dash'),
            name="Oversold"
        ), row=2, col=1)
        
        # Add center line for RSI
        fig.add_trace(go.Scatter(
            x=df.index,
            y=[50] * len(df),
            line=dict(color='gray', width=1, dash='dot'),
            name="RSI Center"
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Technical Analysis",
            yaxis_title="Price",
            xaxis_title="Date",
            height=800,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axis for RSI
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Management")
            
            risk_metrics = {
                "Current Drawdown": f"{metrics['current_drawdown']:.2f}%",
                "Max Drawdown": f"{metrics['max_drawdown']:.2f}%",
                "Volatility": f"{metrics.get('volatility', 0):.2f}%",
                "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
                "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.2f}",
                "Calmar Ratio": f"{metrics.get('calmar_ratio', 0):.2f}",
                "Recovery Factor": f"{metrics.get('recovery_factor', 0):.2f}"
            }
            
            st.table(pd.DataFrame(list(risk_metrics.items()), columns=["Metric", "Value"]))
        
        with col2:
            st.subheader("Trade Statistics")
            
            trade_metrics = {
                "Total Trades": f"{metrics.get('total_trades', 0)}",
                "Winning Trades": f"{metrics.get('winning_trades', 0)}",
                "Losing Trades": f"{metrics.get('losing_trades', 0)}",
                "Win Rate": f"{metrics['win_rate']:.2f}%",
                "Profit Factor": f"{metrics['profit_factor']:.2f}",
                "Average Profit": f"{metrics.get('average_profit', 0):.2f}%",
                "Average Loss": f"{metrics.get('average_loss', 0):.2f}%",
                "Average Duration": f"{metrics.get('average_duration', 0):.2f} days"
            }
            
            st.table(pd.DataFrame(list(trade_metrics.items()), columns=["Metric", "Value"]))
        
        # Recent trades
        st.subheader("Recent Trades")
        if 'recent_trades' in metrics and not metrics['recent_trades'].empty:
            st.dataframe(metrics['recent_trades'], height=200)
        else:
            st.info("No recent trades to display")
        
    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

with tabs[1]:  # Backtest Results
    st.title("Backtest Results")
    
    # Run backtest if button is clicked
    if sidebar_tab == "Backtest" and run_backtest:
        try:
            st.info("Running backtest...")
            
            # Determine backtest period
            end_date = datetime.now()
            if backtest_period == "3 Months":
                start_date = end_date - timedelta(days=90)
            elif backtest_period == "6 Months":
                start_date = end_date - timedelta(days=180)
            elif backtest_period == "1 Year":
                start_date = end_date - timedelta(days=365)
            elif backtest_period == "2 Years":
                start_date = end_date - timedelta(days=365*2)
            else:  # All Data
                start_date = end_date - timedelta(days=365*5)  # 5 years
            
            # Fetch data for backtest
            df = data_handler.fetch_market_data(symbol, timeframe, additional_indicators=True)
            
            # Filter by date range
            df = df[df.index >= pd.Timestamp(start_date)]
            
            # Run backtest
            backtest_results = backtester.run_backtest(
                df, 
                initial_capital=initial_capital,
                position_size=risk_per_trade
            )
            
            # Store in session state
            st.session_state.backtest_results = backtest_results
            
            st.success("Backtest completed successfully!")
        
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            st.error(f"An error occurred during backtest: {str(e)}")
    
    # Display backtest results
    if st.session_state.backtest_results is not None:
        results = st.session_state.backtest_results
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Return", f"{results['total_return']:.2f}%")
        col2.metric("Annual Return", f"{results['annual_return']:.2f}%")
        col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        col4.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
        
        # Equity curve
        st.subheader("Equity Curve")
        
        fig = go.Figure()
        
        if 'equity_curve' in results:
            fig.add_trace(go.Scatter(
                x=results['equity_curve'].index,
                y=results['equity_curve']['Equity'],
                line=dict(color='blue', width=2),
                name="Equity"
            ))
            
            fig.update_layout(
                title="Equity Curve",
                yaxis_title="Equity ($)",
                xaxis_title="Date",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.subheader("Drawdown")
        
        fig_dd = go.Figure()
        
        if 'drawdown_curve' in results:
            fig_dd.add_trace(go.Scatter(
                x=results['drawdown_curve'].index,
                y=results['drawdown_curve']['Drawdown'],
                line=dict(color='red', width=2),
                fill='tozeroy',
                name="Drawdown"
            ))
            
            fig_dd.update_layout(
                title="Drawdown",
                yaxis_title="Drawdown (%)",
                xaxis_title="Date",
                height=300
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        
        # Trade metrics in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            
            perf_metrics = {
                "Total Return": f"{results['total_return']:.2f}%",
                "Annual Return": f"{results['annual_return']:.2f}%",
                "Sharpe Ratio": f"{results['sharpe_ratio']:.2f}",
                "Calmar Ratio": f"{results.get('calmar_ratio', 0):.2f}",
                "Max Drawdown": f"{results['max_drawdown']:.2f}%",
                "Volatility": f"{results.get('volatility', 0):.2f}%",
                "Recovery Factor": f"{results.get('recovery_factor', 0):.2f}"
            }
            
            st.table(pd.DataFrame(list(perf_metrics.items()), columns=["Metric", "Value"]))
        
        with col2:
            st.subheader("Trade Statistics")
            
            trade_stats = {
                "Total Trades": f"{results['trade_count']}",
                "Win Rate": f"{results['win_rate']:.2f}%",
                "Profit Factor": f"{results['profit_factor']:.2f}",
                "Expectancy": f"{results.get('expectancy', 0):.2f}%",
                "System Quality": f"{results.get('system_quality', 0):.2f}",
                "Average Profit": f"{results.get('average_profit', 0):.2f}%",
                "Average Loss": f"{results.get('average_loss', 0):.2f}%"
            }
            
            st.table(pd.DataFrame(list(trade_stats.items()), columns=["Metric", "Value"]))
        
        # Monthly returns heatmap if available
        if 'monthly_returns' in results:
            st.subheader("Monthly Returns Heatmap")
            
            # Convert to DataFrame for heatmap if needed
            try:
                monthly_returns = results['monthly_returns']
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(monthly_returns, annot=True, cmap='RdYlGn', center=0, 
                           fmt=".2f", linewidths=.5, ax=ax)
                ax.set_title('Monthly Returns (%)')
                
                st.pyplot(fig)
            except:
                st.warning("Could not display monthly returns heatmap")
        
        # All trades table
        if 'all_trades' in results and not results['all_trades'].empty:
            st.subheader("Trade List")
            st.dataframe(results['all_trades'])
    else:
        st.info("No backtest results to display. Run a backtest first.")

with tabs[2]:  # Strategy Analysis
    st.title("Strategy Analysis")
    
    try:
        # Display ML Model Information
        st.subheader("ML Model Information")
        
        model_info = {
            "Last Training Date": ml_predictor.last_train_date.strftime("%Y-%m-%d %H:%M:%S") if ml_predictor.last_train_date else "Never",
            "Retraining Frequency": f"{ml_predictor.retrain_frequency} days" if ml_predictor.retrain_frequency > 0 else "Manual only",
            "Model Type": ml_predictor.model.__class__.__name__ if ml_predictor.model else "None"
        }
        
        st.table(pd.DataFrame(list(model_info.items()), columns=["Property", "Value"]))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Force Model Retraining"):
                st.session_state['force_retrain'] = True
                df = data_handler.fetch_market_data(symbol, timeframe, additional_indicators=True)
                predictions = ml_predictor.predict(df, optimize=True, force_retrain=True)
                st.success("Model retrained successfully!")
        
        with col2:
            if st.button("Reset Model"):
                import shutil
                if os.path.exists(ml_predictor.model_path):
                    shutil.rmtree(ml_predictor.model_path)
                    os.makedirs(ml_predictor.model_path)
                ml_predictor.model = RandomForestClassifier(n_estimators=100, random_state=42)
                ml_predictor.scaler = StandardScaler()
                ml_predictor.last_train_date = None
                ml_predictor.feature_importance = None
                st.success("Model reset successfully!")
        
        if 'ml_predictor' in locals() and hasattr(ml_predictor, 'get_feature_importance'):
            feature_imp = ml_predictor.get_feature_importance()
            
            if feature_imp is not None and feature_importance:
                st.subheader("Feature Importance")
                
                # Plot feature importance
                fig = px.bar(feature_imp, x='importance', y='feature', orientation='h',
                           title='Feature Importance in ML Model')
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Signal analysis
        st.subheader("Signal Analysis")
        
        if 'df' in locals() and 'final_signal' in locals():
            # Calculate signal metrics
            signal_df = pd.DataFrame({
                'Date': df.index,
                'Close': df['Close'],
                'Signal': final_signal
            }).set_index('Date')
            
            # Count signal types
            signal_counts = pd.Series(final_signal).value_counts()
            signal_names = {1: 'Buy', -1: 'Sell', 0: 'Hold'}
            signal_counts.index = signal_counts.index.map(signal_names)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a pie chart for signal distribution
                fig = px.pie(values=signal_counts.values, names=signal_counts.index,
                           title='Signal Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Compare signal sources
                if 'signals' in locals() and 'predictions' in locals():
                    agreement = np.sum(signals == predictions) / len(signals) * 100
                    disagreement = 100 - agreement
                    
                    source_comparison = pd.DataFrame({
                        'Percentage': [agreement, disagreement]
                    }, index=['Agreement', 'Disagreement'])
                    
                    fig = px.bar(source_comparison, y='Percentage',
                               title='Technical vs ML Signal Agreement')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Strategy performance by market condition
        st.subheader("Performance by Market Condition")
        
        if 'df' in locals() and 'final_signal' in locals():
            # Identify market conditions (simple trend based)
            df['Trend'] = np.where(df['SMA_20'] > df['SMA_20'].shift(20), 'Uptrend',
                                 np.where(df['SMA_20'] < df['SMA_20'].shift(20), 'Downtrend', 'Sideways'))
            
            # Calculate returns by market condition
            returns_by_condition = pd.DataFrame({
                'Market_Condition': df['Trend'],
                'Strategy_Return': df['Close'].pct_change() * final_signal.shift(1)
            }).dropna()
            
            condition_returns = returns_by_condition.groupby('Market_Condition').agg({
                'Strategy_Return': ['mean', 'std', 'count']
            })
            
            condition_returns.columns = ['Mean Return', 'Std Dev', 'Count']
            condition_returns['Mean Return'] = condition_returns['Mean Return'] * 100
            condition_returns['Std Dev'] = condition_returns['Std Dev'] * 100
            
            st.table(condition_returns)
            
            # Plot returns by market condition
            fig = px.box(returns_by_condition, x='Market_Condition', y='Strategy_Return',
                       title='Strategy Returns by Market Condition')
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        logger.error(f"Error in strategy analysis: {str(e)}")
        st.error(f"An error occurred in strategy analysis: {str(e)}")

with tabs[3]:  # Market Analysis
    st.title("Market Analysis")
    
    try:
        # Multi-timeframe analysis
        st.subheader("Multi-Timeframe Analysis")
        
        timeframes = ["1d", "1h", "15m"]
        tf_data = {}
        
        for tf in timeframes:
            tf_data[tf] = data_handler.fetch_market_data(symbol, tf, additional_indicators=True)
        
        # Display indicators across timeframes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Daily Timeframe")
            daily_signals = trading_strategy.generate_signals(tf_data["1d"])
            daily_trend = "Bullish" if tf_data["1d"]['SMA_20'].iloc[-1] > tf_data["1d"]['SMA_50'].iloc[-1] else "Bearish"
            daily_rsi = tf_data["1d"]['RSI'].iloc[-1]
            
            st.metric("Trend", daily_trend)
            st.metric("RSI", f"{daily_rsi:.2f}")
            st.metric("Signal", "Buy" if daily_signals[-1] == 1 else "Sell" if daily_signals[-1] == -1 else "Neutral")
        
        with col2:
            st.write("Hourly Timeframe")
            hourly_signals = trading_strategy.generate_signals(tf_data["1h"])
            hourly_trend = "Bullish" if tf_data["1h"]['SMA_20'].iloc[-1] > tf_data["1h"]['SMA_50'].iloc[-1] else "Bearish"
            hourly_rsi = tf_data["1h"]['RSI'].iloc[-1]
            
            st.metric("Trend", hourly_trend)
            st.metric("RSI", f"{hourly_rsi:.2f}")
            st.metric("Signal", "Buy" if hourly_signals[-1] == 1 else "Sell" if hourly_signals[-1] == -1 else "Neutral")
        
        with col3:
            st.write("15-Minute Timeframe")
            minute_signals = trading_strategy.generate_signals(tf_data["15m"])
            minute_trend = "Bullish" if tf_data["15m"]['SMA_20'].iloc[-1] > tf_data["15m"]['SMA_50'].iloc[-1] else "Bearish"
            minute_rsi = tf_data["15m"]['RSI'].iloc[-1]
            
            st.metric("Trend", minute_trend)
            st.metric("RSI", f"{minute_rsi:.2f}")
            st.metric("Signal", "Buy" if minute_signals[-1] == 1 else "Sell" if minute_signals[-1] == -1 else "Neutral")
        
        # Correlation analysis with other markets
        st.subheader("Market Correlation Analysis")
        
        # Fetch data for multiple symbols
        correlation_symbols = ["USD/JPY", "EUR/USD", "^IXIC", "^GSPC"]
        correlation_data = {}
        
        for sym in correlation_symbols:
            try:
                correlation_data[sym] = data_handler.fetch_market_data(sym, "1d")['Close']
            except:
                st.warning(f"Could not fetch data for {sym}")
        
        # Create correlation matrix
        if len(correlation_data) > 1:
            corr_df = pd.DataFrame(correlation_data)
            corr_matrix = corr_df.corr()
            
            # Plot correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=.5, ax=ax)
            ax.set_title('Correlation Matrix')
            
            st.pyplot(fig)
        
        # Market volatility and volume analysis
        st.subheader("Volatility Analysis")
        
        if 'df' in locals():
            # Calculate rolling volatility
            df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Volatility_20'],
                line=dict(color='purple', width=2),
                name="20-Day Volatility"
            ))
            
            fig.update_layout(
                title="Rolling Volatility (Annualized)",
                yaxis_title="Volatility (%)",
                xaxis_title="Date",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume analysis if available
            if 'Volume' in df.columns:
                st.subheader("Volume Analysis")
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                  vertical_spacing=0.03, 
                                  row_heights=[0.7, 0.3])
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    line=dict(color='blue', width=2),
                    name="Price"
                ), row=1, col=1)
                
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    marker_color='green',
                    name="Volume"
                ), row=2, col=1)
                
                fig.update_layout(
                    title="Price and Volume",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        logger.error(f"Error in market analysis: {str(e)}")
        st.error(f"An error occurred in market analysis: {str(e)}")

with tabs[4]:  # Live Trading
    st.title("Live Trading Dashboard")
    
    if not st.session_state.trade_executor_initialized:
        st.info("Live trading not initialized. Please configure settings in the sidebar.")
    else:
        trade_executor = st.session_state.trade_executor
        
        # Display trading summary
        performance = trade_executor.get_performance_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Current Capital", f"${performance['current_capital']:.2f}")
        col2.metric("Total P&L", f"${performance['total_pnl']:.2f}")
        col3.metric("Win Rate", f"{performance['win_rate']:.2f}%")
        col4.metric("Open Positions", f"{performance['open_positions']}")
        
        # Display open positions
        st.subheader("Open Positions")
        
        if performance['open_positions'] > 0:
            open_pos_df = pd.DataFrame.from_dict(trade_executor.open_positions.values())
            open_pos_df = open_pos_df[['symbol', 'direction', 'entry_price', 'current_price', 
                                     'stop_loss', 'take_profit', 'position_size', 'pnl', 'pnl_pct', 'open_time']]
            
            st.dataframe(open_pos_df)
            
            # Add position controls
            position_to_close = st.selectbox("Select Position to Close", 
                                          options=[f"{pos['id']} ({pos['symbol']})" for pos in trade_executor.open_positions.values()])
            
            if st.button("Close Selected Position"):
                position_id = position_to_close.split(" ")[0]
                trade_executor.close_position(position_id)
                st.success(f"Position {position_id} closed")
                st.experimental_rerun()
        else:
            st.info("No open positions")
        
        # Display recent trades
        st.subheader("Recent Trades")
        
        if performance['recent_trades']:
            recent_trades_df = pd.DataFrame(performance['recent_trades'])
            st.dataframe(recent_trades_df)
        else:
            st.info("No recent trades")
        
        # New trade entry
        st.subheader("New Trade Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trade_symbol = st.selectbox("Symbol", ["USD/JPY", "EUR/USD", "GBP/USD", "^IXIC", "^GSPC"])
            trade_direction = st.radio("Direction", ["Long", "Short"])
            entry_price = st.number_input("Entry Price", value=0.0, step=0.01)
        
        with col2:
            stop_loss = st.number_input("Stop Loss", value=0.0, step=0.01)
            take_profit = st.number_input("Take Profit", value=0.0, step=0.01)
            position_size = st.number_input("Position Size (optional)", value=0, step=1)
        
        if entry_price > 0 and stop_loss > 0:
            # Calculate estimated risk
            direction_val = 1 if trade_direction == "Long" else -1
            risk_per_unit = abs(entry_price - stop_loss)
            
            if position_size == 0:
                position_size = trade_executor.calculate_position_size(entry_price, stop_loss)
            
            estimated_risk = position_size * risk_per_unit
            estimated_risk_pct = (estimated_risk / trade_executor.capital) * 100
            
            st.write(f"Estimated Risk: ${estimated_risk:.2f} ({estimated_risk_pct:.2f}% of capital)")
            
            if st.button("Submit Trade"):
                position_id = trade_executor.open_position(
                    symbol=trade_symbol,
                    direction=direction_val,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit if take_profit > 0 else None,
                    position_size=position_size if position_size > 0 else None
                )
                
                if position_id:
                    st.success(f"Trade opened with ID: {position_id}")
                    st.experimental_rerun()
                else:
                    st.error("Failed to open trade")
        
        # Show trade statistics
        if len(performance['recent_trades']) > 0:
            st.subheader("Trade Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                stats = {
                    "Total Trades": performance['total_trades'],
                    "Winning Trades": performance['winning_trades'],
                    "Losing Trades": performance['losing_trades'],
                    "Win Rate": f"{performance['win_rate']:.2f}%",
                    "Profit Factor": f"{performance['profit_factor']:.2f}"
                }
                
                st.table(pd.DataFrame(list(stats.items()), columns=["Metric", "Value"]))
            
            with col2:
                profit_stats = {
                    "Initial Capital": f"${performance['initial_capital']:.2f}",
                    "Current Capital": f"${performance['current_capital']:.2f}",
                    "Total P&L": f"${performance['total_pnl']:.2f}",
                    "Total Return": f"{performance['total_return_pct']:.2f}%",
                    "Average Win": f"{performance['average_win']:.2f}%",
                    "Average Loss": f"{performance['average_loss']:.2f}%"
                }
                
                st.table(pd.DataFrame(list(profit_stats.items()), columns=["Metric", "Value"]))
