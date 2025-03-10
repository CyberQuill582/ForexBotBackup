import logging
import pandas as pd
import numpy as np
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def calculate_performance_metrics(df, signals):
    """Calculate trading performance metrics"""
    try:
        # Calculate returns
        returns = df['Close'].pct_change()
        
        # Calculate strategy returns
        strategy_returns = returns * signals
        
        # Calculate metrics
        total_trades = np.sum(np.abs(np.diff(signals) != 0))
        winning_trades = np.sum((strategy_returns > 0) & (signals != 0))
        
        metrics = {
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'profit_factor': np.abs(np.sum(strategy_returns[strategy_returns > 0]) / 
                                  np.sum(strategy_returns[strategy_returns < 0]))
                           if np.sum(strategy_returns[strategy_returns < 0]) != 0 else 0,
            'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                          if np.std(strategy_returns) != 0 else 0,
            'max_drawdown': calculate_max_drawdown(strategy_returns),
            'current_drawdown': calculate_current_drawdown(strategy_returns),
            'recent_trades': get_recent_trades(df, signals)
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating performance metrics: {str(e)}")
        raise

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    return np.min(drawdowns) * 100

def calculate_current_drawdown(returns):
    """Calculate current drawdown"""
    cumulative_returns = (1 + returns).cumprod()
    current_drawdown = (cumulative_returns[-1] / np.max(cumulative_returns) - 1) * 100
    return current_drawdown

def get_recent_trades(df, signals):
    """Get recent trades information"""
    trades = []
    
    for i in range(1, len(signals)):
        if signals[i] != signals[i-1]:
            trades.append({
                'timestamp': df.index[i],
                'action': 'BUY' if signals[i] == 1 else 'SELL',
                'price': df['Close'].iloc[i],
                'return': df['Close'].pct_change().iloc[i]
            })
    
    trades_df = pd.DataFrame(trades[-10:])  # Last 10 trades
    return trades_df
