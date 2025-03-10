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

def calculate_performance_metrics(df, signals, initial_capital=10000.0, risk_per_trade=0.02):
    """
    Calculate comprehensive trading performance metrics

    Args:
        df: DataFrame with market data
        signals: Array of trading signals (-1, 0, 1)
        initial_capital: Initial capital amount
        risk_per_trade: Risk percentage per trade

    Returns:
        Dictionary with performance metrics
    """
    try:
        # Convert signals to pandas Series if it's a numpy array
        if isinstance(signals, np.ndarray):
            signals = pd.Series(signals, index=df.index)

        # Calculate returns
        returns = df['Close'].pct_change().fillna(0)

        # Calculate strategy returns
        strategy_returns = returns * signals.shift(1).fillna(0)

        # Build equity curve
        equity_curve = pd.DataFrame(index=df.index)
        equity_curve['Returns'] = returns
        equity_curve['Strategy_Returns'] = strategy_returns
        equity_curve['Cumulative_Returns'] = (1 + returns).cumprod()
        equity_curve['Strategy_Cumulative_Returns'] = (1 + strategy_returns).cumprod()
        equity_curve['Equity'] = initial_capital * equity_curve['Strategy_Cumulative_Returns']

        # Calculate drawdowns
        equity_curve['Peak'] = equity_curve['Equity'].cummax()
        equity_curve['Drawdown'] = (equity_curve['Equity'] / equity_curve['Peak'] - 1) * 100

        # Identify trades
        trades = pd.DataFrame(index=df.index)
        trades['Signal'] = signals
        trades['Entry_Exit'] = trades['Signal'].diff().fillna(0)

        # Calculate trade statistics
        trade_stats = []
        position = 0
        entry_price = 0
        entry_date = None

        for i in range(1, len(df)):
            # New position
            if signals[i] != 0 and signals[i-1] == 0:
                position = signals[i]
                entry_price = df['Close'].iloc[i]
                entry_date = df.index[i]

            # Exit position
            elif (signals[i] == 0 and signals[i-1] != 0) or (signals[i] != 0 and signals[i] != signals[i-1]):
                if position != 0 and entry_date is not None:
                    exit_price = df['Close'].iloc[i]
                    pnl_pct = ((exit_price / entry_price) - 1) * 100 * position
                    trade_stats.append({
                        'entry_date': entry_date,
                        'exit_date': df.index[i],
                        'position': 'Long' if position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'duration': (df.index[i] - entry_date).total_seconds() / 86400  # in days
                    })
                    position = signals[i]  # Update position if changing direction
                    if position != 0:
                        entry_price = exit_price
                        entry_date = df.index[i]
                    else:
                        entry_date = None

        # Convert trade stats to DataFrame
        trades_df = pd.DataFrame(trade_stats) if trade_stats else pd.DataFrame()

        # Calculate advanced metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0]) if not trades_df.empty else 0
        losing_trades = len(trades_df[trades_df['pnl_pct'] <= 0]) if not trades_df.empty else 0

        avg_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean()) if losing_trades > 0 else 0

        # Risk-adjusted metrics
        total_return = (equity_curve['Equity'].iloc[-1] / initial_capital - 1) * 100

        # Annualized return
        days = (df.index[-1] - df.index[0]).days
        annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0

        # Risk metrics
        volatility = strategy_returns.std() * np.sqrt(252)  # Annualized volatility
        max_drawdown = equity_curve['Drawdown'].min()
        current_drawdown = calculate_current_drawdown(strategy_returns)

        # Sharpe and Sortino ratios
        risk_free_rate = 0.0  # Simplified
        excess_returns = strategy_returns - risk_free_rate/252

        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0

        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0

        # Calculate Calmar ratio (return/max drawdown)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

        # Expectancy and Expectancy Score
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        expectancy = ((avg_profit * winning_trades) - (avg_loss * losing_trades)) / total_trades if total_trades > 0 else 0
        system_quality = expectancy / (strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() != 0 else 0

        # Profit factor
        gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Recovery factor
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        # Monthly returns analysis
        monthly_returns = calculate_monthly_returns(equity_curve)

        # Compile all metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'volatility': volatility * 100,  # Convert to percentage
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'average_profit': avg_profit,
            'average_loss': avg_loss,
            'recovery_factor': recovery_factor,
            'expectancy': expectancy,
            'system_quality': system_quality,
            'monthly_returns': monthly_returns,
            'equity_curve': equity_curve,
            'all_trades': trades_df
        }

        return metrics

    except Exception as e:
        logging.error(f"Error calculating performance metrics: {str(e)}")
        raise

def calculate_current_drawdown(returns):
    """Calculate current drawdown"""
    try:
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        cumulative_returns = (1 + returns).cumprod()
        current_drawdown = (cumulative_returns[-1] / cumulative_returns.max() - 1) * 100
        return current_drawdown
    except Exception as e:
        logging.error(f"Error calculating current drawdown: {str(e)}")
        return 0.0

def calculate_monthly_returns(equity_curve):
    """
    Calculate monthly returns from equity curve

    Args:
        equity_curve: DataFrame with equity values

    Returns:
        DataFrame with monthly returns
    """
    try:
        # Make sure we have a DataFrame with datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            return pd.DataFrame()

        # Calculate monthly returns
        monthly_returns = equity_curve['Equity'].resample('M').last().pct_change()

        # Create a pivot table with years as rows and months as columns
        monthly_pivot = pd.DataFrame()

        if len(monthly_returns) > 0:
            monthly_pivot = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 100  # Convert to percentage
            })

            monthly_pivot = monthly_pivot.pivot(index='Year', columns='Month', values='Return')
            monthly_pivot.columns = [pd.to_datetime(f"2020-{month}-1").strftime('%b') for month in monthly_pivot.columns]

        return monthly_pivot

    except Exception as e:
        logging.error(f"Error calculating monthly returns: {str(e)}")
        return pd.DataFrame()

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
def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    return np.min(drawdowns) * 100