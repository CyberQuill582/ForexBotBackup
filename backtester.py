import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_strategy import TradingStrategy
from ml_predictor import MLPredictor
from utils import calculate_performance_metrics
import logging

class Backtester:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trading_strategy = TradingStrategy()
        self.ml_predictor = MLPredictor()

    def run_backtest(self, df, initial_capital=10000.0, position_size=0.1, start_date = None): #Added start_date parameter
        """
        Run a backtest on historical data

        Args:
            df: DataFrame with OHLC data and indicators
            initial_capital: Initial capital amount
            position_size: Size of each position as percentage of capital
            start_date: Start date for backtest (optional)

        Returns:
            Dictionary with backtest results
        """
        try:
            # Generate signals
            technical_signals = self.trading_strategy.generate_signals(df)
            ml_predictions = self.ml_predictor.predict(df)
            final_signals = self.trading_strategy.combine_signals(technical_signals, ml_predictions)

            # Prepare backtest data
            backtest_data = pd.DataFrame({
                'Close': df['Close'],
                'Signal': final_signals
            })

            # Calculate returns
            backtest_data['Returns'] = df['Close'].pct_change()
            backtest_data['Strategy_Returns'] = backtest_data['Returns'] * backtest_data['Signal'].shift(1)

            # Remove NaN values
            backtest_data = backtest_data.dropna()

            # Calculate equity curve
            backtest_data['Cumulative_Returns'] = (1 + backtest_data['Returns']).cumprod()
            backtest_data['Strategy_Cumulative_Returns'] = (1 + backtest_data['Strategy_Returns']).cumprod()

            # Calculate equity with initial capital
            backtest_data['Equity'] = initial_capital * backtest_data['Strategy_Cumulative_Returns']

            # Calculate position sizes
            backtest_data['Position_Size'] = backtest_data['Equity'] * position_size * backtest_data['Signal']

            # Calculate drawdowns
            backtest_data['Peak'] = backtest_data['Equity'].cummax()
            backtest_data['Drawdown'] = (backtest_data['Equity'] / backtest_data['Peak'] - 1) * 100

            # Calculate metrics
            metrics = {
                'total_return': (backtest_data['Equity'].iloc[-1] / initial_capital - 1) * 100,
                'annual_return': ((backtest_data['Equity'].iloc[-1] / initial_capital) ** 
                                 (252 / len(backtest_data)) - 1) * 100,
                'sharpe_ratio': backtest_data['Strategy_Returns'].mean() / backtest_data['Strategy_Returns'].std() * np.sqrt(252),
                'max_drawdown': backtest_data['Drawdown'].min(),
                'win_rate': (backtest_data['Strategy_Returns'] > 0).sum() / len(backtest_data) * 100,
                'profit_factor': abs(backtest_data['Strategy_Returns'][backtest_data['Strategy_Returns'] > 0].sum() / 
                                  backtest_data['Strategy_Returns'][backtest_data['Strategy_Returns'] < 0].sum()),
                'trade_count': (backtest_data['Signal'] != backtest_data['Signal'].shift(1)).sum(),
                'equity_curve': backtest_data[['Equity']],
                'drawdown_curve': backtest_data[['Drawdown']]
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error in backtesting: {str(e)}")
            raise

    def plot_equity_curve(self, equity_curve, figsize=(12, 6)):
        """Plot equity curve from backtest results"""
        plt.figure(figsize=figsize)
        plt.plot(equity_curve.index, equity_curve['Equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        return plt.gcf()

    def plot_drawdown_curve(self, drawdown_curve, figsize=(12, 6)):
        """Plot drawdown curve from backtest results"""
        plt.figure(figsize=figsize)
        plt.fill_between(drawdown_curve.index, 0, drawdown_curve['Drawdown'], color='red', alpha=0.3)
        plt.plot(drawdown_curve.index, drawdown_curve['Drawdown'], color='red')
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        return plt.gcf()