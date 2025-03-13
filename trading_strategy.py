import numpy as np
import pandas as pd
import logging

import numpy as np
import pandas as pd
import logging

class TradingStrategy:
    def __init__(self, risk_per_trade=0.02, max_drawdown=0.10, trend_filter=True):
        """
        Initialize the trading strategy with risk parameters
        
        Args:
            risk_per_trade: Percentage of capital to risk per trade (default: 2%)
            max_drawdown: Maximum allowable drawdown before reducing position size (default: 10%)
            trend_filter: Whether to use trend filtering for signals (default: True)
        """
        self.logger = logging.getLogger(__name__)
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.trend_filter = trend_filter
        self.current_drawdown = 0
        self.max_experienced_drawdown = 0
        self.position_size_factor = 1.0
    
    def generate_signals(self, df):
        """
        Generate trading signals based on technical indicators with enhanced strategies
        """
        try:
            signals = np.zeros(len(df))
            
            # MACD Strategy
            macd_signals = np.where(
                (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)),
                1,
                np.where(
                    (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)),
                    -1,
                    0
                )
            )
            
            # Moving Average Strategy
            ma_signals = np.where(
                (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1)),
                1,
                np.where(
                    (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1)),
                    -1,
                    0
                )
            )
            
            # RSI Strategy with enhancements
            rsi_signals = np.where(
                (df['RSI'] < 30) & (df['RSI'].shift(1) < df['RSI']),  # Oversold + RSI turning up
                1,
                np.where(
                    (df['RSI'] > 70) & (df['RSI'].shift(1) > df['RSI']),  # Overbought + RSI turning down
                    -1,
                    0
                )
            )
            
            # Bollinger Bands strategy (if calculated in data_handler)
            bb_signals = np.zeros_like(signals)
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                bb_signals = np.where(
                    (df['Close'] < df['BB_Lower']) & (df['Close'].shift(1) >= df['BB_Lower'].shift(1)),
                    1,
                    np.where(
                        (df['Close'] > df['BB_Upper']) & (df['Close'].shift(1) <= df['BB_Upper'].shift(1)),
                        -1,
                        0
                    )
                )
            
            # Price action signals
            candle_signals = np.zeros_like(signals)
            
            # Bullish engulfing
            bullish_engulfing = (df['Close'] > df['Open'].shift(1)) & \
                               (df['Open'] < df['Close'].shift(1)) & \
                               (df['Close'] > df['Open']) & \
                               (df['Close'].shift(1) < df['Open'].shift(1))
            
            # Bearish engulfing
            bearish_engulfing = (df['Close'] < df['Open'].shift(1)) & \
                               (df['Open'] > df['Close'].shift(1)) & \
                               (df['Close'] < df['Open']) & \
                               (df['Close'].shift(1) > df['Open'].shift(1))
            
            candle_signals = np.where(bullish_engulfing, 1, np.where(bearish_engulfing, -1, 0))
            
            # Apply trend filter if enabled
            if self.trend_filter:
                # Simple trend filter: 50-day SMA direction
                # Convert numpy operations to use pandas shift
                sma50 = df['SMA_50'].values
                sma50_shifted = df['SMA_50'].shift(10).values
                
                trend = np.where(sma50 > sma50_shifted, 1, 
                        np.where(sma50 < sma50_shifted, -1, 0))
                
                # Only allow long signals in uptrend and short signals in downtrend
                macd_signals = np.where((trend > 0) & (macd_signals > 0), 1, 
                              np.where((trend < 0) & (macd_signals < 0), -1, 0))
                
                ma_signals = np.where((trend > 0) & (ma_signals > 0), 1, 
                            np.where((trend < 0) & (ma_signals < 0), -1, 0))
                
                rsi_signals = np.where((trend > 0) & (rsi_signals > 0), 1, 
                             np.where((trend < 0) & (rsi_signals < 0), -1, 0))
            
            # Weight the different signals
            signals = (macd_signals * 0.3 + ma_signals * 0.3 + rsi_signals * 0.2 + candle_signals * 0.2)
            
            # Threshold the signals to get clear -1, 0, 1 values
            signals = np.where(signals > 0.2, 1, np.where(signals < -0.2, -1, 0))
            
            # Apply consecutive signal filter (avoid frequent signal changes)
            for i in range(2, len(signals)):
                if signals[i] != 0 and signals[i-1] != 0 and signals[i] != signals[i-1]:
                    signals[i] = 0  # Filter out reversals without pause
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def combine_signals(self, technical_signals, ml_predictions, weights=(0.6, 0.4)):
        """
        Combine technical signals with ML predictions using weighted approach
        
        Args:
            technical_signals: Array of technical signals (-1, 0, 1)
            ml_predictions: Array of ML predictions (-1, 0, 1)
            weights: Tuple of weights for (technical, ML) signals
        """
        try:
            # Validate weights
            if sum(weights) != 1.0:
                weights = tuple(w/sum(weights) for w in weights)
            
            tech_weight, ml_weight = weights
            
            # Validate array lengths
            if len(technical_signals) != len(ml_predictions):
                self.logger.warning(f"Signal arrays have different lengths: technical={len(technical_signals)}, ml={len(ml_predictions)}")
                min_length = min(len(technical_signals), len(ml_predictions))
                technical_signals = technical_signals[-min_length:]
                ml_predictions = ml_predictions[-min_length:]
            
            # Apply weighted combination
            weighted_signals = technical_signals * tech_weight + ml_predictions * ml_weight
            
            # Threshold to get clear signals
            final_signals = np.where(weighted_signals > 0.3, 1, 
                            np.where(weighted_signals < -0.3, -1, 0))
            
            # Filter out potentially weak signals (close to zero)
            final_signals = np.where(abs(weighted_signals) < 0.2, 0, final_signals)
            
            # Apply position sizing based on current drawdown
            self._update_position_sizing(final_signals)
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            raise
    
    def calculate_position_size(self, capital, price, stop_loss):
        """
        Calculate position size based on capital and risk management
        
        Args:
            capital: Available capital
            price: Current price
            stop_loss: Stop loss price
        
        Returns:
            Position size (number of units to trade)
        """
        try:
            if price == stop_loss:
                return 0
            
            # Risk amount based on risk per trade parameter and adjusted for drawdown
            risk_amount = capital * self.risk_per_trade * self.position_size_factor
            
            # Calculate risk per unit
            risk_per_unit = abs(price - stop_loss)
            
            # Calculate position size
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                position_size = 0
                
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            raise
    
    def _update_position_sizing(self, signals):
        """
        Update position sizing factor based on performance and drawdown
        """
        # This would be updated with actual portfolio performance in a live system
        if self.current_drawdown > self.max_experienced_drawdown:
            self.max_experienced_drawdown = self.current_drawdown
        
        # Reduce position size as drawdown increases
        if self.current_drawdown > self.max_drawdown * 0.5:
            self.position_size_factor = 0.75
        elif self.current_drawdown > self.max_drawdown * 0.8:
            self.position_size_factor = 0.5
        elif self.current_drawdown > self.max_drawdown:
            self.position_size_factor = 0.25
        else:
            self.position_size_factor = 1.0
    
    def set_current_drawdown(self, drawdown):
        """Update the current drawdown value"""
        self.current_drawdown = drawdown
