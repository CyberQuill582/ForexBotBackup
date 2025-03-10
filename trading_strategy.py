import numpy as np
import pandas as pd
import logging

class TradingStrategy:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self, df):
        """
        Generate trading signals based on technical indicators
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
            
            # RSI Strategy
            rsi_signals = np.where(
                (df['RSI'] < 30),
                1,
                np.where(
                    (df['RSI'] > 70),
                    -1,
                    0
                )
            )
            
            # Combine signals
            signals = np.where(
                (macd_signals == ma_signals) & (macd_signals != 0),
                macd_signals,
                np.where(
                    (macd_signals == rsi_signals) & (macd_signals != 0),
                    macd_signals,
                    0
                )
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def combine_signals(self, technical_signals, ml_predictions):
        """
        Combine technical signals with ML predictions
        """
        try:
            # Weight distribution: 60% technical, 40% ML
            final_signals = np.zeros(len(technical_signals))
            
            for i in range(len(technical_signals)):
                if technical_signals[i] == ml_predictions[i]:
                    final_signals[i] = technical_signals[i]
                elif technical_signals[i] != 0 and ml_predictions[i] == 0:
                    final_signals[i] = technical_signals[i]
                elif technical_signals[i] == 0 and ml_predictions[i] != 0:
                    final_signals[i] = ml_predictions[i]
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            raise
