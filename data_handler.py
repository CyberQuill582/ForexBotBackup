import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

class DataHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def fetch_market_data(self, symbol, timeframe):
        """
        Fetch market data from Yahoo Finance
        """
        try:
            # Convert forex pair to Yahoo Finance format
            if symbol == "USD/JPY":
                symbol = "USDJPY=X"
            
            # Calculate date range
            end_date = datetime.now()
            if timeframe == "1d":
                start_date = end_date - timedelta(days=365)
                interval = "1d"
            elif timeframe == "1h":
                start_date = end_date - timedelta(days=30)
                interval = "1h"
            else:  # 15m
                start_date = end_date - timedelta(days=7)
                interval = "15m"
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data received for {symbol}")
            
            # Calculate basic technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'], df['Signal'] = self._calculate_macd(df['Close'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
