import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests

class DataHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # Simple cache for data
        
    def fetch_market_data(self, symbol, timeframe, additional_indicators=True):
        """
        Fetch market data from Yahoo Finance with enhanced indicators
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data (1d, 1h, 15m)
            additional_indicators: Whether to calculate additional indicators
            
        Returns:
            DataFrame with market data and indicators
        """
        try:
            # Check cache first (using symbol+timeframe as key)
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.cache:
                # Check if cache is not older than 1 hour
                if (datetime.now() - self.cache[cache_key]['timestamp']).total_seconds() < 3600:
                    return self.cache[cache_key]['data']
            
            # Convert forex pair to Yahoo Finance format
            if symbol == "USD/JPY":
                symbol = "USDJPY=X"
            elif symbol == "EUR/USD":
                symbol = "EURUSD=X"
            elif symbol == "GBP/USD":
                symbol = "GBPUSD=X"
            
            # Calculate date range
            end_date = datetime.now()
            if timeframe == "1d":
                start_date = end_date - timedelta(days=365*2)  # 2 years of data
                interval = "1d"
            elif timeframe == "1h":
                start_date = end_date - timedelta(days=60)  # 60 days
                interval = "1h"
            elif timeframe == "15m":
                start_date = end_date - timedelta(days=10)  # 10 days
                interval = "15m"
            elif timeframe == "5m":
                start_date = end_date - timedelta(days=5)  # 5 days
                interval = "5m"
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data received for {symbol}")
                
            # Convert index to UTC to avoid timezone comparison issues
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert('UTC')
            
            # Calculate basic technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean() if len(df) >= 200 else np.nan
            
            df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
            df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
            
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'], df['Signal'] = self._calculate_macd(df['Close'])
            
            # Calculate additional indicators if requested
            if additional_indicators:
                # Bollinger Bands
                df['BB_Middle'] = df['Close'].rolling(window=20).mean()
                volatility = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['BB_Middle'] + (volatility * 2)
                df['BB_Lower'] = df['BB_Middle'] - (volatility * 2)
                
                # Average True Range (ATR)
                df['ATR'] = self._calculate_atr(df)
                
                # Stochastic Oscillator
                df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df)
                
                # Average Directional Index (ADX)
                df['ADX'] = self._calculate_adx(df)
                
                # On-Balance Volume (if volume data available)
                if 'Volume' in df.columns:
                    df['OBV'] = self._calculate_obv(df)
                
                # Ichimoku Cloud (simplified)
                df['Tenkan_Sen'] = self._calculate_midpoint(df, 9)
                df['Kijun_Sen'] = self._calculate_midpoint(df, 26)
                df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
                df['Senkou_Span_B'] = self._calculate_midpoint(df, 52).shift(26)
                df['Chikou_Span'] = df['Close'].shift(-26)
                
                # Fibonacci Retracement Levels
                highest_high = df['High'].rolling(window=50).max()
                lowest_low = df['Low'].rolling(window=50).min()
                df['Fib_0.0'] = lowest_low
                df['Fib_0.236'] = lowest_low + 0.236 * (highest_high - lowest_low)
                df['Fib_0.382'] = lowest_low + 0.382 * (highest_high - lowest_low)
                df['Fib_0.5'] = lowest_low + 0.5 * (highest_high - lowest_low)
                df['Fib_0.618'] = lowest_low + 0.618 * (highest_high - lowest_low)
                df['Fib_0.786'] = lowest_low + 0.786 * (highest_high - lowest_low)
                df['Fib_1.0'] = highest_high
                
                # Add market sentiment data if available (simplified stub - would need API access)
                # self._add_market_sentiment(df, symbol)
            
            # Cache the data
            self.cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
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
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean()
    
    def _calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        
        stoch_k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        # True Range
        df['TR'] = self._calculate_atr(df, 1)
        
        # Plus Directional Movement (+DM)
        df['Plus_DM'] = np.where(
            (df['High'] - df['High'].shift() > df['Low'].shift() - df['Low']) & 
            (df['High'] - df['High'].shift() > 0),
            df['High'] - df['High'].shift(),
            0
        )
        
        # Minus Directional Movement (-DM)
        df['Minus_DM'] = np.where(
            (df['Low'].shift() - df['Low'] > df['High'] - df['High'].shift()) & 
            (df['Low'].shift() - df['Low'] > 0),
            df['Low'].shift() - df['Low'],
            0
        )
        
        # Smooth these values using Wilder's smoothing technique
        df['Smoothed_TR'] = df['TR'].rolling(window=period).sum()
        df['Smoothed_Plus_DM'] = df['Plus_DM'].rolling(window=period).sum()
        df['Smoothed_Minus_DM'] = df['Minus_DM'].rolling(window=period).sum()
        
        # Calculate plus and minus directional indicators (+DI and -DI)
        df['Plus_DI'] = 100 * (df['Smoothed_Plus_DM'] / df['Smoothed_TR'])
        df['Minus_DI'] = 100 * (df['Smoothed_Minus_DM'] / df['Smoothed_TR'])
        
        # Calculate the directional index (DX)
        df['DX'] = 100 * (abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI']))
        
        # Calculate the average directional index (ADX)
        adx = df['DX'].rolling(window=period).mean()
        
        # Clean up temporary columns
        for col in ['TR', 'Plus_DM', 'Minus_DM', 'Smoothed_TR', 'Smoothed_Plus_DM', 
                   'Smoothed_Minus_DM', 'Plus_DI', 'Minus_DI', 'DX']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        return adx
    
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(df))
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=df.index)
    
    def _calculate_midpoint(self, df, period):
        """Calculate midpoint for Ichimoku"""
        high = df['High'].rolling(window=period).max()
        low = df['Low'].rolling(window=period).min()
        return (high + low) / 2
    
    def _add_market_sentiment(self, df, symbol):
        """
        Add market sentiment data (stub - would need actual API)
        This is a placeholder for integration with sentiment APIs
        """
        # Placeholder - in a real system, this would call a sentiment API
        # Add simple random sentiment for demonstration
        df['Market_Sentiment'] = np.random.normal(0, 1, size=len(df))
        
        return df
    
    def fetch_multiple_timeframes(self, symbol, timeframes=['1d', '1h', '15m']):
        """
        Fetch data for multiple timeframes to enable multi-timeframe analysis
        
        Returns:
            Dictionary of DataFrames keyed by timeframe
        """
        data = {}
        for tf in timeframes:
            data[tf] = self.fetch_market_data(symbol, tf)
        return data
