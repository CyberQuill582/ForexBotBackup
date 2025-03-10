
import os
import logging
import openai
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIOptimizer:
    def __init__(self):
        """Initialize the AI Optimizer"""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
            self.enabled = True
        else:
            logger.warning("No OpenAI API key found. AI optimization disabled.")
            self.enabled = False
        
        self.client = openai.OpenAI() if self.enabled else None
        self.model = "gpt-4" # Can be downgraded to gpt-3.5-turbo for cost savings
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market conditions using OpenAI
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        if not self.enabled:
            return {"error": "OpenAI API key not configured"}
        
        try:
            # Prepare data for analysis
            recent_data = df.tail(20).copy()
            
            # Calculate key metrics
            recent_data['returns'] = recent_data['Close'].pct_change()
            volatility = recent_data['returns'].std() * np.sqrt(252)
            trend = "Uptrend" if recent_data['SMA_20'].iloc[-1] > recent_data['SMA_50'].iloc[-1] else "Downtrend"
            avg_volume = recent_data['Volume'].mean() if 'Volume' in recent_data.columns else "N/A"
            
            # Prepare the prompt
            prompt = f"""
            I have recent market data for analysis. Here are the key metrics:
            
            Current Price: {recent_data['Close'].iloc[-1]:.4f}
            Price Change (1d): {recent_data['returns'].iloc[-1] * 100:.2f}%
            RSI: {recent_data['RSI'].iloc[-1]:.2f}
            MACD: {recent_data['MACD'].iloc[-1]:.4f}
            Signal Line: {recent_data['Signal'].iloc[-1]:.4f}
            Current Trend: {trend}
            Volatility (Annualized): {volatility * 100:.2f}%
            
            Based on this data:
            1. What market conditions are we currently in?
            2. What trading approach would be most suitable?
            3. What are potential risks to watch for?
            4. What technical indicators might be most reliable in these conditions?
            
            Provide your analysis in JSON format with the following keys:
            "market_condition", "recommended_approach", "risks", "reliable_indicators"
            """
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a financial market analysis assistant with expertise in technical analysis and trading strategies."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {"error": str(e)}
    
    def optimize_strategy_parameters(self, historical_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize trading strategy parameters based on historical performance
        
        Args:
            historical_performance: Dictionary with performance metrics
            
        Returns:
            Dictionary with optimized parameters
        """
        if not self.enabled:
            return {"error": "OpenAI API key not configured"}
        
        try:
            # Prepare the prompt with historical performance data
            prompt = f"""
            I have historical performance data for my trading strategy:
            
            Total Return: {historical_performance.get('total_return', 'N/A')}%
            Win Rate: {historical_performance.get('win_rate', 'N/A')}%
            Profit Factor: {historical_performance.get('profit_factor', 'N/A')}
            Sharpe Ratio: {historical_performance.get('sharpe_ratio', 'N/A')}
            Max Drawdown: {historical_performance.get('max_drawdown', 'N/A')}%
            
            Current Strategy Parameters:
            - MACD Fast Period: 12
            - MACD Slow Period: 26
            - MACD Signal Period: 9
            - RSI Period: 14
            - RSI Overbought Level: 70
            - RSI Oversold Level: 30
            - Technical Weight: 0.6
            - ML Weight: 0.4
            
            Based on this performance, suggest optimized parameters to improve the strategy.
            Provide reasoning for each suggestion.
            
            Return your suggestions in JSON format with the following structure:
            {
                "macd_fast": {"value": <int>, "reasoning": "<string>"},
                "macd_slow": {"value": <int>, "reasoning": "<string>"},
                "macd_signal": {"value": <int>, "reasoning": "<string>"},
                "rsi_period": {"value": <int>, "reasoning": "<string>"},
                "rsi_overbought": {"value": <int>, "reasoning": "<string>"},
                "rsi_oversold": {"value": <int>, "reasoning": "<string>"},
                "tech_weight": {"value": <float>, "reasoning": "<string>"},
                "overall_assessment": "<string>"
            }
            """
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a trading strategy optimization assistant with expertise in technical indicators and machine learning."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            optimized_params = json.loads(response.choices[0].message.content)
            
            # Log the optimization
            logger.info(f"AI Optimizer suggested new parameters: {optimized_params}")
            
            return optimized_params
        
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {str(e)}")
            return {"error": str(e)}
    
    def analyze_trade_history(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trade history to identify patterns and improvements
        
        Args:
            trades: DataFrame of trade history
            
        Returns:
            Dictionary with analysis and recommendations
        """
        if not self.enabled or trades.empty:
            return {"error": "OpenAI API key not configured or no trade history"}
        
        try:
            # Prepare trade statistics
            winning_trades = trades[trades['pnl_pct'] > 0]
            losing_trades = trades[trades['pnl_pct'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
            avg_win = winning_trades['pnl_pct'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pnl_pct'].mean() if not losing_trades.empty else 0
            
            # Calculate time-based statistics
            trades['entry_date'] = pd.to_datetime(trades['entry_date'])
            trades['hour'] = trades['entry_date'].dt.hour
            trades['day_of_week'] = trades['entry_date'].dt.day_name()
            
            # Group by hour and day to find patterns
            hour_performance = trades.groupby('hour')['pnl_pct'].mean().to_dict()
            day_performance = trades.groupby('day_of_week')['pnl_pct'].mean().to_dict()
            
            # Prepare the prompt
            trade_examples = trades.head(5).to_dict('records')
            
            prompt = f"""
            I have trade history data with {len(trades)} trades.
            
            Summary Statistics:
            - Win Rate: {win_rate:.2f}%
            - Average Win: {avg_win:.2f}%
            - Average Loss: {avg_loss:.2f}%
            
            Hour Performance (average % return by hour of day):
            {hour_performance}
            
            Day Performance (average % return by day of week):
            {day_performance}
            
            Here are a few example trades:
            {json.dumps(trade_examples, default=str)}
            
            Based on this data:
            1. What patterns do you see in my trading?
            2. When am I most successful?
            3. What specific improvements could I make?
            4. Are there particular market conditions I should avoid?
            
            Provide your analysis in JSON format with the following keys:
            "patterns", "optimal_trading_times", "improvement_suggestions", "conditions_to_avoid"
            """
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a trading performance analysis assistant with expertise in identifying patterns and optimization opportunities."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing trade history: {str(e)}")
            return {"error": str(e)}
