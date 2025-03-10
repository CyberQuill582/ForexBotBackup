
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

class TradeExecutor:
    def __init__(self, risk_per_trade=0.02, max_open_positions=3):
        """
        Initialize trade executor
        
        Args:
            risk_per_trade: Percentage of capital to risk per trade
            max_open_positions: Maximum number of open positions at any time
        """
        self.logger = logging.getLogger(__name__)
        self.risk_per_trade = risk_per_trade
        self.max_open_positions = max_open_positions
        self.open_positions = {}
        self.trade_history = []
        self.capital = 0
        self.initial_capital = 0
        
    def set_capital(self, amount):
        """Set trading capital"""
        self.capital = amount
        self.initial_capital = amount
        self.logger.info(f"Capital set to {amount}")
    
    def calculate_position_size(self, entry_price, stop_loss, risk_amount=None):
        """
        Calculate position size based on risk parameters
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_amount: Optional specific risk amount, otherwise uses risk_per_trade
            
        Returns:
            Position size in units/shares/contracts
        """
        if risk_amount is None:
            risk_amount = self.capital * self.risk_per_trade
            
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
            
        position_size = risk_amount / risk_per_unit
        
        # Round down to prevent exceeding risk
        return int(position_size)
    
    def open_position(self, symbol, direction, entry_price, stop_loss, take_profit=None, position_size=None):
        """
        Open a new trading position
        
        Args:
            symbol: Trading symbol
            direction: 1 for long, -1 for short
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            position_size: Override calculated position size (optional)
            
        Returns:
            Position ID if successful, None otherwise
        """
        try:
            # Check if we can open more positions
            if len(self.open_positions) >= self.max_open_positions:
                self.logger.warning("Maximum number of open positions reached")
                return None
                
            # Calculate position size if not provided
            if position_size is None:
                position_size = self.calculate_position_size(entry_price, stop_loss)
                
            if position_size <= 0:
                self.logger.warning("Position size calculation resulted in zero or negative size")
                return None
                
            # Generate position ID
            position_id = f"{symbol}_{direction}_{int(time.time())}"
            
            # Calculate position value
            position_value = position_size * entry_price
            
            # Check if we have enough capital
            if position_value > self.capital:
                self.logger.warning(f"Insufficient capital for position: {position_value} > {self.capital}")
                return None
                
            # Create position object
            position = {
                'id': position_id,
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'position_value': position_value,
                'open_time': datetime.now(),
                'pnl': 0,
                'pnl_pct': 0,
                'status': 'open'
            }
            
            # Add to open positions
            self.open_positions[position_id] = position
            
            # Update capital (reserve the position value)
            self.capital -= position_value
            
            self.logger.info(f"Opened {position_id}: {direction} {position_size} {symbol} @ {entry_price}")
            
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return None
            
    def update_positions(self, current_prices):
        """
        Update open positions with current prices
        
        Args:
            current_prices: Dictionary of current prices by symbol
        """
        try:
            for pos_id, position in list(self.open_positions.items()):
                symbol = position['symbol']
                
                if symbol not in current_prices:
                    self.logger.warning(f"No price data for {symbol}")
                    continue
                    
                current_price = current_prices[symbol]
                position['current_price'] = current_price
                
                # Calculate unrealized PnL
                if position['direction'] == 1:  # Long
                    position['pnl'] = (current_price - position['entry_price']) * position['position_size']
                    position['pnl_pct'] = (current_price / position['entry_price'] - 1) * 100
                else:  # Short
                    position['pnl'] = (position['entry_price'] - current_price) * position['position_size']
                    position['pnl_pct'] = (position['entry_price'] / current_price - 1) * 100
                
                # Check for stop loss or take profit
                if position['direction'] == 1:  # Long
                    if current_price <= position['stop_loss']:
                        self.close_position(pos_id, current_price, 'stop_loss')
                    elif position['take_profit'] is not None and current_price >= position['take_profit']:
                        self.close_position(pos_id, current_price, 'take_profit')
                else:  # Short
                    if current_price >= position['stop_loss']:
                        self.close_position(pos_id, current_price, 'stop_loss')
                    elif position['take_profit'] is not None and current_price <= position['take_profit']:
                        self.close_position(pos_id, current_price, 'take_profit')
                        
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            
    def close_position(self, position_id, exit_price=None, reason='manual'):
        """
        Close an open position
        
        Args:
            position_id: ID of the position to close
            exit_price: Exit price (if None, uses current_price)
            reason: Reason for closing (manual, stop_loss, take_profit)
            
        Returns:
            Realized PnL if successful, None otherwise
        """
        try:
            if position_id not in self.open_positions:
                self.logger.warning(f"Position {position_id} not found")
                return None
                
            position = self.open_positions[position_id]
            
            # Use provided exit price or current price
            if exit_price is None:
                exit_price = position['current_price']
                
            # Calculate realized PnL
            if position['direction'] == 1:  # Long
                realized_pnl = (exit_price - position['entry_price']) * position['position_size']
                realized_pnl_pct = (exit_price / position['entry_price'] - 1) * 100
            else:  # Short
                realized_pnl = (position['entry_price'] - exit_price) * position['position_size']
                realized_pnl_pct = (position['entry_price'] / exit_price - 1) * 100
                
            # Update capital
            position_value = position['position_size'] * exit_price
            self.capital += position_value + realized_pnl
            
            # Update position
            position.update({
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'duration': (datetime.now() - position['open_time']).total_seconds() / 3600,  # hours
                'realized_pnl': realized_pnl,
                'realized_pnl_pct': realized_pnl_pct,
                'close_reason': reason,
                'status': 'closed'
            })
            
            # Move to trade history
            self.trade_history.append(position)
            
            # Remove from open positions
            del self.open_positions[position_id]
            
            self.logger.info(f"Closed {position_id}: {reason} @ {exit_price}, PnL: {realized_pnl:.2f} ({realized_pnl_pct:.2f}%)")
            
            return realized_pnl
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return None
            
    def close_all_positions(self, current_prices):
        """Close all open positions"""
        for pos_id, position in list(self.open_positions.items()):
            symbol = position['symbol']
            if symbol in current_prices:
                self.close_position(pos_id, current_prices[symbol], 'system')
            else:
                self.close_position(pos_id, position['current_price'], 'system')
                
    def get_performance_summary(self):
        """
        Get performance summary
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'current_capital': self.capital,
                'open_positions': len(self.open_positions)
            }
            
        trades_df = pd.DataFrame(self.trade_history)
        
        winning_trades = trades_df[trades_df['realized_pnl'] > 0]
        losing_trades = trades_df[trades_df['realized_pnl'] <= 0]
        
        total_pnl = trades_df['realized_pnl'].sum()
        gross_profit = winning_trades['realized_pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['realized_pnl'].sum()) if not losing_trades.empty else 0
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Calculate unrealized PnL in open positions
        unrealized_pnl = sum(pos['pnl'] for pos in self.open_positions.values())
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'total_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_return_pct': (self.capital / self.initial_capital - 1) * 100,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': winning_trades['realized_pnl_pct'].mean() if not winning_trades.empty else 0,
            'average_loss': losing_trades['realized_pnl_pct'].mean() if not losing_trades.empty else 0,
            'largest_win': winning_trades['realized_pnl_pct'].max() if not winning_trades.empty else 0,
            'largest_loss': losing_trades['realized_pnl_pct'].min() if not losing_trades.empty else 0,
            'average_trade_duration': trades_df['duration'].mean() if 'duration' in trades_df else 0,
            'open_positions': len(self.open_positions),
            'recent_trades': trades_df.sort_values('exit_time', ascending=False).head(10).to_dict('records') if 'exit_time' in trades_df else []
        }
