
import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
import numpy as np
from data_handler import DataHandler
from trading_strategy import TradingStrategy
from ml_predictor import MLPredictor
import matplotlib.pyplot as plt
import io

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize components
data_handler = DataHandler()
trading_strategy = TradingStrategy()
ml_predictor = MLPredictor(retrain_frequency=7)  # Retrain every 7 days by default

# Helper functions
async def generate_chart(symbol, timeframe):
    """Generate a chart for the given symbol and timeframe"""
    try:
        df = data_handler.fetch_market_data(symbol, timeframe, additional_indicators=True)
        
        # Generate signals
        signals = trading_strategy.generate_signals(df)
        predictions = ml_predictor.predict(df)
        
        # Combine signals
        final_signal = trading_strategy.combine_signals(signals, predictions)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot price
        plt.subplot(2, 1, 1)
        plt.plot(df.index[-30:], df['Close'].iloc[-30:], label='Price')
        plt.plot(df.index[-30:], df['SMA_20'].iloc[-30:], label='SMA 20')
        plt.plot(df.index[-30:], df['SMA_50'].iloc[-30:], label='SMA 50')
        
        # Plot buy/sell signals
        for i in range(len(df)-30, len(df)):
            if final_signal[i] == 1:  # Buy signal
                plt.scatter(df.index[i], df['Low'].iloc[i], marker='^', color='green', s=100)
            elif final_signal[i] == -1:  # Sell signal
                plt.scatter(df.index[i], df['High'].iloc[i], marker='v', color='red', s=100)
        
        plt.title(f'{symbol} - {timeframe} Chart')
        plt.legend()
        
        # Plot RSI
        plt.subplot(2, 1, 2)
        plt.plot(df.index[-30:], df['RSI'].iloc[-30:], color='purple')
        plt.axhline(y=70, color='r', linestyle='-')
        plt.axhline(y=30, color='g', linestyle='-')
        plt.title('RSI')
        
        plt.tight_layout()
        
        # Save plot to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        
        return buffer, df
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        return None, None

async def get_trading_signals(symbol, timeframe):
    """Get trading signals for the given symbol and timeframe"""
    try:
        df = data_handler.fetch_market_data(symbol, timeframe, additional_indicators=True)
        
        # Generate signals
        signals = trading_strategy.generate_signals(df)
        predictions = ml_predictor.predict(df)
        final_signal = trading_strategy.combine_signals(signals, predictions)
        
        # Get latest signal
        latest_signal = "BUY 🟢" if final_signal[-1] == 1 else "SELL 🔴" if final_signal[-1] == -1 else "HOLD ⚪"
        
        # Get current price and indicators
        current_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal'].iloc[-1]
        
        response = (
            f"*{symbol} - {timeframe} Analysis*\n\n"
            f"*Signal:* {latest_signal}\n"
            f"*Current Price:* {current_price:.4f}\n"
            f"*RSI:* {rsi:.2f}\n"
            f"*MACD:* {macd:.4f}\n"
            f"*Signal Line:* {signal:.4f}\n"
        )
        
        return response
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        return f"Error analyzing {symbol}: {str(e)}"

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hi {user.first_name}! I'm your Trading Bot Assistant. "
        "Use /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "Available commands:\n\n"
        "/signals <symbol> <timeframe> - Get trading signals\n"
        "  Example: /signals USD/JPY 1d\n\n"
        "/chart <symbol> <timeframe> - Get price chart with signals\n"
        "  Example: /chart EUR/USD 1h\n\n"
        "/pairs - List available trading pairs\n"
        "/timeframes - List available timeframes\n"
    )
    await update.message.reply_text(help_text)

async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send trading signals for a symbol"""
    if len(context.args) < 2:
        await update.message.reply_text("Please specify a symbol and timeframe. Example: /signals USD/JPY 1d")
        return
    
    symbol = context.args[0]
    timeframe = context.args[1]
    
    await update.message.reply_text(f"Analyzing {symbol} on {timeframe} timeframe...")
    
    response = await get_trading_signals(symbol, timeframe)
    await update.message.reply_text(response, parse_mode='Markdown')

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a chart for a symbol"""
    if len(context.args) < 2:
        await update.message.reply_text("Please specify a symbol and timeframe. Example: /chart USD/JPY 1d")
        return
    
    symbol = context.args[0]
    timeframe = context.args[1]
    
    await update.message.reply_text(f"Generating chart for {symbol} on {timeframe} timeframe...")
    
    chart_buffer, df = await generate_chart(symbol, timeframe)
    if chart_buffer:
        await update.message.reply_photo(chart_buffer)
        
        # Add a text summary
        latest_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change_pct = (latest_close - prev_close) / prev_close * 100
        direction = "🔼" if change_pct > 0 else "🔽"
        
        summary = (
            f"*{symbol} - {timeframe}*\n"
            f"Last price: {latest_close:.4f} {direction} ({change_pct:.2f}%)\n"
            f"SMA 20: {df['SMA_20'].iloc[-1]:.4f}\n"
            f"SMA 50: {df['SMA_50'].iloc[-1]:.4f}\n"
            f"RSI: {df['RSI'].iloc[-1]:.2f}"
        )
        
        await update.message.reply_text(summary, parse_mode='Markdown')
    else:
        await update.message.reply_text(f"Error generating chart for {symbol}")

async def pairs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List available trading pairs"""
    pairs = ["USD/JPY", "EUR/USD", "GBP/USD", "^IXIC", "^GSPC"]
    response = "Available trading pairs:\n\n" + "\n".join(pairs)
    await update.message.reply_text(response)

async def timeframes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List available timeframes"""
    timeframes = ["1d", "1h", "15m", "5m"]
    response = "Available timeframes:\n\n" + "\n".join(timeframes)
    await update.message.reply_text(response)

def main() -> None:
    """Start the bot."""
    # Get token from environment variable
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("No TELEGRAM_BOT_TOKEN found in environment variables")
        return
    
    # Create the Application
    application = Application.builder().token(token).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("chart", chart_command))
    application.add_handler(CommandHandler("pairs", pairs_command))
    application.add_handler(CommandHandler("timeframes", timeframes_command))

    # Run the bot until Ctrl+C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
