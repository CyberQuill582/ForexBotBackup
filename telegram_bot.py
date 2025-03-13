import logging
import os
from telegram import __version__ as TG_VER
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hello {user.first_name}! 👋 Welcome to the Forex Trading Bot.\n\n"
        "Use /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "Available commands:\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
    )
    await update.message.reply_text(help_text)

def main() -> None:
    """Start the bot."""
    try:
        # Log version info
        logger.info(f"Using python-telegram-bot version {TG_VER}")

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

        # Start the bot
        logger.info("Starting bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        raise  # Re-raise the exception for proper error handling

if __name__ == "__main__":
    main()