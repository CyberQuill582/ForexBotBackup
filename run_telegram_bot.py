import logging
import sys
from telegram_bot import main

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        stream=sys.stdout  # Ensure logs are visible in Replit console
    )
    logger = logging.getLogger(__name__)

    try:
        # Run the bot
        print("\nStarting Telegram Bot...\n")
        main()
    except Exception as e:
        logger.error(f"Failed to start Telegram bot: {e}")
        sys.exit(1)