
import os
import logging
from telegram_bot import main

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # Check for Telegram token
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        token = input("Please enter your Telegram bot token: ")
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    
    # Run the bot
    main()
