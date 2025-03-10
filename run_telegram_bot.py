
import logging
from setup_telegram import setup_telegram_token
from telegram_bot import main

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # Setup Telegram token
    setup_telegram_token()
    
    # Run the bot
    print("\nStarting Telegram Bot...\n")
    main()
