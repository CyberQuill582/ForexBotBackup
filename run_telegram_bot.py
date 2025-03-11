import logging
from telegram_bot import main

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # Run the bot
    print("\nStarting Telegram Bot...\n")
    main()