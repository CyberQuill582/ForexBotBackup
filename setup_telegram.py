
import os
import logging

def setup_telegram_token():
    """
    Setup Telegram bot token by prompting the user for input if not already present
    in environment variables.
    
    Returns:
        str: The Telegram bot token
    """
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # Check if token exists in environment variables
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if not token:
        print("\n=== Telegram Bot Setup ===\n")
        print("You need to provide your Telegram bot token.")
        print("You can get a token by talking to @BotFather on Telegram:")
        print("1. Open Telegram and search for @BotFather")
        print("2. Send /newbot and follow the instructions")
        print("3. Copy the HTTP API token provided by BotFather\n")
        
        token = input("Please paste your Telegram bot token here: ")
        
        # Store token in environment variable
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        
        # Let the user know how to set this up in Secrets
        print("\nToken set for this session. For future runs, add it to your Replit Secrets:")
        print("1. Go to the 'Tools' panel in your Replit workspace")
        print("2. Click on 'Secrets'")
        print("3. Add a new secret with key 'TELEGRAM_BOT_TOKEN' and your token as the value\n")
    
    return token

if __name__ == "__main__":
    token = setup_telegram_token()
    print(f"Telegram bot token is {'set' if token else 'not set'}")
