import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Put the repo root on sys.path so we can resolve `from src import ...` properly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.live.predict import generate_predictions

load_dotenv()

# Load the secret Telegram bot token for sending messages automatically
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GET_UPDATES_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"

TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if __name__ == "__main__":
    message = generate_predictions()
    send_message_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"

    print(requests.get(send_message_url).json())
