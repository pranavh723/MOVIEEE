import logging
import sqlite3
import os
import re
import requests
import time
from typing import List, Optional, Tuple
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext
from telegram.error import TelegramError, RetryAfter, BadRequest, Unauthorized, Conflict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging setup
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Load configurations
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
DB_FILE = os.getenv("DATABASE_FILE", "movies.db")  # Default to 'movies.db' if not set
JOB_INTERVAL = int(os.getenv("JOB_INTERVAL", 1800))  # Default to 30 minutes
CHANNEL_CHAT_ID = os.getenv("CHANNEL_CHAT_ID")  # Private/public channel chat ID
CHANNEL_LINK = os.getenv("CHANNEL_LINK")  # Public channel link (optional)
LAST_UPDATE_ID_FILE = "last_update_id.txt"

# Mandatory configuration validation
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set. Please check your .env file.")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY is not set. Please check your .env file.")
if not CHANNEL_CHAT_ID:
    raise ValueError("CHANNEL_CHAT_ID is not set. Please check your .env file.")

# Initialize AI model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for embeddings

# Persistent LAST_UPDATE_ID
def get_last_update_id() -> Optional[int]:
    """Retrieve the last update ID from the file."""
    if os.path.exists(LAST_UPDATE_ID_FILE):
        try:
            with open(LAST_UPDATE_ID_FILE, "r") as file:
                return int(file.read().strip())
        except ValueError:
            logging.error("Error reading LAST_UPDATE_ID_FILE. Resetting to None.")
            return None
    return None

def save_last_update_id(update_id: int) -> None:
    """Save the last update ID to a file."""
    with open(LAST_UPDATE_ID_FILE, "w") as file:
        file.write(str(update_id))

# Database Manager Class
class DatabaseManager:
    """Handles all database operations for the bot."""
    
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_db()

    def init_db(self) -> None:
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_file) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS movies (
                            id INTEGER PRIMARY KEY,
                            normalized_name TEXT UNIQUE,
                            details TEXT,
                            file_id TEXT
                        )''')
            conn.commit()

    def get_all_movies(self) -> List[Tuple[str, str, str]]:
        """Fetch all movies from the database."""
        with sqlite3.connect(self.db_file) as conn:
            c = conn.cursor()
            c.execute("SELECT normalized_name, details, file_id FROM movies")
            return c.fetchall()

    def insert_movies(self, movies: List[Tuple[str, str, str]]) -> None:
        """Insert multiple movies into the database."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                c = conn.cursor()
                c.executemany("INSERT OR IGNORE INTO movies (normalized_name, details, file_id) VALUES (?, ?, ?)", movies)
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database error during batch insert: {e}")

# API Functions
def normalize_movie_name(caption: str) -> str:
    """Normalize a movie name by cleaning up the caption."""
    name = re.sub(r'[^\w\s]', ' ', caption)  # Remove special characters
    name = re.sub(r'\s+', ' ', name).strip()  # Normalize spaces
    match = re.search(r'(\d{4})', name)
    if match:
        year = match.group(1)
        name = re.sub(r'(\d{4})', '', name).strip()
        name = f"{name} ({year})"
    return name

def fetch_movie_details_from_omdb(movie_name: str) -> str:
    """Fetch movie details online using the OMDb API."""
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("Response") == "True":
            details = (
                f"Title: {data.get('Title', 'N/A')}\n"
                f"Year: {data.get('Year', 'N/A')}\n"
                f"Genre: {data.get('Genre', 'N/A')}\n"
                f"Director: {data.get('Director', 'N/A')}\n"
                f"Plot: {data.get('Plot', 'N/A')}"
            )
            return details
        else:
            logging.warning(f"OMDb API error: {data.get('Error', 'Unknown error')}")
            return "Movie details not available."
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from OMDb API: {e}")
        return "Error fetching movie details."

# Job Functions
def fetch_movies_from_channel(context: CallbackContext, db_manager: DatabaseManager) -> None:
    """Fetch movie files and captions from the private Telegram channel."""
    bot = context.bot
    chat_id = CHANNEL_CHAT_ID
    last_update_id = get_last_update_id()

    logging.info(f"Fetching updates from channel: {chat_id}")
    try:
        updates = bot.get_updates(offset=last_update_id)
    except RetryAfter as e:
        logging.error(f"Rate limit exceeded. Retrying after {e.retry_after} seconds.")
        time.sleep(e.retry_after)
        return
    except Conflict:
        logging.error("Another bot instance is running. Exiting.")
        return
    except Unauthorized:
        logging.error(f"Invalid or inaccessible chat_id: {chat_id}.")
        return
    except (TelegramError, BadRequest) as e:
        logging.error(f"Error while fetching updates: {e}")
        return

    movies_to_insert = []
    for update in updates:
        if update.update_id:
            save_last_update_id(update.update_id + 1)
        if update.message and update.message.document:
            document = update.message.document
            caption = update.message.caption or "No details available"
            normalized_name = normalize_movie_name(caption)

            details = caption
            if details == "No details available":
                online_details = fetch_movie_details_from_omdb(normalized_name)
                if online_details:
                    details = online_details

            movies_to_insert.append((normalized_name, details, document.file_id))

    if movies_to_insert:
        db_manager.insert_movies(movies_to_insert)

# Main Function
def main() -> None:
    """Main function to run the bot."""
    db_manager = DatabaseManager(DB_FILE)

    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Add command handlers
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("fetch", lambda update, context: fetch_movies(update, context, db_manager)))
    dp.add_handler(CallbackQueryHandler(lambda update, context: button(update, context, db_manager)))

    # Schedule jobs
    job_queue = updater.job_queue
    job_queue.run_repeating(lambda context: fetch_movies_from_channel(context, db_manager), interval=JOB_INTERVAL, first=10)

    # Start polling
    try:
        updater.start_polling()
    except Conflict:
        logging.error("Bot instance conflict detected. Ensure only one instance is running.")
    updater.idle()

if __name__ == "__main__":
    main()
