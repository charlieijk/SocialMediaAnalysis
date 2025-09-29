import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')

    # Model paths
    MODEL_DIR = 'backend/models/saved'
    DATA_DIR = 'data'

    # API settings
    MAX_TWEETS_PER_REQUEST = 100
    SENTIMENT_THRESHOLD = 0.5