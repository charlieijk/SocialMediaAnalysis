import tweepy
import json
import time
import threading
from queue import Queue
from datetime import datetime
from backend.config import Config

class TwitterStreamListener(tweepy.StreamingClient):
    def __init__(self, bearer_token, tweet_queue, keywords=None):
        super().__init__(bearer_token, wait_on_rate_limit=True)
        self.tweet_queue = tweet_queue
        self.keywords = keywords or []
        self.is_streaming = False

    def on_tweet(self, tweet):
        try:
            tweet_data = {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': datetime.now().isoformat(),
                'author_id': tweet.author_id,
                'public_metrics': getattr(tweet, 'public_metrics', {}),
                'lang': getattr(tweet, 'lang', 'en'),
                'context_annotations': getattr(tweet, 'context_annotations', []),
                'referenced_tweets': getattr(tweet, 'referenced_tweets', [])
            }
            self.tweet_queue.put(tweet_data)
            return True
        except Exception as e:
            print(f"Error processing tweet: {e}")
            return True

    def on_error(self, status_code):
        print(f"Twitter API error: {status_code}")
        if status_code == 420:
            time.sleep(60)
        return True

class TwitterClient:
    def __init__(self):
        self.config = Config()
        self.api = None
        self.stream_listener = None
        self.tweet_queue = Queue()
        self.streaming_thread = None
        self.is_streaming = False
        self.setup_api()

    def setup_api(self):
        try:
            # Setup API v2 client
            self.client = tweepy.Client(
                bearer_token=self.config.TWITTER_BEARER_TOKEN,
                consumer_key=self.config.TWITTER_API_KEY,
                consumer_secret=self.config.TWITTER_API_SECRET,
                access_token=self.config.TWITTER_ACCESS_TOKEN,
                access_token_secret=self.config.TWITTER_ACCESS_TOKEN_SECRET,
                wait_on_rate_limit=True
            )

            # Setup API v1.1 for streaming
            auth = tweepy.OAuthHandler(
                self.config.TWITTER_API_KEY,
                self.config.TWITTER_API_SECRET
            )
            auth.set_access_token(
                self.config.TWITTER_ACCESS_TOKEN,
                self.config.TWITTER_ACCESS_TOKEN_SECRET
            )
            self.api = tweepy.API(auth, wait_on_rate_limit=True)

            print("Twitter API initialized successfully")
        except Exception as e:
            print(f"Error initializing Twitter API: {e}")

    def search_tweets(self, query, max_results=100, lang='en'):
        try:
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang', 'context_annotations']
            ).flatten(limit=max_results)

            tweet_data = []
            for tweet in tweets:
                data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat() if tweet.created_at else None,
                    'author_id': tweet.author_id,
                    'public_metrics': getattr(tweet, 'public_metrics', {}),
                    'lang': getattr(tweet, 'lang', 'en'),
                    'context_annotations': getattr(tweet, 'context_annotations', [])
                }
                tweet_data.append(data)

            return tweet_data
        except Exception as e:
            print(f"Error searching tweets: {e}")
            return []

    def start_streaming(self, keywords, languages=['en']):
        try:
            if self.is_streaming:
                print("Streaming already in progress")
                return

            self.stream_listener = TwitterStreamListener(
                self.config.TWITTER_BEARER_TOKEN,
                self.tweet_queue,
                keywords
            )

            # Add rules for streaming
            for keyword in keywords:
                rule = tweepy.StreamRule(value=f"{keyword} lang:en")
                self.stream_listener.add_rules(rule)

            # Start streaming in a separate thread
            self.streaming_thread = threading.Thread(
                target=self._stream_worker,
                args=(languages,)
            )
            self.streaming_thread.daemon = True
            self.streaming_thread.start()

            self.is_streaming = True
            print(f"Started streaming for keywords: {keywords}")

        except Exception as e:
            print(f"Error starting stream: {e}")

    def _stream_worker(self, languages):
        try:
            self.stream_listener.filter(threaded=True)
        except Exception as e:
            print(f"Streaming error: {e}")
            self.is_streaming = False

    def stop_streaming(self):
        try:
            if self.stream_listener:
                self.stream_listener.disconnect()
            self.is_streaming = False
            print("Streaming stopped")
        except Exception as e:
            print(f"Error stopping stream: {e}")

    def get_streaming_tweets(self, max_tweets=10):
        tweets = []
        count = 0
        while not self.tweet_queue.empty() and count < max_tweets:
            tweets.append(self.tweet_queue.get())
            count += 1
        return tweets

    def get_user_timeline(self, username, max_results=100):
        try:
            user = self.client.get_user(username=username)
            if not user.data:
                return []

            tweets = tweepy.Paginator(
                self.client.get_users_tweets,
                id=user.data.id,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'lang']
            ).flatten(limit=max_results)

            tweet_data = []
            for tweet in tweets:
                data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat() if tweet.created_at else None,
                    'public_metrics': getattr(tweet, 'public_metrics', {}),
                    'lang': getattr(tweet, 'lang', 'en')
                }
                tweet_data.append(data)

            return tweet_data
        except Exception as e:
            print(f"Error getting user timeline: {e}")
            return []

    def get_trending_topics(self, woeid=1):  # 1 = worldwide
        try:
            trends = self.api.get_place_trends(woeid)
            if trends:
                return [trend['name'] for trend in trends[0]['trends'][:10]]
            return []
        except Exception as e:
            print(f"Error getting trending topics: {e}")
            return []

    def analyze_tweet_metrics(self, tweets):
        metrics = {
            'total_tweets': len(tweets),
            'total_retweets': sum(tweet.get('public_metrics', {}).get('retweet_count', 0) for tweet in tweets),
            'total_likes': sum(tweet.get('public_metrics', {}).get('like_count', 0) for tweet in tweets),
            'total_replies': sum(tweet.get('public_metrics', {}).get('reply_count', 0) for tweet in tweets),
            'avg_engagement': 0
        }

        if metrics['total_tweets'] > 0:
            total_engagement = metrics['total_retweets'] + metrics['total_likes'] + metrics['total_replies']
            metrics['avg_engagement'] = total_engagement / metrics['total_tweets']

        return metrics