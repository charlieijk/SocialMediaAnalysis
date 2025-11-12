from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import threading
import time
from datetime import datetime, timedelta
from backend.api.twitter_client import TwitterClient
from backend.models.sentiment_models import SentimentAnalyzer
from backend.utils.text_preprocessor import TextPreprocessor
from backend.utils.feature_extractor import FeatureExtractor
from backend.api.ab_testing_api import register_ab_testing_routes

app = Flask(__name__)
CORS(app)

# Register A/B testing routes
register_ab_testing_routes(app)

# Initialize components
twitter_client = TwitterClient()
text_preprocessor = TextPreprocessor()
feature_extractor = FeatureExtractor()
sentiment_analyzer = SentimentAnalyzer()
sentiment_analyzer.load_dependencies(feature_extractor, text_preprocessor)

# Global variables for real-time data
real_time_data = {
    'tweets': [],
    'sentiment_history': [],
    'last_update': None
}

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/search', methods=['POST'])
def search_tweets():
    try:
        data = request.get_json()
        query = data.get('query', '')
        max_results = data.get('max_results', 100)

        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400

        tweets = twitter_client.search_tweets(query, max_results)

        # Analyze sentiment for each tweet
        analyzed_tweets = []
        for tweet in tweets:
            try:
                predictions, probabilities = sentiment_analyzer.predict([tweet['text']], 'ensemble')
                sentiment_label = ['negative', 'neutral', 'positive'][predictions[0]]
                confidence = float(max(probabilities[0])) if probabilities is not None else 0.0

                tweet['sentiment'] = {
                    'label': sentiment_label,
                    'confidence': confidence,
                    'probabilities': probabilities[0].tolist() if probabilities is not None else []
                }
                analyzed_tweets.append(tweet)
            except Exception as e:
                print(f"Error analyzing sentiment for tweet {tweet['id']}: {e}")
                tweet['sentiment'] = {'label': 'neutral', 'confidence': 0.0}
                analyzed_tweets.append(tweet)

        # Calculate overall sentiment statistics
        sentiment_stats = calculate_sentiment_stats(analyzed_tweets)

        return jsonify({
            'tweets': analyzed_tweets,
            'sentiment_stats': sentiment_stats,
            'total_count': len(analyzed_tweets)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/start', methods=['POST'])
def start_streaming():
    try:
        data = request.get_json()
        keywords = data.get('keywords', [])

        if not keywords:
            return jsonify({'error': 'Keywords are required'}), 400

        twitter_client.start_streaming(keywords)

        # Start background thread for processing streaming data
        processing_thread = threading.Thread(target=process_streaming_data)
        processing_thread.daemon = True
        processing_thread.start()

        return jsonify({
            'message': 'Streaming started successfully',
            'keywords': keywords,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/stop', methods=['POST'])
def stop_streaming():
    try:
        twitter_client.stop_streaming()
        return jsonify({
            'message': 'Streaming stopped successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/data', methods=['GET'])
def get_streaming_data():
    try:
        return jsonify({
            'tweets': real_time_data['tweets'][-100:],  # Last 100 tweets
            'sentiment_history': real_time_data['sentiment_history'][-50:],  # Last 50 sentiment points
            'last_update': real_time_data['last_update'],
            'is_streaming': twitter_client.is_streaming
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        model_name = data.get('model', 'ensemble')

        if not texts:
            return jsonify({'error': 'Texts are required'}), 400

        predictions, probabilities = sentiment_analyzer.predict(texts, model_name)

        results = []
        for i, text in enumerate(texts):
            sentiment_label = ['negative', 'neutral', 'positive'][predictions[i]]
            confidence = float(max(probabilities[i])) if probabilities is not None else 0.0

            results.append({
                'text': text,
                'sentiment': {
                    'label': sentiment_label,
                    'confidence': confidence,
                    'probabilities': probabilities[i].tolist() if probabilities is not None else []
                }
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends', methods=['GET'])
def get_trending_topics():
    try:
        trends = twitter_client.get_trending_topics()
        return jsonify({'trends': trends})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<username>', methods=['GET'])
def get_user_timeline(username):
    try:
        max_results = request.args.get('max_results', 100, type=int)
        tweets = twitter_client.get_user_timeline(username, max_results)

        # Analyze sentiment for user tweets
        analyzed_tweets = []
        for tweet in tweets:
            try:
                predictions, probabilities = sentiment_analyzer.predict([tweet['text']], 'ensemble')
                sentiment_label = ['negative', 'neutral', 'positive'][predictions[0]]
                confidence = float(max(probabilities[0])) if probabilities is not None else 0.0

                tweet['sentiment'] = {
                    'label': sentiment_label,
                    'confidence': confidence
                }
                analyzed_tweets.append(tweet)
            except Exception as e:
                print(f"Error analyzing sentiment: {e}")
                tweet['sentiment'] = {'label': 'neutral', 'confidence': 0.0}
                analyzed_tweets.append(tweet)

        sentiment_stats = calculate_sentiment_stats(analyzed_tweets)

        return jsonify({
            'username': username,
            'tweets': analyzed_tweets,
            'sentiment_stats': sentiment_stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_sentiment_stats(tweets):
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    total_confidence = 0

    for tweet in tweets:
        sentiment = tweet.get('sentiment', {})
        label = sentiment.get('label', 'neutral')
        confidence = sentiment.get('confidence', 0)

        sentiment_counts[label] += 1
        total_confidence += confidence

    total_tweets = len(tweets)

    return {
        'counts': sentiment_counts,
        'percentages': {
            label: (count / total_tweets * 100) if total_tweets > 0 else 0
            for label, count in sentiment_counts.items()
        },
        'average_confidence': total_confidence / total_tweets if total_tweets > 0 else 0,
        'total_tweets': total_tweets
    }

def process_streaming_data():
    while True:
        try:
            if twitter_client.is_streaming:
                new_tweets = twitter_client.get_streaming_tweets(10)

                if new_tweets:
                    # Analyze sentiment for new tweets
                    for tweet in new_tweets:
                        try:
                            predictions, probabilities = sentiment_analyzer.predict([tweet['text']], 'ensemble')
                            sentiment_label = ['negative', 'neutral', 'positive'][predictions[0]]
                            confidence = float(max(probabilities[0])) if probabilities is not None else 0.0

                            tweet['sentiment'] = {
                                'label': sentiment_label,
                                'confidence': confidence
                            }
                        except Exception as e:
                            print(f"Error analyzing streaming tweet sentiment: {e}")
                            tweet['sentiment'] = {'label': 'neutral', 'confidence': 0.0}

                    # Update real-time data
                    real_time_data['tweets'].extend(new_tweets)
                    real_time_data['tweets'] = real_time_data['tweets'][-1000:]  # Keep last 1000 tweets

                    # Calculate sentiment distribution for the last hour
                    current_time = datetime.now()
                    sentiment_point = {
                        'timestamp': current_time.isoformat(),
                        'sentiment_counts': calculate_sentiment_stats(new_tweets)['counts']
                    }
                    real_time_data['sentiment_history'].append(sentiment_point)
                    real_time_data['sentiment_history'] = real_time_data['sentiment_history'][-100:]  # Keep last 100 points

                    real_time_data['last_update'] = current_time.isoformat()

            time.sleep(5)  # Process every 5 seconds

        except Exception as e:
            print(f"Error in streaming data processing: {e}")
            time.sleep(10)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)