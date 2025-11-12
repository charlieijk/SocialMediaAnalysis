#!/usr/bin/env python3
"""
Simplified demo script for Social Media Sentiment Analysis Platform
This version uses only basic dependencies to showcase core functionality.
"""

import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Simple Text Preprocessor
class SimpleTextPreprocessor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return lemmatized_tokens

    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        return ' '.join(tokens)

# Simple Sentiment Analyzer
class SimpleSentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.preprocessor = SimpleTextPreprocessor()

    def prepare_data(self, texts, labels):
        # Preprocess texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]

        # Convert to TF-IDF features
        X = self.vectorizer.fit_transform(preprocessed_texts)
        y = np.array(labels)

        return X, y

    def train_models(self, X, y):
        """Train multiple sentiment analysis models"""
        print("🔧 Training Naive Bayes model...")
        nb_model = MultinomialNB()
        nb_model.fit(X, y)
        self.models['naive_bayes'] = nb_model

        print("🔧 Training SVM model...")
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X, y)
        self.models['svm'] = svm_model

        print("🔧 Training Logistic Regression model...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X, y)
        self.models['logistic_regression'] = lr_model

    def predict(self, texts, model_name):
        """Make predictions using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        # Preprocess texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]

        # Transform to features
        X = self.vectorizer.transform(preprocessed_texts)

        # Make predictions
        model = self.models[model_name]
        predictions = model.predict(X)

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
        else:
            probabilities = None

        return predictions, probabilities

def run_simple_demo():
    """Run a simplified sentiment analysis demo"""
    print("🎭 Social Media Sentiment Analysis - Simple Demo")
    print("=" * 55)

    # Sample tweets for demonstration
    sample_tweets = [
        "I absolutely love this new product! It's amazing! 😍",
        "This is the worst service I've ever experienced. Terrible!",
        "The weather is okay today, nothing special.",
        "Best purchase I've made this year! Highly recommend! 🌟",
        "Not impressed with the quality. Could be better.",
        "Feeling great today! Life is beautiful! ☀️",
        "The movie was boring and too long. Waste of time.",
        "Pretty good overall experience. Satisfied with the results.",
        "Absolutely hate this new update. Ruined everything!",
        "Amazing customer service! They went above and beyond!",
        "This is confusing and hard to use.",
        "Love the new features! Great job! 🎉",
        "It's fine, works as expected.",
        "Incredible performance! Exceeded expectations!",
        "Terrible experience. Very disappointed."
    ]

    # Expected labels (0=negative, 1=neutral, 2=positive)
    expected_labels = [2, 0, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 2, 0]

    print("📝 Sample Tweets for Analysis:")
    sentiment_names = ['😞 Negative', '😐 Neutral', '😊 Positive']
    for i, (tweet, label) in enumerate(zip(sample_tweets, expected_labels), 1):
        print(f"{i:2d}. [{sentiment_names[label]}] {tweet}")

    print(f"\n🔧 Initializing sentiment analyzer...")
    analyzer = SimpleSentimentAnalyzer()

    print("📊 Preparing training data...")
    X, y = analyzer.prepare_data(sample_tweets, expected_labels)
    print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")

    print("\n🚀 Training models...")
    analyzer.train_models(X, y)
    print("✅ All models trained successfully!")

    print("\n🎯 Testing model performance...")
    models_to_test = ['naive_bayes', 'svm', 'logistic_regression']

    results = {}
    for model_name in models_to_test:
        predictions, probabilities = analyzer.predict(sample_tweets, model_name)
        accuracy = accuracy_score(expected_labels, predictions)
        results[model_name] = {
            'predictions': predictions,
            'accuracy': accuracy,
            'probabilities': probabilities
        }
        print(f"📊 {model_name.replace('_', ' ').title():20s}: Accuracy = {accuracy:.1%}")

    print("\n📈 Detailed Predictions Analysis:")
    print("-" * 80)

    for i, tweet in enumerate(sample_tweets):
        print(f"\nTweet {i+1}: {tweet[:60]}{'...' if len(tweet) > 60 else ''}")
        print(f"Expected: {sentiment_names[expected_labels[i]]}")

        for model_name, result in results.items():
            prediction = result['predictions'][i]
            predicted_sentiment = sentiment_names[prediction]

            if result['probabilities'] is not None:
                confidence = np.max(result['probabilities'][i]) * 100
                status = "✅" if prediction == expected_labels[i] else "❌"
                print(f"  {status} {model_name.replace('_', ' ').title():15s}: {predicted_sentiment} ({confidence:.1f}% confidence)")
            else:
                status = "✅" if prediction == expected_labels[i] else "❌"
                print(f"  {status} {model_name.replace('_', ' ').title():15s}: {predicted_sentiment}")

    print("\n🏆 Model Performance Summary:")
    print("-" * 40)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (model_name, result) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name.replace('_', ' ').title():20s}: {result['accuracy']:.1%}")

    print(f"\n🎉 Demo completed successfully!")
    print("\nWhat this demo showed:")
    print("• Text preprocessing and cleaning")
    print("• TF-IDF feature extraction")
    print("• Training multiple ML models (Naive Bayes, SVM, Logistic Regression)")
    print("• Model comparison and performance evaluation")
    print("• Confidence scoring for predictions")

    return results

if __name__ == "__main__":
    print("🚀 Starting Simple Social Media Sentiment Analysis Demo...")
    print("This simplified version uses only basic ML models and dependencies.\n")

    try:
        results = run_simple_demo()

        print("\n💡 Next Steps:")
        print("1. Install additional dependencies (spaCy, TensorFlow) for advanced features")
        print("2. Set up Twitter API credentials for real-time analysis")
        print("3. Start the web application:")
        print("   - Backend: python backend/api/flask_app.py")
        print("   - Frontend: cd frontend && npm start")

    except KeyboardInterrupt:
        print("\n\n⛔ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n👋 Thanks for trying the Social Media Sentiment Analysis Platform!")