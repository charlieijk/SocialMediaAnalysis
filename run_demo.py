#!/usr/bin/env python3
"""
Demo script for Social Media Sentiment Analysis Platform
This script demonstrates the core functionality without requiring Twitter API access.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.models.sentiment_models import SentimentAnalyzer
from backend.utils.text_preprocessor import TextPreprocessor
from backend.utils.feature_extractor import FeatureExtractor
from backend.utils.ab_testing import ABTestingFramework
import numpy as np

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis with sample data"""
    print("🎭 Social Media Sentiment Analysis Demo")
    print("=" * 50)

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
        "Amazing customer service! They went above and beyond!"
    ]

    # Expected labels for training (0=negative, 1=neutral, 2=positive)
    expected_labels = [2, 0, 1, 2, 0, 2, 0, 2, 0, 2]

    print("📝 Sample Tweets:")
    for i, tweet in enumerate(sample_tweets, 1):
        print(f"{i:2d}. {tweet}")

    print("\n🔧 Initializing sentiment analysis components...")

    # Initialize components
    text_preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.load_dependencies(feature_extractor, text_preprocessor)

    print("✅ Components initialized successfully!")

    print("\n🚀 Training models on sample data...")

    # Train models
    try:
        sentiment_analyzer.train_all_models(sample_tweets, expected_labels)
        print("✅ All models trained successfully!")

        # Save models
        sentiment_analyzer.save_models()
        print("💾 Models saved to disk!")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return

    print("\n🔍 Testing individual models...")

    # Test each model
    models_to_test = ['naive_bayes', 'svm', 'logistic_regression', 'neural_network']

    results = {}
    for model_name in models_to_test:
        try:
            predictions, probabilities = sentiment_analyzer.predict(sample_tweets, model_name)

            # Calculate accuracy
            accuracy = np.mean(predictions == np.array(expected_labels))
            results[model_name] = {
                'predictions': predictions,
                'accuracy': accuracy,
                'probabilities': probabilities
            }

            print(f"📊 {model_name:20s}: Accuracy = {accuracy:.2%}")

        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")

    print("\n🎯 Testing ensemble method...")
    try:
        ensemble_predictions, ensemble_probabilities = sentiment_analyzer.predict(sample_tweets, 'ensemble')
        ensemble_accuracy = np.mean(ensemble_predictions == np.array(expected_labels))
        print(f"📊 {'ensemble':20s}: Accuracy = {ensemble_accuracy:.2%}")

        results['ensemble'] = {
            'predictions': ensemble_predictions,
            'accuracy': ensemble_accuracy,
            'probabilities': ensemble_probabilities
        }
    except Exception as e:
        print(f"❌ Error testing ensemble: {e}")

    print("\n📈 Individual Predictions Analysis:")
    print("-" * 80)
    sentiment_labels = ['Negative 😞', 'Neutral 😐', 'Positive 😊']

    for i, tweet in enumerate(sample_tweets):
        print(f"\nTweet {i+1}: {tweet[:50]}{'...' if len(tweet) > 50 else ''}")
        print(f"Expected: {sentiment_labels[expected_labels[i]]}")

        for model_name, result in results.items():
            prediction = result['predictions'][i]
            if result['probabilities'] is not None:
                confidence = np.max(result['probabilities'][i]) * 100
                print(f"{model_name:15s}: {sentiment_labels[prediction]} ({confidence:.1f}% confidence)")
            else:
                print(f"{model_name:15s}: {sentiment_labels[prediction]}")

    return results

def demo_ab_testing(results):
    """Demonstrate A/B testing framework"""
    print("\n\n🧪 A/B Testing Framework Demo")
    print("=" * 50)

    # Initialize A/B testing framework
    ab_framework = ABTestingFramework()

    # Create an experiment
    models_to_test = list(results.keys())
    experiment_id = ab_framework.create_experiment(
        name="Demo Model Comparison",
        description="Comparing sentiment analysis models on sample data",
        models_to_test=models_to_test
    )

    print(f"📋 Created experiment #{experiment_id}: Demo Model Comparison")

    # Sample test data
    test_tweets = [
        "Love the new features! Great job! 🎉",
        "This is confusing and hard to use.",
        "It's fine, works as expected.",
        "Incredible performance! Exceeded expectations!",
        "Terrible experience. Very disappointed."
    ]
    test_labels = [2, 0, 1, 2, 0]

    print(f"🧪 Running experiment with {len(test_tweets)} test samples...")

    try:
        # Run experiment
        experiment = ab_framework.run_experiment(experiment_id, test_tweets, test_labels)

        print("✅ Experiment completed successfully!")

        # Generate report
        report = ab_framework.generate_experiment_report(experiment_id, save_plots=False)

        print("\n📊 Experiment Results:")
        print("-" * 40)

        # Show model rankings
        if report and 'model_rankings' in report:
            print("\n🏆 Model Rankings (Overall):")
            for rank, model, score in report['model_rankings']['overall']:
                print(f"{rank}. {model:15s}: {score:.3f}")

        # Show best models by metric
        print("\n🎯 Best Models by Metric:")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if report and 'detailed_metrics' in report and metric in report['detailed_metrics']:
                best_model = report['detailed_metrics'][metric]['best_model']
                best_value = report['detailed_metrics'][metric]['values'][best_model]
                print(f"{metric.capitalize():10s}: {best_model} ({best_value:.3f})")

        # Show recommendations
        if report and 'recommendations' in report:
            print("\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"• {rec}")

    except Exception as e:
        print(f"❌ Error running A/B test: {e}")

def demo_text_preprocessing():
    """Demonstrate text preprocessing capabilities"""
    print("\n\n🔧 Text Preprocessing Demo")
    print("=" * 50)

    text_preprocessor = TextPreprocessor()

    sample_text = "OMG! I LOVE this new iPhone 📱! Best purchase EVER!!! https://apple.com @apple #iPhone #amazing"

    print(f"Original text: {sample_text}")
    print(f"Cleaned text:  {text_preprocessor.preprocess(sample_text)}")

    # Show preprocessing steps
    print("\n🔍 Preprocessing Steps:")
    print(f"1. Cleaned:     {text_preprocessor.clean_text(sample_text)}")
    print(f"2. Tokenized:   {text_preprocessor.tokenize_and_lemmatize(text_preprocessor.clean_text(sample_text))}")

if __name__ == "__main__":
    print("🚀 Starting Social Media Sentiment Analysis Demo...")
    print("This demo showcases the platform's capabilities without requiring API access.\n")

    try:
        # Demo 1: Text Preprocessing
        demo_text_preprocessing()

        # Demo 2: Sentiment Analysis
        results = demo_sentiment_analysis()

        # Demo 3: A/B Testing (only if sentiment analysis succeeded)
        if results:
            demo_ab_testing(results)

        print("\n\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Set up your Twitter API credentials in .env")
        print("2. Start the Flask backend: python backend/api/flask_app.py")
        print("3. Start the React frontend: cd frontend && npm start")
        print("4. Visit http://localhost:3000 to use the full platform")

    except KeyboardInterrupt:
        print("\n\n⛔ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        print("Please check the requirements and try again.")

    print("\n👋 Thanks for trying the Social Media Sentiment Analysis Platform!")