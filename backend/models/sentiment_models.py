import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.feature_extractor = None
        self.text_preprocessor = None
        self.tokenizer = None
        self.max_sequence_length = 100

    def load_dependencies(self, feature_extractor, text_preprocessor):
        self.feature_extractor = feature_extractor
        self.text_preprocessor = text_preprocessor

    def prepare_data(self, texts, labels):
        preprocessed_texts = [self.text_preprocessor.preprocess(text) for text in texts]
        features = self.feature_extractor.extract_all_features(preprocessed_texts, fit=True)
        return features, labels

    def train_naive_bayes(self, X, y):
        nb_model = MultinomialNB(alpha=1.0)
        nb_model.fit(X, y)
        self.models['naive_bayes'] = nb_model
        return nb_model

    def train_svm(self, X, y):
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        svm_model.fit(X, y)
        self.models['svm'] = svm_model
        return svm_model

    def train_random_forest(self, X, y):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        self.models['random_forest'] = rf_model
        return rf_model

    def train_logistic_regression(self, X, y):
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X, y)
        self.models['logistic_regression'] = lr_model
        return lr_model

    def create_neural_network(self, input_dim, num_classes=3):
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_neural_network(self, X, y, validation_split=0.2, epochs=50):
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))

        nn_model = self.create_neural_network(input_dim, num_classes)

        history = nn_model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )

        self.models['neural_network'] = nn_model
        return nn_model, history

    def create_lstm_model(self, vocab_size, num_classes=3):
        model = Sequential([
            Embedding(vocab_size, 128, input_length=self.max_sequence_length),
            LSTM(128, dropout=0.5, recurrent_dropout=0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_lstm(self, texts, labels, validation_split=0.2, epochs=10):
        # Tokenize texts for LSTM
        self.tokenizer = Tokenizer(num_words=10000)
        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)

        vocab_size = len(self.tokenizer.word_index) + 1
        num_classes = len(np.unique(labels))

        lstm_model = self.create_lstm_model(vocab_size, num_classes)

        history = lstm_model.fit(
            X, labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )

        self.models['lstm'] = lstm_model
        return lstm_model, history

    def train_all_models(self, texts, labels):
        print("Preparing data...")
        X, y = self.prepare_data(texts, labels)

        print("Training traditional ML models...")
        self.train_naive_bayes(X, y)
        self.train_svm(X, y)
        self.train_random_forest(X, y)
        self.train_logistic_regression(X, y)

        print("Training neural network...")
        self.train_neural_network(X, y)

        print("Training LSTM...")
        self.train_lstm(texts, labels)

        print("All models trained successfully!")

    def predict(self, texts, model_name='ensemble'):
        if model_name == 'ensemble':
            return self.ensemble_predict(texts)

        if model_name == 'lstm':
            return self.predict_lstm(texts)

        preprocessed_texts = [self.text_preprocessor.preprocess(text) for text in texts]
        features = self.feature_extractor.extract_all_features(preprocessed_texts, fit=False)

        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            predictions = np.argmax(probabilities, axis=1)
            return predictions, probabilities
        else:
            predictions = model.predict(features)
            return predictions, None

    def predict_lstm(self, texts):
        if self.tokenizer is None:
            raise ValueError("LSTM model not trained or tokenizer not available")

        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)

        probabilities = self.models['lstm'].predict(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities

    def ensemble_predict(self, texts):
        predictions = {}
        probabilities = {}

        # Get predictions from traditional ML models
        for model_name in ['naive_bayes', 'svm', 'random_forest', 'logistic_regression']:
            if model_name in self.models:
                pred, prob = self.predict(texts, model_name)
                predictions[model_name] = pred
                if prob is not None:
                    probabilities[model_name] = prob

        # Get LSTM predictions
        if 'lstm' in self.models:
            pred, prob = self.predict_lstm(texts)
            predictions['lstm'] = pred
            probabilities['lstm'] = prob

        # Ensemble voting
        if predictions:
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=pred_array)

            # Average probabilities if available
            if probabilities:
                prob_array = np.array(list(probabilities.values()))
                ensemble_prob = np.mean(prob_array, axis=0)
            else:
                ensemble_prob = None

            return ensemble_pred, ensemble_prob

        raise ValueError("No models available for prediction")

    def evaluate_models(self, texts, labels):
        results = {}
        X, y = self.prepare_data(texts, labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name, model in self.models.items():
            if model_name == 'lstm':
                continue  # LSTM evaluation handled separately

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred)
            }

        return results

    def save_models(self, model_dir='backend/models/saved'):
        os.makedirs(model_dir, exist_ok=True)

        for model_name, model in self.models.items():
            if model_name == 'lstm' or model_name == 'neural_network':
                model.save(f'{model_dir}/{model_name}_model.h5')
            else:
                joblib.dump(model, f'{model_dir}/{model_name}_model.pkl')

        # Save tokenizer
        if self.tokenizer:
            joblib.dump(self.tokenizer, f'{model_dir}/tokenizer.pkl')

    def load_models(self, model_dir='backend/models/saved'):
        model_files = {
            'naive_bayes': f'{model_dir}/naive_bayes_model.pkl',
            'svm': f'{model_dir}/svm_model.pkl',
            'random_forest': f'{model_dir}/random_forest_model.pkl',
            'logistic_regression': f'{model_dir}/logistic_regression_model.pkl',
            'lstm': f'{model_dir}/lstm_model.h5',
            'neural_network': f'{model_dir}/neural_network_model.h5'
        }

        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                if model_name in ['lstm', 'neural_network']:
                    self.models[model_name] = tf.keras.models.load_model(file_path)
                else:
                    self.models[model_name] = joblib.load(file_path)

        # Load tokenizer
        tokenizer_path = f'{model_dir}/tokenizer.pkl'
        if os.path.exists(tokenizer_path):
            self.tokenizer = joblib.load(tokenizer_path)