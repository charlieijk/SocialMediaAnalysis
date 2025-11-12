import joblib
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)


class TransformerDataset(Dataset):
    """Minimal dataset wrapper for Hugging Face Trainer."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.feature_extractor = None
        self.text_preprocessor = None
        self.tokenizer = None
        self.max_sequence_length = 100
        self.model_selection_results = {}
        self.model_search_space = self._default_model_search_space()
        self.transformer_tokenizer = None
        self.transformer_model_name = 'distilbert-base-uncased'
        self.transformer_max_length = 160

    def load_dependencies(self, feature_extractor, text_preprocessor):
        self.feature_extractor = feature_extractor
        self.text_preprocessor = text_preprocessor

    def prepare_data(self, texts, labels):
        preprocessed_texts = [self.text_preprocessor.preprocess(text) for text in texts]
        features = self.feature_extractor.extract_all_features(preprocessed_texts, fit=True)
        return features, labels

    def _default_model_search_space(self):
        return {
            'naive_bayes': {
                'estimator': MultinomialNB(),
                'search': 'grid',
                'param_grid': {'alpha': [0.5, 1.0, 1.5]},
                'scoring': 'accuracy',
                'cv': 5
            },
            'svm': {
                'estimator': SVC(probability=True),
                'search': 'grid',
                'param_grid': {
                    'kernel': ['linear', 'rbf'],
                    'C': [0.5, 1.0, 2.0],
                    'gamma': ['scale', 'auto']
                },
                'scoring': 'accuracy',
                'cv': 3
            },
            'random_forest': {
                'estimator': RandomForestClassifier(random_state=42),
                'search': 'random',
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 20, 40],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2, 4]
                },
                'n_iter': 6,
                'scoring': 'accuracy',
                'cv': 3
            },
            'logistic_regression': {
                'estimator': LogisticRegression(max_iter=1000, random_state=42),
                'search': 'grid',
                'param_grid': {
                    'C': [0.1, 1.0, 3.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'saga']
                },
                'scoring': 'accuracy',
                'cv': 5
            }
        }

    def configure_model_search(self, overrides):
        """Allow callers to override default search spaces."""
        self.model_search_space.update(overrides)

    def run_model_selection(self, model_key, X, y):
        config = self.model_search_space.get(model_key)
        if not config:
            raise ValueError(f"No configuration available for model '{model_key}'")

        estimator = config['estimator']
        param_grid = config.get('param_grid')
        search_type = config.get('search', 'grid')
        scoring = config.get('scoring', 'accuracy')
        cv = config.get('cv', 5)

        if param_grid:
            if search_type == 'random':
                search = RandomizedSearchCV(
                    estimator,
                    param_distributions=param_grid,
                    n_iter=config.get('n_iter', 10),
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            else:
                search = GridSearchCV(
                    estimator,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    verbose=1
                )
            search.fit(X, y)
            best_model = search.best_estimator_
            self.model_selection_results[model_key] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_
            }
        else:
            best_model = estimator.fit(X, y)
            train_accuracy = accuracy_score(y, best_model.predict(X))
            self.model_selection_results[model_key] = {
                'best_params': {},
                'best_score': train_accuracy
            }

        self.models[model_key] = best_model
        return best_model

    def train_naive_bayes(self, X, y):
        return self.run_model_selection('naive_bayes', X, y)

    def train_svm(self, X, y):
        return self.run_model_selection('svm', X, y)

    def train_random_forest(self, X, y):
        return self.run_model_selection('random_forest', X, y)

    def train_logistic_regression(self, X, y):
        return self.run_model_selection('logistic_regression', X, y)

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
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
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
        y = np.asarray(labels, dtype=np.int32)

        vocab_size = len(self.tokenizer.word_index) + 1
        num_classes = len(np.unique(y))

        lstm_model = self.create_lstm_model(vocab_size, num_classes)

        history = lstm_model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )

        self.models['lstm'] = lstm_model
        return lstm_model, history

    def train_transformer(self, texts, labels, base_model=None, epochs=3, batch_size=16, validation_split=0.2):
        if base_model:
            self.transformer_model_name = base_model

        labels = np.array(labels)
        stratify = labels if len(np.unique(labels)) > 1 else None
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=validation_split,
            random_state=42,
            stratify=stratify
        )

        tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        self.transformer_tokenizer = tokenizer

        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.transformer_max_length
        )
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=self.transformer_max_length
        )

        train_dataset = TransformerDataset(train_encodings, train_labels)
        val_dataset = TransformerDataset(val_encodings, val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.transformer_model_name,
            num_labels=len(np.unique(labels))
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir='backend/models/saved/transformer-checkpoints',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy='epoch',
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_strategy='steps',
            logging_steps=10,
            save_strategy='no',
            report_to=[]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_transformer_metrics
        )

        trainer.train()
        self.models['transformer'] = model
        self.model_selection_results['transformer'] = {
            'base_model': self.transformer_model_name,
            'epochs': epochs,
            'learning_rate': training_args.learning_rate
        }
        return model, trainer.state.log_history

    def train_all_models(self, texts, labels, include_transformer=True):
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

        if include_transformer:
            print("Fine-tuning transformer model...")
            self.train_transformer(texts, labels)

        print("All models trained successfully!")

    def predict(self, texts, model_name='ensemble'):
        if model_name == 'ensemble':
            return self.ensemble_predict(texts)

        if model_name == 'lstm':
            return self.predict_lstm(texts)

        if model_name == 'transformer':
            return self.predict_transformer(texts)

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

    def predict_transformer(self, texts):
        if 'transformer' not in self.models or self.transformer_tokenizer is None:
            raise ValueError("Transformer model not available. Train or load it before prediction.")

        encoding = self.transformer_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.transformer_max_length,
            return_tensors='pt'
        )

        model = self.models['transformer']
        device = next(model.parameters()).device
        encoding = {k: v.to(device) for k, v in encoding.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

        return predictions, probabilities

    def ensemble_predict(self, texts):
        predictions = {}
        probabilities = {}

        # Get predictions from traditional ML models
        for model_name in ['naive_bayes', 'svm', 'random_forest', 'logistic_regression', 'transformer']:
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
            if model_name in ['lstm', 'transformer']:
                continue  # LSTM evaluation handled separately

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred)
            }

        if 'lstm' in self.models:
            lstm_pred, _ = self.predict_lstm(texts)
            results['lstm'] = {
                'accuracy': accuracy_score(labels, lstm_pred),
                'classification_report': classification_report(labels, lstm_pred)
            }

        if 'transformer' in self.models:
            transformer_pred, _ = self.predict_transformer(texts)
            results['transformer'] = {
                'accuracy': accuracy_score(labels, transformer_pred),
                'classification_report': classification_report(labels, transformer_pred)
            }

        return results

    def save_models(self, model_dir='backend/models/saved'):
        os.makedirs(model_dir, exist_ok=True)

        for model_name, model in self.models.items():
            if model_name in ['lstm', 'neural_network']:
                model.save(f'{model_dir}/{model_name}_model.h5')
            elif model_name == 'transformer':
                transformer_model_dir = f'{model_dir}/transformer_model'
                os.makedirs(transformer_model_dir, exist_ok=True)
                model.save_pretrained(transformer_model_dir)
            else:
                joblib.dump(model, f'{model_dir}/{model_name}_model.pkl')

        # Save tokenizer
        if self.tokenizer:
            joblib.dump(self.tokenizer, f'{model_dir}/tokenizer.pkl')

        if self.transformer_tokenizer:
            tokenizer_dir = f'{model_dir}/transformer_tokenizer'
            os.makedirs(tokenizer_dir, exist_ok=True)
            self.transformer_tokenizer.save_pretrained(tokenizer_dir)

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

        transformer_dir = f'{model_dir}/transformer_model'
        tokenizer_dir = f'{model_dir}/transformer_tokenizer'
        if os.path.isdir(transformer_dir):
            self.models['transformer'] = AutoModelForSequenceClassification.from_pretrained(transformer_dir)
        if os.path.isdir(tokenizer_dir):
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        # Load tokenizer
        tokenizer_path = f'{model_dir}/tokenizer.pkl'
        if os.path.exists(tokenizer_path):
            self.tokenizer = joblib.load(tokenizer_path)

    def _compute_transformer_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='weighted',
            zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
