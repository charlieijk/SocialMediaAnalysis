import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=10,
            random_state=42
        )
        self.is_fitted = False

    def extract_tfidf_features(self, texts, fit=True):
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        return tfidf_matrix

    def extract_count_features(self, texts, fit=True):
        if fit:
            count_matrix = self.count_vectorizer.fit_transform(texts)
        else:
            count_matrix = self.count_vectorizer.transform(texts)
        return count_matrix

    def extract_topic_features(self, texts, fit=True):
        count_matrix = self.extract_count_features(texts, fit=fit)
        if fit:
            topic_matrix = self.lda_model.fit_transform(count_matrix)
        else:
            topic_matrix = self.lda_model.transform(count_matrix)
        return topic_matrix

    def extract_text_statistics(self, texts):
        features = []
        for text in texts:
            stats = {
                'char_count': len(text),
                'word_count': len(text.split()),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'upper_case_count': sum(1 for c in text if c.isupper()),
                'digit_count': sum(1 for c in text if c.isdigit())
            }
            features.append(stats)
        return pd.DataFrame(features)

    def extract_all_features(self, texts, fit=True):
        tfidf_features = self.extract_tfidf_features(texts, fit=fit)
        text_stats = self.extract_text_statistics(texts)
        topic_features = self.extract_topic_features(texts, fit=fit)

        # Combine features
        combined_features = np.hstack([
            tfidf_features.toarray(),
            text_stats.values,
            topic_features
        ])

        if fit:
            self.is_fitted = True

        return combined_features

    def get_feature_names(self):
        feature_names = []
        feature_names.extend(self.tfidf_vectorizer.get_feature_names_out())
        feature_names.extend(['char_count', 'word_count', 'avg_word_length',
                             'exclamation_count', 'question_count',
                             'upper_case_count', 'digit_count'])
        feature_names.extend([f'topic_{i}' for i in range(10)])
        return feature_names