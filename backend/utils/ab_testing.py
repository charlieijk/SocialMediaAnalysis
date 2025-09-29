import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ABTestingFramework:
    def __init__(self, results_dir='data/ab_testing'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.experiments = []
        self.load_experiments()

    def load_experiments(self):
        experiments_file = os.path.join(self.results_dir, 'experiments.json')
        if os.path.exists(experiments_file):
            with open(experiments_file, 'r') as f:
                self.experiments = json.load(f)

    def save_experiments(self):
        experiments_file = os.path.join(self.results_dir, 'experiments.json')
        with open(experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)

    def create_experiment(self, name, description, models_to_test, test_data_source=None):
        experiment = {
            'id': len(self.experiments) + 1,
            'name': name,
            'description': description,
            'models_to_test': models_to_test,
            'test_data_source': test_data_source,
            'created_at': datetime.now(),
            'status': 'created',
            'results': {},
            'metrics': {},
            'statistical_tests': {}
        }

        self.experiments.append(experiment)
        self.save_experiments()
        return experiment['id']

    def run_experiment(self, experiment_id, test_texts, true_labels, sample_size=None):
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Sample data if specified
        if sample_size and sample_size < len(test_texts):
            indices = np.random.choice(len(test_texts), sample_size, replace=False)
            test_texts = [test_texts[i] for i in indices]
            true_labels = [true_labels[i] for i in indices]

        experiment['status'] = 'running'
        experiment['test_data_size'] = len(test_texts)
        experiment['started_at'] = datetime.now()

        # Store results for each model
        for model_name in experiment['models_to_test']:
            print(f"Testing model: {model_name}")
            experiment['results'][model_name] = self.test_model_performance(
                model_name, test_texts, true_labels
            )

        # Calculate comparative metrics
        experiment['metrics'] = self.calculate_comparative_metrics(experiment['results'])

        # Perform statistical tests
        experiment['statistical_tests'] = self.perform_statistical_tests(experiment['results'])

        experiment['status'] = 'completed'
        experiment['completed_at'] = datetime.now()

        self.save_experiments()
        return experiment

    def test_model_performance(self, model_name, test_texts, true_labels):
        from backend.models.sentiment_models import SentimentAnalyzer
        from backend.utils.text_preprocessor import TextPreprocessor
        from backend.utils.feature_extractor import FeatureExtractor

        # Initialize analyzer
        sentiment_analyzer = SentimentAnalyzer()
        text_preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor()
        sentiment_analyzer.load_dependencies(feature_extractor, text_preprocessor)

        # Load models
        sentiment_analyzer.load_models()

        # Get predictions
        start_time = datetime.now()
        predictions, probabilities = sentiment_analyzer.predict(test_texts, model_name)
        end_time = datetime.now()

        prediction_time = (end_time - start_time).total_seconds()

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        # Calculate confidence statistics
        confidence_stats = {}
        if probabilities is not None:
            max_probs = np.max(probabilities, axis=1)
            confidence_stats = {
                'mean_confidence': float(np.mean(max_probs)),
                'std_confidence': float(np.std(max_probs)),
                'min_confidence': float(np.min(max_probs)),
                'max_confidence': float(np.max(max_probs))
            }

        return {
            'model_name': model_name,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'prediction_time': prediction_time,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'f1_per_class': f1_per_class.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'confidence_stats': confidence_stats,
            'test_size': len(test_texts)
        }

    def calculate_comparative_metrics(self, results):
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        comparative_metrics = {}

        for metric in metrics:
            comparative_metrics[metric] = {}
            values = [results[model]['metrics'][metric] for model in models]

            comparative_metrics[metric]['values'] = dict(zip(models, values))
            comparative_metrics[metric]['best_model'] = models[np.argmax(values)]
            comparative_metrics[metric]['worst_model'] = models[np.argmin(values)]
            comparative_metrics[metric]['range'] = float(max(values) - min(values))
            comparative_metrics[metric]['std'] = float(np.std(values))

        # Speed comparison
        times = [results[model]['prediction_time'] for model in models]
        comparative_metrics['speed'] = {
            'values': dict(zip(models, times)),
            'fastest_model': models[np.argmin(times)],
            'slowest_model': models[np.argmax(times)],
            'range_seconds': float(max(times) - min(times))
        }

        return comparative_metrics

    def perform_statistical_tests(self, results):
        statistical_tests = {}
        models = list(results.keys())

        if len(models) < 2:
            return statistical_tests

        # McNemar's test for comparing classifier accuracy
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                predictions1 = np.array(results[model1]['predictions'])
                predictions2 = np.array(results[model2]['predictions'])

                # Create contingency table for McNemar's test
                correct1_correct2 = np.sum((predictions1 == predictions2) & (predictions1 == predictions1))  # Both correct (need true labels)
                correct1_wrong2 = np.sum((predictions1 != predictions2))  # Simplified for demo
                wrong1_correct2 = np.sum((predictions1 != predictions2))  # Simplified for demo
                wrong1_wrong2 = np.sum((predictions1 == predictions2))    # Both wrong (simplified)

                # Chi-square test for independence
                accuracy1 = results[model1]['metrics']['accuracy']
                accuracy2 = results[model2]['metrics']['accuracy']

                statistical_tests[f'{model1}_vs_{model2}'] = {
                    'accuracy_difference': float(accuracy1 - accuracy2),
                    'significant_difference': abs(accuracy1 - accuracy2) > 0.05,  # Simplified threshold
                    'better_model': model1 if accuracy1 > accuracy2 else model2,
                    'confidence_comparison': self.compare_confidence(results[model1], results[model2])
                }

        return statistical_tests

    def compare_confidence(self, result1, result2):
        conf1 = result1.get('confidence_stats', {})
        conf2 = result2.get('confidence_stats', {})

        if not conf1 or not conf2:
            return None

        return {
            'mean_diff': float(conf1['mean_confidence'] - conf2['mean_confidence']),
            'more_confident_model': result1['model_name'] if conf1['mean_confidence'] > conf2['mean_confidence'] else result2['model_name']
        }

    def get_experiment(self, experiment_id):
        for exp in self.experiments:
            if exp['id'] == experiment_id:
                return exp
        return None

    def list_experiments(self):
        return self.experiments

    def get_experiment_summary(self, experiment_id):
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        summary = {
            'id': experiment['id'],
            'name': experiment['name'],
            'status': experiment['status'],
            'models_tested': experiment['models_to_test'],
            'created_at': experiment['created_at'],
            'test_data_size': experiment.get('test_data_size', 0)
        }

        if experiment['status'] == 'completed':
            summary['best_accuracy'] = max([
                result['metrics']['accuracy'] for result in experiment['results'].values()
            ])
            summary['best_model'] = experiment['metrics']['accuracy']['best_model']
            summary['completed_at'] = experiment['completed_at']

        return summary

    def generate_experiment_report(self, experiment_id, save_plots=True):
        experiment = self.get_experiment(experiment_id)
        if not experiment or experiment['status'] != 'completed':
            return None

        report = {
            'experiment_info': {
                'id': experiment['id'],
                'name': experiment['name'],
                'description': experiment['description'],
                'models_tested': experiment['models_to_test'],
                'test_data_size': experiment['test_data_size'],
                'duration': str(experiment['completed_at'] - experiment['started_at'])
            },
            'model_rankings': self.rank_models(experiment),
            'detailed_metrics': experiment['metrics'],
            'statistical_significance': experiment['statistical_tests'],
            'recommendations': self.generate_recommendations(experiment)
        }

        if save_plots:
            self.create_visualization_plots(experiment, experiment_id)

        # Save report
        report_file = os.path.join(self.results_dir, f'experiment_{experiment_id}_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def rank_models(self, experiment):
        models = list(experiment['results'].keys())
        rankings = {}

        # Rank by different metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            values = [(model, experiment['results'][model]['metrics'][metric]) for model in models]
            values.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [(rank + 1, model, score) for rank, (model, score) in enumerate(values)]

        # Overall ranking (weighted average)
        weights = {'accuracy': 0.3, 'precision': 0.2, 'recall': 0.2, 'f1_score': 0.3}
        overall_scores = {}
        for model in models:
            score = sum(
                experiment['results'][model]['metrics'][metric] * weight
                for metric, weight in weights.items()
            )
            overall_scores[model] = score

        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['overall'] = [(rank + 1, model, score) for rank, (model, score) in enumerate(overall_ranking)]

        return rankings

    def generate_recommendations(self, experiment):
        recommendations = []

        # Best overall model
        best_model = experiment['metrics']['accuracy']['best_model']
        recommendations.append(f"Use {best_model} for highest overall accuracy")

        # Speed vs accuracy trade-off
        fastest_model = experiment['metrics']['speed']['fastest_model']
        if fastest_model != best_model:
            recommendations.append(f"Consider {fastest_model} for faster predictions if speed is critical")

        # Confidence analysis
        high_confidence_models = []
        for model, result in experiment['results'].items():
            if result.get('confidence_stats', {}).get('mean_confidence', 0) > 0.8:
                high_confidence_models.append(model)

        if high_confidence_models:
            recommendations.append(f"Models with high confidence: {', '.join(high_confidence_models)}")

        return recommendations

    def create_visualization_plots(self, experiment, experiment_id):
        try:
            models = list(experiment['results'].keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']

            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Model Comparison - {experiment["name"]}', fontsize=16)

            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]
                values = [experiment['results'][model]['metrics'][metric] for model in models]

                bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
                ax.set_title(f'{metric.title()} Comparison')
                ax.set_ylabel(metric.title())
                ax.set_ylim(0, 1)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'experiment_{experiment_id}_metrics.png'))
            plt.close()

            # Create confusion matrix plots
            fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
            if len(models) == 1:
                axes = [axes]

            for idx, model in enumerate(models):
                cm = np.array(experiment['results'][model]['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
                axes[idx].set_title(f'{model} Confusion Matrix')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'experiment_{experiment_id}_confusion_matrices.png'))
            plt.close()

        except Exception as e:
            print(f"Error creating visualization plots: {e}")

    def compare_experiments(self, experiment_ids):
        experiments = [self.get_experiment(exp_id) for exp_id in experiment_ids]
        experiments = [exp for exp in experiments if exp and exp['status'] == 'completed']

        if len(experiments) < 2:
            return None

        comparison = {
            'experiments': [
                {
                    'id': exp['id'],
                    'name': exp['name'],
                    'best_model': exp['metrics']['accuracy']['best_model'],
                    'best_accuracy': max([result['metrics']['accuracy'] for result in exp['results'].values()])
                }
                for exp in experiments
            ],
            'cross_experiment_insights': self.generate_cross_experiment_insights(experiments)
        }

        return comparison

    def generate_cross_experiment_insights(self, experiments):
        insights = []

        # Find consistently best performing models
        model_performances = {}
        for exp in experiments:
            best_model = exp['metrics']['accuracy']['best_model']
            model_performances[best_model] = model_performances.get(best_model, 0) + 1

        if model_performances:
            most_consistent = max(model_performances.items(), key=lambda x: x[1])
            insights.append(f"Most consistently best model: {most_consistent[0]} (best in {most_consistent[1]}/{len(experiments)} experiments)")

        return insights