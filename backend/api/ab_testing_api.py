from flask import Blueprint, request, jsonify
from backend.utils.ab_testing import ABTestingFramework
import json

ab_testing_bp = Blueprint('ab_testing', __name__, url_prefix='/api/ab-testing')

# Initialize AB testing framework
ab_framework = ABTestingFramework()

@ab_testing_bp.route('/experiments', methods=['GET'])
def list_experiments():
    try:
        experiments = ab_framework.list_experiments()
        summaries = [ab_framework.get_experiment_summary(exp['id']) for exp in experiments]
        return jsonify({'experiments': summaries})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ab_testing_bp.route('/experiments', methods=['POST'])
def create_experiment():
    try:
        data = request.get_json()

        name = data.get('name')
        description = data.get('description', '')
        models_to_test = data.get('models_to_test', [])
        test_data_source = data.get('test_data_source')

        if not name or not models_to_test:
            return jsonify({'error': 'Name and models_to_test are required'}), 400

        experiment_id = ab_framework.create_experiment(
            name=name,
            description=description,
            models_to_test=models_to_test,
            test_data_source=test_data_source
        )

        return jsonify({
            'experiment_id': experiment_id,
            'message': 'Experiment created successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ab_testing_bp.route('/experiments/<int:experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    try:
        experiment = ab_framework.get_experiment(experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404

        return jsonify(experiment)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ab_testing_bp.route('/experiments/<int:experiment_id>/run', methods=['POST'])
def run_experiment(experiment_id):
    try:
        data = request.get_json()

        test_texts = data.get('test_texts', [])
        true_labels = data.get('true_labels', [])
        sample_size = data.get('sample_size')

        if not test_texts or not true_labels:
            return jsonify({'error': 'test_texts and true_labels are required'}), 400

        if len(test_texts) != len(true_labels):
            return jsonify({'error': 'test_texts and true_labels must have the same length'}), 400

        experiment = ab_framework.run_experiment(
            experiment_id=experiment_id,
            test_texts=test_texts,
            true_labels=true_labels,
            sample_size=sample_size
        )

        return jsonify({
            'message': 'Experiment completed successfully',
            'experiment': experiment
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ab_testing_bp.route('/experiments/<int:experiment_id>/report', methods=['GET'])
def get_experiment_report(experiment_id):
    try:
        save_plots = request.args.get('save_plots', 'true').lower() == 'true'

        report = ab_framework.generate_experiment_report(experiment_id, save_plots)
        if not report:
            return jsonify({'error': 'Experiment not found or not completed'}), 404

        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ab_testing_bp.route('/experiments/compare', methods=['POST'])
def compare_experiments():
    try:
        data = request.get_json()
        experiment_ids = data.get('experiment_ids', [])

        if len(experiment_ids) < 2:
            return jsonify({'error': 'At least 2 experiment IDs are required'}), 400

        comparison = ab_framework.compare_experiments(experiment_ids)
        if not comparison:
            return jsonify({'error': 'Insufficient completed experiments for comparison'}), 400

        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ab_testing_bp.route('/models/available', methods=['GET'])
def get_available_models():
    available_models = [
        'naive_bayes',
        'svm',
        'random_forest',
        'logistic_regression',
        'neural_network',
        'lstm',
        'ensemble'
    ]

    return jsonify({'models': available_models})

@ab_testing_bp.route('/experiments/<int:experiment_id>/quick-test', methods=['POST'])
def quick_test_experiment():
    """Run a quick test with sample data for demonstration"""
    try:
        # Sample test data for demonstration
        sample_texts = [
            "I love this product! It's amazing!",
            "This is terrible, worst purchase ever",
            "It's okay, nothing special",
            "Absolutely fantastic! Highly recommend!",
            "Not great, but not terrible either",
            "Worst experience of my life",
            "Pretty good overall experience",
            "I hate this so much",
            "Really satisfied with the quality",
            "Could be better but acceptable"
        ]

        sample_labels = [2, 0, 1, 2, 1, 0, 2, 0, 2, 1]  # 0=negative, 1=neutral, 2=positive

        experiment = ab_framework.run_experiment(
            experiment_id=experiment_id,
            test_texts=sample_texts,
            true_labels=sample_labels
        )

        return jsonify({
            'message': 'Quick test completed successfully',
            'experiment': experiment
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ab_testing_bp.route('/experiments/<int:experiment_id>/metrics-summary', methods=['GET'])
def get_metrics_summary(experiment_id):
    try:
        experiment = ab_framework.get_experiment(experiment_id)
        if not experiment or experiment['status'] != 'completed':
            return jsonify({'error': 'Experiment not found or not completed'}), 404

        # Extract key metrics for easy consumption
        summary = {
            'experiment_info': {
                'id': experiment['id'],
                'name': experiment['name'],
                'status': experiment['status'],
                'models_tested': experiment['models_to_test'],
                'test_data_size': experiment.get('test_data_size', 0)
            },
            'model_performance': {},
            'rankings': {},
            'best_models': {}
        }

        # Extract performance for each model
        for model_name, results in experiment['results'].items():
            summary['model_performance'][model_name] = {
                'accuracy': results['metrics']['accuracy'],
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'f1_score': results['metrics']['f1_score'],
                'prediction_time': results['prediction_time'],
                'confidence': results.get('confidence_stats', {})
            }

        # Extract rankings
        if 'metrics' in experiment:
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in experiment['metrics']:
                    summary['best_models'][metric] = experiment['metrics'][metric]['best_model']

        return jsonify(summary)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add blueprint registration function
def register_ab_testing_routes(app):
    app.register_blueprint(ab_testing_bp)