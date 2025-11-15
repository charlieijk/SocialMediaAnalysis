# SocialMediaAnalysis

Notebook-first sentiment analysis experiments for social media text.

## What's Included
- **End-to-end notebooks** such as `simple_demo.ipynb`, `insights.ipynb`, `twitter_client.ipynb`, and `ab_testing_api.ipynb` that walk through data exploration, model training, API prototyping, and simulated A/B testing.
- **Backend building blocks** (under `backend/`) with modular notebooks for preprocessing, feature extraction, training, and lightweight Flask endpoints.
- **Saved models** in `backend/models/saved/` ready for quick demos without retraining.
- **Sample data** in `data/processed/` for immediate experimentation.

The original JavaScript/React frontend has been removed so everything now runs inside Jupyter notebooks.

## Getting Started
1. Create a virtual environment (conda, venv, or pyenv) with Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Lab/Notebook:
   ```bash
   jupyter lab
   ```
4. Open any notebook in the project root or within `backend/`, run the cells, and adapt them to your specific analysis.

## Running the Notebooks
After installing the dependencies, activate the environment and start Jupyter Lab from the project root:

```bash
source .venv/bin/activate
jupyter lab
```

In the Lab interface, open and run the notebook that matches what you want to explore:
- `simple_demo.ipynb` – minimal end-to-end sentiment pipeline.
- `insights.ipynb` – focus on visualizations and error analysis.
- `twitter_client.ipynb` – collecting/streaming tweets with Tweepy.
- `ab_testing_api.ipynb` – simulated A/B testing harness.
- `simple_flask_app.ipynb` or `flask_app.ipynb` – quick Flask API prototypes.
- `run_demo.ipynb` – orchestrates saved models for a quick showcase.

## Recommended Notebook Flow
- `simple_demo.ipynb`: Minimal sentiment pipeline on sample tweets.
- `insights.ipynb`: Visualization-heavy exploration of model predictions.
- `twitter_client.ipynb`: Demonstrates Tweepy-based streaming/collection.
- `ab_testing_api.ipynb` or `simple_flask_app.ipynb`: Prototype a Flask API directly from the notebook.
- `backend/utils/*.ipynb`: Mix-and-match preprocessing utilities when building new experiments.

## Directory Overview
- `backend/` – reusable components (API demos, configs, models, utilities).
- `data/` – prepared CSV assets; replace with your own datasets as needed.
- `scripts/` – additional helper notebooks (e.g., corpus preparation).
- `tests/` – placeholder for future automated checks around notebook logic.

## Dependencies
`requirements.txt` now reflects only the libraries actually imported in the notebooks (NLTK, spaCy, scikit-learn, pandas/numpy, Tweepy, Flask/CORS, Matplotlib/Seaborn, TensorFlow, PyTorch, Transformers, python-dotenv, plus JupyterLab/IPyKernel so you can run the notebooks). Plotly, TextBlob, WordCloud, Streamlit, and Requests were removed when the JavaScript frontend was dropped; re-add them if you later build a richer UI.

## Next Steps
- Connect your own Twitter API credentials via `.env` files referenced in the notebooks.
- Swap in new datasets under `data/` and retrain models using `backend/models/sentiment_models.ipynb`.
- Use `backend/models/semi_supervised_training.ipynb` to leverage unlabeled tweets (pseudo-labels + consistency checks) and overwrite the models under `backend/models/saved/`.
- Extend the notebook workflows into production code once experimentation is complete.

## Semi-Supervised Training CLI
Jump straight into self-training without opening a notebook:

```bash
# defaults to the sample CSVs under data/processed/
python backend/models/semi_supervised_training.ipynb \
  --labeled_path data/processed/sample_labeled_tweets.csv \
  --unlabeled_path data/processed/sample_unlabeled_tweets.csv \
  --output_path backend/models/saved/semi_supervised_logreg.pkl \
  --max_iterations 3 \
  --confidence_threshold 0.8 \
  --augment_factor 2
```

The notebook file is still plain Python under the hood, so invoking it with `python backend/models/semi_supervised_training.ipynb --help` works the same way as the old `.py` script while keeping the extension aligned with the rest of the project.

Key flags:
- `--unlabeled_path` can point to any CSV that has a `text` column.
- `--augment_factor` controls how many strongly augmented copies of each labeled sample are created.
- `--confidence_threshold` and `--max_iterations` tune the pseudo-label loop.
- Pass `--disable_consistency` or `--disable_augmentation` to simplify the training loop.

Each run saves a scikit-learn pipeline plus a `.metrics.json` file next to the chosen `--output_path`, making it easy to compare validation reports over time.
