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
- Extend the notebook workflows into production code once experimentation is complete.
