from pathlib import Path

from social_media_analysis import data, model


def test_training_pipeline_trains_and_saves(tmp_path: Path):
    df = data.load_labeled_samples()
    experiment = model.SentimentExperiment(
        config=model.ModelConfig(max_iter=150, max_features=2000, test_size=0.2)
    )
    metrics = experiment.train(df)
    assert metrics["accuracy"] >= 0.6
    model_path = tmp_path / "sentiment.joblib"
    experiment.save(model_path)

    loaded = model.SentimentExperiment.load(model_path)
    preds = loaded.predict(df["text"].head(5).tolist())
    assert len(preds) == 5
