# Sentiment Analysis Web App

Flask serves the backend API and the React UI. The model uses Sentiment140 labels where `0` is negative and `1` is positive.

## Run

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Retrain

The training script looks for `..\training.1600000.processed.noemoticon.csv` by default and rewrites:

- `sentiment_model.pkl`
- `vectorizer.pkl`
- `model_metadata.json`

```powershell
python -m sentiment_app.nlp.train_model
```

For a quicker training pass:

```powershell
python -m sentiment_app.nlp.train_model --sample-per-class 200000
```
