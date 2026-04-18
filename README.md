# Fake News Detector AI

Flask app to classify news text as Fake or Real using TF-IDF + Logistic Regression.

## Run locally
```bash
pip install -r requirements.txt
python train.py
python app.py
```

## Deploy on Render
- Push to GitHub
- New Web Service on Render
- Build Command: `pip install -r requirements.txt && python train.py`
- Start Command: `gunicorn app:app`
