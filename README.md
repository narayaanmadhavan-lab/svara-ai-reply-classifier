Reply Classification System

A comprehensive machine learning pipeline for automatically classifying customer reply sentiments in sales communications.
This system processes text responses and categorizes them as positive, negative, or neutral to help prioritize follow-up actions.

Performance Results

The system implements and compares three different approaches:

Logistic Regression with TF-IDF: 99.76% accuracy, 99.76% weighted F1-score

LightGBM: 98.59% accuracy, 98.59% weighted F1-score

DistilBERT (Fine-tuned): 100% accuracy, 100% weighted F1-score

Features

Multi-model approach with automatic fallback strategy

RESTful API for real-time predictions

Containerized deployment with Docker

Comprehensive evaluation metrics

Production-ready with confidence scoring

Technologies Used

Python 3.x

Transformers (Hugging Face)

FastAPI

scikit-learn

LightGBM

PyTorch

Docker

Quick Start
1. Install dependencies
pip install -r requirements.txt

2. Train models
python train.py

3. Start API server
uvicorn app:app --host 0.0.0.0 --port 8000

4. Test prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"Looking forward to the demo!"}'

API Usage

Endpoint:

POST /predict


Request:

{
  "text": "Your reply text here"
}


Response:

{
  "label": "positive",
  "confidence": 0.9971
}

Model Selection Strategy

CPU-only production: Uses TF-IDF + LightGBM for fast inference

GPU-enabled production: Uses fine-tuned DistilBERT for maximum accuracy

Hybrid approach: Automatic model selection based on available resources
