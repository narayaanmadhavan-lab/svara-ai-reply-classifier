# SvaraAI Reply Classification Assignment

## Files  
- train.py : training code for baselines and DistilBERT
- app.py : FastAPI service exposing /predict
- answers.md : Short answers for Part C
- requirements.txt : Python dependencies
- Dockerfile : Optional containerization

## Quick start
1. Install dependencies: pip install -r requirements.txt
2. Train models: python train.py  
3. Run API: uvicorn app:app --host 0.0.0.0 --port 8000
4. Test: POST to /predict endpoint

## Notes
Use TF-IDF + LightGBM for CPU-only production.
Use DistilBERT for best accuracy with GPU.
