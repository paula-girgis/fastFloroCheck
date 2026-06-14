# fastFloroCheck

FastAPI service for plant-disease classification. Loads a Keras model at startup (with a warm-up pass), filters out non-leaf images by green-pixel ratio, and returns the predicted class.

## Endpoints
- `POST /predict` — multipart `file` image field; returns `{ "predicted_class": "..." }`

## Setup
```bash
pip install -r requirements.txt
# place best_model.keras in the project root (not tracked in git)
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Environment Variables
| Variable          | Description                                        |
| ----------------- | -------------------------------------------------- |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (default `*`)          |
