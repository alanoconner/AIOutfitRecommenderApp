RP20240605 – Outfit Generation Research Prototype
=================================================

This project explores an end-to-end pipeline that recognises garments from an
input photo, asks an LLM for styling ideas, and generates a matching image
through DALL·E. The repository now exposes a clean Flask API for demos and a
training script for refining the apparel classifier.

Project Highlights
------------------
- Python backend with Flask and CORS-enabled endpoints.
- PyTorch convolutional network for multi-class garment recognition.
- OpenAI GPT chat completions for stylistic prompts.
- DALL·E image generation with prompt orchestration and colour detection.
- JSONL persistence layer for capturing qualitative evaluations.

Setup
-----
1. Create and activate a virtual environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Define the required environment variables:
   ```
   export OPENAI_API_KEY=sk-...
   export UPLOADS_DIR=/absolute/path/to/uploads  # optional
   ```

Running the API
---------------
```
flask --app server run --port 8000
```
The service exposes:
- `POST /outfits` (and the legacy alias `POST /post`) for outfit generation.
- `POST /save-evaluations` for submitting UX feedback.
- `GET /health` for lightweight health checks.

Training the Classifier
-----------------------
Use `training/train_classifier.py` to fine-tune the garment recognition model:
```
python training/train_classifier.py \
    --data-dir /path/to/dataset \
    --output-dir recognition-models \
    --epochs 25
```
The dataset should follow the folder structure documented in the script
module docstring.

Gunicorn Deployment (Optional)
------------------------------
```
gunicorn -c gunicorn.conf.py server:app
```

Repository Map
--------------
- `server.py` – Flask application factory and HTTP handlers.
- `repo.py` – Core pipeline orchestration and colour detection utilities.
- `recog.py` – PyTorch model definitions and prediction helpers.
- `styling.py` / `generate.py` – OpenAI integrations for prompts & images.
- `training/train_classifier.py` – CLI for retraining the classifier.
- `evaluator.py` – Utilities for persisting qualitative feedback.
