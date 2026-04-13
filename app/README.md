# Simple Crosstab Application — UI Shell (Local + Koyeb compatible)

## Run locally
From project root (folder containing `app/`):
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open:
- http://127.0.0.1:8000

## Notes
- This is a minimal, Koyeb-friendly interface (FastAPI + Jinja2 + HTMX).
- The crosstab logic here is a fallback preview engine.
- Replace `INTEGRATION POINT (A)` and `INTEGRATION POINT (B)` in `app/main.py` with your real crosstab + export logic.
