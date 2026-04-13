# Simple Crosstab Application (UI) v20

This package fixes the Excel upload 500 error by **including required dependencies** and providing a safer upload handler.

## Quick start (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --reload --log-level debug --access-log
```

Open: http://127.0.0.1:8000

## What was fixed (RCA)
- **Root cause:** RuntimeError "Excel upload requires pandas + openpyxl"
- **Fix:** Added `pandas` + `openpyxl` to requirements and changed upload reader to return a clean 400 error if missing.
- **Extra:** Supports CSV and Excel. Upload endpoint validates non-empty files.

## Notes
- This bundle is a full runnable app skeleton. If you want me to patch *your* existing repo 1:1,
  zip your current project folder and upload it here.
