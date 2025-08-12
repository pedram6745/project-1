
# Thesis Agent – Humanizer Loop (Azure App Service)

Features:
- DOCX ingestion + optional BibTeX upload
- Iterative Undetectable.ai Detect → Humanize → Semantic Guard → Detect loop
- Harvard-style reference preservation check
- Optional Azure OpenAI (AOAI) semantic guard (set env `AOAI_*`)
- Azure Key Vault integration for secrets
- Simple login via `APP_USER` / `APP_PASS`

## Environment variables
- `APP_USER`, `APP_PASS`
- `KEYVAULT_URI` (optional)
- `UNDET_USER_ID`, `UNDET_API_KEY`, `UNDET_THRESHOLD` (e.g., 0.15), `MAX_ITERS` (e.g., 4)
- `CHUNK_SIZE` (e.g., 1000)
- `AOAI_ENDPOINT`, `AOAI_KEY`, `AOAI_DEPLOYMENT`, `AOAI_MAX_TOKENS`

## Run locally
```
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Zip Deploy to Azure
Upload the project root as a zip to your App Service.
