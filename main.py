
import os
import re
import uuid
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Request, UploadFile, Form, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    import docx  # python-docx
except ImportError:
    docx = None  # type: ignore

# Optional Azure Key Vault bootstrap
def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "", "None") else default

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    # Prefer env first
    val = _get_env(name)
    if val:
        return val
    # Try Key Vault if configured
    kv_uri = _get_env("KEYVAULT_URI")
    if not kv_uri:
        return default
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore
        from azure.keyvault.secrets import SecretClient  # type: ignore
        cred = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
        client = SecretClient(vault_url=kv_uri, credential=cred)
        secret = client.get_secret(name)
        return secret.value or default
    except Exception:
        return default

# App & templating
app = FastAPI(title="Thesis Agent – Humanizer Loop")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Basic login (very simple demo – use proper auth for production)
APP_USER = get_secret("APP_USER", "admin")
APP_PASS = get_secret("APP_PASS", "admin")

# Undetectable.ai credentials (user-provided)
UNDET_USER_ID = get_secret("UNDET_USER_ID")
UNDET_API_KEY = get_secret("UNDET_API_KEY")
UNDET_THRESHOLD = float(os.environ.get("UNDET_THRESHOLD", "0.15"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "4"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))

# Azure OpenAI (optional; used for semantic checks / rewrites if configured)
AOAI_ENDPOINT = get_secret("AOAI_ENDPOINT")
AOAI_KEY = get_secret("AOAI_KEY")
AOAI_DEPLOYMENT = os.environ.get("AOAI_DEPLOYMENT", "gpt-4o-mini")
AOAI_MAX_TOKENS = int(os.environ.get("AOAI_MAX_TOKENS", "2048"))

def read_docx(file_bytes: bytes) -> str:
    if not docx:
        raise HTTPException(status_code=500, detail="python-docx is not installed.")
    f = BytesIO(file_bytes)
    d = docx.Document(f)
    paras = []
    for p in d.paragraphs:
        paras.append(p.text)
    return "\n".join(paras)

def chunk_text(text: str, size: int) -> List[str]:
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += 1
        if cur_len >= size:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# Very light Harvard-style reference preservation check
HARVARD_RX = re.compile(r"\(([^)]+?),\s*(\d{4}[a-z]?)\)")

def references_preserved(orig: str, new: str) -> bool:
    orig_refs = set(HARVARD_RX.findall(orig))
    new_refs = set(HARVARD_RX.findall(new))
    # We allow new to be a superset; but must not lose more than 10% of original refs
    if not orig_refs:
        return True
    lost = len(orig_refs - new_refs)
    return lost <= max(1, int(0.1 * len(orig_refs)))

def undetectable_detect(text: str) -> float:
    if not (UNDET_USER_ID and UNDET_API_KEY):
        # If not configured, treat score as 0 (human)
        return 0.0
    url = "https://api.undetectable.ai/detect"
    headers = {"Authorization": f"Bearer {UNDET_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "userId": UNDET_USER_ID,
        "text": text
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # API returns various fields; we try common ones
        # Assume lower means more human; normalize to [0,1]
        score = data.get("aiProbability") or data.get("score") or 0.0
        return float(score)
    except Exception:
        # On failure, do not block
        return 0.0

def undetectable_humanize(text: str) -> str:
    if not (UNDET_USER_ID and UNDET_API_KEY):
        return text
    url = "https://api.undetectable.ai/humanize"
    headers = {"Authorization": f"Bearer {UNDET_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "userId": UNDET_USER_ID,
        "content": text,
        # sensible defaults matching university/article/more human
        "readability": "university",
        "purpose": "article",
        "humanizeLevel": "more"
    }
    try:
        r = requests.post(url, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        out = data.get("content") or data.get("text") or text
        return out
    except Exception:
        return text

def aoai_semantic_guard(original: str, proposal: str) -> str:
    # If no Azure OpenAI configured, fall back to simple guard (keep proposal)
    if not (AOAI_ENDPOINT and AOAI_KEY and AOAI_DEPLOYMENT):
        return proposal
    try:
        import json as _json
        import requests as _req
        url = f"{AOAI_ENDPOINT}openai/deployments/{AOAI_DEPLOYMENT}/chat/completions?api-version=2024-02-15-preview"
        headers = {
            "api-key": AOAI_KEY,
            "Content-Type": "application/json"
        }
        system = "You are a careful academic editor. Keep meaning and citations (Harvard style) intact. If the candidate paraphrase drifts in meaning, return a corrected version that aligns with the original semantics and preserves references. Keep a formal academic tone."
        prompt = (
            "Original:\n```\n" + original + "\n```\n\n"
            "Candidate:\n```\n" + proposal + "\n```\n\n"
            "Return ONLY the final, corrected candidate."
        )
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": min(AOAI_MAX_TOKENS, 2048)
        }
        resp = _req.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        js = resp.json()
        out = js["choices"][0]["message"]["content"]
        return out.strip()
    except Exception:
        return proposal

def process_chunk(original: str) -> Tuple[str, float, int]:
    # Iterative loop: detect -> humanize -> semantic guard -> detect
    iters = 0
    text = original
    while iters < MAX_ITERS:
        iters += 1
        score = undetectable_detect(text)
        if score <= UNDET_THRESHOLD:
            return text, score, iters
        # Humanize
        humanized = undetectable_humanize(text)
        # Semantic guard (keep meaning & refs)
        guarded = aoai_semantic_guard(original, humanized)
        if not references_preserved(original, guarded):
            # soften: merge guarded + original citations
            guarded = guarded + "\n\n" + " ".join(match[0] + ", " + match[1] for match in set(HARVARD_RX.findall(original)))
        text = guarded
    # Return best effort
    final_score = undetectable_detect(text)
    return text, final_score, iters

def process_document_bytes(b: bytes, bib_text: Optional[str] = None) -> Dict:
    original = read_docx(b)
    chunks = chunk_text(original, CHUNK_SIZE)
    results = []
    for ch in chunks:
        out, score, steps = process_chunk(ch)
        results.append(out)
    final_text = "\n\n".join(results)
    return {
        "content": final_text,
        "chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "bib": bib_text or ""
    }

# Routes
@app.get("/", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def do_login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username != APP_USER or password != APP_PASS:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    resp = RedirectResponse("/upload", status_code=302)
    return resp

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process(request: Request, file: UploadFile = File(...), bibfile: Optional[UploadFile] = File(None)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Please upload a .docx file")
    b = await file.read()
    bib_txt = ""
    if bibfile:
        if not bibfile.filename.lower().endswith(".bib"):
            raise HTTPException(status_code=400, detail="BibTeX file must end with .bib")
        bib_txt = (await bibfile.read()).decode("utf-8", errors="ignore")
    data = process_document_bytes(b, bib_txt)
    return templates.TemplateResponse("result.html", {"request": request, "content": data["content"], "bibliography": data["bib"], "chunk_size": data["chunk_size"], "chunks": data["chunks"]})
