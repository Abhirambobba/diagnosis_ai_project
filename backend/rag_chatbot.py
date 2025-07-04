# rag_chatbot.py (modular, no Streamlit)

import re
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
import json
import fitz  # PyMuPDF
import os
from typing import Tuple

# -------------------- CONFIG -------------------- #
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "symptom_index.faiss"
METADATA_FILE = "symptom_metadata.pkl"
CSV_FILE = "diseases.csv"
GEMINI_API_KEY = "AIzaSyDxvUhg2ee_VDnc2l8OpEjGFDJQh4blv2g"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
HEADERS = {"Content-Type": "application/json"}

# -------------------- LOAD MODELS + INDEX -------------------- #
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

symptom_prompts = metadata["prompts"]
diagnoses = metadata["responses"]

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Prompt", "Response"]).to_csv(CSV_FILE, index=False)

# -------------------- DETECTION HELPERS -------------------- #
MEDICAL_PATTERNS = [
    r"\d+\s*(year|yr)s?[\s-]old", r"\d+\s*yo", r"\d+\s*y\.?o\.?",
    r"\b(pain|fever|vomit|cough|rash|dizzy|nausea|ache|bleed|diarrhea|edema|breath|diagnosis|medication)\b"
]
NON_MEDICAL_PATTERNS = {
    "hi", "hello", "thanks", "goodbye", "who are you", "what can you do"
}

def preprocess_input(text: str) -> str:
    text = text.strip().lower()
    for starter in ["hello doctor", "hi doc", "hey dr", "dear doctor"]:
        if text.startswith(starter):
            return text[len(starter):].strip()
    return text

def local_is_medical(text: str) -> Tuple[bool, float]:
    text = text.lower()
    if any(text.startswith(p) for p in NON_MEDICAL_PATTERNS):
        return False, 0.0
    matches = sum(1 for pattern in MEDICAL_PATTERNS if re.search(pattern, text))
    if matches >= 3:
        return True, 1.0
    elif matches > 0:
        return True, min(0.3 * matches, 0.9)
    else:
        return False, 0.0

async def gemini_is_medical(text: str) -> bool:
    prompt = f"""Return 'true' or 'false'. Is the following a medical complaint?

\"\"\"{text}\"\"\"
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1}
    }
    try:
        res = requests.post(GEMINI_URL, headers=HEADERS, data=json.dumps(payload), timeout=3)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip().lower() == "true"
    except Exception:
        return False

async def is_medical_query(text: str) -> bool:
    if not text.strip():
        return False
    text = preprocess_input(text)
    is_med, conf = local_is_medical(text)
    if is_med and conf >= 0.7:
        return True
    if not is_med and conf == 0:
        return False
    return await gemini_is_medical(text)

# -------------------- CORE INFERENCE -------------------- #
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        return f"[Error extracting PDF text: {str(e)}]"

def query_gemini(user_input, context_prompt, known_response, pdf_context=""):
    prompt = f"""
You are a medical AI assistant trained for diagnosis.

Context:
Symptoms: {context_prompt}
Diagnosis: {known_response}

Patient says: {user_input}

Uploaded PDF Notes:
{pdf_context or 'None'}

Return structured advice.
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048
        }
    }
    try:
        res = requests.post(GEMINI_URL, headers=HEADERS, data=json.dumps(payload))
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"❌ Gemini API error: {str(e)}"

def extract_symptoms_as_json(user_input: str):
    prompt = f"""
Extract medical symptoms from the following and output JSON:

Text: \"\"\"{user_input}\"\"\"
Format:
{{
  "symptoms": ["symptom1", "symptom2"]
}}
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 256
        }
    }
    try:
        res = requests.post(GEMINI_URL, headers=HEADERS, data=json.dumps(payload))
        res.raise_for_status()
        raw = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        if "```json" in raw:
            raw = raw.split("```json")[-1].split("```")[0].strip()
        return json.loads(raw)
    except Exception:
        return {"symptoms": []}

# -------------------- FASTAPI UTILITY FUNCTION -------------------- #
async def handle_prompt_and_pdf(prompt: str, pdf_bytes: bytes) -> dict:
    processed = preprocess_input(prompt)
    pdf_text = extract_text_from_pdf(pdf_bytes) if pdf_bytes else ""

    is_med = await is_medical_query(processed)
    if not is_med and not pdf_text.strip():
        return {"type": "non_medical", "message": "This doesn't appear to be a medical query."}

    # Combine prompt + PDF for embedding search
    search_text = f"{processed}\n{pdf_text}"
    embedding = model.encode([search_text])
    D, I = index.search(np.array(embedding).astype("float32"), k=1)
    idx = I[0][0]

    if idx != -1 and D[0][0] < 0.8:
        context_prompt = symptom_prompts[idx]
        context_response = diagnoses[idx]
    else:
        context_prompt = processed
        context_response = "Unknown"

    summary = query_gemini(processed, context_prompt, context_response, pdf_text)
    structured = extract_symptoms_as_json(processed)

    return {
        "type": "medical",
        "matched_prompt": context_prompt,
        "diagnosis": context_response,
        "summary": summary,
        "structured_symptoms": structured,
    }

# -------------------- CSV / FAISS UPDATES -------------------- #
def add_to_faiss(prompt, response):
    symptom_prompts.append(prompt)
    diagnoses.append(response)
    new_embedding = model.encode([prompt])
    index.add(np.array(new_embedding).astype('float32'))
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump({"prompts": symptom_prompts, "responses": diagnoses}, f)

def add_to_csv(prompt, response):
    df = pd.DataFrame([[prompt, response]], columns=["Prompt", "Response"])
    df.to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))
