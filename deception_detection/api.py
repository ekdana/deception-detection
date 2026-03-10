from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import json

from src.predict import load_model, predict_text

app = FastAPI(title="Deception Detection Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
TOKENIZER = None


def startup_load_model():
    global MODEL, TOKENIZER
    model_path = "saved_bert_model"

    if os.path.exists(model_path):
        try:
            MODEL, TOKENIZER = load_model(model_path)
            print(f"Model loaded from: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            MODEL, TOKENIZER = None, None
    else:
        print("saved_bert_model folder not found. API will run in demo-fallback mode.")
        MODEL, TOKENIZER = None, None


@app.on_event("startup")
def on_startup():
    startup_load_model()


def extract_text_from_file_content(filename: str, content: bytes) -> str:
    """
    Very simple demo extraction.
    Best for txt/csv/json/md.
    For binary files like pdf/docx, this demo only returns a placeholder.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".txt", ".csv", ".json", ".md", ".log"]:
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    return f"[Uploaded file: {filename}. Binary/parsing not implemented in this demo.]"


def demo_fallback_score(text: str) -> dict:
    """
    Fallback scoring when trained model is unavailable.
    Only for demo purposes.
    """
    lowered = text.lower()
    score = 18
    triggers = []

    suspicious = [
        "honestly", "trust me", "believe me", "i swear",
        "guaranteed", "100%", "never", "always", "obviously"
    ]

    for word in suspicious:
        if word in lowered:
            score += 7
            triggers.append(f'Contains phrase: "{word}"')

    if text.count("!") >= 3:
        score += 10
        triggers.append("Many exclamation marks")

    if len(text.split()) < 15:
        score += 6
        triggers.append("Very short text")

    score = max(1, min(99, score))

    return {
        "label": "DECEPTIVE ❌" if score >= 50 else "TRUTHFUL ✅",
        "deception_rate": round(score, 2),
        "trustworthiness_score": round(100 - score, 2),
        "flags": triggers[:5]
    }


@app.get("/")
def root():
    return {
        "message": "Deception Detection Demo API is running",
        "model_loaded": MODEL is not None
    }


@app.post("/analyze")
async def analyze(
    text: str = Form(""),
    files: Optional[List[UploadFile]] = File(None)
):
    combined_text = text.strip()
    uploaded_files_info = []

    if files:
        for uploaded_file in files:
            content = await uploaded_file.read()
            extracted = extract_text_from_file_content(uploaded_file.filename, content)

            uploaded_files_info.append({
                "filename": uploaded_file.filename,
                "size_bytes": len(content)
            })

            if extracted.strip():
                combined_text += "\n\n" + extracted.strip()

    if not combined_text.strip():
        return {
            "error": "No text or readable file content provided."
        }

    if MODEL is not None and TOKENIZER is not None:
        try:
            label, trust_score, deception_score = predict_text(
                combined_text,
                MODEL,
                TOKENIZER
            )

            return {
                "label": label,
                "deception_rate": round(float(deception_score) * 100, 2),
                "trustworthiness_score": round(float(trust_score) * 100, 2),
                "confidence": round(max(float(deception_score), float(trust_score)) * 100, 2),
                "model_mode": "bert",
                "uploaded_files": uploaded_files_info
            }
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}"
            }

    demo_result = demo_fallback_score(combined_text)
    demo_result["confidence"] = 70.0
    demo_result["model_mode"] = "demo_fallback"
    demo_result["uploaded_files"] = uploaded_files_info

    return demo_result