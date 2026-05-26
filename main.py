import re
import os
import json
import pickle
import requests

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline as hf_pipeline

MAX_LEN = 300
LABELS = ["legitimate job posting", "fake fraudulent job posting"]

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["lstm"] = tf.keras.models.load_model("lstm_model.keras")
    with open("keras_tokenizer.pkl", "rb") as f:
        models["tokenizer"] = pickle.load(f)
    models["hf"] = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
        framework="pt",
    )
    yield
    models.clear()


app = FastAPI(title="Fake Job Posting Detector", version="1.0.0", lifespan=lifespan)


class JobPosting(BaseModel):
    title: str = ""
    company_profile: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""


def preprocess(posting: JobPosting) -> str:
    text = " ".join([
        posting.title,
        posting.company_profile,
        posting.description,
        posting.requirements,
        posting.benefits,
    ])
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.get("/")
async def root():
    return {
        "message": "Fake Job Posting Detection API",
        "endpoints": [
            "/predict/classic",
            "/predict/opensource",
            "/predict/api",
            "/predict/compare",
        ],
    }


@app.post("/predict/classic")
async def predict_classic(posting: JobPosting):
    text = preprocess(posting)
    seq = pad_sequences(
        models["tokenizer"].texts_to_sequences([text]),
        maxlen=MAX_LEN,
        truncating="post",
    )
    prob = float(models["lstm"].predict(seq, verbose=0)[0][0])
    return {
        "model": "BiLSTM (trained from scratch)",
        "prediction": "Fake" if prob > 0.5 else "Real",
        "fraud_probability": round(prob, 4),
        "confidence": round(prob if prob > 0.5 else 1 - prob, 4),
    }


@app.post("/predict/opensource")
async def predict_opensource(posting: JobPosting):
    text = preprocess(posting)[:512]
    result = models["hf"](text, candidate_labels=LABELS)
    is_fake = result["labels"][0] == LABELS[1]
    return {
        "model": "BART-large-mnli (HuggingFace zero-shot)",
        "prediction": "Fake" if is_fake else "Real",
        "confidence": round(result["scores"][0], 4),
        "scores": {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])},
    }


@app.post("/predict/api")
async def predict_api(posting: JobPosting):
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable not set")

    text = preprocess(posting)
    prompt = (
        "You are an expert fraud analyst specializing in detecting fake job postings.\n"
        "Analyze this job posting and determine if it is FAKE or REAL.\n\n"
        f'Job Posting:\n"""\n{text[:800]}\n"""\n\n'
        "Red flags: vague descriptions, unrealistic salaries, requests for personal info, "
        "poor grammar, no company details.\n\n"
        'Respond ONLY with valid JSON: {"prediction": "FAKE" or "REAL", '
        '"confidence": 0.0 to 1.0, "reason": "brief explanation"}'
    )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.1,
            },
            timeout=30,
        )
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            pred = parsed.get("prediction", "REAL").upper()
            return {
                "model": "Mistral-7B-Instruct (OpenRouter free)",
                "prediction": "Fake" if pred == "FAKE" else "Real",
                "confidence": round(float(parsed.get("confidence", 0.5)), 4),
                "reason": parsed.get("reason", ""),
            }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {exc}")

    raise HTTPException(status_code=502, detail="Could not parse API response")


@app.post("/predict/compare")
async def predict_compare(posting: JobPosting):
    classic = await predict_classic(posting)
    opensource = await predict_opensource(posting)
    api_result = await predict_api(posting)

    votes = [classic["prediction"], opensource["prediction"], api_result["prediction"]]
    majority = max(set(votes), key=votes.count)

    return {
        "classic_ai": classic,
        "open_source_ai": opensource,
        "api_ai": api_result,
        "ensemble_prediction": majority,
        "unanimous": len(set(votes)) == 1,
        "vote_breakdown": {label: votes.count(label) for label in set(votes)},
    }
