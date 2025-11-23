import os
from pathlib import Path
from typing import Dict, Any

import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# ========== COMPONENT 1: TIỀN XỬ LÝ ==========
def preprocess_text(text: str) -> str:
    """Simple, deterministic preprocessing used before inference.

    - lowercase
    - some common ASCII->unicode replacements
    - truncate to 50 chars (keeps behavior from notebook)
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    replace_dict = {
        "rat": "rất",
        "tot": "tốt",
        "xau": "xấu",
        "kho": "khó",
        "de": "dễ",
        "vui": "vui",
        "buon": "buồn",
        "ghet": "ghét",
        "bth": "bình thường",
        "chan": "chán",
        "thich": "thích",
        "do": "đó",
        "toi": "tôi",
        "ban": "bạn",
        "a": "anh",
        "e": "em",
        "t": "tao",
        "m": "mày",
        "yeu": "yêu",
        "iu": "yêu",
        "thuong": "thương",
        "lam": "lắm",
        "dang": "đang",
    }

    for k, v in replace_dict.items():
        text = re.sub(r'\b' + re.escape(k) + r'\b', v, text)

    # Keep parity with original notebook: limit length
    text = text[:50]

    return text


# Fallback mapping in case model returns LABEL_0/1/2
FALLBACK_LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
}


# ========== COMPONENT 2: MODEL LOADING & INFERENCE ==========
# We'll lazily load the pipeline once and reuse it to avoid reloading the model on every call.
_SENTIMENT_PIPELINE = None

# OLD LOCAL MODEL LOADING (commented for rollback)
# _MODEL_DIR = Path(__file__).parent / "model" / "phobert-sentiment-vietnamese-best"
# 
# def _get_model_dir() -> Path:
#     """Return model directory. Allows overriding by environment variable MODEL_DIR."""
#     env = os.environ.get("MODEL_DIR")
#     if env:
#         return Path(env)
#     return _MODEL_DIR

# NEW HUGGINGFACE MODEL PATH
_MODEL_PATH = "duchienmtp/PhoBERT-sentiment-analysis"


def preload_model():
    """Preload the model pipeline during app initialization.
    
    Call this function to warm up the model when the app starts,
    rather than waiting for the first user request.
    """
    try:
        _load_pipeline()
        print(f"Model '{_MODEL_PATH}' preloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to preload model: {e}")
        return False


def _load_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE

    model = AutoModelForSequenceClassification.from_pretrained(_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
    
    # prefer CPU unless CUDA is available
    device = 0 if torch.cuda.is_available() else -1
    
    _SENTIMENT_PIPELINE = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    
    # OLD LOCAL MODEL LOADING (commented for rollback)
    # model_dir = _get_model_dir()
    # if not model_dir.exists():
    #     raise FileNotFoundError(f"Model directory not found: {model_dir}")
    # 
    # tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    # 
    # _SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    
    return _SENTIMENT_PIPELINE


def sentiment_analysis(text: str) -> Dict[str, Any]:
    """Run the sentiment pipeline on `text` and return normalized label and score.

    Returns a dict: {"label": "POSITIVE|NEGATIVE|NEUTRAL", "score": float, "raw_label": str}
    """
    pipe = _load_pipeline()
    results = pipe(text)
    if not results:
        raise RuntimeError("Model returned no results")

    result = results[0]
    raw_label = result.get("label") or result.get("score")
    score = result.get("score")

    # Normalize label
    normalized = None
    if isinstance(raw_label, str):
        rl = raw_label.upper()
        # If model uses LABEL_0 style, map to human labels
        normalized = FALLBACK_LABEL_MAP.get(rl, None)
        if normalized is None:
            # If model already returns 'POSITIVE'/'NEGATIVE' etc, keep them
            if rl in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                normalized = rl
            else:
                # fallback: return raw label as-is
                normalized = rl
    else:
        normalized = str(raw_label)

    return {"label": normalized, "score": float(score) if score is not None else None, "raw_label": raw_label}


# ========== COMPONENT 3: HỢP NHẤT VÀ XỬ LÝ LỖI ==========
def analyze_text(text: str) -> Dict[str, Any]:
    """Public helper used by the app. Returns a dict with text, label and score or an error."""
    try:
        clean_text = preprocess_text(text)
        res = sentiment_analysis(clean_text)
        return {"text": text, "sentiment": res["label"], "score": res.get("score"), "raw_label": res.get("raw_label")}
    except Exception as e:
        return {"text": text, "error": str(e)}

