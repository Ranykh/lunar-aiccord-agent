import pickle, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

BASE = Path(__file__).resolve().parents[1]
INDICES = BASE / "indices"

USE_AZURE = os.getenv("LUNAR_USE_AZURE", "").lower() in ("1","true","yes")
try:
    from agent.llm_client import embed, cosine as cos_sim
except Exception:
    embed = None
    cos_sim = None
    USE_AZURE = False

@dataclass
class EvalResult:
    score: float
    rationale: str

def _material_items(formula_like: Any):
    def _items(lst):
        for x in lst or []:
            if hasattr(x, "name"):
                yield {
                    "name": x.name,
                    "grams": getattr(x, "grams", None),
                    "percent": getattr(x, "percent", None),
                }
            elif isinstance(x, dict):
                yield {
                    "name": x.get("name",""),
                    "grams": x.get("grams"),
                    "percent": x.get("percent"),
                }
    if hasattr(formula_like, "top"):
        return list(_items(formula_like.top)), list(_items(formula_like.mid)), list(_items(formula_like.base))
    # dict fallback
    return list(_items(formula_like.get("top", []))), list(_items(formula_like.get("mid", []))), list(_items(formula_like.get("base", [])))

def formula_to_text(formula_like: Any) -> str:
    top, mid, base = _material_items(formula_like)
    def fmt(x):
        if x.get("grams") is not None:
            return f"{x['name']} {x['grams']}g"
        return f"{x['name']} {x.get('percent',0)}%"
    def part(lst): return ", ".join([fmt(x) for x in lst])
    return f"top: {part(top)} | mid: {part(mid)} | base: {part(base)}"


def _tfidf_score(emotion_text: str, ftxt: str) -> float:
    with open(INDICES/"global.pkl", "rb") as f:
        glob = pickle.load(f)
    vec = glob["vectorizer"]

    # Transform to vectors. These may be scipy sparse (sklearn) or Python lists (TinyTfidf)
    q = vec.transform([emotion_text])
    d = vec.transform([ftxt])

    # ---- sklearn / scipy path (sparse or ndarray) ----
    # If the result has .dot(), use proper vector dot product (cosine if vec is L2-normalized)
    if hasattr(q, "dot"):
        try:
            # q and d are 1xN; q.dot(d.T) is 1x1
            return float(q.dot(d.T)[0, 0])
        except Exception:
            # Some sparse types prefer elementwise then sum
            try:
                return float(q.multiply(d).sum())
            except Exception:
                pass  # fall through to dense fallback

    # ---- TinyTfidf / dense fallback ----
    # TinyTfidf returns lists; normalize above ensures cosine via dot
    try:
        qv = q[0]
        dv = d[0]
    except Exception:
        qv = q
        dv = d
    return float(sum((float(a) * float(b)) for a, b in zip(qv, dv)))

def _embed_score(emotion_text: str, ftxt: str) -> float:
    if not (USE_AZURE and embed and cos_sim):
        return None
    try:
        e = embed([emotion_text, ftxt])
        return float(cos_sim(e[0], e[1]))
    except Exception:
        return None

def _scale_0_100(x: float) -> float:
    # keep your previous scaling feel
    x = max(0.0, min(1.0, x))
    return round(40.0 + 55.0 * x, 1)

def evaluate(formula, emotion_text: str) -> EvalResult:
    ftxt = formula_to_text(formula)
    tfidf = _tfidf_score(emotion_text, ftxt)
    emb = _embed_score(emotion_text, ftxt)

    if emb is None:
        final = _scale_0_100(tfidf)
        rationale = "Alignment via TF-IDF cosine on local corpus."
    else:
        # Blend: equal weight; tweak if desired
        blended = 0.5*tfidf + 0.5*emb
        final = _scale_0_100(blended)
        rationale = "Blended TF-IDF + Azure embedding cosine alignment."

    return EvalResult(score=final, rationale=rationale)
