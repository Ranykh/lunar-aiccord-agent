# agent/note_rag.py
from typing import List, Dict, Any
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue  # noqa: F401
except Exception:
    QdrantClient = None

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, TOP_K_NOTES


from tokens_count.token_meter import TokenMeter
from agent.llm_io import embed_texts

from pathlib import Path
import pickle
import numpy as np
import os

USE_QDRANT = os.getenv("LUNAR_USE_QDRANT", "1").lower() in ("1", "true", "yes")

def _to_candidates_by_family(byfam: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """
    Flatten {'by_family': {'top':[...], 'mid':[...], 'base':[...]}}
    to {'top':[note,...], 'mid':[...], 'base':[...]} with **global** de-dup
    (a note appears at most once across all roles).
    """
    seen = set()

    def uniq_notes(items: List[Dict[str, Any]]) -> List[str]:
        out = []
        for it in items or []:
            n = (it.get("note") or it.get("name") or "").strip()
            if not n:
                continue
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    return {
        "top":  uniq_notes(byfam.get("top", [])),
        "mid":  uniq_notes(byfam.get("mid", [])),
        "base": uniq_notes(byfam.get("base", [])),
    }


def _fallback_local(seed_terms: List[str], emotion_text: str, top_k: int) -> Dict[str, Any]:
    pk = Path("indices") / "notes.pkl"
    if not pk.exists():
        empty = {"top": [], "mid": [], "base": []}
        return {
            "candidates": [],
            "by_family": empty,
            "candidates_by_family": _to_candidates_by_family(empty),
        }

    blob = pickle.load(open(pk, "rb"))
    vec = blob["vectorizer"]; X = blob["X"]; rows = blob["rows"]
    query = " ".join(seed_terms or []) + " " + (emotion_text or "")
    qv = vec.transform([query]).toarray()[0]
    scores = (X @ qv)
    order = np.argsort(-scores)[:max(top_k, 1)]

    cands = []
    for i in order:
        r = rows[int(i)]
        fam = (r.get("family") or "").lower()
        cands.append({
            "note": r.get("note",""),
            "family": fam,
            "score": float(scores[i]),
            "source": (r.get("description","") or "")[:100]
        })
    byfam = {"top": [], "mid": [], "base": []}
    for c in cands:
        byfam[c["family"] if c["family"] in byfam else "mid"].append(c)

    return {
        "candidates": cands,
        "by_family": byfam,
        "candidates_by_family": _to_candidates_by_family(byfam),  # cross-role de-dup
    }

def retrieve_candidates(seed_terms: List[str], emotion_text: str, top_k: int = TOP_K_NOTES) -> Dict[str, Any]:
    if not USE_QDRANT or QdrantClient is None or not QDRANT_URL:
        return _fallback_local(seed_terms, emotion_text, top_k)

    meter = TokenMeter()
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, check_compatibility=False)
        qtext = (" ".join(seed_terms or [])) + " " + (emotion_text or "")
        qvec = embed_texts([qtext], meter)[0]
        hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k)
        cands = []
        for h in hits:
            p = h.payload or {}
            fam = (p.get("family") or "").lower()
            cands.append({
                "note": p.get("note",""),
                "family": fam,
                "score": float(h.score),
                "source": (p.get("description","") or "")[:100]
            })
        byfam = {"top": [], "mid": [], "base": []}
        for c in cands:
            byfam[c["family"] if c["family"] in byfam else "mid"].append(c)
        meter.flush("qdrant_search")
        return {
            "candidates": cands,
            "by_family": byfam,
            "candidates_by_family": _to_candidates_by_family(byfam),  # cross-role de-dup
        }
    except Exception:
        return _fallback_local(seed_terms, emotion_text, top_k)
