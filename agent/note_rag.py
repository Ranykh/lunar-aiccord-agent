# agent/note_rag.py
from typing import List, Dict, Any
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
except Exception:
    QdrantClient = None  # enables fallback path
from agent.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, TOP_K
from tokens_count.token_meter import TokenMeter
from agent.llm_io import embed_texts

# Fallback: your local notes.pkl TF-IDF path is left as-is elsewhere
from pathlib import Path, PurePath
import pickle
import numpy as np
import os
USE_QDRANT = os.getenv("LUNAR_USE_QDRANT", "1").lower() in ("1","true","yes")

def _fallback_local(seed_terms: List[str], emotion_text: str, top_k: int) -> Dict[str, Any]:
    pk = Path("indices") / "notes.pkl"
    if not pk.exists():
        return {"candidates": [], "by_family": {"top":[], "mid":[], "base":[]}}
    blob = pickle.load(open(pk, "rb"))
    vec = blob["vectorizer"]; X = blob["X"]; rows = blob["rows"]
    query = " ".join(seed_terms or []) + " " + (emotion_text or "")
    qv = vec.transform([query]).toarray()[0]
    scores = (X @ qv)
    order = np.argsort(-scores)[:max(top_k,1)]
    cands = []
    for i in order:
        r = rows[int(i)]
        fam = (r.get("family") or "").lower()
        cands.append({"note": r.get("note",""), "family": fam, "score": float(scores[i]), "source": r.get("description","")[:80]})
    byfam = {"top": [], "mid": [], "base": []}
    for c in cands:
        fam = c["family"]
        if fam in byfam: byfam[fam].append(c)
        else: byfam["mid"].append(c)
    return {"candidates": cands, "by_family": byfam}

def retrieve_candidates(seed_terms: List[str], emotion_text: str, top_k: int = TOP_K):
    """
    Returns:
      {
        "candidates": [ {note, family, score, source}, ... ],
        "by_family": { "top": [...], "mid":[...], "base":[...] }
      }
    """
    if not USE_QDRANT:
        return _fallback_local(seed_terms, emotion_text, top_k)
    meter = TokenMeter()
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
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
                "source": p.get("description","")[:100]
            })
        byfam = {"top": [], "mid": [], "base": []}
        for c in cands:
            if c["family"] in byfam:
                byfam[c["family"]].append(c)
            else:
                byfam["mid"].append(c)
        meter.flush("qdrant_search")
        return {"candidates": cands, "by_family": byfam}
    except Exception:
        # fallback to local TF-IDF if Qdrant is unavailable
        return _fallback_local(seed_terms, emotion_text, top_k)
