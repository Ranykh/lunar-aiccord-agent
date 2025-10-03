# agent/material_rag.py
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import pickle
import os

try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None  # ok: we'll fall back to local RAG

from agent.llm_io import embed_text  # Azure embeddings
from config import (
    API_KEY, QDRANT_URL,
    QDRANT_COLLECTION_MATERIALS,
    TOP_K_MATERIALS,
)

INDICES_DIR = Path("/Users/ranykhirbawi/Desktop/LunarAIccord/indices")
MATERIALS_PKL = INDICES_DIR / "materials.pkl"


def _load_materials_index():
    """
    Expects pickle with keys: 'vectorizer', 'X', 'rows'
    Each row like:
      id, name, family, role, descriptors(list|str), usage_max(float), edp_min(float?), ...
    """
    if not MATERIALS_PKL.exists():
        return None
    with open(MATERIALS_PKL, "rb") as f:
        return pickle.load(f)


def _row_to_candidate(r: Dict[str, Any], score: float, vec_row) -> Dict[str, Any]:
    # Map your schema -> pipeline schema
    mid = r.get("material_id") or r.get("id") or r.get("name")
    usage_max = r.get("usage_max")  # your field
    usage_max_pct = float(r.get("usage_max_pct", usage_max if usage_max is not None else 100.0))

    edp_min = r.get("edp_min", None)
    # a soft weight used by allocators/deduper; if missing, default 1.0
    if edp_min is not None:
        try:
            target_mid = 0.5 * (float(edp_min) + float(usage_max_pct))
        except Exception:
            target_mid = 1.0
    else:
        target_mid = 1.0

    desc = r.get("descriptors", "")
    if isinstance(desc, list):
        desc = ", ".join(map(str, desc))

    # expose a vector for mmr() diversity (dense np.array)
    try:
        tfv = vec_row.toarray()[0]
    except Exception:
        try:
            tfv = np.array(vec_row).ravel()
        except Exception:
            tfv = None

    return {
        "material_id": str(mid),
        "name": r.get("name") or str(mid),
        "family": r.get("family") or r.get("role") or "",
        "descriptors": desc,
        "usage_max_pct": float(usage_max_pct),
        "target_usage_mid_pct": float(target_mid),
        "emb": None,
        "tfidf_vec": tfv,
        "score": float(score),
    }


def _tfidf_search(conceptual_note: str, emotion_text: str, top_k: int) -> List[Dict[str, Any]]:
    blob = _load_materials_index()
    if not blob:
        raise RuntimeError(
            "indices/materials.pkl not found. Build it (see step 6) or configure Qdrant via QDRANT_URL."
        )

    vec = blob["vectorizer"]
    X   = blob["X"]
    rows= blob["rows"]

    query = f"{conceptual_note or ''} {emotion_text or ''}".strip()

    # vectorize and score robustly
    try:
        q = vec.transform([query])
        try:
            scores = (X @ q.T)
            # handle sparse/ndarray
            scores = scores.A.ravel() if hasattr(scores, "A") else scores.ravel()
        except Exception:
            scores = (X.multiply(q)).sum(axis=1)
            scores = scores.A.ravel() if hasattr(scores, "A") else scores.ravel()
    except Exception:
        qv = vec.transform([query]).toarray()[0]
        scores = (X @ qv)

    order = np.argsort(-scores)[:max(1, top_k)]
    out: List[Dict[str, Any]] = []
    for i in order:
        r = rows[int(i)]
        desc = r.get("descriptors", "")
        if isinstance(desc, list):
            desc = ", ".join(desc)
        out.append({
            "material_id": r.get("material_id") or r.get("id") or r.get("name"),
            "name": r.get("name"),
            "family": r.get("family", ""),
            "role": r.get("role", ""),
            "descriptors": desc or "",
            "usage_max_pct": float(r.get("usage_max_pct", r.get("usage_max", 100.0)) or 100.0),
            # a light default weight; if you later compute better priors, map here
            "target_usage_mid_pct": float(r.get("target_usage_mid_pct", r.get("edp_min", 1.0)) or 1.0),
            "score": float(scores[i]),
        })
    return out


class MaterialRAG:
    def __init__(self, online: bool = True):
        self.online = online and (QdrantClient is not None) and bool(QDRANT_URL)
        self.client = QdrantClient(QDRANT_URL, api_key=API_KEY, check_compatibility=False) if self.online else None
        self._has_local = MATERIALS_PKL.exists()

    def _online_search(self, query_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
        res = self.client.search(
            collection_name=QDRANT_COLLECTION_MATERIALS,
            query_vector=query_vec,
            limit=top_k,
        )
        out = []
        for r in res:
            p = r.payload or {}
            out.append({
                "material_id": p.get("material_id") or p.get("id") or p.get("name"),
                "name": p.get("name"),
                "family": p.get("family") or p.get("role") or "",
                "descriptors": p.get("descriptors", ""),
                "usage_max_pct": float(p.get("usage_max_pct", p.get("usage_max", 100.0))),
                "target_usage_mid_pct": float(p.get("target_usage_mid_pct", 1.0)),
                "aliases": p.get("aliases", []),
                "emb": getattr(r, "vector", None),
                "score": float(r.score),
            })
        return out

    def retrieve(self, conceptual_note: str, emotion_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or TOP_K_MATERIALS
        qtext = f"{conceptual_note} || emotion: {emotion_text}".strip()

        # 1) online (if configured)
        if self.online:
            try:
                qvec = embed_text(qtext)
                hits = self._online_search(qvec, k)
                if hits:
                    return hits
            except Exception:
                pass  # fall back to local

        # 2) local TF-IDF materials index
        if self._has_local:
            return _tfidf_search(conceptual_note, emotion_text, k)

        # 3) nothing available
        return []


def mmr_select(cands: List[Dict[str, Any]], already: List[Dict[str, Any]], lambda_: float = 0.7, k: int = 1):
    """
    Maximal Marginal Relevance over 'score' and cosine similarity of vectors in 'emb' or 'tfidf_vec'.
    """
    chosen = []
    already_ids = {a.get("material_id") for a in (already or [])}
    pool = [c for c in cands if c.get("material_id") not in already_ids]

    def v(x):
        arr = x.get("emb") or x.get("tfidf_vec") or []
        try:
            return np.array(arr, dtype=float)
        except Exception:
            return np.array([], dtype=float)

    while pool and len(chosen) < k:
        scores = []
        for c in pool:
            rel = float(c.get("score", 0.0))
            if not chosen and not already:
                div = 0.0
            else:
                comp = (chosen + (already or []))
                sims = []
                vc = v(c)
                for y in comp:
                    vy = v(y)
                    if vc.size and vy.size:
                        denom = (np.linalg.norm(vc) + 1e-9) * (np.linalg.norm(vy) + 1e-9)
                        sims.append(float(np.dot(vc, vy) / denom))
                div = max(sims) if sims else 0.0
            mmr = lambda_ * rel - (1 - lambda_) * div
            scores.append((mmr, c))
        scores.sort(key=lambda t: t[0], reverse=True)
        best = scores[0][1]
        chosen.append(best)
        pool = [x for x in pool if x.get("material_id") != best.get("material_id")]
    return chosen
