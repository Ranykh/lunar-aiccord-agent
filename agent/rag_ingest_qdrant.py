# rag_ingest_qdrant.py
import pickle, math
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, MAX_DOCS_INGEST, EMBED_BATCH
from tokens_count.token_meter import TokenMeter
from .llm_io import embed_texts

INDICES = Path("/Users/ranykhirbawi/Desktop/LunarAIccord/indices")
def _load_notes() -> List[Dict[str, Any]]:
    p = INDICES / "notes.pkl"
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Build indices first.")
    blob = pickle.load(open(p, "rb"))
    return blob["rows"]

def main():
    notes = _load_notes()
    if MAX_DOCS_INGEST:
        notes = notes[:MAX_DOCS_INGEST]
    print(f"Ingesting {len(notes)} notes into Qdrant collection {QDRANT_COLLECTION}")

    # Initialize client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

    # Create collection if needed; vector size discovered from a probe embedding
    meter = TokenMeter()
    probe_vec = embed_texts(["probe"], meter)[0]
    dim = len(probe_vec)

    try:
        client.get_collection(QDRANT_COLLECTION)
        print(f"Collection '{QDRANT_COLLECTION}' exists.")
    except Exception:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created collection '{QDRANT_COLLECTION}' with dim={dim}.")

    # Prepare payloads
    texts = [r.get("text") or f"{r.get('note','')} ; {r.get('description','')}" for r in notes]
    ids = list(range(1, len(texts)+1))

    # Batch embed + upsert
    for i in range(0, len(texts), EMBED_BATCH):
        chunk_texts = texts[i:i+EMBED_BATCH]
        vecs = embed_texts(chunk_texts, meter)
        points = [
            PointStruct(
                id=ids[i+j],
                vector=vecs[j],
                payload={
                    "note": notes[i+j].get("note",""),
                    "family": notes[i+j].get("family",""),
                    "description": notes[i+j].get("description",""),
                    "text": chunk_texts[j],
                }
            )
            for j in range(len(vecs))
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"Upserted {i+len(points)}/{len(texts)}")

    summary = meter.flush("qdrant_ingest")
    print(f"Token summary (this ingest): {summary}")

if __name__ == "__main__":
    main()
