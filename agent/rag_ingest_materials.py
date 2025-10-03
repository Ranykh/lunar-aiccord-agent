# agent/rag_ingest_materials.py
import json
import argparse
import uuid
import re
from typing import Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from agent.llm_io import embed_text
from config import API_KEY, QDRANT_URL, QDRANT_COLLECTION_MATERIALS

# If you know your Azure embedding size, set it here to avoid dimension mismatches.
# For text-embedding-3-large it's 3072; for older ada-002 it's 1536.
EMBED_DIM = 1536  # <-- adjust to your embedding deployment dim if needed


def _slug(s: str) -> str:
    s = (s or "").lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "material"


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    return [str(x)]


def _normalize_material(i: int, m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure we have a complete payload: material_id, name, aliases, family, descriptors, usage caps, etc.
    Generates a stable material_id if missing.
    """
    name = m.get("name") or m.get("material") or m.get("title") or f"material-{i}"
    base_id = m.get("material_id") or m.get("id") or _slug(name)
    # Guarantee uniqueness (catalogs sometimes repeat names)
    material_id = f"{base_id}-{i}" if base_id == _slug(name) else str(base_id)

    aliases = _as_list(m.get("aliases"))
    family = m.get("family", "")
    descriptors = m.get("descriptors", m.get("description", "")) or ""

    # Numeric caps, with safe defaults
    try:
        usage_max = float(m.get("usage_max_pct", 100.0))
    except Exception:
        usage_max = 100.0
    try:
        target_mid = float(m.get("target_usage_mid_pct", 1.0))
    except Exception:
        target_mid = 1.0

    return {
        "material_id": material_id,
        "name": name,
        "aliases": aliases,
        "family": family,
        "descriptors": descriptors,
        "usage_max_pct": usage_max,
        "target_usage_mid_pct": target_mid,
    }


def _text_of(m: Dict[str, Any]) -> str:
    aliases = ", ".join(m.get("aliases", []))
    return f"{m['name']} | aliases: {aliases} | family: {m.get('family','')} | desc: {m.get('descriptors','')}"


def ingest_materials(catalog_path: str):
    client = QdrantClient(QDRANT_URL, api_key=API_KEY, prefer_grpc=False)

    # Create collection if it doesn't exist (avoid deprecated recreate_collection)
    try:
        exists = client.collection_exists(QDRANT_COLLECTION_MATERIALS)
    except Exception:
        # Older clients may not have collection_exists; fallback to get_collection
        try:
            client.get_collection(QDRANT_COLLECTION_MATERIALS)
            exists = True
        except Exception:
            exists = False

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_MATERIALS,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    # Load items
    with open(catalog_path, "r", encoding="utf-8") as f:
        if catalog_path.endswith(".jsonl"):
            items = [json.loads(line) for line in f if line.strip()]
        else:
            items = json.load(f)

    # Normalize and embed
    points: List[PointStruct] = []
    for i, raw in enumerate(items):
        mat = _normalize_material(i, raw)
        text = _text_of(mat)
        vec = embed_text(text)  # single vector (list[float])
        if not vec:
            # If embedding failed, skip to avoid empty vectors in Qdrant
            continue
        if EMBED_DIM and len(vec) != EMBED_DIM:
            raise ValueError(
                f"Embedding dim mismatch: got {len(vec)}, expected {EMBED_DIM}. "
                f"Set EMBED_DIM in rag_ingest_materials.py to match your Azure embedding deployment."
            )

        payload = {
            "material_id": mat["material_id"],
            "name": mat["name"],
            "aliases": mat["aliases"],
            "family": mat["family"],
            "descriptors": mat["descriptors"],
            "usage_max_pct": mat["usage_max_pct"],
            "target_usage_mid_pct": mat["target_usage_mid_pct"],
        }
        points.append(PointStruct(id=i, vector=vec, payload=payload))

        # Batch upserts
        if len(points) >= 256:
            client.upsert(collection_name=QDRANT_COLLECTION_MATERIALS, points=points)
            points = []

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION_MATERIALS, points=points)

    print(f"Ingested {len(items)} materials into {QDRANT_COLLECTION_MATERIALS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest materials into Qdrant.")
    ap.add_argument("--path", required=True, help="Path to materials_catalog.jsonl or .json")
    args = ap.parse_args()
    ingest_materials(args.path)
