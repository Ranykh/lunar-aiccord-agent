# agent/material_mapper.py
import json, pickle, re, random
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE = Path(__file__).resolve().parents[1]
# Try both common locations; adjust if your repo uses a different path.
CATALOG_CANDIDATES = [
    BASE / "data" / "materials_catalog.jsonl"
]
INDEX_MATERIALS = BASE / "indices" / "materials.pkl"

# -------- Loaders --------
def _load_catalog() -> List[Dict[str, Any]]:
    for p in CATALOG_CANDIDATES:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    return []

def _load_index_materials() -> Optional[List[Dict[str, Any]]]:
    if INDEX_MATERIALS.exists():
        with open(INDEX_MATERIALS, "rb") as f:
            return pickle.load(f)
    return None

_CATALOG = _load_catalog()
_INDEX_MATS = _load_index_materials()  # optional

def _text_of(m: Dict[str, Any]) -> str:
    return " ".join([
        str(m.get("name","")),
        " ".join(m.get("aliases", [])),
        " ".join(m.get("descriptors", [])),
        str(m.get("family",""))
    ]).lower()

def _usage_mid_pct(m: Dict[str, Any]) -> float:
    # Prefer EDP range if present; fall back to any 'fine' range
    rng = m.get("usage_hint_pct") or {}
    low = rng.get("edp_min") or rng.get("fine_min") or 0.2
    high= rng.get("edp_max") or rng.get("fine_max") or 1.0
    try:
        return float((float(low) + float(high)) / 2.0)
    except Exception:
        return 1.0

def _usage_max_pct(m: Dict[str, Any]) -> Optional[float]:
    rng = m.get("usage_hint_pct") or {}
    hi = rng.get("edp_max") or rng.get("fine_max")
    try:
        return float(hi) if hi is not None else None
    except Exception:
        return None

def _stock_solution_pct(m: Dict[str, Any]) -> Optional[float]:
    u = m.get("usage_hint") or {}
    # Catalog field name: recommended_%solution (string like "0.1" or "10")
    val = u.get("recommended_%solution")
    try:
        return float(val)
    except Exception:
        return None

_SPECIALS = {
    "muguet": ["Hydroxycitronellal", "Lilyflore", "Cyclamen Aldehyde"],
    "marine notes": ["Calone", "Helional", "Floralozone", "Ultrazur", "Melonal"],
    "fresh linen": ["Aldehyde C-11 Undecylenic", "Aldehyde C-12 MNA", "Aldehyde C-10"],
    "aldehydes": ["Aldehyde C-10", "Aldehyde C-11 Undecylenic", "Aldehyde C-12 MNA"],
    "blue lotus": ["Lilyflore"],
    "white musk": ["Galaxolide", "Habanolide", "Ethylene Brassylate"],
    "amber": ["Ambroxan", "Amber Extreme", "Cetalox"],
    "musk": ["Habanolide", "Galaxolide", "Ethylene Brassylate"],
    "cedar": ["Cedramber", "Iso E Super", "Vertofix"],
    "bergamot": ["Bergamot Oil", "Limonene", "Linalyl Acetate"],
}

def _find_in_catalog(name: str) -> Optional[Dict[str, Any]]:
    nl = name.lower()
    for m in _CATALOG:
        if m.get("name","").lower() == nl:
            return m
    return None

def _pick_from_special(note: str) -> Optional[Dict[str, Any]]:
    for name in _SPECIALS.get(note.lower().strip(), []):
        m = _find_in_catalog(name)
        if m:
            return m
    return None

def _score_material(note: str, m: Dict[str, Any]) -> int:
    note = note.lower()
    toks = re.findall(r"[a-z]+", note)
    txt = _text_of(m)
    score = 0
    if note in txt: score += 3
    for t in toks:
        if t in txt: score += 1
    # tiny boost if family/role notionally matches the word
    fam = (m.get("family") or "").lower()
    if any(t in fam for t in toks): score += 1
    return score

def choose_materials_for_note(note: str, k: int = 1) -> List[Dict[str, Any]]:
    # 1) curated specials
    m = _pick_from_special(note)
    if m:
        return [m]

    # 2) catalog heuristic
    if _CATALOG:
        scored = [( _score_material(note, m), m ) for m in _CATALOG]
        scored = [x for x in scored if x[0] > 0]
        if scored:
            scored.sort(key=lambda z: -z[0])
            return [z[1] for z in scored[:max(1,k)]]

    # 3) optional indices/materials.pkl (if you have it)
    if _INDEX_MATS:
        # assume items in this index already material-like dicts with 'name','family'
        hits = [m for m in _INDEX_MATS if _score_material(note, m) > 0]
        if hits:
            return hits[:max(1,k)]

    # 4) last resort: empty
    return []
