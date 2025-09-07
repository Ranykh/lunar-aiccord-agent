# agent/scent_composer.py
from dataclasses import dataclass
from typing import List, Dict, Any
from agent.local_utils import pick_role_from_family
from agent.material_mapper import choose_materials_for_note, _usage_mid_pct, _usage_max_pct, _stock_solution_pct

BATCH_TOTAL_G = 100.0  # grams per batch

@dataclass
class MaterialUse:
    name: str
    role: str
    grams: float
    percent: float   # computed from grams so downstream code still works
    info: Dict[str, Any]   # include catalog meta: family, usage ranges, stock solution, etc.

@dataclass
class Formula:
    top: List[MaterialUse]
    mid: List[MaterialUse]
    base: List[MaterialUse]

def _attach_material_meta(cand) -> Dict[str, Any]:
    """
    cand has .name or .note (string); return a meta dict from catalog if found.
    """
    note = getattr(cand, "name", None) or getattr(cand, "note", None) or ""
    note = str(note)
    mats = choose_materials_for_note(note, k=1)
    if not mats:
        return {"from_note": note}
    m = mats[0]
    meta = {
        "from_note": note,
        "material": m.get("name"),
        "family": m.get("family"),
        "descriptors": m.get("descriptors", []),
        "aliases": m.get("aliases", []),
        "usage_hint_pct": m.get("usage_hint_pct", {}),
        "usage_max_pct": _usage_max_pct(m),
        "stock_solution_pct": _stock_solution_pct(m),  # keep as in catalog (no scaling)
    }
    return meta

def allocate_roles(candidates, max_per_role=5):
    buckets = {"top":[], "mid":[], "base":[]}
    for c in candidates:
        fam = getattr(c, "family", "") or ""
        role = getattr(c, "role", "") or pick_role_from_family(fam)
        buckets[role].append(c)
    # backfill if empty
    allc = candidates[:]
    for r in ("top","mid","base"):
        if not buckets[r] and allc:
            buckets[r].append(allc[0])
    for r in buckets:
        buckets[r] = buckets[r][:max_per_role]
    return buckets

def _share_split() -> Dict[str, float]:
    return {"top": 0.30, "mid": 0.40, "base": 0.30}

def compose(candidates) -> Formula:
    # 1) bucket
    buckets = allocate_roles(candidates)

    # 2) for each candidate, attach material meta and a “target weight” = mid of usage range
    meta_buckets: Dict[str, List[Dict[str, Any]]] = {"top": [], "mid": [], "base": []}
    for role in ("top","mid","base"):
        for c in buckets[role]:
            meta = _attach_material_meta(c)
            meta["target_usage_mid_pct"] = _usage_mid_pct({"usage_hint_pct": meta.get("usage_hint_pct", {})})
            meta_buckets[role].append(meta)

    # 3) allocate grams by role: split the batch, then proportion by target_usage_mid_pct
    split = _share_split()
    out_rows: Dict[str, List[MaterialUse]] = {"top": [], "mid": [], "base": []}
    for role in ("top","mid","base"):
        items = meta_buckets[role]
        if not items:
            continue
        role_grams = BATCH_TOTAL_G * split[role]
        weights = [max(1e-4, float(x.get("target_usage_mid_pct", 1.0))) for x in items]
        s = sum(weights) or 1.0
        grams_list = [round(role_grams * w / s, 2) for w in weights]
        # Compute percents for compatibility downstream
        for meta, g in zip(items, grams_list):
            pct = round(100.0 * g / BATCH_TOTAL_G, 3)
            out_rows[role].append(MaterialUse(
                name=meta.get("material") or meta.get("from_note") or "Unknown",
                role=role,
                grams=g,
                percent=pct,
                info=meta
            ))

    return Formula(
        top = out_rows["top"],
        mid = out_rows["mid"],
        base= out_rows["base"],
    )
