# agent/proportioner_llm.py
from typing import Dict, List, Any
from config import ROLE_SPLIT, ROLE_TOLERANCE
from agent.llm_io import chat_json  # NEW (already exists in your repo)
import os, json, glob
from pathlib import Path


def _load_reference_formulas(max_refs=5):
    refs = []
    for p in glob.glob("data/reference_formulas/*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                J = json.load(f)
            refs.append(J)
        except Exception:
            pass
    return refs[:max_refs]

def _pick_refs(emotion_text: str, materials_by_role, max_refs=3):
    # extremely simple: prefer files that share tags/notes
    refs = _load_reference_formulas(50)
    if not refs: return []
    want = set(str(emotion_text or "").lower().split())
    # also throw in conceptual notes if you want:
    # want |= {... from materials_by_role names, etc.}
    scored = []
    for J in refs:
        tags = set(map(str.lower, J.get("tags", [])))
        notes= set(map(str.lower, J.get("notes", [])))
        score = len(want & tags) + len(want & notes)
        scored.append((score, J))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [J for s,J in scored[:max_refs]]



def llm_proportion(materials_by_role: Dict[str, List[Dict[str, Any]]],
                   emotion_text: str,
                   role_split: Dict[str, float] = None,
                   tolerance: float = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Intentionally no LLM here by default. We delegate to MILP (or deterministic fallback)
    to avoid any chance of invented material names.
    """
    refs = _pick_refs(emotion_text, materials_by_role, max_refs=3)
    return llm_proportion_strict(materials_by_role, emotion_text, role_split, tolerance, reference_formulas=refs)




def llm_proportion_strict(materials_by_role, emotion_text, role_split=None, tolerance=None,
                          reference_formulas: List[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Strict LLM allocator that:
      - never invents materials,
      - must select only from provided materials (by material_id),
      - respects caps and role splits,
      - may look at optional `reference_formulas` (list of JSON formulas) as guidance.
    """
    role_split = role_split or ROLE_SPLIT
    tolerance  = tolerance  or ROLE_TOLERANCE
    reference_formulas = reference_formulas or []

    # flatten + metadata maps to re-attach (same idea as MILP)
    flat = []
    meta_by_id, meta_by_name = {}, {}
    for role, arr in materials_by_role.items():
        for m in arr:
            mid = str(m.get("material_id") or m.get("name"))
            name = m.get("name") or mid
            flat.append({
                "material_id": mid,
                "name": name,
                "role": role,
                "usage_max_pct": float(m.get("usage_max_pct", 100.0)),
                "target_usage_mid_pct": float(m.get("target_usage_mid_pct", 1.0)),
                "family": m.get("family"),
                "desc": m.get("descriptors", ""),
            })
            meta = {
                "usage_max_pct": float(m.get("usage_max_pct", 100.0)),
                "target_usage_mid_pct": float(m.get("target_usage_mid_pct", 1.0)),
                "family": m.get("family"),
                "role": role,
                "descriptors": m.get("descriptors", ""),
                "aliases": m.get("aliases", []),
            }
            meta_by_id[mid] = meta
            meta_by_name[name.lower()] = meta

    system = """
    You are an expert perfumery chemist.
    TASK: Allocate GRAMS/PERCENT across the provided materials ONLY, to total exactly 100.0 grams.
    HARD CONSTRAINTS:
    - Use ONLY materials provided in `materials` list. Do not invent or rename.
    - Each material's percent/grams must be <= its `usage_max_pct`.
    - Total grams = 100.0 exactly.
    - Per-role totals must be within role_split ± tolerance.
    OUTPUT FORMAT: strict JSON with keys: top, mid, base, total_grams; each item has material_id, name, grams, percent.
    """

    user = {
        "emotion_text": emotion_text,
        "materials": flat,
        "role_split": role_split,
        "tolerance": tolerance,
        "reference_formulas": reference_formulas
    }

    out = chat_json(system=system, user_json=user, response_schema={
        "top":"list","mid":"list","base":"list","total_grams":"number"
    })

    # normalize + re-attach metadata and info (same as MILP)
    for role in ("top","mid","base"):
        out.setdefault(role, [])
        for x in out[role]:
            x["grams"] = float(x.get("grams", 0.0))
            x["percent"] = float(x.get("percent", 0.0))
            mid = str(x.get("material_id") or "")
            name = (x.get("name") or "").lower()
            meta = meta_by_id.get(mid) or meta_by_name.get(name) or {}
            if "usage_max_pct" in meta: x["usage_max_pct"] = meta["usage_max_pct"]
            if "target_usage_mid_pct" in meta: x["target_usage_mid_pct"] = meta["target_usage_mid_pct"]
            if "family" in meta: x["family"] = meta["family"]
            x["role"] = role
            x["info"] = {
                "descriptors": meta.get("descriptors", ""),
                "aliases": meta.get("aliases", []),
            }
    if float(out.get("total_grams", 0.0)) == 0.0:
        out["total_grams"] = sum(x["grams"] for r in ("top","mid","base") for x in out[r])
    return out















def milp_proportion(materials_by_role, emotion_text, role_split=None, tolerance=None):
    try:
        import pulp
    except Exception:
        return _simple_caps(materials_by_role, role_split or ROLE_SPLIT)

    role_split = role_split or ROLE_SPLIT
    tolerance = tolerance or ROLE_TOLERANCE

    # --- ADDED: build metadata map so we can re-attach details in the output ---
    meta_by_id: Dict[str, Dict[str, Any]] = {}   # <<< ADDED
    meta_by_name: Dict[str, Dict[str, Any]] = {} # <<< ADDED

    mats = []
    id_to_name = {}
    for role, arr in materials_by_role.items():
        for m in arr:
            mid = str(m.get("material_id") or m.get("name"))
            name = m.get("name") or mid
            id_to_name[mid] = name
            # keep a compact meta snapshot to re-attach later                 # <<< ADDED
            meta = {                                                          # <<< ADDED
                "usage_max_pct": float(m.get("usage_max_pct", 100.0)),        # <<< ADDED
                "target_usage_mid_pct": float(m.get("target_usage_mid_pct", 1.0)),  # <<< ADDED
                "family": m.get("family"),                                    # <<< ADDED
                "role": role,                                                 # <<< ADDED
                "descriptors": m.get("descriptors", ""),                      # <<< ADDED
                "aliases": m.get("aliases", []),                              # <<< ADDED
            }                                                                  # <<< ADDED
            meta_by_id[mid] = meta                                            # <<< ADDED
            meta_by_name[name.lower()] = meta                                 # <<< ADDED

            mats.append({
                "id": mid,
                "name": name,
                "role": role,
                "cap": float(m.get("usage_max_pct", 100.0)),
                "w": float(m.get("target_usage_mid_pct", 1.0)),
            })

    prob = pulp.LpProblem("proportioning", pulp.LpMaximize)
    x = {m["id"]: pulp.LpVariable(f"x_{_safe_var(m['id'])}", lowBound=0) for m in mats}
    prob += pulp.lpSum(m["w"] * x[m["id"]] for m in mats)
    prob += pulp.lpSum(x.values()) == 100.0

    for m in mats:
        prob += x[m["id"]] <= m["cap"]

    for role, target in role_split.items():
        ids = [m["id"] for m in mats if m["role"] == role]
        if not ids:
            continue
        prob += pulp.lpSum(x[i] for i in ids) >= target - tolerance
        prob += pulp.lpSum(x[i] for i in ids) <= target + tolerance

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    out = {"top": [], "mid": [], "base": [], "total_grams": 100.0}
    for m in mats:
        grams = float(x[m["id"]].value() or 0.0)
        pct = grams  # 100g total

        # --- ADDED: re-attach metadata as top-level + info -------------------
        meta = meta_by_id.get(m["id"]) or meta_by_name.get(m["name"].lower(), {})  # <<< ADDED
        row = {                                                                     # <<< ADDED
            "material_id": m["id"],                                                 # <<< ADDED
            "name": m["name"],                                                      # <<< ADDED
            "grams": grams,                                                         # <<< ADDED
            "percent": pct,                                                         # <<< ADDED
        }                                                                           # <<< ADDED
        if "usage_max_pct" in meta: row["usage_max_pct"] = meta["usage_max_pct"]    # <<< ADDED
        if "target_usage_mid_pct" in meta: row["target_usage_mid_pct"] = meta["target_usage_mid_pct"]  # <<< ADDED
        if "family" in meta: row["family"] = meta["family"]                          # <<< ADDED
        # keep role stable if present (MILP preserves role via mats)               # <<< ADDED
        row["role"] = m["role"]                                                     # <<< ADDED
        # put reference-only details into info                                     # <<< ADDED
        row["info"] = {                                                             # <<< ADDED
            "descriptors": meta.get("descriptors", ""),                             # <<< ADDED
            "aliases": meta.get("aliases", []),                                      # <<< ADDED
        }                                                                            # <<< ADDED

        out[m["role"]].append(row)
    return out

def _simple_caps(materials_by_role, role_split):
    out = {"top": [], "mid": [], "base": []}

    # --- ADDED: meta lookup for the fallback path too -------------------------
    meta_by_id: Dict[str, Dict[str, Any]] = {}   # <<< ADDED
    meta_by_name: Dict[str, Dict[str, Any]] = {} # <<< ADDED

    role_split = role_split or ROLE_SPLIT
    for role, target in role_split.items():
        arr = materials_by_role.get(role, []) or []
        if not arr:
            continue
        weights = [max(1e-6, float(m.get("target_usage_mid_pct", 1.0))) for m in arr]
        s = sum(weights) or 1.0
        for m, w in zip(arr, weights):
            pct = min(float(m.get("usage_max_pct", 100.0)), target * (w / s))

            mid = str(m.get("material_id") or m.get("name"))                      # <<< ADDED
            name = m.get("name") or mid                                           # <<< ADDED
            meta = {                                                              # <<< ADDED
                "usage_max_pct": float(m.get("usage_max_pct", 100.0)),            # <<< ADDED
                "target_usage_mid_pct": float(m.get("target_usage_mid_pct", 1.0)),# <<< ADDED
                "family": m.get("family"),                                        # <<< ADDED
                "role": role,                                                     # <<< ADDED
                "descriptors": m.get("descriptors", ""),                          # <<< ADDED
                "aliases": m.get("aliases", []),                                  # <<< ADDED
            }                                                                      # <<< ADDED
            meta_by_id[mid] = meta                                                 # <<< ADDED
            meta_by_name[name.lower()] = meta                                      # <<< ADDED

            row = {
                "material_id": mid,
                "name": name,
                "grams": pct,
                "percent": pct,
                "usage_max_pct": meta["usage_max_pct"],            # <<< ADDED
                "target_usage_mid_pct": meta["target_usage_mid_pct"],  # <<< ADDED
                "family": meta["family"],                          # <<< ADDED
                "role": role,                                      # <<< ADDED
                "info": {                                          # <<< ADDED
                    "descriptors": meta.get("descriptors", ""),    # <<< ADDED
                    "aliases": meta.get("aliases", []),            # <<< ADDED
                }                                                  # <<< ADDED
            }
            out[role].append(row)

    out["total_grams"] = sum(x["grams"] for r in ("top","mid","base") for x in out[r])
    return out

def _safe_var(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", s or "m")
