# agent/scent_composer.py
from typing import Dict, Any, List
import re

from config import ROLE_SPLIT
from agent.material_rag import MaterialRAG, mmr_select
# from agent.proportioner_llm import milp_proportion  # <- use deterministic solver only
from agent.proportioner_llm import llm_proportion, milp_proportion  # uses llm_proportion_strict under the hood

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")

def _to_note_list(seq):
    out, seen = [], set()
    for it in (seq or []):
        if isinstance(it, dict):
            n = (it.get("note") or it.get("name") or "").strip()
        else:
            n = str(it).strip()
        if n and n not in seen:
            out.append(n); seen.add(n)
    return out

def compose(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strictly returns materials (never notes). If no materials can be retrieved,
    falls back to a fixed set of *real* marine/woody materials.
    """
    emotion_text = state.get("emotion_text", "") or ""
    byfam        = state.get("candidates_by_family", {}) or {"top": [], "mid": [], "base": []}
    online       = bool(state.get("online", True))

    matrag = MaterialRAG(online=online)
    chosen_by_role: Dict[str, List[Dict[str, Any]]] = {"top": [], "mid": [], "base": []}

    def _as_note(x):
        if isinstance(x, dict):
            return (x.get("note") or x.get("name") or "").strip()
        return str(x).strip()

    # ---------- Try to select *materials* per conceptual note ----------
    for role in ("top", "mid", "base"):
        already = chosen_by_role["top"] + chosen_by_role["mid"] + chosen_by_role["base"]
        for cand in byfam.get(role, []):
            conceptual_note = _as_note(cand)
            if not conceptual_note:
                continue
            cands = matrag.retrieve(conceptual_note, emotion_text, top_k=6)
            if not cands:
                continue

            mmr = mmr_select(cands, already, lambda_=0.7, k=3)
            pick = max(
                mmr,
                key=lambda m: (float(m.get("target_usage_mid_pct", 1.0)), float(m.get("score", 0.0)))
            )
            pick = dict(pick)
            pick["role"] = role
            chosen_by_role[role].append(pick)
            already = chosen_by_role["top"] + chosen_by_role["mid"] + chosen_by_role["base"]

    # ---------- If still nothing, do a *material-based* fallback ----------
    if not any(chosen_by_role.values()):
        # A small, real-material set that works for "marine linen"
        # (replace with catalog IDs if you have them)
        fallback_queries = {
            "top":  ["Calone", "Linalyl acetate", "Citronellal"],
            "mid":  ["Hedione", "Dihydromyrcenol", "Iso E Super"],
            "base": ["Ambroxan", "Cashmeran", "Galaxolide"]
        }
        for role, qs in fallback_queries.items():
            for q in qs:
                hits = matrag.retrieve(q, emotion_text, top_k=1)
                if not hits:
                    continue
                best = dict(hits[0])
                best["role"] = role
                chosen_by_role[role].append(best)

    # If STILL nothing, return an explicit empty (never invent notes)
    if not any(chosen_by_role.values()):
        return {"draft_formula": {"top": [], "mid": [], "base": [], "total_grams": 100.0}}

    # draft = milp_proportion(chosen_by_role, emotion_text, role_split=ROLE_SPLIT)
    try:
        draft = llm_proportion(chosen_by_role, emotion_text, role_split=ROLE_SPLIT)
    except Exception:
        # safe fallback if LLM is unavailable or returns bad JSON
        draft = milp_proportion(chosen_by_role, emotion_text, role_split=ROLE_SPLIT)


    # Ensure we keep material metadata for downstream normalization/compliance
    for role in ("top", "mid", "base"):
        for x in draft.get(role, []):
            # try to copy usage caps and ids from the picked materials
            src = next((m for m in chosen_by_role[role] if m.get("material_id") == x.get("material_id") or m.get("name") == x.get("name")), None)
            if src:
                if "material_id" in src: x["material_id"] = src["material_id"]
                if "usage_max_pct" in src: x["usage_max_pct"] = float(src["usage_max_pct"])
                if "target_usage_mid_pct" in src: x["target_usage_mid_pct"] = float(src["target_usage_mid_pct"])
                if "family" in src: x["family"] = src["family"]
                x["role"] = role

    return {"draft_formula": draft}
