"""
Lunar AIccord – end-to-end runner with LangGraph (NEW with transform path).

Pipeline:
  Path A (new brief):
    1) intent_parser.parse_intent(brief, colors=None)
    2) sensory_moodboard.build_seed(intent_tone, color_terms, story_text=brief)
    3) note_rag.retrieve_candidates(seed_terms, emotion_text, top_k=24)
    4) scent_composer.compose(candidates_by_family, emotion_text, online)
    5) compliance_agent.check(formula) [+ apply_fixes if available]
    6) evaluator.evaluate(formula, emotion_text)
    7) brand_styler.style(intent, formula)

  Path B (transform/flanker, when --base is provided):
    T) transformer.transform_formula(base_formula_obj, brief_mod, intent, emotion_text)
    5) compliance_agent.check(formula) [+ apply_fixes if available]
    6) evaluator.evaluate(formula, emotion_text)
    7) brand_styler.style(intent, formula)
"""
import os, sys, json, pickle, argparse, inspect
from importlib import import_module
from dataclasses import is_dataclass, asdict
from types import SimpleNamespace
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

# -----------------------------------------------------------------------------
# Environment bridge (unchanged)
# -----------------------------------------------------------------------------
os.environ.setdefault("LUNAR_USE_AZURE", "1")
os.environ.setdefault("LUNAR_USE_QDRANT", "1")

try:
    from config import (
        API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION,
        AZURE_CHAT_DEPLOYMENT, AZURE_EMBED_DEPLOYMENT,
    )
    os.environ.setdefault("AZURE_OPENAI_API_KEY", API_KEY or "")
    os.environ.setdefault("AZURE_OPENAI_ENDPOINT", AZURE_ENDPOINT)
    os.environ.setdefault("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION)
    os.environ.setdefault("AZURE_OPENAI_CHAT_DEPLOYMENT", AZURE_CHAT_DEPLOYMENT)
    os.environ.setdefault("AZURE_OPENAI_EMBED_DEPLOYMENT", AZURE_EMBED_DEPLOYMENT)
except Exception:
    pass

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJ = Path("/Users/ranykhirbawi/Desktop/LunarAIccord").resolve()
INDICES = PROJ / "indices"

INTRO = """\
Hi! I’m Lunar AIccord — an autonomous multi-agent that turns your creative brief
(text + brand colors) into a complete, IFRA-aware fragrance formula, a name, and a story.
Tell me your brand and vibe (e.g., “minimalist wellness, spring launch, #CFE8FF, marine fresh linen”),
and I’ll do the rest.
"""

# -----------------------------------------------------------------------------
# Safe import helpers
# -----------------------------------------------------------------------------
def _try(dotted: str):
    try:
        mod_path, fn = dotted.rsplit(".", 1)
        mod = import_module(mod_path)
        return getattr(mod, fn)
    except Exception as e:
        # Optional debug: set LUNAR_DEBUG=1 to see why imports fail
        if os.getenv("LUNAR_DEBUG", "").lower() in ("1","true","yes"):
            import traceback
            print(f"[LUNAR_DEBUG] Failed to import {dotted}: {e}", file=sys.stderr)
            traceback.print_exc()
        return None


def _asdict_safe(x):
    if x is None:
        return None
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return x
    try:
        return dict(x)
    except Exception:
        pass
    try:
        return vars(x)  # handles SimpleNamespace
    except Exception:
        return x

def _listify(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def _ensure_formula_object(x: Any) -> Any:
    """
    Object with .top/.mid/.base lists of items having .name/.percent/.grams/.info.
    Accepts dataclass/dict-like or already-correct object.
    """
    if hasattr(x, "top") and hasattr(x, "mid") and hasattr(x, "base"):
        return x
    d = _asdict_safe(x) or {}
    if not (isinstance(d, dict) and all(k in d for k in ("top", "mid", "base"))):
        return SimpleNamespace(top=[], mid=[], base=[])
    def to_item(it):
        dd = _asdict_safe(it) or {}
        return SimpleNamespace(
            name=dd.get("name") or dd.get("note") or "",
            percent=float(dd.get("percent", 0.0)) if dd.get("percent") is not None else 0.0,
            grams=float(dd.get("grams", 0.0)) if dd.get("grams") is not None else 0.0,
            info=dd.get("info") or {}
        )
    return SimpleNamespace(
        top=[to_item(z) for z in (d.get("top") or [])],
        mid=[to_item(z) for z in (d.get("mid") or [])],
        base=[to_item(z) for z in (d.get("base") or [])],
    )

# -----------------------------------------------------------------------------
# Indices (TF-IDF fallback for notes)
# -----------------------------------------------------------------------------
def _load_notes_index():
    pk = INDICES / "notes.pkl"
    if not pk.exists():
        raise RuntimeError(f"Missing index file: {pk}. Build indices first (data_ingest/build_collections.py).")
    with open(pk, "rb") as f:
        blob = pickle.load(f)
    return blob["vectorizer"], blob["X"], blob["rows"]

# -----------------------------------------------------------------------------
# Fallbacks (kept from your older version)
# -----------------------------------------------------------------------------
def fallback_parse_intent(brief: str, colors: Optional[List[str]] = None, **kwargs):
    txt = " " + brief.lower() + " "
    season = next((s for s in ["spring","summer","autumn","fall","winter"] if s in txt), None)
    context = next((c for c in ["spa","gym","office","hotel","retail","lobby","event","wellness"] if c in txt), None)
    brand_tone = [w for w in txt.split() if w in (
        "calm","fresh","clean","warm","airy","marine","luminous","minimalist","confident","polished","soft","cool","dark","green","woody","citrus","floral","spicy","herbal","ozonic","linen","amber","musk"
    )]
    return {
        "season": season,
        "context": context,
        "target_audience": None,
        "brand_tone": brand_tone or ["clean","fresh"],
        "constraints": [],
        "color_terms": colors or [],
        "raw": brief
    }

def fallback_build_seed(intent_tone: List[str], color_terms: List[str], story_text: Optional[str] = None, **kwargs):
    base = ["bergamot","aldehydes","muguet","rose","cedar","musk","amber"]
    em = " ".join(list(dict.fromkeys((intent_tone or []) + (color_terms or []))))
    return {"seed_notes": base, "emotion_text": em or "clean fresh", "emotion_terms": (intent_tone or [])}

def fallback_retrieve_candidates(seed_terms: List[str], emotion_text: str, top_k: int = 24):
    vec, X, rows = _load_notes_index()
    qtext = " ".join(_listify(seed_terms)) + " " + str(emotion_text or "")
    q = vec.transform([qtext])
    import numpy as np
    scores = q.dot(X.T).toarray()[0]
    idx = np.argsort(-scores)[:max(top_k, 1)]
    out = []
    for i in idx:
        r = rows[int(i)]
        out.append({"note": r.get("note",""), "family": r.get("family",""), "score": float(scores[i]), "source": r.get("description","")[:80]})
    buckets = {"top": [], "mid": [], "base": []}
    for it in out:
        fam = (it["family"] or "").lower()
        if fam in buckets:
            buckets[fam].append(it)
        else:
            buckets["mid"].append(it)
    return {"candidates": out, "by_family": buckets}

def fallback_compose(candidates: List[Any]):
    top = [c for c in candidates if getattr(c, "role", getattr(c, "family", "")).startswith("top")][:6]
    mid = [c for c in candidates if getattr(c, "role", getattr(c, "family", "")).startswith("mid")][:6]
    base= [c for c in candidates if getattr(c, "role", getattr(c, "family", "")).startswith("base")][:6]
    def torow(x):
        name = getattr(x, "name", None) or getattr(x, "note", None) or (x.get("name") if isinstance(x, dict) else str(x))
        return {"name": str(name), "percent": 2.0}
    if not top and not mid and not base:
        grab = [candidates[i] for i in range(min(12, len(candidates)))]
        top, mid, base = grab[:4], grab[4:8], grab[8:]
    return {"top": list(map(torow, top)), "mid": list(map(torow, mid)), "base": list(map(torow, base))}

def fallback_compliance(formula_like: Any):
    return {"ok": True, "issues": [], "formula": formula_like}

def fallback_evaluate(formula_like: Any, emotion_text: str):
    return {"score": 72.5, "rationale": "Heuristic baseline score."}

def fallback_brand(intent_like: Any, formula_like: Any):
    return {"name": "Lunar Accord", "story": "A clean, airy signature with marine lift and soft woods."}

def fallback_transform(base_formula_obj: Any, brief_mod: str, intent: Dict[str, Any], emotion_text: str):
    """Very light fallback: return base formula unchanged; real agent should exist in agent.transformer."""
    return _asdict_safe(base_formula_obj) or {"top": [], "mid": [], "base": [], "total_grams": 100.0}

# -----------------------------------------------------------------------------
# Import your agents (preferred)
# -----------------------------------------------------------------------------
parse_intent_fn   = _try("agent.intent_parser.parse_intent")         or fallback_parse_intent
build_seed_fn     = _try("agent.sensory_moodboard.build_seed")       or fallback_build_seed
retrieve_fn       = _try("agent.note_rag.retrieve_candidates")       or fallback_retrieve_candidates
compose_fn        = _try("agent.scent_composer.compose")             
compliance_fn     = _try("agent.compliance_agent.check")             or fallback_compliance
evaluate_fn       = _try("agent.evaluator.evaluate")                 or fallback_evaluate
brand_fn          = _try("agent.brand_styler.style")                 or fallback_brand
transform_fn      = _try("agent.transformer.transform_formula")      # may be None

# -----------------------------------------------------------------------------
# Adapters / Normalizers
# -----------------------------------------------------------------------------
def _intent_to_fields(intent: Any) -> Dict[str, Any]:
    x = _asdict_safe(intent) or {}
    return {
        "season": x.get("season"),
        "context": x.get("context"),
        "target_audience": x.get("target_audience"),
        "brand_tone": x.get("brand_tone") or [],
        "constraints": x.get("constraints") or [],
        "color_terms": x.get("color_terms") or [],
        "raw": x.get("raw"),
    }

def _seed_to_fields(seed: Any) -> Dict[str, Any]:
    x = _asdict_safe(seed) or {}
    return {
        "seed_notes": x.get("seed_notes") or x.get("notes") or [],
        "emotion_text": x.get("emotion_text") or " ".join(x.get("emotion_terms", [])),
        "emotion_terms": x.get("emotion_terms") or [],
    }

def _normalize_candidates(ret: Any) -> Dict[str, Any]:
    """
    Accepts:
      - {'candidates':[...], 'by_family': {'top':[], 'mid':[], 'base':[]}}
      - list[...] -> will group into by_family
    """
    x = _asdict_safe(ret)
    if isinstance(x, dict) and ("candidates" in x or "by_family" in x):
        cands = x.get("candidates")
        if not cands and "by_family" in x:
            cands = (x["by_family"].get("top", []) + x["by_family"].get("mid", []) + x["by_family"].get("base", []))
        byfam = x.get("by_family") or {"top": [], "mid": [], "base": []}
        return {"candidates": _listify(cands), "by_family": byfam}

    lst = _listify(x)
    byfam = {"top": [], "mid": [], "base": []}
    for it in lst:
        d = _asdict_safe(it) or {}
        fam = (d.get("family") or d.get("role") or "").lower()
        if fam.startswith("top"): byfam["top"].append(d)
        elif fam.startswith("mid"): byfam["mid"].append(d)
        elif fam.startswith("base"): byfam["base"].append(d)
        else: byfam["mid"].append(d)
    return {"candidates": lst, "by_family": byfam}

def _normalize_formula(obj: Any) -> Dict[str, Any]:
    x = _asdict_safe(obj)

    def _row_common(dd_in: Any) -> Dict[str, Any]:
        dd = _asdict_safe(dd_in) or {}
        if not isinstance(dd, dict):
            dd = {}
        # IMPORTANT: do NOT map note->name; name must be the material name
        name = dd.get("name") or ""  # no fallback to "note"
        grams = float(dd.get("grams", 0.0)) if dd.get("grams") is not None else 0.0
        pct   = float(dd.get("percent", 0.0)) if dd.get("percent") is not None else 0.0
        out = {"name": name, "grams": grams, "percent": pct}
        # carry material metadata through
        if "material_id" in dd: out["material_id"] = dd["material_id"]
        if "usage_max_pct" in dd: out["usage_max_pct"] = float(dd["usage_max_pct"])
        if "target_usage_mid_pct" in dd: out["target_usage_mid_pct"] = float(dd["target_usage_mid_pct"])
        if "family" in dd: out["family"] = dd["family"]
        if "role" in dd: out["role"] = dd["role"]
        if isinstance(dd.get("info"), dict): out["info"] = dd["info"]
        return out

    if isinstance(x, dict) and all(k in x for k in ("top","mid","base")):
        return {
            "top":  [_row_common(z) for z in (x.get("top") or [])],
            "mid":  [_row_common(z) for z in (x.get("mid") or [])],
            "base": [_row_common(z) for z in (x.get("base") or [])],
        }

    if isinstance(x, dict) and "formula" in x:
        f = _asdict_safe(x["formula"]) or {}
        if all(k in f for k in ("top","mid","base")):
            return _normalize_formula(f)

    if hasattr(obj, "top") and hasattr(obj, "mid") and hasattr(obj, "base"):
        return {
            "top":  [_row_common(z) for z in (getattr(obj, "top", []) or [])],
            "mid":  [_row_common(z) for z in (getattr(obj, "mid", []) or [])],
            "base": [_row_common(z) for z in (getattr(obj, "base", []) or [])],
        }

    x = _asdict_safe(obj)
    if isinstance(x, dict) and "draft_formula" in x:
        f = _asdict_safe(x["draft_formula"]) or {}
        if all(k in f for k in ("top","mid","base")):
            return _normalize_formula(f)

    return {"top": [], "mid": [], "base": []}




def _call_compose_with_iterable(seed_notes: List[str], by_family: Dict[str, List[Dict[str, Any]]]):
    """
    Build a single iterable of candidate objects/dicts for composer.compose(candidates).
    """
    def mk(role, it):
        d = _asdict_safe(it) or {}
        note = d.get("note") or d.get("name") or ""
        name = d.get("name") or note
        fam  = d.get("family") or role
        return SimpleNamespace(note=note, name=name, family=fam, role=role, weight=float(d.get("weight", 1.0)))
    ordered = []
    for role in ("top","mid","base"):
        for it in by_family.get(role, []):
            ordered.append(mk(role, it))
    if not ordered:
        ordered = [SimpleNamespace(note=s, name=s, family="mid", role="mid", weight=1.0) for s in seed_notes[:12]]
    shapes = [
        ordered,
        [ {"note":c.note,"name":c.name,"family":c.family,"role":c.role,"weight":c.weight} for c in ordered ],
        [ f"{c.role}:{c.note}" for c in ordered ],
    ]
    last_err = None
    sig = inspect.signature(compose_fn)
    for payload in shapes:
        try:
            return compose_fn(payload)
        except Exception as e:
            last_err = e
    raise TypeError(f"compose() did not accept any iterable payload; signature={sig}, last_error={last_err}")

# -----------------------------------------------------------------------------
# LangGraph wiring
# -----------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END

class LunarState(TypedDict, total=False):
    # inputs
    brief: str
    colors: List[str]
    online: bool
    # transform path inputs
    base_formula_obj: Any
    brief_mod: Optional[str]

    # shared outputs
    intent: Dict[str, Any]
    seed: Dict[str, Any]
    candidates: Dict[str, Any]
    candidates_by_family: Dict[str, List[Dict[str, Any]]]
    draft_formula: Dict[str, Any]
    draft_formula_obj: Any
    compliance: Dict[str, Any]
    formula: Dict[str, Any]
    formula_obj: Any
    evaluation: Dict[str, Any]
    branding: Dict[str, Any]
    telemetry: Dict[str, Any]
    __has_base: bool

def node_intent(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    brief_txt = out.get("brief") or ""
    if not brief_txt:
        # synthesize a minimal brief so the graph never KeyErrors
        brief_txt = "clean fresh minimalist"
        out["brief"] = brief_txt
    cols = out.get("colors") or []
    intent = parse_intent_fn(brief_txt, colors=cols)
    out["intent"] = _intent_to_fields(intent)
    out.setdefault("telemetry", {})["intent_ms"] = round((perf_counter()-t0)*1000, 1)
    return out


def node_sensory(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    it = out["intent"]
    seed = build_seed_fn(intent_tone=it.get("brand_tone") or [],
                         color_terms=it.get("color_terms") or (state.get("colors") or []),
                         story_text=state["brief"])
    out["seed"] = _seed_to_fields(seed)
    out.setdefault("telemetry", {})["sensory_ms"] = round((perf_counter()-t0)*1000, 1)
    return out

def node_retrieval(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    s = out["seed"]
    ret = retrieve_fn(seed_terms=s.get("seed_notes") or [], emotion_text=s.get("emotion_text",""), top_k=24)
    norm = _normalize_candidates(ret)
    out["candidates"] = norm
    out["candidates_by_family"] = norm.get("by_family", {"top": [], "mid": [], "base": []})
    # NEW: ensure emotion_text is present on state for composer
    out["emotion_text"] = s.get("emotion_text","")
    out.setdefault("telemetry", {})["retrieval_ms"] = round((perf_counter()-t0)*1000, 1)
    return out


def node_compose(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)

    composer_input = {
        "candidates_by_family": out.get("candidates_by_family", {"top": [], "mid": [], "base": []}),
        "emotion_text": out.get("emotion_text", out.get("seed", {}).get("emotion_text", "")),
        "online": bool(out.get("online", True)),
    }

    def _extract_draft(composed_obj):
        """
        Accept any of:
          - {"top": [...], "mid": [...], "base": [...]}
          - {"draft_formula": {"top": [...], "mid": [...], "base": [...]} }
          - SimpleNamespace with .top/.mid/.base
        Return a dict with top/mid/base or raise.
        """
        x = _asdict_safe(composed_obj)

        # direct dict
        if isinstance(x, dict):
            if all(k in x for k in ("top","mid","base")):
                return x
            if "draft_formula" in x and isinstance(x["draft_formula"], dict) and \
               all(k in x["draft_formula"] for k in ("top","mid","base")):
                return x["draft_formula"]

        # object with .top/.mid/.base
        if hasattr(composed_obj, "top") and hasattr(composed_obj, "mid") and hasattr(composed_obj, "base"):
            return {
                "top":  list(getattr(composed_obj, "top") or []),
                "mid":  list(getattr(composed_obj, "mid") or []),
                "base": list(getattr(composed_obj, "base") or []),
            }

        raise ValueError(f"compose() returned unexpected shape: {type(composed_obj)}")

    try:
        if compose_fn is None:
            raise RuntimeError(
                "agent.scent_composer.compose not importable. "
                "Make sure agent/scent_composer.py exists and is on sys.path."
            )
        # 1) Preferred: our composer expects a DICT state
        composed = compose_fn(composer_input)

        # extract the draft dict from whatever the composer returned
        draft = _extract_draft(composed)

    except Exception as e_dict:
        # 2) Secondary: support composers that expect an iterable
        try:
            seed_notes = (out.get("seed") or {}).get("seed_notes") or []
            byfam = out.get("candidates_by_family", {"top": [], "mid": [], "base": []})
            composed = _call_compose_with_iterable(seed_notes, byfam)
            draft = _extract_draft(composed)
        except Exception as e_iter:
            print(f"[LUNAR_DEBUG] compose() failed; falling back to MATERIAL-BASED fallback", file=sys.stderr)
            from agent.material_rag import MaterialRAG
            from agent.proportioner_llm import milp_proportion
            matrag = MaterialRAG(online=bool(out.get("online", True)))
            emotion_text = out.get("emotion_text", out.get("seed", {}).get("emotion_text", "")) or ""
            defaults = {
                "top":  ["Calone", "Linalyl acetate", "Citronellal"],
                "mid":  ["Hedione", "Dihydromyrcenol", "Iso E Super"],
                "base": ["Ambroxan", "Cashmeran", "Galaxolide"]
            }
            chosen = {"top": [], "mid": [], "base": []}
            for role, qs in defaults.items():
                for q in qs:
                    hits = matrag.retrieve(q, emotion_text, top_k=1)
                    if hits:
                        best = dict(hits[0]); best["role"] = role
                        chosen[role].append(best)
            draft = milp_proportion(chosen, emotion_text) if any(chosen.values()) else {"top": [], "mid": [], "base": [], "total_grams": 100.0}


    # Normalize to object + dict for downstream
    draft_obj = _ensure_formula_object(draft)
    out["draft_formula_obj"] = draft_obj
    out["draft_formula"]     = _normalize_formula(draft_obj)

    out.setdefault("telemetry", {})["compose_ms"] = round((perf_counter()-t0)*1000, 1)
    return out



def node_compliance(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    draft_obj = _ensure_formula_object(out.get("draft_formula_obj", out.get("draft_formula")))
    comp = compliance_fn(draft_obj)
    compd = _asdict_safe(comp) or {}
    try:
        from agent.compliance_agent import apply_fixes as _apply_fixes
    except Exception:
        _apply_fixes = None
    fixes = compd.get("fixes") if isinstance(comp, dict) else getattr(comp, "fixes", [])
    final_obj = draft_obj
    if _apply_fixes and fixes:
        final_obj = _apply_fixes(draft_obj, fixes)
    out["compliance"]  = compd
    out["formula_obj"] = _ensure_formula_object(final_obj)
    out["formula"]     = _normalize_formula(final_obj)
    out.setdefault("telemetry", {})["compliance_ms"] = round((perf_counter()-t0)*1000, 1)
    return out

def node_evaluate(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    ev = evaluate_fn(out.get("formula_obj", out["formula"]), out.get("seed", {}).get("emotion_text", ""))
    out["evaluation"] = _asdict_safe(ev) or {"score": None, "rationale": "N/A"}
    out.setdefault("telemetry", {})["evaluate_ms"] = round((perf_counter()-t0)*1000, 1)
    return out

def node_brand(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    br = brand_fn(out.get("intent", {}), out.get("formula_obj", out.get("formula")))
    out["branding"] = _asdict_safe(br) or {"name": "Untitled", "story": ""}
    out.setdefault("telemetry", {})["branding_ms"] = round((perf_counter()-t0)*1000, 1)
    return out

def node_check_mode(state: LunarState) -> LunarState:
    out = dict(state)
    base = out.get("base_formula_obj")
    base_ok = isinstance(base, dict) and all(k in base for k in ("top","mid","base")) and \
              any(base.get(k) for k in ("top","mid","base"))  # at least one list non-empty
    has_transform = transform_fn is not None
    out["__has_base"] = bool(base_ok and has_transform)
    return out

def node_transform(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    base = out.get("base_formula_obj")
    brief_mod = out.get("brief_mod") or out.get("brief") or ""
    intent = out.get("intent", {})
    emotion_text = out.get("seed", {}).get("emotion_text", "")

    result = None
    if transform_fn:
        result = transform_fn(out) if hasattr(transform_fn, "__call__") and \
                 len(inspect.signature(transform_fn).parameters) == 1 \
                 else transform_fn(base, brief_mod, intent, emotion_text)

    # If transform unavailable/returned empty, synthesize via retrieval+compose
    def _is_empty_formula(f):
        d = _asdict_safe(f) or {}
        return not (isinstance(d, dict) and any(d.get(k) for k in ("top","mid","base")))

    if not result or _is_empty_formula(result):
        # Ensure we have a seed/emotion target
        if not out.get("seed"):
            seed = build_seed_fn(
                intent_tone=(intent.get("brand_tone") or []),
                color_terms=(intent.get("color_terms") or []),
                story_text=out.get("brief") or brief_mod or ""
            )
            out["seed"] = _seed_to_fields(seed)
        s = out["seed"]
        ret = retrieve_fn(
            seed_terms=s.get("seed_notes") or [],
            emotion_text=s.get("emotion_text",""),
            top_k=24
        )
        norm = _normalize_candidates(ret)
        byfam = norm.get("by_family", {"top": [], "mid": [], "base": []})
        result = _call_compose_with_iterable(s.get("seed_notes") or [], byfam)

    draft_obj = _ensure_formula_object(result)
    out["draft_formula_obj"] = draft_obj
    out["draft_formula"]     = _normalize_formula(draft_obj)
    out.setdefault("telemetry", {})["transform_ms"] = round((perf_counter()-t0)*1000, 1)
    return out


def build_app():
    g = StateGraph(LunarState)
    # Core nodes
    g.add_node("check_mode",  node_check_mode)
    g.add_node("intent",      node_intent)
    g.add_node("sensory",     node_sensory)
    g.add_node("retrieval",   node_retrieval)
    g.add_node("compose",     node_compose)
    g.add_node("transform",   node_transform)
    g.add_node("compliance",  node_compliance)
    g.add_node("evaluate",    node_evaluate)
    g.add_node("branding",    node_brand)

    g.add_edge(START, "check_mode")

    # Conditional route: transform if base formula exists, else go through intent path
    def _route(state: LunarState) -> str:
        return "transform" if state.get("__has_base") else "intent"

    g.add_conditional_edges("check_mode", _route, {"transform": "transform", "intent": "intent"})

    # Path A (new brief)
    g.add_edge("intent", "sensory")
    g.add_edge("sensory", "retrieval")
    g.add_edge("retrieval", "compose")
    g.add_edge("compose", "compliance")

    # Path B (transform/flanker)
    g.add_edge("transform", "compliance")

    # Merge tail
    g.add_edge("compliance", "evaluate")
    g.add_edge("evaluate", "branding")
    g.add_edge("branding", END)

    return g.compile()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _read_text_file(p: str) -> str:
    pth = Path(p)
    if not pth.exists():
        raise FileNotFoundError(f"Brief file not found: {p}")
    return pth.read_text(encoding="utf-8").strip()

def _read_json(p: str) -> dict:
    pth = Path(p)
    if not pth.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(pth.read_text(encoding="utf-8"))

def _autosave_path(brief: str) -> str:
    from datetime import datetime
    slug = "-".join((brief or "brief").lower().split())[:40] or "brief"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("examples") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{ts}_{slug}.json")

if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

def main():
    ap = argparse.ArgumentParser(description="Run Lunar AIccord pipeline.")
    # Standard brief path
    ap.add_argument("--brief", default="", help="Creative brief text (required unless --interactive or --file or --base)")
    ap.add_argument("--file",  default="", help="Path to a text file containing the brief (e.g., examples/01_brief.txt)")
    ap.add_argument("--colors", default="", help="Comma-separated color terms or hexes (e.g. '#CFE8FF, white')")
    # NEW: transform path inputs
    ap.add_argument("--base", default="", help="Path to existing formula JSON to modify (enables transform path)")
    ap.add_argument("--brief_mod", default="", help="Modification brief if --base is provided (e.g., 'summer flanker')")
    # Operational flags
    ap.add_argument("--online", action="store_true", help="If set, use online services (Qdrant/Azure) where available")
    ap.add_argument("--save",  default="", help="Optional path to write JSON result; interactive will auto-save if omitted")
    ap.add_argument("--no-stream", action="store_true", help="Disable step-by-step streaming prints")
    ap.add_argument("--interactive", action="store_true", help="Explain functionality then wait for a user prompt")
    args = ap.parse_args()

    # Interactive input
    if args.interactive:
        print(INTRO, flush=True)
        brief = input("Your brief: ").strip()
        colors_str = input("Optional colors (comma-separated, hex or names): ").strip()
        colors = [c.strip() for c in colors_str.split(",") if c.strip()] if colors_str else []
        payload = {"brief": brief, "colors": colors, "online": bool(args.online)}
        autosave = not bool(args.save)
    else:
        payload: Dict[str, Any] = {"online": bool(args.online)}
        # Transform path if --base provided
        if args.base:
            base_blob = _read_json(args.base)
            # accept either {"formula": {...}} or direct top/mid/base
            base_formula = base_blob.get("formula") if isinstance(base_blob, dict) else None
            if not base_formula and all(k in base_blob for k in ("top","mid","base")):
                base_formula = base_blob
            payload["base_formula_obj"] = base_formula
            payload["brief_mod"] = (args.brief_mod or args.brief or "").strip()
            # intent/sensory may still run (intent optional for branding later), but check_mode will route to transform
            # colors optional
            payload["colors"] = [c.strip() for c in args.colors.split(",") if c.strip()] if args.colors else []
            autosave = False
        else:
            # Non-transform path: require a brief (from --file or --brief)
            if args.file:
                brief = _read_text_file(args.file)
            else:
                brief = (args.brief or "").strip()
            if not brief:
                ap.error("Provide --brief text OR --file path, or use --interactive. Or pass --base for transform mode.")
            colors = [c.strip() for c in args.colors.split(",") if c.strip()] if args.colors else []
            payload.update({"brief": brief, "colors": colors})
            autosave = False

    app = build_app()

    if not args.no_stream:
        print(">>> STREAM")
        for event in app.stream(payload):
            for node_name, state in event.items():
                if node_name in ("__start__", "__end__"):
                    continue
                keys = [k for k in state.keys() if k not in ("telemetry",)]
                print(f"[{node_name}] -> {sorted(keys)}")

    result: Dict[str, Any] = app.invoke(payload)

    # Pretty summary
    intent = result.get("intent", {}) or {}
    branding = result.get("branding", {}) or {}
    evaluation = result.get("evaluation", {}) or {}
    formula = result.get("formula", {}) or {}

    print("\n=== SUMMARY ===")
    print("Name   :", branding.get("name"))
    print("Season :", intent.get("season"), "| Context:", intent.get("context"))
    tone_list = intent.get("brand_tone") or intent.get("tone") or []
    print("Tone   :", ", ".join([str(x) for x in tone_list][:6]))
    if isinstance(evaluation, dict):
        print("Score  :", evaluation.get("score"), "|", evaluation.get("rationale"))
    story = branding.get("story", "")
    if isinstance(story, str):
        print("Story  :", story[:240], "..." if len(story) > 240 else "")

    # Save result
    def _save(out_path: str):
        p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({
                "candidates": result.get("candidates"),
                "formula": formula,
                "compliance": result.get("compliance"),
                "evaluation": evaluation,
                "branding": branding,
                "brief": payload.get("brief") or payload.get("brief_mod") or "",
                "colors": payload.get("colors", []),
                "mode": "transform" if payload.get("base_formula_obj") else "brief",
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved result to {p}")

    save_path = args.save or (_autosave_path(payload.get("brief") or payload.get("brief_mod") or "run") if autosave else "")
    if save_path:
        _save(save_path)

if __name__ == "__main__":
    try:
        os.chdir(PROJ)
    except Exception:
        pass
    sys.path.insert(0, str(PROJ))
    sys.path.insert(0, str(PROJ / "agent"))
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
