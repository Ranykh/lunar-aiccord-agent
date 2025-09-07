#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lunar AIccord – end-to-end runner with LangGraph.

Pipeline:
  1) intent_parser.parse_intent(brief, colors=None) -> Intent dataclass/dict
  2) sensory_moodboard.build_seed(intent_tone, color_terms, story_text=brief) -> SensorySeed dataclass/dict
  3) note_rag.retrieve_candidates(seed_terms, emotion_text, top_k=24) -> dict (preferred) or list
     (fallback provided: TF-IDF over indices/notes.pkl)
  4) scent_composer.compose(candidates) -> Formula dataclass/dict
  5) compliance_agent.check(formula) -> ComplianceResult dataclass/dict or raw formula
  6) evaluator.evaluate(formula, emotion_text) -> EvalResult dataclass/dict
  7) brand_styler.style(intent, formula) -> Branding dataclass/dict

Outputs a JSON blob with: intent, seed, candidates, formula, compliance, evaluation, branding.
"""
from tokens_count.token_meter import TokenMeter
from agent.chain import lunar_chain
import os, sys, json, pickle, argparse, inspect
from importlib import import_module
from dataclasses import is_dataclass, asdict
from types import SimpleNamespace
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# --- Default online (Azure + Qdrant) unless explicitly disabled ---
# --- put this near the top of run.py (after imports of os, sys, etc.) ---
# Make online (Azure) the default unless explicitly disabled
# --- Make ONLINE the default and bridge config → env ---
os.environ.setdefault("LUNAR_USE_AZURE", "1")
os.environ.setdefault("LUNAR_USE_QDRANT", "1")

# Bridge our config constants to env expected by agent.llm_client
try:
    from agent.config import (
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

# Make stdout unbuffered-ish so prompts always show
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _default_save_path(brief: str) -> str:
    # examples/outputs/YYYYmmdd_HHMMSS_slug.json
    from datetime import datetime
    import re
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", (brief or "").lower()).strip("-")[:40] or "run"
    outdir = PROJ / "examples" / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir / f"{ts}_{slug}.json")



INTRO = """\
Hi! I’m Lunar AIccord — an autonomous multi-agent that turns your creative brief
(text + brand colors) into a complete, IFRA-aware fragrance formula, a name, and a story.
Tell me your brand and vibe (e.g., “minimalist wellness, spring launch, #CFE8FF, marine fresh linen”),
and I’ll do the rest.
"""

def interactive_prompt() -> dict:
    print(INTRO)
    brief = input("Your brief: ").strip()
    colors = input("Optional colors (comma-separated, hex or names): ").strip()
    colors_list = [c.strip() for c in colors.split(",")] if colors else []
    return {"brief": brief, "colors": colors_list}


# ----------------------------- Project paths -----------------------------
PROJ = Path("/Users/ranykhirbawi/Desktop/LunarAIccord").resolve()
INDICES = PROJ / "indices"

# ---------------------------- Safe import helpers -------------------------
def _try(dotted: str):
    try:
        mod_path, fn = dotted.rsplit(".", 1)
        mod = import_module(mod_path)
        return getattr(mod, fn)
    except Exception:
        return None

def _asdict_safe(x):
    if x is None: return None
    if is_dataclass(x): return asdict(x)
    if isinstance(x, dict): return x
    try: return dict(x)
    except Exception: return x

def _listify(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]


def _ensure_formula_object(x: Any) -> Any:
    """
    Make sure we have an object with attributes:
      .top/.mid/.base  each a list of items with .name/.percent/.grams/.info
    Accepts either a dataclass Formula, a dict-like, or already-correct object.
    """
    # If it's already an object with .top/.mid/.base, keep it
    if hasattr(x, "top") and hasattr(x, "mid") and hasattr(x, "base"):
        return x

    d = _asdict_safe(x) or {}
    if not (isinstance(d, dict) and all(k in d for k in ("top", "mid", "base"))):
        # last resort: empty shell
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


# ----------------------------- Load indices ------------------------------
def _load_notes_index():
    pk = INDICES / "notes.pkl"
    if not pk.exists():
        raise RuntimeError(f"Missing index file: {pk}. Build indices first (data_ingest/build_collections.py).")
    with open(pk, "rb") as f:
        blob = pickle.load(f)
    # Expected: {"vectorizer": TfidfVectorizer, "X": csr, "rows": [{"note","family","description","text"}]}
    return blob["vectorizer"], blob["X"], blob["rows"]

# ----------------------------- Fallbacks ---------------------------------
def fallback_parse_intent(brief: str, colors: Optional[List[str]] = None, **kwargs):
    # ultra-minimal; your intent_parser should supersede this.
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
    # minimal heuristic
    base = ["bergamot","aldehydes","muguet","rose","cedar","musk","amber"]
    em = " ".join(list(dict.fromkeys((intent_tone or []) + (color_terms or []))))
    return {"seed_notes": base, "emotion_text": em or "clean fresh", "emotion_terms": (intent_tone or [])}

def fallback_retrieve_candidates(seed_terms: List[str], emotion_text: str, top_k: int = 24):
    # TF-IDF cosine query over notes.pkl
    vec, X, rows = _load_notes_index()
    qtext = " ".join(_listify(seed_terms)) + " " + str(emotion_text or "")
    q = vec.transform([qtext])
    import numpy as np
    scores = q.dot(X.T).toarray()[0]  # cosine if vectors are l2-normalized
    idx = np.argsort(-scores)[:max(top_k, 1)]
    out = []
    for i in idx:
        r = rows[int(i)]
        out.append({"note": r.get("note",""), "family": r.get("family",""), "score": float(scores[i]), "source": r.get("description","")[:80]})
    # group into top/mid/base suggestion buckets
    buckets = {"top": [], "mid": [], "base": []}
    for it in out:
        fam = (it["family"] or "").lower()
        if fam in buckets:
            buckets[fam].append(it)
        else:
            # shove unknowns into mid
            buckets["mid"].append(it)
    return {"candidates": out, "by_family": buckets}

def fallback_compose(candidates: List[Any]):
    # Ridiculously small composer to keep flow alive: pick ~6/6/6 and assign rough percents
    from math import isfinite
    top = [c for c in candidates if getattr(c, "role", getattr(c, "family", "")).startswith("top")][:6]
    mid = [c for c in candidates if getattr(c, "role", getattr(c, "family", "")).startswith("mid")][:6]
    base= [c for c in candidates if getattr(c, "role", getattr(c, "family", "")).startswith("base")][:6]
    def torow(x): 
        name = getattr(x, "name", None) or getattr(x, "note", None) or (x.get("name") if isinstance(x, dict) else str(x))
        return {"name": str(name), "percent": 2.0}
    if not top and not mid and not base:
        # if roles unknown, just slice head
        grab = [candidates[i] for i in range(min(12, len(candidates)))]
        top, mid, base = grab[:4], grab[4:8], grab[8:]
    return {
        "top":  list(map(torow, top)),
        "mid":  list(map(torow, mid)),
        "base": list(map(torow, base)),
    }

def fallback_compliance(formula_like: Any):
    # Identity compliance result
    return {"ok": True, "issues": [], "formula": formula_like}

def fallback_evaluate(formula_like: Any, emotion_text: str):
    # Naive score
    return {"score": 72.5, "rationale": "Heuristic baseline score."}

def fallback_brand(intent_like: Any, formula_like: Any):
    return {"name": "Lunar Accord", "story": "A clean, airy signature with marine lift and soft woods."}

# ------------------------ Import your agents (preferred) ------------------
parse_intent_fn   = _try("agent.intent_parser.parse_intent")      or fallback_parse_intent
build_seed_fn     = _try("agent.sensory_moodboard.build_seed")    or fallback_build_seed
retrieve_fn       = _try("agent.note_rag.retrieve_candidates")    or fallback_retrieve_candidates
compose_fn        = _try("agent.scent_composer.compose")          or fallback_compose
compliance_fn     = _try("agent.compliance_agent.check")          or fallback_compliance
evaluate_fn       = _try("agent.evaluator.evaluate")              or fallback_evaluate
brand_fn          = _try("agent.brand_styler.style")              or fallback_brand


# ---------------------------- Adapters / Normalizers ----------------------
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
      - list[...] -> will group into by_family = {'top','mid','base'}
    """
    x = _asdict_safe(ret)
    if isinstance(x, dict) and ("candidates" in x or "by_family" in x):
        # ensure both keys exist
        cands = x.get("candidates")
        if not cands and "by_family" in x:
            cands = (x["by_family"].get("top", []) + x["by_family"].get("mid", []) + x["by_family"].get("base", []))
        byfam = x.get("by_family") or {"top": [], "mid": [], "base": []}
        return {"candidates": _listify(cands), "by_family": byfam}

    # List form: bucket by 'family' if possible
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
    """
    Accept either dataclass/dict with keys top/mid/base (lists of {name, percent}),
    or a compliance wrapper containing 'formula'.
    """
    x = _asdict_safe(obj)
    if isinstance(x, dict) and all(k in x for k in ("top","mid","base")):
        return x
    if isinstance(x, dict) and "formula" in x:
        f = _asdict_safe(x["formula"]) or {}
        if all(k in f for k in ("top","mid","base")):
            return f
    # last resort: empty structure to avoid crashes
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
    # if empty, at least pass seed strings as mid
    if not ordered:
        ordered = [SimpleNamespace(note=s, name=s, family="mid", role="mid", weight=1.0) for s in seed_notes[:12]]
    # try different payload shapes
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

# -------------------------- LangGraph wiring ------------------------------
from langgraph.graph import StateGraph, START, END

class LunarState(TypedDict, total=False):
    brief: str
    colors: List[str]
    intent: Dict[str, Any]
    seed: Dict[str, Any]
    candidates: Dict[str, Any]
    draft_formula: Dict[str, Any]
    draft_formula_obj: Any     
    compliance: Dict[str, Any]
    formula: Dict[str, Any]
    formula_obj: Any           
    evaluation: Dict[str, Any]
    branding: Dict[str, Any]
    telemetry: Dict[str, Any]

def node_intent(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    cols = state.get("colors") or []
    intent = parse_intent_fn(state["brief"], colors=cols)
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
    out.setdefault("telemetry", {})["retrieval_ms"] = round((perf_counter()-t0)*1000, 1)
    return out

def node_compose(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    seed_notes = out["seed"].get("seed_notes") or []
    byfam = out["candidates"]["by_family"]
    draft_raw = _call_compose_with_iterable(seed_notes, byfam)  # could be dict OR object
    draft_obj = _ensure_formula_object(draft_raw)               
    out["draft_formula_obj"] = draft_obj
    out["draft_formula"]     = _normalize_formula(draft_obj)    # dict for display/save
    out.setdefault("telemetry", {})["compose_ms"] = round((perf_counter()-t0)*1000, 1)
    return out


def node_compliance(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    draft_obj = _ensure_formula_object(out.get("draft_formula_obj", out.get("draft_formula")))
    comp = compliance_fn(draft_obj)          # always pass an object with .top/.mid/.base
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
    out["formula_obj"] = _ensure_formula_object(final_obj)  # keep object for downstream
    out["formula"]     = _normalize_formula(final_obj)      # dict for display/save
    out.setdefault("telemetry", {})["compliance_ms"] = round((perf_counter()-t0)*1000, 1)
    return out




def node_evaluate(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    ev = evaluate_fn(out.get("formula_obj", out["formula"]), out["seed"]["emotion_text"])
    out["evaluation"] = _asdict_safe(ev) or {"score": None, "rationale": "N/A"}
    out.setdefault("telemetry", {})["evaluate_ms"] = round((perf_counter()-t0)*1000, 1)
    return out

def node_brand(state: LunarState) -> LunarState:
    t0 = perf_counter(); out = dict(state)
    br = brand_fn(out["intent"], out.get("formula_obj", out["formula"]))
    out["branding"] = _asdict_safe(br) or {"name": "Untitled", "story": ""}
    out.setdefault("telemetry", {})["branding_ms"] = round((perf_counter()-t0)*1000, 1)
    return out



def build_app():
    g = StateGraph(LunarState)
    g.add_node("intent",     node_intent)
    g.add_node("sensory",    node_sensory)
    g.add_node("retrieval",  node_retrieval)
    g.add_node("compose",    node_compose)
    g.add_node("compliance", node_compliance)
    g.add_node("evaluate",   node_evaluate)
    g.add_node("branding",   node_brand)
    g.add_edge(START, "intent")
    g.add_edge("intent", "sensory")
    g.add_edge("sensory", "retrieval")
    g.add_edge("retrieval", "compose")
    g.add_edge("compose", "compliance")
    g.add_edge("compliance", "evaluate")
    g.add_edge("evaluate", "branding")
    g.add_edge("branding", END)
    return g.compile()
# ------------------------------- CLI --------------------------------------
def _read_text_file(p: str) -> str:
    from pathlib import Path
    pth = Path(p)
    if not pth.exists():
        raise FileNotFoundError(f"Brief file not found: {p}")
    return pth.read_text(encoding="utf-8").strip()

def _autosave_path(brief: str) -> str:
    from datetime import datetime
    from pathlib import Path
    slug = "-".join(brief.lower().split())[:40] or "brief"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("examples") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{ts}_{slug}.json")

if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


def main():
    ap = argparse.ArgumentParser(description="Run Lunar AIccord pipeline.")
    ap.add_argument("--brief", default="", help="Creative brief text (required unless --interactive or --file)")
    ap.add_argument("--file",  default="", help="Path to a text file containing the brief (e.g., examples/01_brief.txt)")
    ap.add_argument("--colors", default="", help="Comma-separated color terms or hexes (e.g. '#CFE8FF, white')")
    ap.add_argument("--save",  default="", help="Optional path to write JSON result; interactive will auto-save if omitted")
    ap.add_argument("--no-stream", action="store_true", help="Disable step-by-step streaming prints")
    ap.add_argument("--interactive", action="store_true", help="Explain functionality then wait for a user prompt")
    args = ap.parse_args()

    # Build payload (intro + prompt for interactive)
    if args.interactive:
        # Intro + prompt
        print(INTRO, flush=True)
        brief = input("Your brief: ").strip()
        colors_str = input("Optional colors (comma-separated, hex or names): ").strip()
        colors = [c.strip() for c in colors_str.split(",") if c.strip()] if colors_str else []
        payload = {"brief": brief, "colors": colors}
        autosave = not bool(args.save)
    else:
        # Non-interactive: brief can come from --file or --brief
        if args.file:
            brief = _read_text_file(args.file)
        else:
            brief = (args.brief or "").strip()
        if not brief:
            ap.error("Provide --brief text OR --file path, or use --interactive.")
        colors = [c.strip() for c in args.colors.split(",") if c.strip()] if args.colors else []
        payload = {"brief": brief, "colors": colors}
        autosave = False

    # Build LangGraph app and run
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

    # Save
    save_path = args.save or (_autosave_path(payload["brief"]) if autosave else "")
    if save_path:
        out = {
            "candidates": result.get("candidates") or result.get("note_material_suggestions"),
            "formula": formula,
            "compliance": result.get("compliance"),
            "evaluation": evaluation,
            "branding": branding,
            "brief": payload["brief"],
            "colors": payload.get("colors", []),
        }
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)   # << Ensure directory exists
        with p.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved result to {p}")

if __name__ == "__main__":
    # Ensure relative imports work no matter where you run from
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
