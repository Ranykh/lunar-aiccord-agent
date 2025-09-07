import os, json, re, random
from dataclasses import dataclass
from typing import Any

USE_AZURE = os.getenv("LUNAR_USE_AZURE", "").lower() in ("1","true","yes")
try:
    from agent.llm_client import chat
except Exception:
    chat = None
    USE_AZURE = False

@dataclass
class Branding:
    name: str
    story: str

ADJ = ["Lunar","Equinox","Nocturne","Aurea","Aqua","Verde","Obsidian","Nimbus","Serene","Solstice"]
NUUN = ["Accord","Veil","Current","Silhouette","Tide","Ember","Whisper","Halo","Pulse","Drift"]

def _intent_to_fields(intent: Any):
    # supports dataclass or dict-like
    def get(attr, default=None):
        if hasattr(intent, attr): return getattr(intent, attr)
        if isinstance(intent, dict): return intent.get(attr, default)
        return default
    return {
        "season": get("season"),
        "context": get("context"),
        "brand_tone": get("brand_tone", []),
        "color_terms": get("color_terms", []),
        "constraints": get("constraints", []),
    }

def _formula_summary(formula_like: Any) -> str:
    def items(lst):
        for x in lst or []:
            if hasattr(x, "name") and hasattr(x, "percent"):
                yield f"{x.name} {getattr(x,'percent',0)}%"
            elif isinstance(x, dict):
                yield f"{x.get('name','')} {x.get('percent',0)}%"
    if hasattr(formula_like, "top"):
        top, mid, base = formula_like.top, formula_like.mid, formula_like.base
    else:
        top, mid, base = formula_like.get("top",[]), formula_like.get("mid",[]), formula_like.get("base",[])
    return (
        f"Top: {', '.join(items(top))} | "
        f"Mid: {', '.join(items(mid))} | "
        f"Base: {', '.join(items(base))}"
    )

def _azure_brand(intent: Any, formula: Any) -> Branding:
    if not (USE_AZURE and chat):
        raise RuntimeError("Azure not available")
    fields = _intent_to_fields(intent)
    summary = _formula_summary(formula)
    user = (
        "Create a product **name** (max 3 words) and a short **story** (2–3 sentences) "
        "for a fragrance. Keep it tasteful and brandable.\n\n"
        f"Season: {fields['season']}\n"
        f"Context: {fields['context']}\n"
        f"Brand tone: {', '.join(fields['brand_tone'] or [])}\n"
        f"Colors: {', '.join(fields['color_terms'] or [])}\n"
        f"Constraints: {', '.join(fields['constraints'] or [])}\n"
        f"Formula summary: {summary}\n\n"
        "Return **ONLY** JSON with keys: name, story."
    )
    txt = chat(
        [{"role":"system","content":"You are a senior brand copywriter for a luxury perfumery."},
         {"role":"user","content":user}],
        temperature=0.7
    )
    # Parse JSON; tolerate stray text
    try:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        data = json.loads(m.group(0)) if m else json.loads(txt)
        name = str(data.get("name","")).strip() or None
        story = str(data.get("story","")).strip() or None
        if name and story:
            return Branding(name=name, story=story)
    except Exception:
        pass
    # fallback minimal extraction
    lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
    name = (lines[0][:40] if lines else None) or (random.choice(ADJ)+" "+random.choice(NUUN))
    story = " ".join(lines[1:3]) or f"{name} captures a modern, clean aura."
    return Branding(name=name, story=story)

def style(intent, formula) -> Branding:
    # Optional Azure path
    if USE_AZURE and chat:
        try:
            return _azure_brand(intent, formula)
        except Exception:
            pass
    # Offline fallback (unchanged behavior)
    base = random.choice(ADJ) + " " + random.choice(NUUN)
    tone = ", ".join(list(dict.fromkeys(getattr(intent, "brand_tone", [])[:4]))) or "modern, clean"
    line = f"{base} captures a {tone} aura, translating your brief into an elegant balance of top, heart, and base."
    return Branding(name=base, story=line)
