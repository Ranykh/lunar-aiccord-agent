from dataclasses import dataclass
from typing import List, Optional
import os, re

from agent.local_utils import map_hex_to_color_words, normalize_text
from agent.story2notes import StoryToNotes

# --- Optional Azure ---
USE_AZURE = os.getenv("LUNAR_USE_AZURE", "").lower() in ("1","true","yes")
try:
    from agent.llm_client import chat
except Exception:
    chat = None
    USE_AZURE = False

@dataclass
class SensorySeed:
    seed_notes: List[str]
    emotion_text: str
    emotion_terms: List[str]

COLOR_HINTS = {
    "teal": ["marine","green","fresh"],
    "gold": ["amber","warm","luminous"],
    "black": ["leathery","smoky","dark"],
    "white": ["clean","airy","minimalist"],
    "blue": ["marine","ozonic","fresh","linen"],
    "green": ["green","herbal","fresh"],
    "red": ["spicy","warm","amber"],
    "pink": ["floral","soft"],
    "beige": ["soft","powdery","clean"],
}

MOOD_WHITELIST = {
    "calm","fresh","clean","warm","airy","marine","luminous","minimalist",
    "uplifting","confident","polished","soft","cool","dark","green","woody",
    "citrus","floral","spicy","herbal","aldehydic","amber","musk","leathery",
    "balsamic","incense","ozonic","linen"
}

def _heuristic_seed(intent_tone: List[str], color_terms: List[str]) -> SensorySeed:
    hints = []
    for c in color_terms or []:
        if isinstance(c, str) and c.startswith("#"):
            hints += map_hex_to_color_words(c)
        else:
            hints += COLOR_HINTS.get((c or "").lower(), [])
    uniq = []
    for w in (hints + (intent_tone or [])):
        w = normalize_text(w)
        if w and w not in uniq:
            uniq.append(w)
    emotion_terms = [w for w in uniq if w in MOOD_WHITELIST] or (uniq[:6] or ["clean","fresh"])
    emotion_text = " ".join(emotion_terms)

    # Very light default seed notes by mood keywords (keeps your existing behavior stable)
    SEED_MAP = {
        "marine": ["sea notes","sea water","calone","cyclamen"],
        "ozonic": ["aldehydes","cyclamen","muguet","floralozone"],
        "clean":  ["aldehydes","muguet","white musk","fresh linen"],
        "fresh":  ["bergamot","grapefruit","basil","green notes"],
        "floral": ["rose","jasmine","magnolia","lily of the valley"],
        "woody":  ["cedar","vetiver","sandalwood","patchouli"],
        "amber":  ["amber","labdanum","benzoin","vanilla"],
        "musk":   ["musk","white musk"],
        "leathery":["leather","suede"],
        "spicy":  ["cardamom","black pepper","clove","coriander"],
        "herbal": ["rosemary","sage","basil","thyme"],
        "linen":  ["fresh linen","aldehydes","muguet"],
    }
    seeds: List[str] = []
    for t in emotion_terms:
        seeds += SEED_MAP.get(t, [])
    if not seeds:
        seeds = ["bergamot","aldehydes","muguet","rose","cedar","musk","amber"]
    # de-dupe and trim
    seen, out = set(), []
    for s in seeds:
        ss = normalize_text(s)
        if ss and ss not in seen:
            out.append(s); seen.add(ss)
    return SensorySeed(seed_notes=out[:12], emotion_text=emotion_text, emotion_terms=emotion_terms)

def _refine_with_gpt(seed: SensorySeed, color_terms: List[str]) -> SensorySeed:
    """Optional refinement using GPT-4o (keeps offline seed as base)."""
    if not (USE_AZURE and chat):
        return seed
    prompt = (
        "You are a perfumer's assistant. Given emotion terms and brand colors, "
        "propose 10-14 concise fragrance notes (single or two-word names). "
        "Focus on realism (materials or common perfume notes).\n\n"
        f"Emotion terms: {', '.join(seed.emotion_terms)}\n"
        f"Colors: {', '.join(color_terms or [])}\n"
        f"Current seed: {', '.join(seed.seed_notes)}\n\n"
        "Return a plain list, one note per line, no numbering."
    )
    try:
        text = chat([
            {"role":"system","content":"You are an experienced perfumer."},
            {"role":"user","content":prompt}
        ], temperature=0.5)
        # parse lines
        cand = [re.sub(r"^[\-\*\d\.\)\s]+","", ln).strip() for ln in (text or "").splitlines()]
        cand = [c for c in cand if c]
        # merge with current seed
        merged = []
        seen = set()
        for s in (seed.seed_notes + cand):
            k = normalize_text(s)
            if k and k not in seen:
                merged.append(s); seen.add(k)
        seed.seed_notes = merged[:12]
        # optionally tighten emotion_text a bit
        seed.emotion_text = " ".join(seed.emotion_terms)
    except Exception:
        pass
    return seed

def build_seed(intent_tone: List[str], color_terms: List[str], story_text: Optional[str] = None, **kwargs) -> SensorySeed:
    # Try your indexed story→notes path first (keeps prior behavior)
    try:
        if story_text:
            s2n = StoryToNotes.load()
            res = s2n.candidates_from_story(story_text, color_terms or [], top_k=30)
            seed = SensorySeed(
                seed_notes=res["seed_notes"],
                emotion_text=res["emotion_text"],
                emotion_terms=list(dict.fromkeys(res["emotion_text"].split()))
            )
            return _refine_with_gpt(seed, color_terms)
    except Exception:
        pass
    # Fallback heuristic + optional GPT refinement
    seed = _heuristic_seed(intent_tone or [], color_terms or [])
    return _refine_with_gpt(seed, color_terms or [])
