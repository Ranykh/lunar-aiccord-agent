# agent/intent_parser.py
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from agent.local_utils import normalize_text

SEASONS  = ["spring", "summer", "autumn", "fall", "winter", "all-season"]
CONTEXTS = ["spa","gym","office","hotel","retail","boutique","menswear","womenswear",
            "wellness","lobby","app","lounge","event","store","pop-up","campaign"]

@dataclass
class Intent:
    season: Optional[str]
    context: Optional[str]
    target_audience: Optional[str]
    brand_tone: List[str]
    constraints: List[str]
    color_terms: List[str]
    raw: str

def parse_colors(raw: str, extra_colors: Optional[List[str]] = None) -> List[str]:
    extra_colors = extra_colors or []
    color_words: List[str] = []

    # hex colors like #A1B2C3
    color_words += re.findall(r"#(?:[0-9a-fA-F]{6})", raw)

    # common named colors
    named = re.findall(
        r"\b(black|white|gold|amber|teal|green|blue|red|violet|purple|pink|sand|beige|silver|navy|indigo)\b",
        raw, flags=re.IGNORECASE
    )
    color_words += [c.lower() for c in named]
    color_words += [str(c).lower() for c in extra_colors if c]

    # de-dupe (preserve order)
    seen = set()
    out: List[str] = []
    for c in color_words:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def parse_intent(brief: str, colors: Optional[List[str]] = None, **kwargs: Any) -> Intent:
    """
    Make 'colors' optional so callers can pass just the brief.
    **kwargs is accepted to stay forward-compatible with different call sites.
    """
    txt = normalize_text(brief)

    # season
    season = next((s for s in SEASONS if s in txt), None)
    if season == "autumn":
        season = "fall"

    # context
    context = next((c for c in CONTEXTS if c in txt), None)

    # target audience
    target: Optional[str] = None
    padded = f" {txt} "
    if "menswear" in txt or " men " in padded or " male " in padded:
        target = "men"
    elif "womenswear" in txt or " women " in padded or " female " in padded:
        target = "women"

    # brand tone: simple heuristic — adjectives & curated tokens
    tone_vocab = {"calm","fresh","clean","confident","polished","minimalist","uplifting",
                  "warm","airy","marine","sparkling","cozy","bold","luminous","soft",
                  "crisp","modern","classic","sensual"}
    words = re.findall(r"[a-zA-Z]+", txt)
    brand_tone = []
    for w in words:
        wl = w.lower()
        if wl in tone_vocab or wl.endswith(("y","al","ive")):
            brand_tone.append(wl)
    # keep order, de-dupe
    seen = set(); brand_tone = [t for t in brand_tone if not (t in seen or seen.add(t))]

    # constraints
    constraints: List[str] = []
    if "avoid animalic" in txt or "no animalic" in txt:
        constraints.append("no_animalic")
    if "no musk" in txt:
        constraints.append("no_musk")
    if "hypoallergenic" in txt:
        constraints.append("hypoallergenic")

    color_terms = parse_colors(txt, colors)

    return Intent(
        season=season,
        context=context,
        target_audience=target,
        brand_tone=brand_tone,
        constraints=constraints,
        color_terms=color_terms,
        raw=brief,
    )

# Optional helper if other parts of your app expect a dict:
def parse_intent_as_dict(brief: str, colors: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
    return asdict(parse_intent(brief, colors=colors, **kwargs))
