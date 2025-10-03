# data_ingest/dataset_builder.py
import os, json, uuid, random
from typing import Dict, Any, List, Optional
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from config import DATASET_OUT_DIR, DATASET_SHARD_SIZE
from run import build_app

def _ensure_dir(p): os.makedirs(p, exist_ok=True)
def _save_shard(rows: List[Dict[str, Any]], outdir: str, shard_idx: int):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, os.path.join(outdir, f"shard_{shard_idx:05d}.parquet"))

# ---------------------------- Brief synthesis ----------------------------
_TONES = [
    "minimalist","clean","fresh","airy","luminous","soft","warm","crisp","elegant",
    "modern","bold","comforting","refined","vibrant","serene","noir","sport","zen",
]
_FACETS = [
    "marine","ozonic","citrus","green","floral","woody","amber","musk","herbal",
    "spicy","fruity","gourmand","smoky","leathery","powdery","tea","incense","aldehydic",
]
_CONTEXTS = [
    "spa","lobby","boutique","office","gym","hotel room","retail floor","event hall",
    "home diffuser","laundry care","body mist","EDT","EDP","candle",
]
_SEASONS = ["spring","summer","autumn","winter","all-season"]
_ADDONS = [
    "long-lasting yet subtle","IFRA-safe for skin contact","gender-neutral signature",
    "day-to-night versatility","premium but approachable","elevated everyday",
    "sparkling opening, transparent heart, soft base",
]
_TEMPLATES = [
    "{tones} {facet} {context} for a {season} launch {colors}",
    "{tones} {facet} concept for {context} ({season}) {colors}",
    "brand vibe: {tones}; goal: {facet} impression in {context} ({season}) {colors}",
    "{season} story with {facet} facets for {context}; tone: {tones} {colors}",
    "{tones} {facet} profile for {context}; {addon} ({season}) {colors}",
]

_COLOR_NAMES = [
    "white","silver","charcoal","black","ivory","navy","sky blue","cobalt",
    "teal","mint","emerald","olive","chartreuse","lemon","gold","amber",
    "peach","coral","rose","magenta","fuchsia","lavender","violet","plum",
    "taupe","beige","sand","coffee","copper",
]

def _rand_hex(rng: random.Random) -> str:
    return f"#{rng.randrange(0, 0xFFFFFF):06X}"

def _color_block(rng: random.Random) -> str:
    if rng.random() < 0.30:
        return ""  # no color
    parts = []
    n_named = rng.choice([1,1,2,2,3])
    parts += rng.sample(_COLOR_NAMES, k=n_named)
    if rng.random() < 0.5:
        parts.append(_rand_hex(rng))
    return ", colors: " + ", ".join(parts)

def _choose_multi(rng: random.Random, pool: List[str], kmin=1, kmax=3, sep=" "):
    k = rng.randint(kmin, kmax)
    return sep.join(rng.sample(pool, k=k))

def _synth_brief(rng: random.Random) -> str:
    tones = _choose_multi(rng, _TONES, 1, 2, sep=" ")
    facet = rng.choice(_FACETS)
    context = rng.choice(_CONTEXTS)
    season  = rng.choice(_SEASONS)
    addon   = rng.choice(_ADDONS)
    colors  = _color_block(rng)
    tmpl    = rng.choice(_TEMPLATES)
    brief = tmpl.format(tones=tones, facet=facet, context=context, season=season, addon=addon, colors=colors).strip()
    return " ".join(brief.split())

def _load_seed_formulas(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                items.append(json.loads(line))
            except:
                pass
    return items

def _has_formula(blob):
    if not isinstance(blob, dict): return False
    f = blob.get("formula")
    if isinstance(f, dict) and all(k in f for k in ("top","mid","base")):
        return any(f.get(k) for k in ("top","mid","base"))
    return all(k in blob for k in ("top","mid","base")) and any(blob.get(k) for k in ("top","mid","base"))

# ---------------------- NEW: export to reference JSON ----------------------
def _rows_from_formula(formula: Dict[str, Any]) -> List[Dict[str, Any]]:
    def rows(role):
        arr = (formula or {}).get(role, []) or []
        out = []
        for i, r in enumerate(arr, 1):
            name = r.get("name") or r.get("material")
            if not name:
                continue
            out.append({
                "material": name,
                "grams": float(r.get("grams", r.get("percent", 0.0))),
                "role": role,
                "material_id": r.get("material_id") or name,
                "usage_max_pct": r.get("usage_max_pct"),
                "target_usage_mid_pct": r.get("target_usage_mid_pct"),
                "line_index": i,
                "supplier": None,
                "dilution_percent": None
            })
        return out
    return rows("top") + rows("mid") + rows("base")

def _export_reference_json(out: Dict[str, Any], export_dir: str) -> Optional[str]:
    """
    Normalize a run output to a single JSON document suitable for formula_priors.
    Returns the file path if written.
    """
    form = out.get("formula") or out.get("draft_formula") or {}
    rows = _rows_from_formula(form)
    if not rows:
        return None

    intent   = out.get("intent") or {}
    branding = out.get("branding") or {}
    brief    = out.get("brief") or out.get("brief_mod") or ""

    # id: prefer name if exists; make unique
    base_id = branding.get("name") or f"gen-{uuid.uuid4().hex[:8]}"
    slug = "".join(c.lower() if c.isalnum() else "-" for c in base_id).strip("-")
    doc_id = f"{slug}-{uuid.uuid4().hex[:6]}"

    doc = {
        "schema_version": "0.2",
        "id": doc_id,
        "title": branding.get("name", "Untitled"),
        "season": intent.get("season"),
        "mood": intent.get("brand_tone", []),
        "style": "unisex",
        "formula": rows,
        "meta": {
            "author": "Lunar AIccord",
            "year": datetime.utcnow().year,
            "license": "internal",
            "total_grams_reported": sum(x["grams"] for x in rows),
            "brief": brief
        }
    }

    _ensure_dir(export_dir)
    out_path = os.path.join(export_dir, f"{doc_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return out_path

# ------------------------------ Generator ------------------------------
def generate(source_path: str,
             out_dir: str = DATASET_OUT_DIR,
             n: int = 100000,
             mix: str = "0.5:brief,0.3:flanker,0.2:programmatic",
             online: bool = False,
             seed: int = 42,
             brief_file: Optional[str] = None,
             export_json: bool = False,
             export_dir: str = "data/reference_formulas"):
    rng = random.Random(seed)
    _ensure_dir(out_dir)
    if export_json:
        _ensure_dir(export_dir)
    app = build_app()

    # Optional user-provided brief pool
    user_briefs: List[str] = []
    if brief_file and os.path.exists(brief_file):
        with open(brief_file, "r", encoding="utf-8") as bf:
            user_briefs = [ln.strip() for ln in bf if ln.strip()]
        rng.shuffle(user_briefs)

    # Real seeds for flanker
    seeds = _load_seed_formulas(source_path)

    # Build the mode deck once
    modes: List[str] = []
    for part in mix.split(","):
        p, name = part.split(":")
        modes += [name] * int(float(p) * 100)

    seen_briefs = set()

    def next_brief() -> str:
        for _ in range(5):
            if user_briefs:
                base = rng.choice(user_briefs)
                maybe = []
                if rng.random() < 0.5: maybe.append(rng.choice(_FACETS))
                if rng.random() < 0.5: maybe.append(rng.choice(_SEASONS))
                cblk = _color_block(rng)
                text = base + (" " + " ".join(maybe) if maybe else "") + (cblk if cblk else "")
            else:
                text = _synth_brief(rng)

            if text not in seen_briefs:
                seen_briefs.add(text)
                return text

        fallback = _synth_brief(rng) + f" ::{rng.randrange(10**9)}"
        seen_briefs.add(fallback)
        return fallback

    rows, shard_idx = [], 0
    for i in range(n):
        mode = rng.choice(modes)
        state: Dict[str, Any] = {"online": online}

        if mode == "brief" or not seeds:
            state["brief"] = next_brief()
        else:
            base = rng.choice(seeds)
            base_formula = base.get("formula") if isinstance(base, dict) else None
            if base_formula is None and _has_formula(base):
                base_formula = base

            if _has_formula(base_formula):
                state["base_formula_obj"] = base_formula
                state["brief_mod"] = rng.choice([
                    "make it woodier","summer flanker","remove musk","increase freshness",
                    "lighter daytime version","intense night version","citrus boost","amber twist",
                ])
                state["brief"] = next_brief()
            else:
                mode = "brief"
                state["brief"] = next_brief()

        out = app.invoke(state)

        # Row assembly (Parquet)
        row = {
            "id": str(uuid.uuid4()),
            "mode": mode,
            "intent": out.get("intent"),
            "emotion_text": (out.get("seed") or {}).get("emotion_text"),
            "formula": out.get("formula") or out.get("draft_formula"),
            "constraints": {"notes": "", "ifra_ver": "UNK"},
            "compliance": out.get("compliance"),
            "evaluation": out.get("evaluation"),
            "branding": out.get("branding"),
            "meta": {"created_at": datetime.utcnow().isoformat() + "Z", "online": online}
        }
        rows.append(row)

        # NEW: write a normalized reference JSON for priors (one file per result)
        if export_json:
            try:
                _export_reference_json({**out, **state}, export_dir)
            except Exception:
                # keep dataset generation robust even if export for one row fails
                pass

        if len(rows) >= DATASET_SHARD_SIZE:
            _save_shard(rows, out_dir, shard_idx)
            rows, shard_idx = [], shard_idx + 1

    if rows:
        _save_shard(rows, out_dir, shard_idx)

# ------------------------------ CLI --------------------------------------
if __name__ == "__main__":
    import argparse
    def str2bool(v: str) -> bool:
        return str(v).lower() in ("1","true","yes","y","on")

    ap = argparse.ArgumentParser(description="Build Lunar AIccord dataset shards (Parquet) and optional reference JSONs.")
    ap.add_argument("--source", default="", help="Path to real formulas .jsonl (optional for flanker/programmatic).")
    ap.add_argument("--out", default=DATASET_OUT_DIR, help=f"Output dir (default from config: {DATASET_OUT_DIR})")
    ap.add_argument("--n", type=int, default=100000, help="Total records to generate.")
    ap.add_argument("--mix", default="0.5:brief,0.3:flanker,0.2:programmatic", help="e.g. '0.5:brief,0.3:flanker,0.2:programmatic'")
    ap.add_argument("--online", default="false", help="Use online services (True/False).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--brief_file", default="", help="Optional path to a file with one brief per line.")
    ap.add_argument("--export_json", default="false", help="Also write normalized JSON docs for priors (True/False).")
    ap.add_argument("--export_dir", default="data/reference_formulas", help="Directory for the normalized JSON docs.")
    args = ap.parse_args()

    generate(
        source_path=args.source,
        out_dir=args.out,
        n=args.n,
        mix=args.mix,
        online=str2bool(args.online),
        seed=args.seed,
        brief_file=args.brief_file or None,
        export_json=str2bool(args.export_json),
        export_dir=args.export_dir,
    )
    print(f"Done. Shards (if any) are in: {args.out}")
