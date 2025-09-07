# data_ingest/build_collections.py
import os, json, pickle, ast, re
from pathlib import Path
import pandas as pd
import numpy as np

# Try to import your helper; fall back to a local implementation if absent
try:
    from agent.local_utils import flatten_json  # optional but nice to have
except Exception:
    def flatten_json(x):
        """Minimal flatten: turn nested dict/list into a single text string."""
        parts = []
        def walk(v, k=None):
            if isinstance(v, dict):
                for kk, vv in v.items():
                    walk(vv, kk)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    walk(vv, k)
            else:
                s = str(v).strip()
                if s:
                    parts.append(s)
        walk(x)
        return " ".join(parts)

# ----------------- paths & config -----------------
BASE = Path(__file__).resolve().parents[1]
INDICES = BASE / "indices"

DATA_DIR = Path(os.environ.get("LUNAR_DATA_DIR", "data"))
MATERIALS = DATA_DIR / "materials_catalog.jsonl"
FORMULAS  = DATA_DIR / "formulas.jsonl"
NOTES1    = DATA_DIR / "nlp_notes_data.csv"
FRA_PATH  = DATA_DIR / "fra_cleaned.csv"  # semicolon CSV

# FRA subset controls (environment overrides)
FRA_MAX_PERFUMES = int(os.environ.get("LUNAR_FRA_MAX_PERFUMES", "1000"))    # ingest first N perfumes
FRA_CHUNK_SIZE   = int(os.environ.get("LUNAR_FRA_CHUNK", "50000"))          # streaming chunk size

# TF-IDF knobs (environment overrides allowed)
TF_TOKEN_PATTERN = r"(?u)\b\w[\w\-]+\b"  # keep hyphenated tokens
TF_STRIP_ACCENTS = "unicode"

def tfidf(corpus, *, max_features=80000, min_df=1, max_df=0.9, ngram_range=(1,2)):
    """Build a bounded TF-IDF vectorizer + matrix (float32)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Try: pip install scikit-learn") from e
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        strip_accents=TF_STRIP_ACCENTS,
        token_pattern=TF_TOKEN_PATTERN,
        lowercase=True,
        dtype=np.float32,
    )
    X = vec.fit_transform(corpus)
    return vec, X

# ----------------- small utils -----------------
def read_jsonl(path: Path):
    """Lenient JSONL reader (tolerates trailing commas)."""
    items = []
    if not path.exists(): return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                try:
                    items.append(json.loads(line.rstrip(", ")))
                except Exception:
                    pass
    return items

def _clean_piece(s): return (str(s) if s is not None else "").strip()
def _split_list_field(val):
    if val is None: return []
    if isinstance(val, list): return [str(x).strip() for x in val if str(x).strip()]
    return [p.strip() for p in str(val).split(",") if p.strip()]

# ================= MATERIALS (robust) =================
MATERIALS_REQUIRED = ["id", "name", "family"]  # role/usage may be missing

ROLE_MAP = {
    # top-leaning
    "citrus": "top", "aldehydic": "top", "green": "top", "herbal": "top",
    "aromatic": "top", "ozonic": "top", "aquatic": "top",
    # mid-leaning
    "floral": "mid", "fruity": "mid", "spicy": "mid", "gourmand": "mid",
    "powdery": "mid", "soapy": "mid",
    # base-leaning
    "woody": "base", "amber": "base", "musk": "base", "leath": "base",
    "balsamic": "base", "incense": "base", "resin": "base", "vanill": "base", "smok": "base"
}

def _parse_float(x):
    try:
        if x is None: return None
        if isinstance(x, str):
            s = x.strip().replace("%", "")
            if s == "" or s.lower() in ("null","none","nan"): return None
            return float(s)
        return float(x)
    except Exception:
        return None

def _parse_hours(s):
    """'168hrs', '24h', '2 days' -> hours (float)."""
    if not s: return None
    s = str(s).strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(d|day|days|h|hr|hrs|hour|hours)?", s)
    if not m: return None
    val = float(m.group(1)); unit = (m.group(2) or "h").lower()
    if unit.startswith("d"): val *= 24.0
    return val

def _infer_role(family, descriptors):
    t = (" ".join([family or ""] + (descriptors or []))).lower()
    for key, role in ROLE_MAP.items():
        if key in t: return role
    return "mid"

def extract_material_record_strict(raw: dict):
    missing = [k for k in MATERIALS_REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"materials_catalog.jsonl missing required keys {missing}. Got: {list(raw.keys())}")

    mid    = _clean_piece(raw.get("id"))
    name   = _clean_piece(raw.get("name"))
    family = _clean_piece(raw.get("family"))
    role   = _clean_piece(raw.get("role") or "") or None

    desc_list = raw.get("descriptors") or []
    if not isinstance(desc_list, list):
        desc_list = _split_list_field(desc_list)
    descriptors = [str(x).strip() for x in desc_list if str(x).strip()]
    descriptors_str = ", ".join(descriptors)

    comp = raw.get("compliance") or {}
    allergen_keys = comp.get("allergen_keys") or []
    if not isinstance(allergen_keys, list):
        allergen_keys = _split_list_field(allergen_keys)

    usage_hint = raw.get("usage_hint") or {}
    tenacity_100_raw = _clean_piece(usage_hint.get("tenacity_for_100%", ""))
    tenacity_hours   = _parse_hours(tenacity_100_raw)
    rec_solution     = _parse_float(usage_hint.get("recommended_%solution"))

    usage_pct = raw.get("usage_hint_pct") or {}
    edp_min = _parse_float(usage_pct.get("edp_min"))
    edp_max = _parse_float(usage_pct.get("edp_max"))

    if role is None:
        role = _infer_role(family, descriptors)

    embedding_text = _clean_piece(raw.get("embedding_text", ""))

    text_blob = " ; ".join(filter(None, [
        f"{name} ({family}, {role})",
        f"descriptors={descriptors_str}" if descriptors_str else "",
        f"tenacity_100={tenacity_100_raw}" if tenacity_100_raw else "",
        f"recommended_solution%={rec_solution}" if rec_solution is not None else "",
        f"edp_range={edp_min}..{edp_max}" if (edp_min is not None or edp_max is not None) else "",
        embedding_text
    ]))

    return {
        "id": mid, "name": name, "family": family, "role": role,
        "descriptors": descriptors, "allergens": allergen_keys,
        "usage_max": edp_max, "edp_min": edp_min,
        "tenacity_for_100%": tenacity_100_raw or None, "tenacity_hours": tenacity_hours,
        "recommended_%solution": rec_solution,
        "text": text_blob
    }

# ================= NOTES (NLP CSV) =================
NLP_REQUIRED = ["brand", "name", "description", "notes"]

def load_notes_from_nlp_csv(path: Path):
    """
    Explode 'notes' list into rows: {note, family, description, text}.
    family remains "" (unknown) for NLP rows.
    """
    if not path.exists(): return []
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"Failed to read {path} with common encodings.")
    cols = list(df.columns)
    missing = [c for c in NLP_REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}. Found: {cols}")

    rows = []
    df = df.fillna("")
    for _, r in df.iterrows():
        brand = _clean_piece(r["brand"])
        name  = _clean_piece(r["name"])
        desc  = _clean_piece(r["description"])
        long_desc = f"{brand} {name}".strip()
        long_desc = f"{long_desc} — {desc}".strip(" —")

        notes_field = r["notes"]
        note_list = []
        if isinstance(notes_field, list):
            note_list = [str(x).strip() for x in notes_field if str(x).strip()]
        else:
            s = str(notes_field).strip()
            if s:
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        note_list = [str(x).strip() for x in parsed if str(x).strip()]
                    else:
                        note_list = _split_list_field(s)
                except Exception:
                    note_list = _split_list_field(s)

        for note_name in note_list:
            n = note_name.strip()
            if not n: continue
            text = " ; ".join([n, long_desc])
            rows.append({"note": n, "family": "", "description": long_desc, "text": text})
    return rows

# ================= FRA CSV (semicolon) — SUBSET ONLY =================
def load_notes_from_fra_subset(path: Path, max_perfumes: int = FRA_MAX_PERFUMES, chunksize: int = FRA_CHUNK_SIZE):
    """
    Stream FRA CSV (latin-1; sep=';') and STOP after 'max_perfumes' unique perfumes.
    Extract Top/Middle/Base → rows {note, family, description, text}.
    """
    if not path.exists(): return []
    import io, gc
    usecols = ["Perfume", "Brand", "Top", "Middle", "Base"]
    try:
        reader = pd.read_csv(
            path, sep=";", engine="python", encoding="latin-1",
            on_bad_lines="skip", usecols=usecols,
            chunksize=chunksize, iterator=True
        )
    except Exception:
        raw = path.read_bytes()
        txt = raw.decode("latin-1", errors="ignore")
        reader = pd.read_csv(
            io.StringIO(txt), sep=";", engine="python",
            on_bad_lines="skip", usecols=usecols,
            chunksize=chunksize, iterator=True
        )

    rows, perfumes = [], set()
    for chunk in reader:
        chunk = chunk.fillna("")
        for _, r in chunk.iterrows():
            perfume = _clean_piece(r["Perfume"])
            brand   = _clean_piece(r["Brand"])
            if not perfume and not brand: 
                continue
            base_desc = f"{brand} {perfume}".strip()
            perfumes.add(base_desc)

            for role, col in (("top","Top"), ("mid","Middle"), ("base","Base")):
                for n in _split_list_field(r[col]):
                    text = " ; ".join([n, f"family={role}", base_desc])
                    rows.append({"note": n, "family": role, "description": base_desc, "text": text})

            if len(perfumes) >= max_perfumes:
                break
        if len(perfumes) >= max_perfumes:
            break
        if len(rows) % (chunksize * 2) == 0:
            gc.collect()
    return rows

# ----------------- main build -----------------
def main():
    INDICES.mkdir(exist_ok=True, parents=True)

    # ---- Materials ----
    materials_raw = read_jsonl(MATERIALS)
    materials = [extract_material_record_strict(x) for x in materials_raw]
    materials_corpus = [x["text"] for x in materials] if materials else ["placeholder"]
    mat_vec, mat_X = tfidf(
        materials_corpus,
        max_features=int(os.environ.get("LUNAR_MAT_MAX_FEAT", "50000")),
        min_df=1, max_df=0.95, ngram_range=(1,2),
    )
    pickle.dump({"vectorizer": mat_vec, "X": mat_X, "rows": materials}, open(INDICES/"materials.pkl","wb"))

    # ---- NOTES: NLP + FRA SUBSET, collapse to (note, family) ----
    notes_nlp = load_notes_from_nlp_csv(NOTES1)
    notes_fra = load_notes_from_fra_subset(FRA_PATH, max_perfumes=FRA_MAX_PERFUMES, chunksize=FRA_CHUNK_SIZE)
    print(f"Loaded NLP notes: {len(notes_nlp)} | FRA subset ({FRA_PATH.name}): {len(notes_fra)}")

    from collections import OrderedDict
    collapsed = OrderedDict()
    def key_of(r): return (r["note"].strip().lower(), r["family"].strip().lower())
    for r in notes_nlp + notes_fra:
        k = key_of(r)
        if k not in collapsed:
            desc = (r.get("description","") or "")
            fam  = r.get("family","")
            text = " ; ".join([r["note"], f"family={fam}" if fam else "", desc]).strip(" ;")
            collapsed[k] = {"note": r["note"], "family": fam, "description": desc, "text": text}
    notes = list(collapsed.values())
    print("Notes after collapse:", len(notes))

    notes_corpus = [x["text"] for x in notes] if notes else ["placeholder"]
    note_vec, note_X = tfidf(
        notes_corpus,
        max_features=int(os.environ.get("LUNAR_NOTES_MAX_FEAT", "80000")),
        min_df=1, max_df=0.9, ngram_range=(1,2),
    )
    pickle.dump({"vectorizer": note_vec, "X": note_X, "rows": notes}, open(INDICES/"notes.pkl","wb"))
    print(f"✅ notes.pkl written | rows: {len(notes)} | vocab: {len(note_vec.vocabulary_)}")

    # ---- Formulas (text only) ----
    formulas_raw = read_jsonl(FORMULAS)
    formulas_rows = [{"raw": fr, "text": flatten_json(fr)} for fr in formulas_raw]
    formulas_corpus = [x["text"] for x in formulas_rows] if formulas_rows else [
        "top bergamot 2 aldehydes 0.3 mid rose 4 jasmine 2 base cedar 3 amber 2 musk 0.5"
    ]
    form_vec, form_X = tfidf(
        formulas_corpus,
        max_features=int(os.environ.get("LUNAR_FORM_MAX_FEAT", "30000")),
        min_df=1, max_df=0.95, ngram_range=(1,2),
    )
    pickle.dump({"vectorizer": form_vec, "X": form_X, "rows": formulas_rows}, open(INDICES/"formulas.pkl","wb"))

    # ---- Global vectorizer (for evaluator) ----
    global_corpus = materials_corpus + notes_corpus + formulas_corpus + [
        "calm airy sea breeze marine tranquil luminous confident polished green fresh clean warm woody leathery musk amber balsamic incense citrus floral spicy herbal aldehydic lactonic"
    ]
    glob_vec, _ = tfidf(
        global_corpus,
        max_features=int(os.environ.get("LUNAR_GLOB_MAX_FEAT", "120000")),
        min_df=1, max_df=0.95, ngram_range=(1,2),
    )
    pickle.dump({"vectorizer": glob_vec}, open(INDICES/"global.pkl","wb"))

    print("Indices built in", INDICES)

if __name__ == "__main__":
    main()
