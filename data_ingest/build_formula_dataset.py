#!/usr/bin/env python3
# tools/build_formula_dataset.py
import argparse, json, os, glob, pickle, random, math
from pathlib import Path
import statistics as stats
from typing import Dict, Any, List

def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in (s or "")).strip("-") or "formula"

def _role_or_default(x: str) -> str:
    r = (x or "").strip().lower()
    if r in ("top","mid","base"): return r
    return "mid"  # default if unknown

def _normalize_one_jsonl_record(J: Dict[str, Any]) -> Dict[str, Any]:
    """Map your JSONL schema -> our reference JSON format."""
    rows = J.get("formula", []) or []
    if not rows:
        return {"top":[], "mid":[], "base":[], "total_grams":0.0, "tags":[], "notes":[]}

    total = sum(float(r.get("grams", 0.0) or 0.0) for r in rows) or 1.0
    top, mid, base = [], [], []

    for r in rows:
        name = (r.get("material") or r.get("material_original") or "").strip()
        grams = float(r.get("grams", 0.0) or 0.0)
        pct = (grams / total) * 100.0
        role = _role_or_default(r.get("role"))
        item = {
            "material_id": _slug(name),   # best-effort stable id
            "name": name,
            "grams": round(pct, 6),       # store % as grams; our system assumes 100g total
            "percent": round(pct, 6),
        }
        if role == "top":   top.append(item)
        elif role == "base": base.append(item)
        else:                mid.append(item)

    out = {"top": top, "mid": mid, "base": base, "total_grams": 100.0}

    # Light metadata → helpful for Option B (LLM strict references)
    tags = []
    if J.get("season"): tags.append(str(J["season"]).lower())
    if J.get("style"):  tags.append(str(J["style"]).lower())
    title = (J.get("title") or "").strip()
    if title: tags.append(_slug(title))
    out["tags"] = sorted(list(set(tags)))

    # Optional: add a naive "notes" field from title/id keywords (can be extended)
    notes = []
    if "blue" in title.lower(): notes += ["marine","ozonic","mineral"]
    out["notes"] = sorted(list(set(notes)))
    return out

def _write_reference_json(out_dir: Path, rec_id: str, data: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{_slug(rec_id)}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p

def _jitter_percentages(items: List[Dict[str,Any]], jitter=0.18) -> List[Dict[str,Any]]:
    """Return a new list with small random perturbation to each 'percent' (±jitter),
    then re-normalized to keep the role sum intact."""
    if not items: return items[:]
    vals = [max(0.0, it["percent"] * (1.0 + random.uniform(-jitter, jitter))) for it in items]
    s = sum(vals) or 1.0
    scaled = [v * (sum(it["percent"] for it in items)/s) for v in vals]
    out = []
    for it, v in zip(items, scaled):
        x = dict(it); x["percent"] = x["grams"] = round(v, 6)
        out.append(x)
    return out

def _synthesize_from_base(baseJ: Dict[str,Any], n_variants=5, seed=42, jitter=0.18):
    random.seed(seed)
    outs = []
    for i in range(n_variants):
        J = {"top":[], "mid":[], "base":[], "total_grams": 100.0}
        for role in ("top","mid","base"):
            J[role] = _jitter_percentages(baseJ.get(role, []), jitter=jitter)
        # renormalize grand total to 100 and preserve role share
        total = sum(x["percent"] for r in ("top","mid","base") for x in J[r]) or 1.0
        scale = 100.0 / total
        for r in ("top","mid","base"):
            for x in J[r]:
                x["percent"] = x["grams"] = round(x["percent"] * scale, 6)
        J["total_grams"] = 100.0
        # carry tags/notes through
        J["tags"] = list(baseJ.get("tags", []))
        J["notes"] = list(baseJ.get("notes", []))
        outs.append(J)
    return outs

def _collect_files(dir_path: Path) -> List[str]:
    return sorted(glob.glob(str(dir_path / "*.json")))

def _build_priors(files: List[str]) -> Dict[str, Any]:
    by_mat_role = {}  # (key, role) -> [pct,...]
    by_mat = {}       # key -> [pct,...]
    def app(d,k,v):
        if k not in d: d[k] = []
        d[k].append(v)

    tag_index, note_index = {}, {}

    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                J = json.load(f)
        except Exception:
            continue
        tags = [str(t).lower() for t in (J.get("tags") or [])]
        notes= [str(n).lower() for n in (J.get("notes") or [])]
        for t in tags:
            tag_index.setdefault(t, set()).add(os.path.basename(p))
        for n in notes:
            note_index.setdefault(n, set()).add(os.path.basename(p))

        for role in ("top","mid","base"):
            for r in J.get(role, []):
                key = str(r.get("material_id") or r.get("name") or "").strip().lower()
                pct = float(r.get("percent", r.get("grams", 0.0)) or 0.0)
                app(by_mat, key, pct)
                app(by_mat_role, (key, role), pct)

    priors: Dict[str, Dict[str, float]] = {}
    for k, arr in by_mat.items():
        if not arr: continue
        priors.setdefault(k, {})["prior_pct_mean"] = float(stats.mean(arr))
        # P95-ish cap; fall back to max if small sample
        priors[k]["prior_pct_p95"] = float(stats.quantiles(arr, n=20)[-1]) if len(arr) >= 20 else float(max(arr))
    for (k, role), arr in by_mat_role.items():
        if not arr: continue
        priors.setdefault(k, {})[f"prior_{role}_mean"] = float(stats.mean(arr))
        priors[k][f"prior_{role}_p95"] = float(max(arr))

    return {
        "priors": priors,
        "tag_index": {k: sorted(list(v)) for k, v in tag_index.items()},
        "note_index": {k: sorted(list(v)) for k, v in note_index.items()},
    }

def main():
    ap = argparse.ArgumentParser(description="Build reference formula dataset + priors (and optional synthetic variants).")
    ap.add_argument("--jsonl", required=True, help="Path to your base JSONL with real formulas.")
    ap.add_argument("--out_dir", default="data/reference_formulas", help="Where normalized JSON files will be written.")
    ap.add_argument("--indices_dir", default="indices", help="Where to write formula_priors.pkl")
    ap.add_argument("--synth_per", type=int, default=5, help="Synthetic variants to create per real formula (0 to disable).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jitter", type=float, default=0.18, help="±jitter for proportion perturbation (0.0-0.5 reasonable).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    idx_dir = Path(args.indices_dir); idx_dir.mkdir(parents=True, exist_ok=True)

    # 1) Normalize JSONL → JSON files
    real_files = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                J = json.loads(line)
            except Exception:
                continue
            rec_id = J.get("id") or J.get("title") or "formula"
            norm = _normalize_one_jsonl_record(J)
            p = _write_reference_json(out_dir, rec_id, norm)
            real_files.append(str(p))


    # 3) Build priors across ALL files in out_dir
    all_files = _collect_files(out_dir)
    blob = _build_priors(all_files)

    # 4) Write formula_priors.pkl
    out_pkl = idx_dir / "formula_priors.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(blob, f)
    print(f"Wrote {out_pkl} with {len(blob['priors'])} materials.")
    print(f"Reference formulas in {out_dir} → {len(all_files)} files (real: {len(real_files)}).")

if __name__ == "__main__":
    main()
