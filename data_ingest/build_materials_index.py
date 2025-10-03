# tools/build_materials_index.py
import json, pickle, argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def _text_of(row):
    aliases = row.get("aliases") or []
    if isinstance(aliases, list):
        aliases = ", ".join(map(str, aliases))
    return f"{row.get('name','')} | aliases: {aliases} | family: {row.get('family','')} | desc: {row.get('descriptors','')}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="materials_catalog.json or .jsonl")
    ap.add_argument("--out", default="indices/materials.pkl")
    args = ap.parse_args()

    p = Path(args.catalog)
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        rows = json.loads(p.read_text(encoding="utf-8"))

    texts = [_text_of(r) for r in rows]
    vec = TfidfVectorizer(min_df=1, max_df=0.95)
    X = vec.fit_transform(texts)
    out = {"vectorizer": vec, "X": X, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"Wrote {args.out} with {len(rows)} rows")
