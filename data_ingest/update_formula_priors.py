# tools/update_formula_priors.py
import json, pickle, math
from pathlib import Path
from collections import defaultdict, Counter

REF_DIR = Path("/Users/ranykhirbawi/Desktop/LunarAIccord/data/reference_formulas")
OUT = Path("/Users/ranykhirbawi/Desktop/LunarAIccord/indices/formula_priors.pkl")
OUT.parent.mkdir(parents=True, exist_ok=True)

materials = defaultdict(lambda: {
    "name": None,
    "n": 0,
    "sum_pct": 0.0,
    "sum_pct_sq": 0.0,
    "role_counts": Counter(),
    "max_cap_pct": 100.0,   # keep if present in rows
})
pairs = Counter()
role_totals = Counter()     # total % per role across formulas (used to refine ROLE_SPLIT)
n_formulas = 0

# helper to convert grams→percent assuming runs are scaled to 100g (your pipeline does)
def as_pct(g): return float(g)

for p in sorted(REF_DIR.glob("*.json")):
    doc = json.loads(p.read_text(encoding="utf-8"))
    rows = doc.get("formula", [])
    if not rows: 
        continue
    n_formulas += 1

    # per-formula materials and roles for co-occurrence
    mat_ids = []
    role_share = Counter()

    for r in rows:
        name = r.get("material") or r.get("name")
        if not name: 
            continue
        mid = str(r.get("material_id") or name)
        role = (r.get("role") or "").lower() or "mid"
        grams = float(r.get("grams", 0.0))
        pct = as_pct(grams)

        m = materials[mid]
        m["name"] = name
        m["n"] += 1
        m["sum_pct"] += pct
        m["sum_pct_sq"] += pct * pct
        m["role_counts"][role] += 1
        if r.get("usage_max_pct") is not None:
            try:
                m["max_cap_pct"] = min(m["max_cap_pct"], float(r["usage_max_pct"]))
            except Exception:
                pass

        mat_ids.append(mid)
        role_share[role] += pct

    # update co-occurrence pairs (unordered)
    uniq = sorted(set(mat_ids))
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            pairs[(uniq[i], uniq[j])] += 1

    # accumulate role totals
    for role, pct in role_share.items():
        role_totals[role] += pct

# finalize stats
priors = {"materials": {}, "pairs": {}, "role_defaults": {}}
for mid, v in materials.items():
    n = max(1, v["n"])
    mean = v["sum_pct"] / n
    var = max(0.0, (v["sum_pct_sq"] / n) - mean*mean)
    priors["materials"][mid] = {
        "name": v["name"],
        "priors_mean_pct": mean,
        "priors_std_pct": math.sqrt(var),
        "role_counts": dict(v["role_counts"]),
        "max_cap_pct": v["max_cap_pct"],
    }

# normalize role defaults to ~100
role_sum = sum(role_totals.values()) or 1.0
priors["role_defaults"] = {r: (v/role_sum)*100.0 for r, v in role_totals.items()}
priors["pairs"] = {f"{a}||{b}": c for (a,b), c in pairs.items()}
priors["n_formulas"] = n_formulas

OUT.write_text("")  # ensure file exists even if pickle fails early
with OUT.open("wb") as f:
    pickle.dump(priors, f)
print(f"Built priors from {n_formulas} formulas -> {OUT}")
