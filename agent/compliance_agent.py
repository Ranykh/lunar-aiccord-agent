from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ComplianceResult:
    ok: bool
    warnings: List[str]
    fixes: List[Dict[str, Any]]

def check(formula):
    warnings = []
    fixes = []

    # total grams if available for % computation
    all_items = getattr(formula, "top", []) + getattr(formula, "mid", []) + getattr(formula, "base", [])
    total_g = sum(getattr(mu, "grams", 0.0) for mu in all_items) or 0.0

    for role_list in [formula.top, formula.mid, formula.base]:
        for mu in role_list:
            meta = mu.info if isinstance(mu.info, dict) else {}
            usage_max = meta.get("usage_max_pct")  # from catalog edp_max
            # derive percent from grams if needed
            pct = getattr(mu, "percent", None)
            if (pct is None or pct == 0) and total_g > 0:
                pct = 100.0 * float(getattr(mu, "grams", 0.0)) / total_g

            try:
                if usage_max is not None:
                    um = float(usage_max)
                    if pct is not None and float(pct) > um:
                        # propose clamp in percent (convert to grams)
                        tgt_pct = um
                        tgt_g = round(total_g * tgt_pct / 100.0, 2) if total_g > 0 else None
                        fixes.append({"material": mu.name, "from_pct": pct, "to_pct": tgt_pct,
                                      "from_g": getattr(mu,"grams",None), "to_g": tgt_g,
                                      "reason": "usage_max clamp"})
                if meta.get("descriptors"):
                    # Example soft warning; keep your earlier allergens logic if you have it elsewhere
                    pass
            except Exception:
                continue

    ok = len([fx for fx in fixes if fx.get("to_pct", fx.get("to_g")) is not None]) == 0
    return ComplianceResult(ok=ok, warnings=warnings, fixes=fixes)

def apply_fixes(formula, fixes):
    # apply in grams if present; otherwise percents
    all_items = formula.top + formula.mid + formula.base
    name_to_target_g = {fx["material"]: fx["to_g"] for fx in fixes if fx.get("to_g") is not None}
    name_to_target_pct = {fx["material"]: fx["to_pct"] for fx in fixes if fx.get("to_g") is None and fx.get("to_pct") is not None}

    if not name_to_target_g and not name_to_target_pct:
        return formula

    total_g = sum(getattr(mu,"grams",0.0) for mu in all_items) or 0.0

    for mu in all_items:
        if mu.name in name_to_target_g:
            mu.grams = float(name_to_target_g[mu.name])
        elif mu.name in name_to_target_pct and total_g > 0:
            mu.grams = round(total_g * float(name_to_target_pct[mu.name]) / 100.0, 2)
        # re-compute percent
        mu.percent = round(100.0 * mu.grams / (total_g if total_g > 0 else 100.0), 3)

    return formula
