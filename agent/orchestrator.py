from pathlib import Path
from agent.intent_parser import parse_intent
from agent.sensory_moodboard import build_seed
from agent.note_rag import retrieve_candidates
from agent.scent_composer import compose
from agent.compliance_agent import check, apply_fixes
from agent.evaluator import evaluate
from agent.brand_styler import style

BASE = Path(__file__).resolve().parents[1]
OUTPUTS = (BASE / "examples" / "outputs")
OUTPUTS.mkdir(exist_ok=True, parents=True)

def generate(brief: str, colors=None):
    colors = colors or []
    intent = parse_intent(brief, colors)
    seed = build_seed(intent.brand_tone, intent.color_terms, story_text=brief)
    cands = retrieve_candidates(seed.seed_notes, seed.emotion_text, top_k=24)
    formula = compose(cands)
    comp = check(formula)
    if comp.fixes:
        formula = apply_fixes(formula, comp.fixes)
    evalr = evaluate(formula, seed.emotion_text)
    brand = style(intent, formula)
    res = {
        "intent": vars(intent),
        "seed": {"seed_notes": seed.seed_notes, "emotion_text": seed.emotion_text},
        "formula": {
            "top": [vars(x) for x in formula.top],
            "mid": [vars(x) for x in formula.mid],
            "base": [vars(x) for x in formula.base],
        },
        "compliance": {
            "ok": comp.ok,
            "warnings": comp.warnings,
            "fixes": comp.fixes,
        },
        "evaluation": {"score": evalr.score, "rationale": evalr.rationale},
        "branding": {"name": brand.name, "story": brand.story},
    }
    return res

def save_output(res: dict, name: str):
    import json
    out_json = OUTPUTS / f"{name}.json"
    out_md   = OUTPUTS / f"{name}.md"
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    def part(lst):
        return ", ".join([f"{x['name']} {x['percent']}%" for x in lst])
    md = (
    "# Lunar AIccord — {name}\n\n"
    "**Score:** {score} / 100  \n"
    "**Story:** {story}\n\n"
    "## Formula\n"
    "- **Top:** {top}\n"
    "- **Mid:** {mid}\n"
    "- **Base:** {base}\n\n"
    "## Compliance\n"
    "- OK: {ok}\n"
    "- Warnings: {warnings}\n"
    "- Fixes: {fixes}\n"
    ).format(
        name=res['branding']['name'],
        score=res['evaluation']['score'],
        story=res['branding']['story'],
        top=part(res['formula']['top']),
        mid=part(res['formula']['mid']),
        base=part(res['formula']['base']),
        ok=res['compliance']['ok'],
        warnings="; ".join(res['compliance']['warnings']) or "None",
        fixes=res['compliance']['fixes'] or "None",
    )

    out_md.write_text(md)
    return str(out_json), str(out_md)
