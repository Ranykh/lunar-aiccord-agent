# agent/chain.py
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda

from agent.intent_parser import parse_intent
from agent.sensory_moodboard import build_seed
from agent.note_rag import retrieve_candidates
from agent.scent_composer import compose
from agent.compliance_agent import check as compliance_check
from agent.evaluator import evaluate as eval_formula
from agent.brand_styler import style as brand_style

def _intent_step(inp: Dict[str, Any]) -> Dict[str, Any]:
    brief = inp["brief"]
    colors = inp.get("colors") or []
    intent = parse_intent(brief, colors=colors)
    return {**inp, "intent": intent}

def _sensory_step(inp: Dict[str, Any]) -> Dict[str, Any]:
    it = inp["intent"]
    seed = build_seed(intent_tone=it.get("brand_tone", []), color_terms=it.get("color_terms", []), story_text=inp["brief"])
    return {**inp, "seed": seed}

def _retrieve_step(inp: Dict[str, Any]) -> Dict[str, Any]:
    s = inp["seed"]
    ret = retrieve_candidates(seed_terms=s.get("seed_notes", []), emotion_text=s.get("emotion_text",""), top_k=24)
    return {**inp, "candidates": ret}

def _compose_step(inp: Dict[str, Any]) -> Dict[str, Any]:
    byfam = inp["candidates"]["by_family"]
    # Build a single iterable of candidates for composer
    ordered = byfam.get("top", []) + byfam.get("mid", []) + byfam.get("base", [])
    formula = compose(ordered)
    return {**inp, "draft_formula": formula}

def _compliance_step(inp: Dict[str, Any]) -> Dict[str, Any]:
    comp = compliance_check(inp["draft_formula"])
    # normalize (support both raw or wrapped)
    if isinstance(comp, dict) and "formula" in comp:
        formula = comp["formula"]
    else:
        formula = comp
    return {**inp, "compliance": comp, "formula": formula}

def _evaluate_step(inp: Dict[str, Any]) -> Dict[str, Any]:
    ev = eval_formula(inp["formula"], inp["seed"]["emotion_text"])
    return {**inp, "evaluation": ev}

def _brand_step(inp: Dict[str, Any]) -> Dict[str, Any]:
    br = brand_style(inp["intent"], inp["formula"])
    return {**inp, "branding": br}

# LCEL chain
lunar_chain = (
    RunnableLambda(_intent_step)
    | RunnableLambda(_sensory_step)
    | RunnableLambda(_retrieve_step)
    | RunnableLambda(_compose_step)
    | RunnableLambda(_compliance_step)
    | RunnableLambda(_evaluate_step)
    | RunnableLambda(_brand_step)
)
