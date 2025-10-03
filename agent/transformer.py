# agent/transformer.py
from typing import Dict, Any
from agent.llm_io import chat_json
from agent.local_utils import ensure_formula_object

def transform_formula(state: Dict[str, Any]) -> Dict[str, Any]:
    base = state.get("base_formula_obj") or state.get("base_formula")  # support either key
    brief = state.get("brief_mod") or state.get("modification_prompt") or state.get("brief", "")
    intent = state.get("intent", {})
    emotion_text = state.get("emotion_text", "")

    user = {
        "base_formula": base,
        "intent": intent,
        "emotion_text": emotion_text,
        "modification": brief,
        "constraints": "Respect usage_max_pct for each material; totals must sum to 100g."
    }
    schema = {"top":"list","mid":"list","base":"list","total_grams":"number"}
    out = chat_json(
        system=("You are a perfumery chemist. Modify the given base formula to satisfy the modification brief "
                "while respecting safety caps and summing to 100g. Return structured JSON only."),
        user_json=user,
        response_schema=schema
    )
    state["draft_formula"] = ensure_formula_object(out)
    return state
