# agent/llm_io.py
from typing import List, Dict, Any, Optional
import json
from config import API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_EMBED_DEPLOYMENT, AZURE_CHAT_DEPLOYMENT
from tokens_count.token_meter import TokenMeter
from openai import AzureOpenAI

_client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

def embed_texts(texts: List[str], meter: TokenMeter) -> List[List[float]]:
    if not texts:
        return []
    resp = _client.embeddings.create(
        input=texts,
        model=AZURE_EMBED_DEPLOYMENT,
    )
    try:
        meter.add_embedding(resp.usage.prompt_tokens)
    except Exception:
        pass
    return [d.embedding for d in resp.data]

def embed_text(text: str) -> List[float]:
    meter = TokenMeter()
    vecs = embed_texts([text], meter)
    return vecs[0] if vecs else []

# ---------- NEW: chat + chat_json ----------
def chat(messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: int = 800) -> str:
    """Thin wrapper around Azure Chat Completions."""
    resp = _client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

def chat_json(system: str, user_json: Dict[str, Any], response_schema: Optional[Dict[str, str]] = None,
              temperature: float = 0.2, max_tokens: int = 800) -> Dict[str, Any]:
    """
    Ask the model for strict JSON. We use response_format for coercion and still
    parse defensively.
    """
    user_text = json.dumps(user_json, ensure_ascii=False)
    resp = _client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=[
            {"role":"system","content": system + " Return ONLY valid JSON."},
            {"role":"user","content": user_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content or "{}"
    try:
        return json.loads(txt)
    except Exception:
        # final fallback: extract {...}
        import re
        m = re.search(r"\{.*\}", txt, flags=re.S)
        return json.loads(m.group(0)) if m else {}
