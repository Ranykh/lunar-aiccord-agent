# agent/llm_client.py
import os
from typing import List, Union
from openai import AzureOpenAI
from tokens_count.token_meter import TokenMeter  # <-- add

_REQUIRED = ["AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_API_VERSION"]

def _client() -> AzureOpenAI:
    missing = [k for k in _REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}. Export AZURE_OPENAI_* first.")
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )

def chat(messages, temperature=0.7) -> str:
    client = _client()
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    out = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    # --- token accounting ---
    try:
        u = out.usage
        meter = TokenMeter()
        meter.add_chat(u.prompt_tokens or 0, u.completion_tokens or 0)
        meter.flush("chat")
    except Exception:
        pass
    return out.choices[0].message.content

def embed(texts: Union[str, List[str]]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]
    client = _client()
    deployment = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
    out = client.embeddings.create(model=deployment, input=texts)
    # --- token accounting ---
    try:
        meter = TokenMeter()
        meter.add_embedding(out.usage.prompt_tokens or 0)
        meter.flush("embed")
    except Exception:
        pass
    return [d.embedding for d in out.data]

def cosine(a: List[float], b: List[float]) -> float:
    import math
    da = math.sqrt(sum(x*x for x in a)) or 1.0
    db = math.sqrt(sum(x*x for x in b)) or 1.0
    return sum(x*y for x,y in zip(a,b)) / (da*db)
