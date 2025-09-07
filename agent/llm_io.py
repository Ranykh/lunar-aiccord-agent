# llm_io.py
from typing import List
from agent.config import API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_EMBED_DEPLOYMENT
from tokens_count.token_meter import TokenMeter

from openai import AzureOpenAI

_client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

def embed_texts(texts: List[str], meter: TokenMeter) -> List[List[float]]:
    """
    Returns a list of embedding vectors for the given texts.
    Increments the token meter for prompt tokens used.
    """
    if not texts:
        return []
    resp = _client.embeddings.create(
        input=texts,
        model=AZURE_EMBED_DEPLOYMENT,  # deployment name
    )
    # OpenAI embeddings response includes usage.prompt_tokens
    try:
        ptoks = resp.usage.prompt_tokens
        meter.add_embedding(ptoks)
    except Exception:
        pass

    return [d.embedding for d in resp.data]
