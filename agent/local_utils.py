import re, math
from collections import Counter

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9#]+", " ", s)
    return " ".join(s.split())

def tokenize(s: str):
    s = normalize_text(s)
    return [t for t in s.split() if t]

class TinyTfidf:
    def __init__(self):
        self.vocab = {}
        self.idf = []
        self.fitted = False

    def fit(self, corpus):
        docs = [tokenize(x) for x in corpus]
        vocab = {}
        for doc in docs:
            for t in set(doc):
                if t not in vocab:
                    vocab[t] = len(vocab)
        N = len(docs)
        df = [0]*len(vocab)
        for doc in docs:
            seen = set(doc)
            for t in seen:
                df[vocab[t]] += 1
        idf = [math.log((N+1)/(dfi+1)) + 1.0 for dfi in df]
        self.vocab = vocab
        self.idf = idf
        self.fitted = True
        return self

    def transform(self, corpus):
        if not self.fitted:
            raise RuntimeError("TinyTfidf not fitted")
        X = []
        for text in corpus:
            tokens = tokenize(text)
            counts = Counter(tokens)
            vec = [0.0]*len(self.vocab)
            for t, c in counts.items():
                j = self.vocab.get(t)
                if j is not None:
                    vec[j] = (c) * self.idf[j]
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            vec = [v/norm for v in vec]
            X.append(vec)
        return X

def cosine(u, v):
    return sum(a*b for a,b in zip(u,v))

def flatten_json(obj, prefix=""):
    parts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            parts.append(f"{k}: {flatten_json(v, k)}")
    elif isinstance(obj, list):
        parts.append(", ".join([flatten_json(x, prefix) for x in obj]))
    else:
        parts.append(str(obj))
    return " ".join(str(p) for p in parts if p is not None)

def map_hex_to_color_words(hex_code: str):
    try:
        if not hex_code.startswith("#") or len(hex_code) != 7:
            return []
        r = int(hex_code[1:3], 16)/255.0
        g = int(hex_code[3:5], 16)/255.0
        b = int(hex_code[5:7], 16)/255.0
        mx, mn = max(r,g,b), min(r,g,b)
        d = mx - mn
        if d == 0:
            h = 0
        elif mx == r:
            h = (60 * ((g - b)/d) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r)/d) + 120) % 360
        else:
            h = (60 * ((r - g)/d) + 240) % 360
        if 170 <= h <= 250:
            return ["blue", "marine"]
        if 80 <= h <= 160:
            return ["green"]
        if 20 <= h <= 50:
            return ["gold", "amber"]
        if h < 20 or h > 300:
            return ["red", "warm"]
        if 250 < h <= 300:
            return ["violet"]
        return []
    except Exception:
        return []

FAMILY_TO_ROLE = {
    "citrus": "top", "green": "top", "aldehydic": "top", "marine": "top",
    "lactonic": "mid", "floral": "mid", "spicy": "mid", "herbal": "mid",
    "woody": "base", "amber": "base", "musk": "base", "leathery": "base",
    "oud": "base", "balsamic": "base", "incense": "base",
}

def pick_role_from_family(family: str):
    if not family:
        return "mid"
    fam = normalize_text(family).replace("/", " ").split()[0]
    return FAMILY_TO_ROLE.get(fam, "mid")
