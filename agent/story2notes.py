import pickle
from dataclasses import dataclass
from typing import List
from pathlib import Path
from agent.local_utils import normalize_text, map_hex_to_color_words, cosine

BASE = Path(__file__).resolve().parents[1]
INDICES = BASE / "indices"

# You can extend this list anytime without retraining.
MOOD_WHITELIST = {
    "calm","fresh","clean","warm","airy","marine","luminous","minimalist",
    "uplifting","confident","polished","soft","cool","dark","green","woody",
    "citrus","floral","spicy","herbal","aldehydic","amber","musk","leathery","balsamic","incense"
}

COLOR_HINTS = {
    "teal": ["marine","green","fresh"],
    "gold": ["amber","warm","luminous"],
    "black": ["leathery","smoky","dark"],
    "white": ["clean","airy","minimalist"],
    "green": ["green","herbal","fresh"],
    "blue": ["marine","cool","calm"],
    "sand": ["amber","soft"],
    "amber": ["amber","warm"],
}

@dataclass
class StoryToNotes:
    _vec: object
    _X: list
    _rows: list

    @classmethod
    def load(cls):
        with open(INDICES/"notes.pkl", "rb") as f:
            notes = pickle.load(f)
        return cls(_vec=notes["vectorizer"], _X=notes["X"], _rows=notes["rows"])

    @staticmethod
    def _extract_emotion_terms(text: str, colors: List[str]) -> List[str]:
        txt = normalize_text(text)
        # adjectives / moods + explicit whitelist
        terms = [w for w in txt.split() if w.endswith("y") or w.endswith("al") or w in MOOD_WHITELIST]
        # add color-derived hints
        for c in colors or []:
            if c.startswith("#"):
                terms += map_hex_to_color_words(c)
            else:
                terms += COLOR_HINTS.get(c.lower(), [])
        # keep whitelist + dedupe while preserving order
        out = []
        for w in terms:
            if w in MOOD_WHITELIST and w not in out:
                out.append(w)
        return out or ["modern","clean","balanced"]

    def build_emotion_text_and_vector(self, story: str, colors: List[str]):
        terms = self._extract_emotion_terms(story, colors)
        emotion_text = " ".join(terms)
        v = self._vec.transform([emotion_text])[0]
        return emotion_text, v

    def _top_k(self, qvec, k=30):
        sims = [(i, cosine(qvec, x)) for i, x in enumerate(self._X)]
        sims.sort(key=lambda z: z[1], reverse=True)
        out, seen = [], set()
        for i, s in sims[:max(1, k)]:
            r = self._rows[i]
            name = r.get("note") or r.get("name") or (r.get("text","")[:40])
            if not name:
                continue
            key = normalize_text(name)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "name": name,
                "family": r.get("family",""),
                "description": r.get("description",""),
                "score": float(s)
            })
        return out

    def candidates_from_story(self, story: str, colors: List[str], top_k=30):
        emotion_text, qvec = self.build_emotion_text_and_vector(story, colors)
        cands = self._top_k(qvec, k=top_k)
        seed_notes = [c["name"] for c in cands[:12]]
        return {
            "emotion_text": emotion_text,
            "emotion_vector": qvec,
            "seed_notes": seed_notes,
            "candidates": cands
        }
