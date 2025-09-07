# lunar-aiccord-agent
Lunar AIccord is a production-ready, LangGraph-based multi-agent system that turns creative briefs (text + color palettes) for your bussiness into complete, IFRA-aware perfume/fragrance formulas, a brandable name, and a short narrative. It integrates RAG over Qdrant, Azure OpenAI (chat + embeddings), and lightweight local fallbacks, with built-in token accounting and a reproducible pipeline.

- a complete, IFRA-aware perfume formula,

- a brandable product name and short story,

- an emotion-fit score (evaluation).



## Features (what the agents do)

- Intent Parsing – extract season/context/tone/constraints from free-text.

- Sensory/Moodboard – convert mood + colors into seed notes & emotion text (LLM + prompt engineering).

- RAG Retrieval – query notes/materials from a vector DB (Qdrant).

- Scent Composition – build top/mid/base using structured allocation .

- Compliance – validate and (optionally) clamp materials to IFRA limits.

- Evaluation – score emotion–formula alignment (TF-IDF and/or embeddings).

- Branding – generate name + short story (LLM - openai).


### Quickstart

### 1) Clone & create a virtual env

```bash

git clone <your-repo-url> LunarAIccord
cd LunarAIccord

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
# py -m venv .venv
# .\.venv\Scripts\Activate.ps1

```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) (Online mode) Ingest notes into Qdrant (once)
```bash
python rag_ingest_qdrant.py
```
This embeds your notes corpus and upserts into Qdrant (token usage counted).


## Running the Agent
The runner (run.py) defaults to online mode (Azure + Qdrant).
Here’s the exact command that will show the intro in the terminal (interactive mode) and save the result to your examples folder:
```bash
python -u run.py --interactive \
  --save examples/outputs/online_sample_01.json
```
What you’ll see next:

```bash
Hi! I’m Lunar AIccord — an autonomous multi-agent that turns your creative brief
(text + brand colors) into a complete, IFRA-aware fragrance formula, a name, and a story.
Tell me your brand and vibe (e.g., “minimalist wellness, spring launch, #CFE8FF, marine fresh linen”),
and I’ll do the rest.

Your brief:  <type your brief here>
Optional colors (comma-separated, hex or names):  <type colors or press Enter>
```

After you answer those two prompts, the pipeline will run and save the JSON to the path you provided.

### Example runs (inputs → outputs)

Each example is a full 100 g EDP formula produced by the LangGraph multi-agent pipeline. Results include: mapped raw materials (with grams), IFRA/usage-limit compliance, an emotion-fit score, and branding (name + story).

| # | Brief (input)                                                                                       | Product name       | Compliance                      | Emotion-fit score | File path                                       |
| - | --------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------- | ----------------- | ----------------------------------------------- |
| 1 | Fresh, clean fragrance for a wellness app launch (minimalist, green, uplifting).                    | **Verdant Aura**   | ✅ OK                            | **51.4**          | `examples/outputs/fresh_wellness_launch.json`   |
| 2 | Night spa; sweet, dark fragrance; black palette.                                                    | **Obsidian Veil**  | ✅ OK                            | **50.3**          | `examples/outputs/dark_spa.json`                |
| 3 | Winter scent for a luxury menswear boutique (confident, polished; warm woods + a touch of leather). | **Velour Éclipse** | ✅ OK                            | **52.5**          | `examples/outputs/winter_luxury_men_scent.json` |

### What output shows : 

- Note→Material mapping with grams (not just “notes”) and built-in usage hints/limits. Example: citrus & neroli are mapped to concrete materials with target/maximum usage; the JSON includes usage_hint_pct and usage_max_pct for transparency.
- Automatic compliance checks + proposed fixes when any material exceeds a max usage guideline; fixes include both percent and gram targets so you can apply them deterministically.
- Distinct branding (name + story) tailored to the brief, e.g., “Verdant Aura”, “Obsidian Veil”, and “Velour Éclipse.” with a story like : ""In the sanctuary of stillness, where shadows and serenity intertwine, Obsidian Veil envelops the senses. A deep, smoky blend of leather and agarwood unfurls, grounding you in a moment of luxurious introspection.""
### Reproduce any example locally
```bash
# Run interactively (shows the intro, then prompts):
python run.py --interactive --save examples/outputs/online_sample_03.json

# Or run from a text brief:
python run.py --file examples/inputs/01_brief.txt --save examples/outputs/01_result.json

# Optional: stream step-by-step node progress
python run.py --file examples/inputs/01_brief.txt --save examples/outputs/01_result.json --no-stream false
```

### flowchart LR
  Start((Start)) --> intent
  intent --> sensory
  sensory --> retrieval
  retrieval --> compose
  compose --> compliance
  compliance --> evaluate
  evaluate --> branding
  branding --> End((End))

  
