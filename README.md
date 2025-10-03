# lunar-aiccord-agent
Lunar AIccord is a production-ready, Fully autonomous, retrieval-augmented agent built with LangGraph (LangChain) and the OpenAI API.
Given a short user brief (plus optional brand colors), the system orchestrates multiple tools—LLM chat/embeddings, vector databases (Qdrant), and local indices—to retrieve domain knowledge, synthesize candidates, and produce a validated fragrance formula with evaluation and branding copy. The project emphasizes prompt engineering, clean modular code, and effective API/database usage to demonstrate practical generative-AI patterns (RAG, tool orchestration, schema-constrained generation, and fault-tolerant fallbacks).


## Features 

### **Autonomous Orchestration (LangGraph/LangChain)**
A state graph (run.py) coordinates independent agents (intent parsing, moodboard, retrieval, composition, compliance, evaluation, branding). The graph supports conditional routing (new-brief vs. transform), deterministic ordering, and resilient fallbacks.

### **Dual-Layer RAG**
- **Note-RAG** retrieves conceptual notes aligned to the brief using Azure embeddings and Qdrant (with a local TF-IDF fallback).
- **Material-RAG** maps conceptual notes to real catalog materials (with usage caps and metadata) via vector search and MMR diversity.

### **Schema-Constrained LLM Calls**
The strict proportioner (agent/proportioner_llm.py) uses chat_json(...) to force structured JSON output (top/mid/base with grams/percent). Hard constraints are enforced in the prompt: no invented materials, respect usage caps, role split ± tolerance, total = 100g. If the LLM fails, a deterministic MILP (PuLP) fallback guarantees completion.

### **Reference-Aware Generation (priors)**
The proportioner optionally loads reference formulas (real & synthetic) from data/reference_formulas/ and **formula_priors.pkl` to bias allocations toward realistic shapes—without ever introducing unseen materials.

- The terminal snippet for creating the reference_formula and the syntithic dataset:
```bash
(base) ranykhirbawi@Ranys-MacBook-Pro LunarAIccord & python data_ingest/build_formula_dataset.py \
-jsonl data/base_formulas-json \
--out_dir data/reference_formulas \
-indices_dir indices \
-synth_per 0 \
--seed 123
Wrote indices/formula_priors.pkl with 51 materials.
Reference formulas in data/reference_formulas → 10 files (real: 10).
```
```bash
(base) ranykhirbawi@Ranys-MacBook-Pro LunarAIccord & python -m data_ingest.dataset_builder \
--source data/base_formulas-jsonl \
-—n 200 \
—-mix "0.6: brief,0.3: flanker,0.1:programmatic" \
--online true \
--export_json true \
--export_dir
```
### **Effective API & DB Usage**
- **OpenAI**: chat completions (strict JSON) and embeddings.
- **Qdrant:** vector search over materials & notes; compatibility guards and offline TF-IDF indices for robustness.
- **Clean adapters** convert between tool outputs and agent inputs to keep code composable and testable.

### **Prompt Engineering Built-In**
Carefully scoped system prompts isolate responsibilities (intent extraction, moodboard, proportioning) and inject hard constraints, few-shot priors, and strict schemas to reduce hallucination and improve controllability.

### **Compliance & Evaluation Layers**
Post-composition checks ensure IFRA-style constraints (with optional auto-fix hooks). A lightweight evaluator scores fit-to-brief, and a branding agent generates name and copy for end-to-end usability.

### **Dataset & Index Tooling**
Utilities under data_ingest/ generate large synthetic datasets (Parquet/JSON), aggregate formula priors, and build local indices—useful for offline runs, analysis, and iterative improvement.

### **Observability & Safety Nets**
Step-wise telemetry, debug messages, and clearly defined fallbacks keep the pipeline stable under API/network variance while maintaining deterministic behavior when needed.



## How the agent runs - Autonomous Flow
1) **Your single input** → you type a natural-language brief + optional colors (e.g., “vegas vibes… (yellow)”).

2) **Intent parser** (rule/LLM, depending on your setup) extracts season, context, and tone + color terms.

3) **Sensory/Moodboard** distills those into emotion_text and seed notes (e.g., citrus, fresh, sporty).

4) **Note-RAG** retrieves conceptual notes relevant to the brief.

5) **Material-RAG** maps notes → real, safe materials from your Qdrant collection (or local TF-IDF index).

6) **Proportioning (LLM-assisted, strict)** allocates exact grams across only the retrieved materials, honoring usage caps and role targets (top/mid/base). Here the LLM is constrained and augmented with your reference formulas (both real and synthetic).

7) **Compliance** checks and (optionally) auto-fixes.

8) **Evaluator** scores fit-to-brief.

9) **Branding** names the scent + writes a short story using LLM.

The whole thing runs end-to-end after your single brief; you don’t micromanage any step.



## Repo Layout (high-level)
```bash
agent/
  intent_parser.py
  sensory_moodboard.py
  note_rag.py
  material_rag.py
  scent_composer.py
  proportioner_llm.py
  compliance_agent.py
  evaluator.py
  brand_styler.py
  llm_io.py
data_ingest/
  dataset_builder.py
  build_materials_index.py
  update_formula_priors.py
  rag_ingest_materials.py
  rag_ingest_qdrant.py
indices/
  notes.pkl
  materials.pkl
data/
  base_formulas.jsonl
  reference_formulas/  (generated JSON formulas)
  formula_priors.pkl   (aggregated priors)
examples/
  outputs/
    20251003_001918_late-night-date-in-cozy-resturant.json
    20251003_003740_vegas-vibes-in-the-summer-inside-a-gym-n.json
    20251003_004627_relaxing-spa-fragrance-for-a-luxury-hote.json
    20251003_010212_feminine-cherry-fragrance-for-women-clot.json
run.py
config.py
requirements.txt

```

## Quickstart

### 1) Clone & create a virtual env

```bash

git clone https://github.com/Ranykh/lunar-aiccord-agent LunarAIccord
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

### 3) Index / DB prep
### Option A — Qdrant (online RAG):
```bash
# Ingest your materials catalog to Qdrant
python -m agent.rag_ingest_materials --path data/materials_catalog.jsonl
# -> "Ingested N materials into lunar_materials_v1"
```
This embeds your notes corpus and upserts into Qdrant (token usage counted).

### Option B — Local indices (offline RAG fallback):
```bash
# Build local TF-IDF indices (notes/materials)
python -m data_ingest.build_materials_index
python -m data_ingest.build_collections   # if you have a script to build notes.pkl

```
Ensure you have:
- indices/materials.pkl
- indices/notes.pkl



## Running the Agent
Interactive (recommended to see the autonomous flow)
Here’s the exact command that will show the intro in the terminal (interactive mode) and save the result to your examples folder:
```bash
python run.py --interactive --online
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
**Sample I/O**
```bash
Your brief: vegas vibes in the summer inside a gym near the beach
Optional colors (comma-separated, hex or names): yellow

>>> STREAM
[check_mode] -> ...
[intent]     -> ...
[sensory]    -> ...
[retrieval]  -> ...
[compose]    -> ...
[compliance] -> ...
[evaluate]   -> ...
[branding]   -> ...

=== SUMMARY ===
Name   : Citrus Elevé
Season : summer | Context: gym
Score  : 50.7 | Blended TF-IDF + Azure embedding cosine alignment.
Saved result to examples/outputs/...
```
The saved JSON includes:
- retrieved candidates (notes),
- the material-level formula (with grams/percent),
- compliance, evaluation, branding,
- your brief/colors.

### saved JSON file looks like this :
```bash
{
  "candidates": {
    "candidates": [
      {
        "note": "birch tar",
        "family": "base",
        "score": 0.2636602222919464,
        "source": "heretic-parfums poltergeist"
      },
      {
        "note": "black pepper",
        "family": "mid",
        "score": 0.20404505729675293,
        "source": "ricardo-ramos-perfumes-de-autor al-misk"
      },
      {
        "note": "black pepper",
        "family": "top",
        "score": 0.19694413244724274,
        "source": "o-boticario men-brahma"
      },
      {
        "note": "dark patchouli",
        "family": "base",
        "score": 0.18007630109786987,
        "source": "ds-durga deep-dark-vanilla"
      },
      ......... (24 Candidate Notes)

  },
  "formula": {
    "top": [
      {
        "name": "pepper essential oil, black",
        "grams": 10.0,
        "percent": 10.0,
        "info": {
          "descriptors": [
            "warm",
            "piquant",
            "exotic",
            "dry",
            "pepper",
            "tar",
            "gin"
          ],
          "aliases": []
        }
      },
      {
        "name": "cocoa essence",
        "grams": 10.0,
        "percent": 10.0,
        "info": {
          "descriptors": [
            "edible",
            "dessert-like",
            "chocolate",
            "caramel",
            "bitter",
            "sharp cocoa note",
            "subtly lactonic",
            "bitter chocolate",
            "rum"
          ],
          "aliases": []
        }
      },
      {
        "name": "labdanum absolute, clear",
        "grams": 10.0,
        "percent": 10.0,
        "info": {
          "descriptors": [
            "sweet",
            "resinous",
            "warm",
            "vanillic",
            "oriental"
          ],
          "aliases": []
        }
      }
    ],
    "mid": [
      {
        "name": "russian leather",
        "grams": 15.0,
        "percent": 15.0,
        "info": {
          "descriptors": [
            "musky",
            "civet",
            "leathery",
            "dirty",
            "leather",
            "musk",
            "aldehydic",
            "musky leather",
            "citrus at top",
            "definitely aldehydic",
            "spicy"
          ],
        ..............

 "compliance": {
    "ok": true,
    "warnings": [],
    "fixes": []
  },
  "evaluation": {
    "score": 49.3,
    "rationale": "Blended TF-IDF + Azure embedding cosine alignment."
  },
  "branding": {
    "name": "Nocturne Veil",
    "story": "In the stillness of twilight, Nocturne Veil unfolds like a whispered secret. Smoky Russian leather and Haitian vetiver embrace the warmth of cocoa and patchouli, while a trace of pepper lingers, igniting intrigue. Envelop yourself in its dark, velvety cocoon—mysterious, comforting, unforgettable."
  },
  "brief": "late night date in cozy resturant",
  "colors": [
    "black"
  ],
  "mode": "brief"
```

## Optional! - Transform/Flanker Mode
Start from an existing formula JSON:
```bash
python run.py \
  --base examples/seed_formula.json \
  --brief_mod "summer flanker with more citrus and less musk" \
  --colors "yellow, white" \
  --online
```

## How RAG + LLM work together

- **Note-RAG** retrieves conceptual notes aligned to the brief.
- **Material-RAG** converts those notes into real materials using Qdrant (or local TF-IDF).
- **LLM Proportioner (strict)** receives only those materials (with caps + roles) plus a few reference_formulas to learn “shapes” of real formulas; it must output exact 100 g allocations within per-role targets.
- If the LLM fails or is disabled, MILP does the allocation deterministically.

**Strict proportioner system prompt**
```bash
    You are an expert perfumery chemist.
    TASK: Allocate GRAMS/PERCENT across the provided materials ONLY, to total exactly 100.0 grams.
    HARD CONSTRAINTS:
    - Use ONLY materials provided in `materials` list. Do not invent or rename.
    - Each material's percent/grams must be <= its `usage_max_pct`.
    - Total grams = 100.0 exactly.
    - Per-role totals must be within role_split ± tolerance.
    OUTPUT FORMAT: strict JSON with keys: top, mid, base, total_grams; each item has material_id, name, grams, percent.
```



### Example run (inputs → outputs)
```bash
(base) ranykhirbawi@Ranys-MacBook-Pro LunarAIccord % python run.py --interactive --online
Hi! I’m Lunar AIccord — an autonomous multi-agent that turns your creative brief
(text + brand colors) into a complete, IFRA-aware fragrance formula, a name, and a story.
Tell me your brand and vibe (e.g., “minimalist wellness, spring launch, #CFE8FF, marine fresh linen”),
and I’ll do the rest.
```
```bash
Your brief: vegas vibes in the summer inside a gym near the beach
Optional colors (comma-separated, hex or names): yellow
```
```bash
=== SUMMARY ===
Name   : Citrus Elevé
Season : summer | Context: gym
Tone   : 
Score  : 50.7 | Blended TF-IDF + Azure embedding cosine alignment.
Story  : Bask in the energy of summer with Citrus Elevé, a vibrant blend of zesty lemon and pink grapefruit that invigorates your senses. A cooling minty heart and soft musky base uplift your post-workout glow, leaving you refreshed and radiant all  ...

Saved result to examples/outputs/20251003_003740_vegas-vibes-in-the-summer-inside-a-gym-n.json
```

In  “vegas vibes in the summer inside a gym near the beach (yellow)” run, the system acted autonomously end-to-end after single, natural-language input. First, brief and color were parsed into an intent (season/context/tone + color terms). That intent fed a sensory/moodboard step that distilled emotion terms and seed notes. Using those, the note-RAG retrieved relevant note ideas; then the material-RAG mapped those notes to real, safe materials from the Qdrant-backed catalog. With that candidate pool, the LLM proportioner (constrained by role targets and usage caps, and informed by reference formulas) allocated exact grams to reach a 100 g formula—no invented names, just selections from the retrieved materials. Next, a compliance agent verified IFRA-style constraints and the evaluator scored fit to the brief. Finally, a branding agent named the scent and wrote the story. You didn’t have to orchestrate any of this: once you gave the brief, the agents coordinated the retrieval, reasoning, proportioning, validation, and storytelling on their own, streaming progress and saving the finished JSON output - ready for creating the fragrance.

## Configuration

**Adjust these in config.py:**

- ROLE_SPLIT (default e.g., {"top": 30, "mid": 40, "base": 30})

- ROLE_TOLERANCE (± % per role)

- TOP_K_MATERIALS for Material-RAG

- QDRANT_COLLECTION_MATERIALS

- Dataset constants (DATASET_OUT_DIR, DATASET_SHARD_SIZE, …)

## License
This project is provided for research & prototyping purposes. Please review supplier SDS/IFRA restrictions before producing anything for real-world use.

## Acknowledgements
Thanks to the open-source fragrance-making community and dataset contributors. Real formulas used as references are credited in your data/base_formulas.jsonl, /data/reference_formulas/, /datamaterials_catalog.jsonl sources.
ATTRIBUTION - SHARE-ALIKE, provided by CREATIVE COMMONS
CC BY-SA License
(This license lets others remix, tweak, and build upon your work even for commercial purposes, as long as they credit you and license their new creations under the identical terms. This license is often compared to "copyleft" free and open source software licenses. All new works based on yours will carry the same license, so any derivatives will also allow commercial use. This is the license used by Wikipedia, and is recommended for materials that would benefit from incorporating content from Wikipedia and similarly licensed projects.)
LEGAL CODE: https://creativecommons.org/licenses/by-sa/4.0/legalcode
LICENSE DEED: httos://creativecommons.ora/licenses/bv.sa/4.0/

https://docs.google.com/spreadsheets/d/1U4XKFcUypBs0ruJybO9oXKNgIs6_fUDATj4U7gjuZr4/edit?gid=0#gid=0
