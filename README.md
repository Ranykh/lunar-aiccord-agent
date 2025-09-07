# lunar-aiccord-agent
Lunar AIccord is a production-ready, LangGraph-based multi-agent system that turns creative briefs (text + color palettes) into complete, IFRA-aware perfume formulas, a brandable name, and a short narrative. It integrates RAG over Qdrant, optional Azure OpenAI (chat + embeddings), and lightweight local fallbacks, with built-in token accounting and a reproducible pipeline.

flowchart LR
  %% Direction & groups
  classDef ext fill:#f5f7ff,stroke:#6b7cff,color:#1b1f3b
  classDef agent fill:#f9fafb,stroke:#bbb,color:#111
  classDef data fill:#fffbe6,stroke:#f0c36d,color:#6b4e00

  subgraph U[User]
    B[(Brief: string)]:::data -->|input| A
    C[(Colors: [string])]:::data -->|input| A
  end

  subgraph P[Agent Pipeline (LangGraph)]
    direction LR
    A[Intent Parser\n(agent.intent_parser.parse_intent)\n— Extracts season/context/tone/colors]:::agent
    S[Sensory Moodboard\n(agent.sensory_moodboard.build_seed)\n— Seed notes + emotion_text]:::agent
    R[Note RAG\n(agent.note_rag.retrieve_candidates)\n— Qdrant → note candidates]:::agent
    K[Scent Composer\n(agent.scent_composer.compose)\n— Materials/grams/role split]:::agent
    L[Compliance Agent\n(agent.compliance_agent.check/apply_fixes)\n— IFRA caps → adjusted]:::agent
    E[Evaluator\n(agent.evaluator.evaluate)\n— Emotion fit score]:::agent
    Y[Brand Styler\n(agent.brand_styler.style)\n— Name + story]:::agent
  end

  subgraph X[External / Data Sources]
    Q[(Qdrant\ncollection: lunar_notes_v1)]:::ext
    O[(Azure OpenAI\nchat: team6-gpt4o\nembed: team6-embedding)]:::ext
    I[(Local indices\nindices/*.pkl)]:::ext
  end

  %% Flow
  A -->|intent{season, context, brand_tone, color_terms}| S
  S -->|seed{seed_notes[], emotion_text}| R
  R -->|candidates{by_family}| K
  K -->|draft_formula (dataclass)| L
  L -->|formula (final)| E
  E -->|score,rationale| Y

  %% I/O edges
  R --- Q
  S --- O
  E --- O
  R -.fallback.-> I

  %% Final outputs
  Y --> OUT[(Output JSON)\n{formula, name, story,\ncompliance, score}]:::data

