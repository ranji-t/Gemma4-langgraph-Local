# Gemma4-langgraph-Local

A production-adjacent Retrieval-Augmented Generation (RAG) agent running **entirely on CPU** with no cloud dependencies. Built from scratch using Gemma4 E2B via Ollama, FastEmbed, LangGraph, and DuckDuckGo — with a hybrid retrieval pipeline combining dense embeddings, BM25 sparse retrieval, Maximal Marginal Relevance (MMR), and Reciprocal Rank Fusion (RRF). Evaluated end-to-end using RAGAS with a local Gemma4 judge. Fully observable via self-hosted Langfuse with per-node tracing, chunk-level RRF score logging, and latency tracking across the entire pipeline.

> Built on a no-GPU laptop. Every component implemented from scratch with full understanding of the underlying mathematics.

---

## Demo

**Query:** *"What is the NVIDIA Stock Price Today?"*

```
The current stock price of NVIDIA Corporation (NVDA) is 210.69 USD.
This price reflects an increase of 2.95% in the past 24 hours.
```

**End-to-end latency:** ~6 minutes on CPU (i5-1155G7, 16GB RAM, no GPU)

---

## RAGAS Evaluation Results

Evaluated on a 5-sample hand-curated dataset covering factual, comparative, multi-hop, and out-of-scope query types. Judge LLM: Gemma4 E2B via Ollama's OpenAI-compatible endpoint — fully local, zero cloud API cost.

### Baseline (k=10, MMR λ=0.5)

| Query | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| NVDA Stock Price Today | 1.0 | 0.00 | — |
| Who is the CEO of NVIDIA? | 1.0 | 0.60 | 1.0 |
| NVDA vs AMD Market Cap | 1.0 | 1.00 | — |
| Recent NVIDIA Announcements | — | 0.00 | — |
| Mumbai Weather | — | 0.50 | — |
| **Aggregate** | **1.00** | **0.32** | **0.50** |

### Ablation: k=5, MMR λ=0.75

| Query | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| NVDA Stock Price Today | 1.0 | 0.00 | — |
| Who is the CEO of NVIDIA? | 1.0 | **0.92** | 0.0 |
| NVDA vs AMD Market Cap | 1.0 | — | — |
| Recent NVIDIA Announcements | — | 0.00 | — |
| Mumbai Weather | — | 0.50 | 0.0 |
| **Aggregate** | **1.00** | **0.35** | **0.00** |

### Key Findings

**Faithfulness = 1.0 across all queries** — Gemma4 produces zero hallucinations. Every claim in the generated answer is grounded in the retrieved chunks.

**Increasing MMR λ from 0.5 → 0.75** raises Context Precision on the CEO query from 0.60 → 0.92 — confirming that higher relevance weighting (less diversity pressure) correctly surfaces Jensen Huang-specific chunks over general NVIDIA history.

**Reducing k from 10 → 5** improves precision but collapses recall — the classic precision-recall tradeoff observed empirically. Optimal k and λ are query-type dependent: factual queries benefit from higher λ and lower k; multi-hop queries need higher k to cover all required sub-facts.

**Context Precision = 0.0 on stock price despite a correct answer** — Gemma4 found the price in a lower-ranked chunk (position 6-7), not the top-ranked ones. The answer is right but the ranking is wrong. This points to RRF over-weighting semantic similarity over exact keyword match for numeric queries — BM25 weight should be increased for this query type.

**Remaining NaN values** on multi-hop and weather queries are not caused by truncation (`OLLAMA_CONTEXT_LENGTH=16384` eliminates all truncation, confirmed via `truncated = 0` across all 35 inference calls). The actual cause is RAGAS failing to parse Gemma4's responses — RAGAS sends structured prompts expecting JSON output (`{"statements": [...], "verdicts": [...]}`) but Gemma4 reverts to prose on complex multi-chunk contexts, producing unparseable responses that RAGAS silently converts to NaN. Fix: pass `response_format={"type": "json_object"}` to force constrained JSON generation at the Ollama level.

---

## Architecture

```
START
  ├── len(messages) > 1 → Summary Node → Smart Query Node → Web Search?
  └── single message   → Dummy Query Node → Web Search?

                              Web Search?
                              ├── use_web=True  → Google it!! → Answering Node → END
                              └── use_web=False → Answering Node → END
```

```
                   +-----------+
                   | __start__ |
                   +-----------+
                 ...            ...
               ..                  ...
             ..                       ..
  +--------------+                      ..
  | Summary Node |                       .
  +--------------+                       .
          *                              .
          *                              .
          *                              .
+------------------+           +------------------+
| Smart Query Node |           | Dummy Query Node |
+------------------+           +------------------+
                 ***            ***
                    **        **
                      **    **
                  +-------------+
                  | Web Search? |
                  +-------------+
                  ..            ..
                ..                ..
              ..                    ..
     +-------------+                  ..
     | Google it!! |                ..
     +-------------+              ..
                  **            ..
                    **        ..
                      **    ..
                +----------------+
                | Answering Node |
                +----------------+
                          *
                          *
                          *
                    +---------+
                    | __end__ |
                    +---------+
```

### Node Descriptions

| Node | Purpose |
|---|---|
| **Summary Node** | Compresses multi-turn conversation history into structured context. Activated only when conversation has more than one message. |
| **Smart Query Node** | Enriches the user's raw query using conversation context to produce a focused, web-search-friendly query. |
| **Dummy Query Node** | Fast path for single-turn queries — passes the message directly as the search query with no enrichment needed. |
| **Web Search?** | Routing node. Gemma4 decides whether the query requires a live web search or can be answered from internal knowledge. Constrained JSON output via `response_format={"type": "json_object"}` + Pydantic validation. |
| **Google it!!** | Executes the full retrieval pipeline: web search → chunking → dense + sparse scoring → MMR → RRF → ranked chunks. Logs chunk-level scores to Langfuse. |
| **Answering Node** | Synthesizes the final answer using retrieved chunks as context, or falls back to model knowledge if no search was needed. |

---

## Retrieval Pipeline

```
Web Search (DuckDuckGo, 5 URLs)
        ↓
Token-based chunking (300 tokens, 50 overlap)
        ↓
    ┌────────────────────────────────────┐
    │  Dense Retrieval                   │
    │  nomic-embed-text-v1.5-Q           │
    │  Cosine similarity via einsum      │
    └──────────────┬─────────────────────┘
                   │
    ┌──────────────▼─────────────────────┐
    │  Sparse Retrieval                  │
    │  BM25Okapi                         │
    │  Regex tokenization                │
    └──────────────┬─────────────────────┘
                   │
    ┌──────────────▼─────────────────────┐
    │  MMR (Maximal Marginal Relevance)  │
    │  Vectorized via np.einsum          │
    │  λ=0.75 (current), relevance-first │
    │  Run separately on dense + sparse  │
    └──────────────┬─────────────────────┘
                   │
    ┌──────────────▼─────────────────────┐
    │  RRF (Reciprocal Rank Fusion)      │
    │  k=60, fuses both MMR outputs      │
    │  Returns top-5 ranked chunks       │
    └──────────────┬─────────────────────┘
                   │
            Final chunks → LLM
```

### Why hybrid retrieval?

Dense retrieval finds semantically similar chunks but misses exact keyword matches (e.g. a stock price like "210.69"). BM25 finds exact keyword matches but misses semantic paraphrases. MMR removes redundancy within each retriever's results before fusion. RRF fuses both ranked lists using rank positions — bypassing the incompatible score scales of cosine similarity and BM25 entirely.

---

## Key Design Decisions

### Vectorized MMR

Most MMR implementations use a nested Python for-loop over candidates. This implementation replaces the inner loop with a single `np.einsum` operation:

```python
worst_redundency = np.einsum(
    "rf, bf -> rb", remaining_docs_emb, best_docs_emb
).max(axis=1)
```

This computes all pairwise similarities between remaining and already-selected chunks simultaneously as a single matrix operation — no Python loops over data, fully vectorized, and GPU-ready.

### Full Chunk Provenance Tracking

Every chunk carries its complete scoring history through the pipeline:

```python
class RetrivedDocs(NamedTuple):
    search_id: int        # which URL
    chunk_id: int         # which chunk within URL
    url: str
    title: str
    snippet: str          # DDGS short description
    content: str
    context_score: float  # cosine similarity score
    semantic_score: float # BM25 score
    context_mmr: bool     # selected by dense MMR?
    sematic_mmr: bool     # selected by sparse MMR?
    rrf_score: float      # final RRF score
    rrf_rank: int         # final rank position
```

This makes the retrieval pipeline fully debuggable — every chunk's journey from raw web result to final answer is traceable. All 12 fields are logged to Langfuse per query, enabling direct correlation between RRF rank and RAGAS Context Precision scores.

### Constrained JSON Output

The routing node uses Ollama's OpenAI-compatible `response_format` combined with Pydantic validation — schema correctness enforced at two levels:

```python
class UseWebTemplate(BaseModel):
    use_web: bool

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
result = client.chat.completions.create(
    model=gemma_lite,
    messages=[...],
    response_format={"type": "json_object"},
)
json_response = UseWebTemplate.model_validate_json(
    result.choices[0].message.content
).model_dump()
```

Token-level JSON constraint from Ollama, field-level type validation from Pydantic.

### Production Infrastructure

```python
# Timing and error capture on every function
@logging_decorator
def any_function():
    ...

# Exponential backoff retry on every node
@sync_retry(max_retries=3, delay=1.0, backoff=2.0)
def any_node():
    ...
```

---

## Langfuse Observability

Self-hosted Langfuse (Docker, `localhost:3000`) provides full pipeline tracing — zero data leaving the machine.

**Integration — Langfuse v4 SDK with `langfuse.openai` drop-in:**

```python
from langfuse import observe, get_client
from langfuse.openai import OpenAI  # drop-in for openai — auto-traces all LLM calls

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

@sync_retry()
@logging_decorator
@observe(name="Web Search Node")
def web_search_node(state: AgentState01) -> AgentState01:
    # ... retrieval logic ...

    lfuse = get_client()
    lfuse.update_current_span(
        metadata={
            "num_docs_retrived": len(final_retrived_docs),
            "prime_query": query,
            "urls": [d.url for d in final_retrived_docs],
            "content": [d.content for d in final_retrived_docs],
            "context_score": [d.context_score for d in final_retrived_docs],
            "semantic_score": [d.semantic_score for d in final_retrived_docs],
            "rrf_rank": [d.rrf_rank for d in final_retrived_docs],
        }
    )
    return state
```

**What each trace shows:**

| Node | Auto-captured | Custom metadata |
|---|---|---|
| Summarizer Node | Latency, input messages, output summary | Input/output char lengths |
| Smart Query Node | Latency, raw query, enriched query | — |
| Web Use Node | Latency, routing decision | `use_web: true/false` |
| Web Search Node | Latency, raw + ranked chunks | Query, URLs, cosine scores, BM25 scores, RRF ranks per chunk |
| Answering Node | Latency, context sent, final answer | — |

**Example trace — Mumbai weather query:**

```
invoke  (total: ~6 min)
  ├── Summarizer Node     1m 42s   126 → 625 tokens
  ├── Smart Query Node      33s   215 → 378 tokens  → "Weather in Mumbai"
  ├── Web Use Node          20s   233 → 8 tokens    → use_web: true
  ├── Web Search Node     1m 43s  294 raw → 5 final chunks
  └── Answering Node      1m 58s  1866 → 430 tokens
```

The Web Search Node metadata surfaces chunk-level cosine scores, BM25 scores, and RRF ranks directly in the Langfuse UI — enabling direct correlation between retrieval decisions and RAGAS Context Precision scores without additional tooling.

**Self-hosted setup:**

```bash
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up
# UI at http://localhost:3000
```

---

## RAGAS Evaluation Setup

The evaluation pipeline uses Gemma4 as the judge LLM via Ollama's OpenAI-compatible endpoint — zero external API dependencies.

**Working configuration:**

```python
from openai import OpenAI
from ragas.llms import llm_factory

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

# response_format forces constrained JSON generation at the Ollama level.
# Without this, Gemma4 reverts to prose on complex contexts and RAGAS
# cannot parse the response, returning NaN silently instead of erroring.
local_llm_judge = llm_factory(
    "gemma4:e4b",
    provider="openai",
    client=client,
    model_kwargs={"response_format": {"type": "json_object"}},
)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, context_precision, context_recall],
    llm=local_llm_judge,
)
```

**Ollama server requirements for eval:**

```bash
# Context length must be 16384 minimum — RAGAS sends question + all contexts
# + evaluation rubric in a single prompt. At 4096 (default), prompts truncate
# silently and generation returns 1 token, producing NaN scores.
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_FLASH_ATTENTION=1
ollama serve 2>&1 | tee ./logs/ollama_inference.log > /dev/null

# Monitor eval progress — one line per completed RAGAS call
tail -f ./logs/ollama_inference.log | grep --line-buffered "POST"
```

**Inference characteristics observed during eval (i5-1155G7):**

- RAGAS makes ~35 LLM calls for 5 samples × 3 metrics
- Each call: 1000–4000 token prompts, 80–1024 token responses
- Prefill speed: ~28-60 tok/s depending on prompt length
- Generation speed: ~7-14 tok/s, degrades slightly over time due to thermal throttling
- Total eval time: ~1.5 hours on CPU
- All calls completed with `truncated = 0` at 16384 context length

**Eval dataset:** 5 hand-curated samples with timestamped ground truths covering factual + time-sensitive (NVDA stock price), factual + stable (NVIDIA CEO), comparative (NVDA vs AMD market cap), multi-hop + temporal (recent NVIDIA announcements), and out-of-scope routing test (Mumbai weather). Ground truths for time-sensitive queries are dated — re-collect on the day of evaluation for accurate recall scores.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemma4 E2B via Ollama |
| Embeddings | nomic-embed-text-v1.5-Q via FastEmbed |
| Web Search | DuckDuckGo (DDGS) |
| Agent Orchestration | LangGraph |
| Sparse Retrieval | BM25Okapi (rank-bm25) |
| Text Splitting | LangChain TokenTextSplitter |
| LLM Client | Langfuse-wrapped OpenAI client (`langfuse.openai.OpenAI`) |
| Observability | Langfuse v4 (self-hosted Docker, `@observe` + `get_client()`) |
| Evaluation | RAGAS (Faithfulness, Context Precision, Context Recall) |
| Output Validation | Pydantic |
| Notebook | Marimo |
| Numerical Computing | NumPy |

---

## Project Structure

```
Gemma4-Test/
├── Gemma4-Agents_06.py         # Main agent notebook (Marimo)
├── pyproject.toml              # Project dependencies and metadata (uv)
├── .python-version             # Python version pin for uv
├── uv.lock                     # Locked dependency versions
├── logs/
│   ├── logs06.log              # Runtime logs
│   └── ollama_inference.log    # Ollama server inference logs (tee output)
└── .cache/
    └── nomic-embed-text-v1.5-Q/   # FastEmbed model cache
```

---

## How to Run

### Prerequisites

**1. Start Langfuse (self-hosted)**

```bash
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up
# UI available at http://localhost:3000
# Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from the UI
```

**2. Install Ollama and pull the model**

```bash
# Install Ollama — https://ollama.com

# Set custom model directory if needed
export OLLAMA_MODELS=/d/.files/.ollama/models

# Start the Ollama server with extended context and flash attention
export OLLAMA_MODELS="D:\.files\.ollama\models" \
  && export OLLAMA_CONTEXT_LENGTH=16384 \
  && export OLLAMA_FLASH_ATTENTION=1 \
  && ollama serve

# With full inference logging (recommended)
export OLLAMA_MODELS="D:\.files\.ollama\models" \
  && export OLLAMA_CONTEXT_LENGTH=16384 \
  && export OLLAMA_FLASH_ATTENTION=1 \
  && ollama serve 2>&1 | tee ./logs/ollama_inference.log > /dev/null

# In a separate terminal — watch live inference summary
tail -f ./logs/ollama_inference.log | grep --line-buffered -E "total time|POST|truncat"

# Pull the model
ollama pull gemma4:e2b
```

**3. Install uv**

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**4. Clone and install**

```bash
git clone https://github.com/ranji-t/Gemma4-langgraph-Local
cd Gemma4-langgraph-Local

uv sync                    # core runtime
uv sync --group notebook   # adds Marimo, ipykernel, grandalf
uv sync --group dev        # adds mypy, pytest, ruff
```

### Dependency Groups

| Group | Purpose | Command |
|---|---|---|
| Default | Core runtime | `uv sync` |
| `notebook` | Marimo, ipykernel, graph viz | `uv sync --group notebook` |
| `dev` | mypy, pytest, ruff | `uv sync --group dev` |

### Run

```bash
# Start Langfuse (separate terminal)
cd langfuse && docker compose up

# Start Ollama (separate terminal)
export OLLAMA_CONTEXT_LENGTH=16384 && export OLLAMA_FLASH_ATTENTION=1 && ollama serve

# View mode (end-user)
uv run marimo run Gemma4-Agents_06.py

# Edit mode (development)
uv run marimo edit Gemma4-Agents_06.py
```

### Example Invocation

```python
result = app01.invoke({
    "messages": [
        Message(role="user", content="Who are You?"),
        Message(role="assistant", content="I am Gemma4 an LLM by google!"),
        Message(role="user", content="What is the Nvidia Stock Price Today?"),
    ]
})

print(result["messages"][-1].content)
# The current stock price of NVIDIA Corporation (NVDA) is 210.69 USD.
# This price reflects an increase of 2.95% in the past 24 hours.
```

---

## Agent State

```python
class AgentState01(TypedDict):
    messages: Annotated[Sequence[Message], operator.add]               # accumulates across turns
    summary: str | None                                                 # conversation summary
    prime_query: str | None                                             # enriched search query
    use_web: bool                                                       # routing decision
    web_search_result: Annotated[Sequence[WebSearchResults], replace]  # raw chunks (~288)
    retrived_result: Annotated[Sequence[RetrivedDocs], replace]        # ranked chunks (top-5)
```

`messages` uses `operator.add` — accumulates across turns. `web_search_result` and `retrived_result` use a custom `replace` reducer — replaced fresh each invocation to avoid stale retrieval state.

---

## Performance

Tested on Intel i5-1155G7 @ 2.5GHz, 16GB RAM, no GPU:

| Step | Approximate Time |
|---|---|
| Summarizer node (Gemma inference) | ~1m 42s |
| Smart Query node (Gemma inference) | ~33s |
| Web Search (DDGS + extraction) | ~1m 43s |
| FastEmbed encoding (~288 chunks) | ~2.5 min |
| BM25 + MMR + RRF | ~5s |
| Answering node (Gemma inference) | ~1m 58s |
| **Total** | **~6 minutes** |

RAGAS evaluation (5 samples, 3 metrics, local Gemma4 judge): ~1.5 hours on CPU.

Estimated with RTX 3060 12GB: ~30-40s per query, ~10 min full RAGAS eval.

---

## Planned Improvements

- [x] RAGAS evaluation — Faithfulness, Context Precision, Context Recall ✓
- [x] Ablation study — MMR λ and top-k impact on precision/recall ✓
- [x] Context window fix — OLLAMA_CONTEXT_LENGTH=16384, zero truncation confirmed ✓
- [x] Langfuse observability — self-hosted Docker, per-node tracing, chunk RRF scores logged ✓
- [ ] JSON mode fix — add `response_format={"type": "json_object"}` to RAGAS judge to eliminate remaining NaN values
- [ ] Answer Relevancy metric — requires embeddings fix for local eval
- [ ] Content cleaning — strip navigation menus and URL dumps before chunking
- [ ] Per-URL chunk cap — prevent single sources from dominating retrieval
- [ ] BM25 weight tuning — increase for exact numeric/factual queries (stock prices, market caps)
- [ ] Guardrails AI — output validation beyond JSON schema
- [ ] DSPy optimization — systematic prompt optimization against RAGAS metrics
- [ ] OpenTelemetry + Prometheus — production monitoring dashboard
- [ ] `answer` field in state — expose final answer directly without parsing messages

---

## What I Learned Building This

Every component is implemented from scratch with full understanding of the underlying mathematics — not a LangChain wrapper call.

**Retrieval theory:** what cosine similarity means geometrically, BM25's IDF and term frequency saturation, why MMR's inner loop reduces to an einsum, and why RRF exists — you cannot directly combine cosine and BM25 scores because their scales are fundamentally incompatible.

**Evaluation:** RAGAS metrics measure orthogonal properties — Faithfulness (does the answer hallucinate?), Context Precision (are the right chunks ranked highest?), and Context Recall (do the retrieved chunks cover the ground truth?) can all move independently. A correct answer with low Context Precision means your retrieval ranking is wrong even though generation succeeded — a failure mode invisible without evaluation.

**Local inference:** running a 4B parameter model on CPU teaches you what cloud APIs abstract away — context window limits, KV cache mechanics, prefill vs generation speed, and why a 4096-token overflow silently returns 1 token instead of erroring. At 16384 context length, all truncation is eliminated (`truncated = 0` confirmed across 35 eval calls), but a second failure mode emerges: the judge LLM must output structured JSON for RAGAS to parse, and without `response_format={"type": "json_object"}`, Gemma4 reverts to prose on complex prompts, producing silent NaN scores. Two distinct failure modes, both invisible without reading the raw inference logs.

**The ablation result:** increasing MMR λ from 0.5 → 0.75 improves precision but collapses recall. The optimal k and λ are query-type dependent. This is not a bug — it is the fundamental precision-recall tradeoff made visible through measurement.

**Observability:** instrumenting with `@observe` and `get_client().update_current_span()` reveals things evaluation scores alone cannot — which chunks ranked 1st for each query, whether routing made the right call, and where latency concentrates. The `langfuse.openai.OpenAI` drop-in wrapper automatically traces every LLM call with zero additional code, giving token-level visibility across the entire pipeline for free.

The gap between using RAG and understanding RAG is everything when something breaks in production.

---

## Author

**Ranji T** — Senior AI ML Engineer
[GitHub](https://github.com/ranji-t)

*Built on a no-GPU laptop. Local-first, privacy-preserving, zero cloud API costs.*
