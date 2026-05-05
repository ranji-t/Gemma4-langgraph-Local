# Gemma4 Local RAG Agent

A production-adjacent Retrieval-Augmented Generation (RAG) agent running **entirely on CPU** with no cloud dependencies. Built from scratch using Gemma4 E2B via Ollama, FastEmbed, LangGraph, and DuckDuckGo — with a hybrid retrieval pipeline combining dense embeddings, BM25 sparse retrieval, Maximal Marginal Relevance (MMR), and Reciprocal Rank Fusion (RRF).

> Built in a weekend on a no-GPU laptop. Every component implemented from scratch with full understanding of the underlying mathematics.

---

## Demo

**Query:** *"What is the NVIDIA Stock Price Today?"*

```
As of May 5, 2026, the NVIDIA Corporation share price is reported as $198.56.

- The stock reached a daily high of $198.81 and a low of $198.27 during the trading session.
- At a current price of $198.66, the stock was +0.2% higher than the low and -0.1% under the high.
- The 52-week high was $216.83 and the 52-week low was $110.82.
```

**End-to-end latency:** ~6 minutes on CPU (i5-1155G7, 16GB RAM, no GPU)

---

## Architecture

```
START
  ├── len(messages) > 1 → Summary Node → Smart Query Node → Web Search?
  └── single message   → Dummy Query Node → Web Search?
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
| **Web Search?** | Routing node. Gemma4 decides whether the query requires a live web search or can be answered from internal knowledge. |
| **Google it!!** | Executes the full retrieval pipeline: web search → chunking → dense + sparse scoring → MMR → RRF → ranked chunks. |
| **Answering Node** | Synthesizes the final answer using retrieved chunks as context, or falls back to model knowledge if no search was needed. |

---

## Retrieval Pipeline

The heart of this project is a hybrid retrieval pipeline built entirely from scratch:

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
    │  Vectorized via einsum             │
    │  Removes redundant chunks          │
    │  Run separately on dense + sparse  │
    └──────────────┬─────────────────────┘
                   │
    ┌──────────────▼─────────────────────┐
    │  RRF (Reciprocal Rank Fusion)      │
    │  k=60, fuses both MMR outputs      │
    │  Returns top-N diverse chunks      │
    └──────────────┬─────────────────────┘
                   │
            Final chunks → LLM
```

### Why hybrid retrieval?

Dense retrieval finds semantically similar chunks but misses exact keyword matches. BM25 finds exact keyword matches but misses semantic paraphrases. MMR removes redundancy within each retriever's results. RRF fuses both lists using rank positions — bypassing the incompatible score scales of cosine similarity and BM25.

---

## Key Design Decisions

### Vectorized MMR

Most MMR implementations use a nested Python for loop. This implementation replaces the inner loop with a single `np.einsum` operation:

```python
worst_redundancy = np.einsum(
    "rf, bf -> rb", remaining_docs_emb, best_docs_emb
).max(axis=1)
```

This computes all pairwise similarities between remaining and selected chunks simultaneously as a single matrix operation — significantly faster on large chunk sets and GPU-ready.

### Full Provenance Tracking

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

This makes the retrieval pipeline fully debuggable — you can trace exactly why any chunk made the final cut.

### Constrained JSON Output

The routing node uses Ollama's constrained decoding combined with Pydantic validation:

```python
class UseWebTemplate(BaseModel):
    use_web: bool

json_response = UseWebTemplate.model_validate_json(
    response.message.content
).model_dump()
```

Schema correctness is enforced at two levels — token-level JSON constraint from Ollama, field-level type validation from Pydantic.

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

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemma4 E2B via Ollama |
| Embeddings | nomic-embed-text-v1.5-Q via FastEmbed |
| Web Search | DuckDuckGo (DDGS) |
| Agent Orchestration | LangGraph |
| Sparse Retrieval | BM25Okapi (rank-bm25) |
| Text Splitting | LangChain TokenTextSplitter |
| Output Validation | Pydantic |
| Notebook | Marimo |
| Numerical Computing | NumPy |

---

## Project Structure

```
Gemma4-Test/
├── Gemma4-Agents_04.py     # Main agent notebook (Marimo)
├── pyproject.toml          # Project dependencies and metadata (uv)
├── .python-version         # Python 3.14 pin for uv
├── uv.lock                 # Locked dependency versions
├── logs/
│   └── logs04.log          # Runtime logs
└── .cache/
    └── nomic-embed-text-v1.5-Q/   # FastEmbed model cache
```

---

## How to Run

### Prerequisites

**1. Install Ollama and pull the model**

```bash
# Install Ollama — https://ollama.com

# If your models are stored in a custom location (recommended for large model files)
export OLLAMA_MODELS=/d/.files/.ollama/models

# Start the Ollama server
clear && export OLLAMA_MODELS=/d/.files/.ollama/models && ollama serve

# In a separate terminal — pull the model
ollama pull gemma4:e2b
```

**2. Install uv**

This project uses [uv](https://docs.astral.sh/uv/) for dependency management — significantly faster than pip.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**3. Clone and install**

```bash
git clone https://github.com/ranji-t/gemma4-test
cd gemma4-test

# Install all dependencies from pyproject.toml
uv sync

# Install with notebook extras (Marimo, ipykernel)
uv sync --group notebook

# Install with dev extras (mypy, pytest, ruff)
uv sync --group dev
```

### Python Version

This project requires **Python 3.14+** as specified in `.python-version`:

```
3.14
```

uv automatically manages the correct Python version — no manual installation needed if uv is configured with Python version management.

### Dependency Groups

The `pyproject.toml` defines three dependency groups:

| Group | Purpose | Install command |
|---|---|---|
| Default | Core runtime (fastembed, langgraph, ddgs, etc.) | `uv sync` |
| `notebook` | Marimo, ipykernel, grandalf for graph viz | `uv sync --group notebook` |
| `dev` | mypy, pytest, ruff for development | `uv sync --group dev` |

### Run

```bash
# Start Ollama first (in a separate terminal)
clear && export OLLAMA_MODELS=/d/.files/.ollama/models && ollama serve

# Run notebook in view mode
uv run marimo run Gemma4-Agents_04.py

# Run notebook in edit mode
uv run marimo edit Gemma4-Agents_04.py
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
```

---

## Agent State

```python
class AgentState01(TypedDict):
    messages: Annotated[Sequence[Message], operator.add]           # accumulates
    summary: str | None                                             # conversation summary
    prime_query: str | None                                         # enriched search query
    use_web: bool                                                   # routing decision
    web_search_result: Annotated[Sequence[WebSearchResults], replace]  # raw chunks
    retrived_result: Annotated[Sequence[RetrivedDocs], replace]        # ranked chunks
```

`messages` uses `operator.add` — accumulates across turns. `web_search_result` and `retrived_result` use a custom `replace` reducer — replaced fresh each invocation.

---

## Performance

Tested on Intel i5-1155G7 @ 2.5GHz, 16GB RAM, no GPU:

| Step | Approximate Time |
|---|---|
| Summarizer node (Gemma inference) | ~19s |
| Smart Query node (Gemma inference) | ~19s |
| Web Search (DDGS + extraction) | ~15s |
| FastEmbed encoding (~1000 chunks) | ~2.5 min |
| BM25 + MMR + RRF | ~5s |
| Answering node (Gemma inference) | ~19s |
| **Total** | **~6 minutes** |

With RTX 3060 12GB the estimated total drops to ~30-40 seconds.

---

## Planned Improvements

- [ ] RAGAS evaluation — context precision, faithfulness, answer relevancy
- [ ] LangSmith tracing — full observability into every node
- [ ] Guardrails AI — output validation beyond JSON schema
- [ ] DSPy optimization — systematic prompt optimization against RAGAS metrics
- [ ] Content cleaning — strip navigation menus and URL dumps before chunking
- [ ] Per-URL chunk cap — prevent single sources from dominating
- [ ] OpenTelemetry — distributed tracing
- [ ] Prometheus + Grafana — production monitoring
- [ ] `answer` field in state — expose final answer directly
- [ ] Error tracking — `error` and `failed_node` fields in state

---

## What I Learned Building This

Every component in this pipeline is implemented from scratch with full understanding of the underlying mathematics — not a LangChain wrapper call.

- **Dense retrieval** — what cosine similarity means geometrically and why normalized dot product equals it
- **BM25** — IDF, term frequency saturation, and length normalization
- **MMR** — the relevance-diversity tradeoff and why the inner loop can be replaced with einsum
- **RRF** — why you cannot directly combine cosine and BM25 scores and why rank-based fusion solves this
- **LangGraph state** — reducers, conditional edges, and state flow across nodes

The gap between using RAG and understanding RAG is everything when something breaks in production.

---

## Author

**Ranji T** — Senior Analyst
[GitHub](https://github.com/ranji-t)

*Built on a no-GPU laptop. Local-first, privacy-preserving, zero cloud API costs.*