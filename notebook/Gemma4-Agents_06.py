import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **01. Gemma4 Test Note 06**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This note books uses.
    #### *1.1 Tools*
    - Gemma4
    - Fast Embedder
    - LangGraph
    - DuckDuckGo

    #### *1.2 Techniques*
    - Dense Similarity
    - BM25
    - MMR on Sense and Sparce
    - RRF k=60
    - Logging Decorator
    """)
    return


@app.cell
def _():
    import re
    import logging
    import operator
    from itertools import chain
    from functools import wraps
    from warnings import filterwarnings
    from time import perf_counter, sleep
    from dataclasses import dataclass, field
    from typing import (
        TypedDict,
        NamedTuple,
        Sequence,
        Annotated,
        Callable,
        Any,
        Literal,
        TypeVar,
        Self,
    )

    # Third party Imports
    import numpy as np
    import instructor
    import marimo as mo
    from ddgs.ddgs import DDGS, DDGSException

    ## LLM Build & Harness
    from ollama import chat
    from pydantic import BaseModel
    from langgraph.graph import START, END
    from langgraph.graph.state import StateGraph
    from rank_bm25 import BM25Okapi
    from fastembed import TextEmbedding
    from langchain_text_splitters import TokenTextSplitter

    ## Observibility & Evlualtion
    from datasets import Dataset
    from langfuse import observe, get_client
    from langfuse.openai import OpenAI
    from ragas import evaluate
    from ragas.metrics.collections import (
        faithfulness,
        context_recall,
        context_precision,
        answer_relevancy,
    )
    from ragas.llms import llm_factory
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return (
        Annotated,
        Any,
        BM25Okapi,
        BaseModel,
        Callable,
        DDGS,
        DDGSException,
        END,
        Literal,
        NamedTuple,
        OpenAI,
        START,
        Self,
        Sequence,
        StateGraph,
        TextEmbedding,
        TokenTextSplitter,
        TypeVar,
        TypedDict,
        dataclass,
        field,
        filterwarnings,
        get_client,
        logging,
        mo,
        np,
        observe,
        operator,
        perf_counter,
        re,
        sleep,
        wraps,
    )


@app.cell
def _():
    # result_old = chat(
    #     model=gemma_lite,
    #     messages=[
    #         {"role": "system", "content": "Help if you fell like it"},
    #         {"role": "user", "content": "Who are you?"},
    #     ],
    #     # format="json",
    #     think=True,
    # )

    # client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    # result_new = client.chat.completions.create(
    #     model=gemma_lite,
    #     messages=[
    #         {"role": "system", "content": "Help if you fell like it"},
    #         {"role": "user", "content": "Who are you?"},
    #     ],
    #     # response_format={"type": "json_object"},  # replaces format="json"
    # )
    return


@app.cell
def _():
    # result_old2 = chat(
    #     model=gemma_lite,
    #     messages=[
    #         {"role": "system", "content": "Help if you fell like it"},
    #         {"role": "user", "content": "Who are you?"},
    #     ],
    #     format="json",
    #     think=True,
    # )

    # # client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    # result_new2 = client.chat.completions.create(
    #     model=gemma_lite,
    #     messages=[
    #         {"role": "system", "content": "Help if you fell like it"},
    #         {"role": "user", "content": "Who are you?"},
    #     ],
    #     response_format={"type": "json_object"},  # replaces format="json"
    # )
    return


@app.cell
def _():
    # result_old.message.content
    return


@app.cell
def _():
    # result_new.choices[0].message.content
    return


@app.cell
def _():
    # result_old2.message.content
    return


@app.cell
def _():
    # result_new2.choices[0].message.content
    return


@app.cell
def _(filterwarnings):
    filterwarnings("ignore")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **02. Evaluation**
    """)
    return


@app.cell
def _():
    # from ragas.embeddings import BaseRagasEmbeddings


    # class FastEmbeddingRagas(BaseRagasEmbeddings):
    #     def __init__(
    #         self,
    #         model_name: str = "nomic-ai/nomic-embed-text-v1.5-Q",
    #         cache_dir: str | None = r"D:\Codebase\Gemma4-Test\.cache",
    #     ) -> None:
    #         self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    #     def embed_query(self, text: str) -> np.ndarray:
    #         embedding = list(self.model.embed([f"search_query: {text}"]))[0]
    #         return embedding

    #     def embed_documents(self, texts: list[str], batch_size: int = 32):
    #         #  Check if Iterable == List
    #         if not isinstance(texts, list):
    #             raise TypeError(
    #                 "embedder expected lost of text. pls check the input to embed_documents."
    #             )
    #         # Prefix text, embed and return
    #         prefixed = [f"search_document: {t}" for t in texts]
    #         embeddings = np.array(
    #             list(self.model.embed(prefixed, batch_size=batch_size))
    #         )
    #         return embeddings
    return


@app.cell
def _():
    # # Fast Embedder
    # embed_test = FastEmbeddingRagas()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **03. Start Up processes**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *2.1 Logging*
    """)
    return


@app.cell
def _(logging):
    logging.basicConfig(
        level=logging.INFO, filename=r"D:\Codebase\Gemma4-Test\logs\logs04.log"
    )
    # Get Loggers
    logger = logging.getLogger(name="notebook04")
    return (logger,)


@app.cell
def _(Any, Callable, logger, perf_counter, wraps):
    def logging_decorator(func: Callable[Any, Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Counter start
                start_time = perf_counter()
                # Call the func
                result = func(*args, **kwargs)
                # Run Time log
                logger.info(
                    f"func: {func.__name__} ran for {perf_counter() - start_time:.3f} seconds."
                )
            except Exception as e:
                # log the error
                logger.error(f"func: {func.__name__} ran exception: {e}")
                # Reraise
                raise e
            # Return Resutlts
            return result

        # Return wrapper
        return wrapper

    return (logging_decorator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **2.2 Retry Logic**
    """)
    return


@app.cell
def _(logger, sleep, wraps):
    def sync_retry(
        max_retries: int = 3,
        delay: float = 1.0,
        backoff=2.0,
        exceptions=(Exception,),
    ):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # The current dealy time
                current_delay = delay
                # interate through the tries
                for attempt in range(1, max_retries + 1):
                    try:
                        # Call the function
                        result = func(*args, **kwargs)
                        # Return the data
                        return result

                    except exceptions as e:
                        # If the attempt failed
                        if attempt == max_retries:
                            logger.error(
                                f"Function '{func.__name__}' failed after {max_retries} attempts."
                            )
                            raise e
                        # Log the retry
                        logger.warning(
                            f"Attempt {attempt}/{max_retries} for '{func.__name__}' failed: {e}. "
                            f"Retrying in {current_delay} seconds..."
                        )
                        # Sleep then Update the time
                        sleep(current_delay)
                        current_delay *= backoff

            # The Return of wrapper
            return wrapper

        # Return the decorator
        return decorator

    return (sync_retry,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *2.3 Embedder*
    """)
    return


@app.cell
def _(TextEmbedding):
    fe_encoder_model = TextEmbedding(
        model_name="nomic-ai/nomic-embed-text-v1.5-Q",  # Embedding Dimension of 768
        cache_dir=r"D:\Codebase\Gemma4-Test\.cache",
    )
    return (fe_encoder_model,)


@app.cell
def _(Literal, Sequence, fe_encoder_model, np):
    def embed_with_fs(
        *,
        texts: Sequence[str],
        prompt: Literal["search_document", "search_query"],
        batch_size: int = 32,
    ) -> np.ndarray:
        # Check the prompt
        if prompt not in {"search_document", "search_query"}:
            raise ValueError(
                f"prompt can only be 'search_document' or 'search_query' "
                f"but instead got '{prompt}'"
            )
        # Check if the
        if not (
            isinstance(texts, list)
            or isinstance(texts, tuple)
            or isinstance(texts, set)
        ):
            raise TypeError(f"texts is to be an iterable. But {type(texts)}")

        # Norm of the data
        test_embed_not_norm = np.array(
            list(
                fe_encoder_model.embed(
                    [f"{prompt}: {_}" for _ in texts], batch_size=batch_size
                )
            )
        )

        # Inverse of L2 Norm
        l2_norm_inv = 1 / np.linalg.norm(test_embed_not_norm, axis=1).clip(
            min=1e-9
        )

        # Shpaes of the data
        return np.einsum("nf, n ->nf", test_embed_not_norm, l2_norm_inv)

    return (embed_with_fs,)


@app.cell
def _(embed_with_fs):
    embed_docs = embed_with_fs(
        texts=["Where is Gamora?", "Who is Quinn?", "Why is Drax?"],
        prompt="search_document",
    )

    embed_docs.shape
    return (embed_docs,)


@app.cell
def _(embed_with_fs):
    embed_query = embed_with_fs(
        texts=["what is Gamora?"],
        prompt="search_query",
    )[0]

    embed_query.shape
    return (embed_query,)


@app.cell
def _(embed_docs, embed_query, np):
    np.einsum("ne, e -> n", embed_docs, embed_query)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *2.4 Regex Pattern Compile*
    """)
    return


@app.cell
def _(re):
    BM25_TOKENIZER = re.compile(r"\b\w+\b")
    return (BM25_TOKENIZER,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **04. LLM Functions**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *3.1 Message Layout*
    """)
    return


@app.cell
def _(Literal, Self, dataclass, field):
    print("Whatttt!!!!")

    @dataclass(frozen=True)
    class Message:
        role: Literal["system", "user", "assistant"]
        content: str
        text: str = field(init=False, repr=False)

        def __post_init__(self: Self) -> None:
            object.__setattr__(self, "text", f"[{self.role}]: {self.content}")

        def __str__(self: Self) -> str:
            return self.text

        def __eq__(self: Self, other) -> bool:
            return (self.role == other.role) & (self.content == other.content)

    return (Message,)


@app.cell
def _():
    gemma_lite = "gemma4:e2b"
    return (gemma_lite,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *3.2 Use Web Decion Function*
    """)
    return


@app.cell
def _(OpenAI, gemma_lite):
    def use_web(user_query: str = "Hello Gemma!"):
        # The Sys Prompt
        system_prompt = """
    Role: Act as a routing agent and decide whether a web search is required to answer the implicit request.
    Format the Output: The output must be in valid JSON format!!!!.

    ----------------------------------------------------------------------
    Output SCHEMA: {
      "use_web": True or False  ## You can only answer in Boolian
    }
    ----------------------------------------------------------------------

    # Procedure
    1. Analyze the Request
    2. Determine the Intent
    3. Check for Information Need
    4. Formulate the Decision
    5. Format the Output: The required output format is JSON.
    6. Conclusion: <The Requested JSON Output>


    # Exmaples:
    1. Correct Answers:
    {"use_web": true}

    2. Correct Answers:
    {"use_web": false}

    3. Wrong Answers:
    { "tool_required": false, "reason": "The user has only said 'Hello Gemma!'. This is a simple greeting that doesn't require external information." }
    """
        # # Response from user
        # result = chat(
        #     model=gemma_lite,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_query},
        #     ],
        #     format="json",
        #     think=True,
        # )
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        result = client.chat.completions.create(
            model=gemma_lite,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format={"type": "json_object"},  # replaces format="json"
        )
        # return data
        return result

    return (use_web,)


@app.cell
def _():
    # # result = use_web("What is the Nvidia Stock prices today?")
    # result = use_web(
    #     "What are the questions asked in coupa interviews for data scientist?"
    # )
    return


@app.cell
def _():
    # # Convert the string to JSON
    # json.loads(result.message.content)
    return


@app.cell
def _():
    # # The Thinking Source
    # mo.md(result.message.thinking)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *3.3 Summarize Function*
    """)
    return


@app.cell
def _(Message, OpenAI, gemma_lite):
    def summarizer(messages: list[Message]):
        # The purpose of this node
        system_prompt = """
    # Role & Constraints
    - You are a expert Summarizer.
    - Do not Give answers just summarise the  context the user provides
    - you will summarise in structured format.
    - Keep Track of the Core intent of the user
    - Actions be kept in temproal summaries order
    - later messages will be given pririty
    - chages in the intent will be noted"""
        # Changes in the data
        content = "\n\n".join([msg.text for msg in messages])

        # Response from user
        # result = chat(
        #     model=gemma_lite,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": content},
        #     ],
        #     format=None,
        #     think=True,
        # )
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        result = client.chat.completions.create(
            model=gemma_lite,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            # response_format={"type": "json_object"},  # replaces format="json"
        )
        # retur ndata
        return result

    return (summarizer,)


@app.cell
def _():
    # # This the test message
    # test_messages_01 = [
    #     Message("user", "Who are you?"),
    #     Message("assistant", "Hi I am Gemini?"),
    #     Message("user", "I have an interview lined up with Coupa."),
    #     Message("assistant", "Congratulation!!!! How can I help you?"),
    #     Message(
    #         "user",
    #         "I want to know what kind of questions will be asked in the first round of Data Scientist interview",
    #     ),
    #     Message(
    #         "assistant",
    #         """
    # After that recruiter screen, there may be an online technical assessment covering coding problems and data structures, typically 3 questions within 45–60 minutes. GeeksforGeeks

    # Round 1 — Technical Questions Actually Asked
    # ML Concepts

    # You are tasked with building a decision tree to predict if a borrower will repay a loan — how would you evaluate if it's the right choice, and how would you assess performance before and after deployment? PR Newswire
    # Explain how a random forest generates its ensemble of trees, and why you might choose it over logistic regression. PR Newswire
    # Compare bagging vs boosting — describe scenarios where you'd prefer one over the other and discuss the tradeoffs. PR Newswire
    # Your manager asks you to build a neural network model — how would you justify its complexity and explain its predictions to non-technical stakeholders? PR Newswire
    # Calculating the number of trainable parameters for a CNN. ArcWeb

    # Data Engineering / SQL

    # ETL pipeline design questions, Snowflake data warehouse tools, PySpark coding, and data transformation. iMocha

    # Coding / DSA

    # DSA questions at easy to medium LeetCode level — basic LeetCode 75 is sufficient. Live coding or pseudo code expected. iMocha
    # """,
    #     ),
    #     Message("user", "I am nervous!"),
    # ]

    # result2 = summarizer(messages=test_messages_01)
    return


@app.cell
def _():
    # # Result
    # mo.md(result2.message.content)
    return


@app.cell
def _():
    # # The Thinking Source
    # mo.md(result2.message.thinking)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *3.4 Update Query Node*
    """)
    return


@app.cell
def _(OpenAI, gemma_lite):
    def update_query(context: str, query: str):
        system_prompt = """
    # Role
    - Taking the context form the user, update the query wuch that it enriches the query.
    - Only return the modified query
    - make it consise and potent and also web search engine frendly.

    # The procedure
    1. Analyze the Request
    2. Analyze the Goal
    3. Evaluate the Context
    4. Determine the Enrichment
    5. Formulate the New Query
    6. Initial Query
    7. Refine for Conciseness and Potency (Web Search Friendly)
    8. Final Output Generation: Generate only the modified query.
    """
        user_prompt = f"""
    # Context:
    {context}
    -------------------------------------------

    # Query:
    {query}
    """
        # Response from user
        # result = chat(
        #     model=gemma_lite,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     format=None,
        #     think=True,
        # )
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        result = client.chat.completions.create(
            model=gemma_lite,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # response_format={"type": "json_object"},  # replaces format="json"
        )

        # Return
        return result

    return (update_query,)


@app.cell
def _():
    # result3 = update_query(
    #     result2.message.content,
    #     "Where this companies office situated?",
    # )
    return


@app.cell
def _():
    # #  Show the updated query
    # mo.md(result3.message.content)
    return


@app.cell
def _():
    # # Show the Thinking
    # mo.md(result3.message.thinking)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **05. Web Search, Embedding, BM25, MMR & RRF**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *4.1 Web Search With DDG*
    """)
    return


@app.cell
def _(NamedTuple):
    class WebSearchResults(NamedTuple):
        search_id: int
        chunk_id: int
        url: str
        title: str
        snippet: str
        content: str

    return (WebSearchResults,)


@app.cell
def _(
    DDGS,
    DDGSException,
    Sequence,
    TokenTextSplitter,
    WebSearchResults,
    logger,
):
    def perform_web_search(
        *, query: str, max_results: int = 5
    ) -> Sequence[WebSearchResults]:
        # Show web results
        web_results: list[WebSearchResults] = []
        chunked_web_results: list[WebSearchResults] = []
        counter = 0

        # Search the web
        with DDGS() as d:
            # Get the urls from the search
            for url_dict in d.text(query=query, max_results=max_results):
                try:
                    # Get url form this
                    url = url_dict.get("href")
                    # Get Title
                    title = url_dict.get("title")
                    # Get Snippet of the site
                    snippet = url_dict.get("body")
                    # Extract the data
                    data = d.extract(url, fmt="text_markdown").get("content")
                    # update counter
                    counter += 1

                    # Add data to container
                    web_results.append(
                        WebSearchResults(
                            counter, None, url, title, snippet, data
                        )
                    )

                except DDGSException as e:
                    print(
                        f"func: perform_web_search, url extraction failed{e}"
                    )
                    logger.warning(
                        f"func: perform_web_search, url extraction failed{e}"
                    )

        # The Token based filter
        tts = TokenTextSplitter(chunk_size=300, chunk_overlap=50)

        # Chunk the objects
        for res in web_results:
            # Split the content
            split_content = tts.split_text(res.content)
            # Iterate Trrought
            for chunk_id, chunk in enumerate(split_content, start=1):
                chunked_web_results.append(
                    WebSearchResults(
                        res.search_id,
                        chunk_id,
                        res.url,
                        res.title,
                        res.snippet,
                        chunk,
                    )
                )

        # Return resutls
        return chunked_web_results

    return (perform_web_search,)


@app.cell
def _():
    # # Web Seerch Test with DDG
    # web_search_results = perform_web_search(
    #     query="Who is Billie Eilish?", max_results=5
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *4.2 Score for Embedder & BM25 full text search*
    """)
    return


@app.cell
def _(NamedTuple):
    class RetrivedDocs(NamedTuple):
        # The Web Chunked docs
        search_id: int
        chunk_id: int
        url: str
        title: str
        snippet: str
        content: str

        # The Scores
        context_score: float
        semantic_score: float

        # MMR select
        context_mmr: bool
        sematic_mmr: bool

        # RRF Score
        rrf_score: float
        rrf_rank: int

    return (RetrivedDocs,)


@app.cell
def _(
    BM25Okapi,
    BM25_TOKENIZER,
    Sequence,
    embed_with_fs,
    logging_decorator,
    np,
):
    def context_scorer_fe(
        query: str, docs: Sequence[str], batch_size: int = 32
    ):
        # Create Document Encoding
        doc_encoding = embed_with_fs(
            texts=docs, prompt="search_document", batch_size=batch_size
        )

        # Create query Embedding
        query_encoding = embed_with_fs(
            texts=[query],
            prompt="search_query",
        )[0]

        # Similarity
        sim_score = np.einsum("de, e -> d", doc_encoding, query_encoding)

        # return
        return (sim_score, doc_encoding, query_encoding)

    @logging_decorator
    def semantic_scorer(query: str, docs: Sequence[str]):
        # Tokenize the query
        tokenized_docs = [
            BM25_TOKENIZER.findall(doc.lower().strip()) for doc in docs
        ]
        tokenized_query = BM25_TOKENIZER.findall(query.lower().strip())

        # Build BM25 index
        bm25 = BM25Okapi(corpus=tokenized_docs)

        # Score all docs against query
        scores = bm25.get_scores(tokenized_query)

        # return
        return scores

    return context_scorer_fe, semantic_scorer


@app.cell
def _():
    # # Number of chunks
    # print(len(web_search_results))

    # # Show The Results
    # web_search_results[:3]
    return


@app.cell
def _():
    # # The Context Scorer Sentence Trasnfomerms
    # context_score, doc_encoding, query_encoding = context_scorer_fe(
    #     "Who is Billie Eilish?", [_.content for _ in web_search_results]
    # )
    return


@app.cell
def _():
    # # Semantic Score
    # semantic_score = semantic_scorer(
    #     "Who is Billie Eilish?", [_.content for _ in web_search_results]
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *4.3 Perform MMR*
    """)
    return


@app.cell
def _(np):
    def mmr(
        scores: np.ndarray,
        doc_encoding: np.ndarray,
        top_k: int = 10,
        lambda_param: float = 0.75,
        score_threshold: float = -np.inf,
    ):
        # Get the number of samples
        n_docs = len(scores)

        # Tracking IDs
        selected_ids = []
        remaining_ids = list(range(n_docs))
        remaining_ids = [
            i for i in remaining_ids if (scores[i] > score_threshold)
        ]
        red_n_docs = len(remaining_ids)

        if red_n_docs == 0:
            raise ValueError(
                "No docs selected reduce the threshold or check if you are inputing empty data"
            )

        # Looop the TOPK
        for _ in range(min(red_n_docs, top_k)):
            if not selected_ids:
                max_idx = np.argmax(scores[remaining_ids])
                best_idx = remaining_ids[max_idx]

            else:
                # Get embedding and scores
                best_docs_emb = doc_encoding[selected_ids]
                remaining_docs_emb = doc_encoding[remaining_ids]
                remaining_docs_score = scores[remaining_ids]

                # Compute the remaing best scores the nthe wors case
                worst_redundency = np.einsum(
                    "rf, bf -> rb", remaining_docs_emb, best_docs_emb
                ).max(axis=1)

                # Compute MRR
                mmr_score = (lambda_param * remaining_docs_score) - (
                    (1 - lambda_param) * worst_redundency
                )

                # get the index of the best MMR array the nget the index of the main array
                max_idx = np.argmax(mmr_score)
                best_idx = remaining_ids[max_idx]

            # Update the tacking indices
            selected_ids.append(best_idx)
            remaining_ids.remove(best_idx)

        # Return selected index
        return selected_ids

    return (mmr,)


@app.cell
def _():
    # # The MMR Context IDs
    # context_ids = mmr(context_score, doc_encoding)

    # # The MMR Semantic IDs
    # semantic_ids = mmr(
    #     semantic_score / np.maximum(semantic_score.max(), 1e-9), doc_encoding
    # )
    return


@app.cell
def _():
    # # Update scores
    # retrived_docs = [
    #     RetrivedDocs(*res, cs, ss, False, False, None, None)
    #     for (res, cs, ss) in zip(web_search_results, context_score, semantic_score)
    # ]

    # # Update MMR Context
    # retrived_docs = [
    #     RetrivedDocs(
    #         doc.search_id,
    #         doc.chunk_id,
    #         doc.url,
    #         doc.title,
    #         doc.snippet,
    #         doc.content,
    #         doc.context_score,
    #         doc.semantic_score,
    #         (idx in context_ids),
    #         False,
    #         None,
    #         None,
    #     )
    #     for idx, doc in enumerate(retrived_docs)
    # ]

    # # Update MMR Semantic
    # retrived_docs = [
    #     RetrivedDocs(
    #         doc.search_id,
    #         doc.chunk_id,
    #         doc.url,
    #         doc.title,
    #         doc.snippet,
    #         doc.content,
    #         doc.context_score,
    #         doc.semantic_score,
    #         doc.context_mmr,
    #         (idx in semantic_ids),
    #         None,
    #         None,
    #     )
    #     for idx, doc in enumerate(retrived_docs)
    # ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *4.4 RRF (Recursive Rank Regression)*
    """)
    return


@app.function
def rrf(ranked_lists: list[list[int]], k: int = 60, top_n: int = 5):
    # The sore tracker for each index
    scores = dict()

    # Iterate through lists
    for ranked_list in ranked_lists:
        for i, idx in enumerate(ranked_list, start=1):
            if idx in scores:
                scores[idx] += 1 / (k + i)
            else:
                scores[idx] = 1 / (k + i)

    # Sort the values based on the value
    rrf_ranks = sorted(
        scores.keys(), key=lambda k: scores.get(k), reverse=True
    )

    # Return sorted inidices
    return (
        list(rrf_ranks)[:top_n],
        sorted(scores.values(), reverse=True)[:top_n],
        list(range(1, len(list(rrf_ranks)[:top_n]) + 1)),
    )


@app.cell
def _():
    # final_retrived_docs = []

    # for idx, score, rank in zip(*rrf([context_ids, semantic_ids])):
    #     part_doc = retrived_docs[idx]
    #     final_retrived_docs.append(
    #         RetrivedDocs(
    #             part_doc.search_id,
    #             part_doc.search_id,
    #             part_doc.url,
    #             part_doc.title,
    #             part_doc.snippet,
    #             part_doc.content,
    #             part_doc.context_score,
    #             part_doc.semantic_score,
    #             part_doc.context_mmr,
    #             part_doc.sematic_mmr,
    #             score,
    #             rank,
    #         )
    #     )

    # final_retrived_docs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *4.5 Test Functions*
    """)
    return


@app.cell
def _():
    # # Chunk the model
    # query = "Current Nvidia stock price"
    # search_results = perform_web_search("Latest Nvidia stock price")

    # # Rank the documents Context Wise
    # doc_context_score, doc_encoding, query_encoding = context_scorer(
    #     query, search_results
    # )

    # # Rank the documents Context Wise
    # doc_semantic_score = semantic_scorer(query, search_results)

    # # Get MMR IDs based on
    # mmr_dense = mmr(doc_context_score, doc_encoding, score_threshold=0.5)

    # # Get MMR IDs based on Sparsity
    # mmr_sparse = mmr(
    #     doc_semantic_score / doc_semantic_score.max(),
    #     doc_encoding,
    #     score_threshold=0.5,
    # )

    # final_docs_index = rrf([mmr_dense, mmr_sparse])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **06. The Agent Graph**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *5.1 Graph State*
    """)
    return


@app.cell
def _(TypeVar):
    T = TypeVar("T")

    def replace(x: T, y: T) -> T:
        return y

    return (replace,)


@app.cell
def _(
    Annotated,
    Message,
    RetrivedDocs,
    Sequence,
    TypedDict,
    WebSearchResults,
    operator,
    replace,
):
    class AgentState01(TypedDict):
        # Mwssages
        messages: Annotated[Sequence[Message], operator.add]
        summary: str | None

        # Query
        prime_query: str | None

        # Wed Search Results
        use_web: bool
        web_search_result: Annotated[Sequence[WebSearchResults], replace]
        retrived_result: Annotated[Sequence[RetrivedDocs], replace]

    return (AgentState01,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *5.2 The Node Functions*
    """)
    return


@app.cell
def _(
    AgentState01,
    BaseModel,
    Message,
    OpenAI,
    RetrivedDocs,
    Sequence,
    context_scorer_fe,
    gemma_lite,
    get_client,
    logging_decorator,
    mmr,
    np,
    observe,
    perform_web_search,
    semantic_scorer,
    summarizer,
    sync_retry,
    update_query,
    use_web,
):
    class UseWebTemplate(BaseModel):
        use_web: bool

    # Test the model output vlaidation with json
    print(UseWebTemplate.model_validate_json('{"use_web": true}').model_dump())

    @sync_retry()
    @logging_decorator
    @observe(name="Summarizer Node")
    def summarizer_node(state: AgentState01) -> AgentState01:
        # Extrac messages
        messsages = state["messages"]
        # Pass it though the summarizer
        result = summarizer(messages=messsages)

        # Add metadata to langfuese
        lfuse = get_client()
        lfuse.update_current_span(
            metadata={
                "Input Char Length": len(messsages),
                "Output Char Length": len(result.choices[0].message.content),
            }
        )

        # Content summarized
        # return {"summary": result.message.content}
        return {"summary": result.choices[0].message.content}

    @sync_retry()
    @logging_decorator
    @observe(name="Smart Query Node")
    def smart_query_node(state: AgentState01) -> AgentState01:
        # Extract the query
        query = state["messages"][-1].content
        # Extract context
        context = state["summary"]
        # Return
        result = update_query(context=context, query=query)
        # Update the query
        # return {"prime_query": result.message.content}
        return {"prime_query": result.choices[0].message.content}

    @sync_retry()
    @logging_decorator
    @observe(name="Web Use Node")
    def web_use_node(state: AgentState01) -> AgentState01:
        # Message extraction
        messages: Sequence[Message] = state["messages"]

        # Extract the last query from the node
        query: str = messages[-1].content

        # Does the node need web
        response = use_web(user_query=query)

        # Gather the JSON String result and parse it into dict
        json_response = UseWebTemplate.model_validate_json(
            response.choices[0].message.content
        ).model_dump()

        # Structured Response
        return json_response

    @sync_retry()
    @logging_decorator
    @observe(name="Web Search Node")
    def web_search_node(state: AgentState01) -> AgentState01:
        # Extract Query from sate
        query = state["prime_query"]

        # Search The web
        search_results = perform_web_search(query=query, max_results=5)

        # Extract contents form websearch reesults
        content = [_.content for _ in search_results]
        # Rank the documents Context Wise
        doc_context_score, doc_encoding, _ = context_scorer_fe(query, content)
        # Rank the documents Context Wise
        doc_semantic_score = semantic_scorer(query, content)

        # Get MMR IDs based on
        mmr_dense = mmr(doc_context_score, doc_encoding, score_threshold=0.5)
        # Get MMR IDs based on Sparsity
        mmr_sparse = mmr(
            doc_semantic_score / np.maximum(doc_semantic_score.max(), 1e-9),
            doc_encoding,
            score_threshold=0.5,
        )

        # Convert to set
        mmr_dense_set = set(mmr_dense)
        mmr_sparse_set = set(mmr_sparse)

        # Update MMR Context
        retrived_docs = [
            RetrivedDocs(
                *res,
                cs,
                ss,
                (idx in mmr_dense_set),
                (idx in mmr_sparse_set),
                None,
                None,
            )
            for idx, (res, cs, ss) in enumerate(
                zip(search_results, doc_context_score, doc_semantic_score)
            )
        ]

        final_retrived_docs = []
        for idx, score, rank in zip(*rrf([mmr_dense, mmr_sparse])):
            part_doc = retrived_docs[idx]
            final_retrived_docs.append(
                RetrivedDocs(
                    part_doc.search_id,
                    part_doc.chunk_id,
                    part_doc.url,
                    part_doc.title,
                    part_doc.snippet,
                    part_doc.content,
                    part_doc.context_score,
                    part_doc.semantic_score,
                    part_doc.context_mmr,
                    part_doc.sematic_mmr,
                    score,
                    rank,
                )
            )

        # The Data to be put in span
        lfuse = get_client()
        lfuse.update_current_span(
            metadata={
                "num_docs_retrived": len(final_retrived_docs),
                "prime_query": query,
                "urls": [d.url for d in final_retrived_docs],
                "content": [d.content for d in final_retrived_docs],
                "context_score": [
                    d.context_score for d in final_retrived_docs
                ],
                "semantic_score": [
                    d.semantic_score for d in final_retrived_docs
                ],
                "rrf_rank": [d.rrf_rank for d in final_retrived_docs],
            }
        )

        # Return the data
        return {
            "web_search_result": search_results,
            "retrived_result": final_retrived_docs,
        }

    @sync_retry()
    @logging_decorator
    @observe(name="Answering Node")
    def answering_node(state: AgentState01) -> AgentState01:
        # Get enghances query
        user_prompt = state["prime_query"]

        # If not None
        if state["web_search_result"] is not None:
            # Convert this to string
            consolidates_results = "\n-----\n".join(
                [_.content for _ in state["retrived_result"]]
            )
            system_prompt = f"""
    # Role
    You are an expert research analyst. Your job is to answer the user query accurately using the provided web search results.

    # Instructions
    - Answer directly and concisely based on the provided context
    - Synthesize information across multiple sources — do not just repeat one source
    - If sources contradict each other, acknowledge it and present both perspectives
    - If the context does not contain enough information to answer fully, say so clearly
    - Never fabricate information not present in the context
    - Cite which part of the context supports your answer naturally in prose

    # Context
    The following are web search results relevant to the query:
    -----
    {consolidates_results}
    -----

    # Output Format
    Answer in clear prose. Be direct. Lead with the answer, then support it with details from context.
    """
        else:
            consolidates_results = None
            system_prompt = """
    # Role
    You are a knowledgeable assistant answering from your training knowledge.

    # Instructions
    - Answer the query using your internal knowledge
    - Be clear about the limits of your knowledge — if something may have changed recently, say so
    - Do not fabricate specific facts like prices, dates, or statistics you are uncertain about
    - Keep the answer concise and direct

    # Output Format
    Answer in clear prose. Lead with the answer, then add supporting detail.
    """

        # Response from user
        # result = chat(
        #     model=gemma_lite,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     format=None,
        #     think=True,
        # )
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        result = client.chat.completions.create(
            model=gemma_lite,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # response_format={"type": "json_object"},  # replaces format="json"
        )

        # Message
        return {
            "messages": [
                Message(
                    role="assistant", content=result.choices[0].message.content
                )
            ]
        }

    return (
        answering_node,
        smart_query_node,
        summarizer_node,
        web_search_node,
        web_use_node,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *5.3 Graph Structure*
    """)
    return


@app.cell
def _(
    AgentState01,
    END,
    START,
    StateGraph,
    answering_node,
    smart_query_node,
    summarizer_node,
    web_search_node,
    web_use_node,
):
    builder01 = StateGraph(AgentState01)

    # Build Node & Edges of the app
    builder01.add_node("Summary Node", summarizer_node)
    builder01.add_node("Smart Query Node", smart_query_node)
    builder01.add_node(
        "Dummy Query Node",
        lambda state: {"prime_query": state["messages"][0].content},
    )
    builder01.add_node("Web Search?", web_use_node)
    builder01.add_node("Google it!!", web_search_node)
    builder01.add_node("Answering Node", answering_node)

    # Add Edges to the Graph
    builder01.add_conditional_edges(
        source=START,
        path=lambda state: len(state["messages"]) > 1,
        path_map={True: "Summary Node", False: "Dummy Query Node"},
    )
    # With Context Path
    builder01.add_edge("Summary Node", "Smart Query Node")
    builder01.add_edge("Smart Query Node", "Web Search?")
    # With out context path
    builder01.add_edge("Dummy Query Node", "Web Search?")
    builder01.add_conditional_edges(
        "Web Search?",
        lambda state: state["use_web"],
        {True: "Google it!!", False: "Answering Node"},
    )
    builder01.add_edge("Google it!!", "Answering Node")
    builder01.add_edge("Answering Node", END)

    # build the app
    app01 = builder01.compile()
    return (app01,)


@app.cell
def _(app01):
    print(app01.get_graph().draw_ascii())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *5.4 Test the App*
    """)
    return


@app.cell
def _(Message, app01):
    result4 = app01.invoke(
        {
            "messages": [
                Message(role="user", content="Who are You?"),
                Message(
                    role="assistant", content="I am Gemma4 an LLM by google!"
                ),
                Message(
                    role="user",
                    # content="What is the Nvidia Stock Price Today?",
                    # content="Who is the CEO of NVIDIA?",
                    # content="How does NVDA compare to AMD in market cap?",
                    # content="What were the recent major announcements by NVIDIA that impacted its stock price?",
                    content="What is the weather in Mumbai?",
                ),
            ]
        }
    )

    # Show reslults
    result4
    return (result4,)


@app.cell
def _(mo, result4):
    mo.md(result4["messages"][-1].content)
    return


@app.cell
def _():
    # (result4["messages"][-1].content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **07. Evaluations**
    """)
    return


@app.cell
def _():
    # [_.content for _ in result4["retrived_result"]]
    return


@app.cell
def _():
    # data = {
    #     "question": [
    #         "What is the Nvidia Stock Price Today?",
    #         "Who is the CEO of NVIDIA?",
    #         "Market capitalization comparison: NVDA vs AMD",
    #         "What were the recent major announcements by NVIDIA that impacted its stock price?",
    #         "Current weather in Mumbai",
    #     ],
    #     "answer": [
    #         "'The current stock price for NVIDIA (NVDA) is $195.74 USD. This price was noted as of June 26, 2026, where the previous close was also $195.74 USD.\n\nAdditionally, the stock has recently seen a decrease:\n*   The current price is $195.74 USD, reflecting a decrease of −1.75% in the past 24 hours.'",
    #         "Jensen Huang is the Co-founder, President, and Chief Executive Officer of NVIDIA.",
    #         "Based on the provided information, the market capitalization comparison between NVDA and AMD is as follows:\n\n*   **NVIDIA (NVDA):** Market capitalization is $4.74 Trillion (T).\n*   **AMD:** Market capitalization is $868.41 Billion (B).",
    #         'The major recent event that significantly impacted NVIDIA\'s stock price was the announcement of its Q1 FY2027 earnings.\n\nKey points regarding this event and the market reaction include:\n\n*   **Record Earnings:** Revenue for the quarter reached $81.6 billion (up 85% year-over-year), data center revenue hit $75.2 billion (up 92%), and non-GAAP earnings per share were $1.87, with free cash flow at $48.6 billion.\n*   **Stock Volatility:** Following the announcement on May 20, the stock experienced significant downward pressure, falling on every trading day until May 27. A broader sector sell-off further impacted the stock on June 5.\n*   **Market Behavior:** This behavior was described as "buy the rumor, sell the news," as the actual results confirmed what the market had already priced in.\n\nAdditionally, the context notes that while the strong earnings were reported, some factors like China headwinds, compute price decline, and export control tightening are putting pressure on the stock valuation.',
    #         "The current weather in Mumbai, Maharashtra is Rain with a temperature of 31°C, and the feels-like temperature is 38°C (Hot) based on observed conditions (RealFeel® Temperature).\n\nAs of 05:00 PM on June 26, 2026, the detailed weather parameters for Mumbai are as follows:\n*   **Condition:** Mist\n*   **Temperature:** 31°C (Feels like 38°C)\n*   **Humidity:** 71%\n*   **Wind Speed:** 18.4 km/h from the SW (231 degrees), with a gust speed of 7.1 m/s\n*   **Cloud Cover:** 75%\n*   **Visibility:** 3.5 km\n*   **Precipitation:** 0.07 mm, with a 74% chance of rain\n*   **UV Index:** 1.1 (Low)\n*   **Pressure:** 1005mb",
    #     ],  # Gemma4's response
    #     "contexts": [
    #         [
    #             "valued Stocks ][319]\nAdvertisement\n\n# NVIDIA Corporation (NVDA)\n\nView the NVIDIA Corporation stock price live, this page displays NASDAQ NVDA stock exchange data.\nView the NVDA premarket stock price ahead of the market session or assess the after hours quote.\nMonitor the latest movements within the NVIDIA Corporation real time stock price chart below. You\ncan find more details by visiting the additional pages to view historical data, charts, latest news,\nanalysis or visit the forum to view opinions on the NVDA quote. (ISIN: US67066G1040)\n\n**What Is the Current Stock Price of NVIDIA?**\nThe NVIDIA Corporation stock price is 195.74.\n\n**Who Founded NVIDIA?**\nNVIDIA was founded on April 5, 1993, by Jensen Huang, Chris Malachowsky, and Curtis Priem. They\nstarted the company with a vision to revolutionize computer graphics.\n\n**Where is NVIDIA HQ?**\nWhile this company has operations around the globe, the current NVDA headquarters is located in\nSanta Clara, California (U.S.).\n\n**When Did NVIDIA Stock Go Public?**\nNVIDIA went public on January 22, 1999. The stock was initially offered at $12 per share (adjusted\nfor subsequent splits).\n\n**When Was the Last Time NVIDIA Stock Split?**\nNVIDIA's most recent stock split occurred on July 20, 2021. This was a 4-",
    #             " the current trading status of NVIDIA (NVDA)?\n\nAs of Jun 26, 2026, NVIDIA (NVDA) is trading at a price of 195.74 USD, with a previous close of\n195.74 USD. The stock has fluctuated within a day range of 192.13 USD to 200.80 USD, while its\n52-week range spans from 151.49 USD to 236.54 USD.\n\n### What Is NVIDIA (NVDA) Price Target According to Analysts?\n\nThe average 12-month price target for NVIDIA is 298.93 USD, with a high estimate of 500 USD and a\nlow estimate of 180 USD. 59 analysts recommend buying the stock, while 1 suggest selling, leading to\nan overall rating of Strong Buy. The stock has an +52.72% Upside potential.\n\n### What Is the NVDA Premarket Price?\n\nNVDA's last pre-market stock price is 193.66 USD. The pre-market share volume is 1,250,000.00, and\nthe stock has decreased by -2.08, or -1.06%.\n\n### What Is the NVDA After Hours Price?\n\nNVDA's last after hours stock price is 193.78 USD, the stock has decreased by -1.96, or -1.00%.\n\n## Industry Spotlight\n\n[5 Things You Need to Know About the Gold and Silver Selloff][",
    #             "VanEck Semiconductor ETF][165]\nWeight 18.50%\nMarket value ‪13.24 B‬\n[\n0A3D\niShares Core S&P 500 UCITS ETF][166]\nWeight 7.65%\nMarket value ‪11.51 B‬\n[\n\nExplore more ETFs \n\n][167]\n\n## Frequently asked questions\n\n﻿\nWhat is NVIDIA Corporation stock price today?\nThe current price of NVDA is 195.74 USD — it has decreased by −1.75% in the past 24 hours. Watch\nNVIDIA Corporation stock price performance more closely on the [chart][168].\nWhat is NVIDIA Corporation stock ticker?\nDepending on the exchange, the stock ticker may vary. For instance, on NASDAQ exchange NVIDIA\nCorporation stocks are traded under the ticker NVDA.\nIs NVIDIA Corporation stock price growing?\nNVDA stock has fallen by −5.70% compared to the previous week, the month change is a −9.71% fall,\nover the last year NVIDIA Corporation has showed a 31.13% increase.\nWhat is NVDA stock forecast?\nWe've gathered analysts' opinions on NVIDIA Corporation future price: according to them, NVDA price\nhas a max estimate of 743.10 USD and a min estimate of 180.00 USD. Watch [NVDA chart][",
    #             ' is 195.74 USD.\n\n### What Stock Exchange Does NVIDIA Trade On?\n\nNVIDIA is listed and trades on the Nasdaq Stock Exchange.\n\n### What Is the Stock Symbol for NVIDIA?\n\nThe stock symbol for NVIDIA is "NVDA."\n\n### Does NVIDIA Pay Dividends? What’s The Current Dividend Yield?\n\nThe NVIDIA dividend yield is 0.51%.\n\n### What Is the NVIDIA Market Cap?\n\nAs of today, NVIDIA market cap is 4.74T USD.\n\n### What Is NVIDIA\'s Earnings Per Share (TTM)?\n\nThe NVIDIA EPS (TTM) is 6.53.\n\n### When Is the Next NVIDIA Earnings Date?\n\nNVIDIA will release its next earnings report on Aug 26, 2026.\n\n### From a Technical Analysis Perspective, Is NVDA a Buy or Sell?\n\nBased on moving averages and other technical indicators, the daily buy/sell signal is Strong Sell.\n\n### How Many Times Has NVIDIA Stock Split?\n\nNVIDIA has split 6 times.\n\n### How Many Employees Does NVIDIA Have?\n\nNVIDIA has 42000 employees.\n\n### What is the current trading status of NVIDIA (NVDA)?\n\nAs of Jun 26, 2026, NVIDIA (NVDA) is trading at a price of 195.74 USD, with a previous close of\n195.74 USD. The stock has',
    #             "?\nWe've gathered analysts' opinions on NVIDIA Corporation future price: according to them, NVDA price\nhas a max estimate of 743.10 USD and a min estimate of 180.00 USD. Watch [NVDA chart][169] and read\na more detailed [NVIDIA Corporation stock forecast][170]: see what analysts think of NVIDIA\nCorporation and suggest that you do with its stocks.\nWhat are NVIDIA Corporation stock highest and lowest prices ever?\nNVDA reached its all-time high on May 14, 2026 with the price of 236.54 USD, and its all-time low\nwas 0.03 USD and was reached on Apr 26, 1999. View more price dynamics on [NVDA chart][171].\nSee other stocks reaching their [highest][172] and [lowest][173] prices.\nHow volatile is NVDA stock?\nNVDA stock is 4.51% volatile and has beta coefficient of 1.40. Track NVIDIA Corporation stock price\non the [chart][174] and check out the list of the [most volatile stocks][175] — is NVIDIA\nCorporation there?\nWhat is NVIDIA Corporation market cap?\nToday NVIDIA Corporation has the market capitalization of ‪4.74 T‬, it has decreased by −5.55% over\nthe last week.\nDoes NVIDIA Corporation release reports?\nYes, you can track [NVIDIA Corporation financials][",
    #         ],
    #         [
    #             "  \n  `` ``\n  4,853\n  `` `` `` `` `` `` ``\n  425 Comments\n  ][225]\n* [\n  \n  Matthew Kimball\n  \n  Moor Insights & Strategy • 4K followers\n  \n  ][226] [\n  \n  \n  Wylie Wong at Data Center Knowledge delivers a really good breakdown of NVIDIA CEO Jensen Huang's\n  keynote. With so much news flooding the airwaves (I know, I'm dating myself) - Wylie cuts through\n  a lot of the noise. Check out his coverage below ⬇️\n  \n  `` ``\n  5\n  `` `` `` `` `` `` ``\n  ][227]\n* [\n  \n  Mark Wade\n  \n  Ayar Labs • 4K followers\n  \n  ][228] [\n  \n  \n  I am excited to welcome Sankara Venkateswaran to the Ayar Labs team as our new VP of Engineering.\n  As we move into 2026, the industry is hitting a critical inflection point. The \"copper wall\" is a\n  real-world bottleneck for hyperscale AI, and co-packaged optics has become an infrastructure\n  necessity in future roadmaps. Sankara joins us at a pivotal moment as we move towards mass\n  production. His deep",
    #             " [Benefits][389]\n* News\n  * News\n  * [Newsroom][390]\n  * [NVIDIA Blog][391]\n  * [AI Podcast][392]\n  * [Stay Informed][393]\n* Events\n  * Events\n  * [Events Calendar][394]\n  * [Webinars][395]\n  * [NVIDIA GTC][396]\nThis site requires Javascript in order to view all its content. Please enable Javascript in order to\naccess all the functionality of this web site. Here are the [instructions how to enable JavaScript\nin your web browser.][397]\n\n## Jensen Huang\n\n### Co-founder, President and Chief Executive Officer\n\nJensen Huang founded NVIDIA in 1993 and has served since its inception as president, chief executive\nofficer, and a member of the board of directors. Prior to founding NVIDIA, Mr. Huang held a variety\nof positions from 1985 to 1993 at LSI Logic Corp., a computer chip manufacturer, and from 1984 to\n1985 at Advanced Micro Devices, Inc., a semiconductor company. In 2017, he was named Fortune’s\nBusinessperson of the Year. In 2019, Harvard Business Review ranked him No. 1 on its list of the\nworld’s 100 best-performing CEOs over the lifetime of their tenure. Mr. Huang holds a BSEE degree\nfrom Oregon State University and an MSEE degree",
    #             ', 2024). ["Nvidia CEO Jensen Huang says robots are the next\n     wave of AI — and 2 kinds will dominate"][1826]. Business Insider. [Archived][1827] from the\n     original on June 15, 2024.\n256. **[^][1828]** Coleman, Julie (March 20, 2024). ["Nvidia CEO Jensen Huang explains why he\'s all\n     in on humanoid robotics"][1829]. CNBC. Retrieved June 19, 2024.\n257. **[^][1830]** Liszewski, Andrew (March 18, 2025). ["Nvidia says \'the age of generalist robotics\n     is here\'"][1831]. The Verge. [Archived][1832] from the original on April 4, 2025. Retrieved\n     April 7, 2025.\n258. **[^][1833]** Moore, Mike (March 18, 2025). ["\'The age of generalist robotics is here\' -\n     Nvidia\'s latest GROOT AI model just took us another step closer to fully humanoid\n     robots"][1834]. TechRadar. [Archived][1835] from the original on April 9, 2025. Retrieved April\n     7, 2025.\n259. **[^][1836]** ["How Nvidia\'s Simulation Tech is Used',
    #             "───┼─────────────────┼─────\nUnited States  │61.3             │46.9%\n───────────────┼─────────────────┼─────\nSingapore      │23.7             │18.2%\n───────────────┼─────────────────┼─────\nTaiwan         │20.6             │15.8%\n───────────────┼─────────────────┼─────\nChina          │17.1             │13.1%\n───────────────┼─────────────────┼─────\nOther countries│7.9              │6.0% \n───────────────┴─────────────────┴─────\n\n### Leadership\n\n[[edit][659]]\n\nNvidia's key management as of March 2024 consists of:^{[[189]][660]}\n* [Jensen Huang][661], founder, president and chief executive officer\n* [Chris Malachowsky][662], founder and Nvidia fellow\n* [Colette Kress][663], executive vice",
    #             " where Nvidia's three co-founders\nagreed to start the company in late 1992 Nvidia's former headquarters which was home to the company\nthrough most of its pre-AI period (still in use) Aerial view of Endeavor, the first of the two new\nNvidia headquarters buildings, in [Santa Clara, California][247], in 2017 Entrance of Endeavor\nheadquarters building in 2018\n\nNvidia was founded on April 5, 1993,^{[[21]][248]}^{[[22]][249]}^{[[23]][250]} by Jensen Huang, a\n[Taiwanese-American][251] electrical engineer who was previously the director of CoreWare at [LSI\nLogic][252] and a [microprocessor][253] designer at [AMD][254]; [Chris Malachowsky][255], an\nengineer who worked at [Sun Microsystems][256]; and [Curtis Priem][257], who was previously a senior\nstaff engineer and graphics chip designer at [IBM][258] and Sun\nMicrosystems.^{[[24]][259]}^{[[25]][260]} In late 1992, the three men agreed to start the company in\na meeting at a [Denny's][261] roadside diner on Berryessa Road in East [San\nJose][262].^{[[26]][263]}^{[[27]][264]}^{[[28]][265",
    #         ],
    #         [
    #             " we present long-term analysis, their fundamental ratings and make\n## comparative short-term technical analysis which are presented below. The conclusion is AMD is a\n## Buy and NVDA is a Buy.\n\nInteract to see\nAdvertisement\nCOMPARISON\nComparison\nJun 26, 2026\nStock price -- (**AMD**: $532.57 vs. **NVDA**: $195.74)\nBrand notoriety: **AMD** and **NVDA** are both notable\nBoth companies represent the Semiconductors industry\nCurrent volume relative to the 65-day Moving Average: **AMD**: 70% vs. **NVDA**: 91%\nMarket capitalization -- **AMD**: $868.41B vs. **NVDA**: $4.74T\nAMD [@Semiconductors] is valued at $868.41B. NVDA’s [@Semiconductors] market capitalization is\n$4.74T. The market cap for tickers in the [@Semiconductors] industry ranges from $4.74T to $0. The\naverage market capitalization across the [@Semiconductors] industry is $194.88B.\n\n### Long-Term Analysis\n\nIt is best to consider a long-term outlook for a ticker by using Fundamental Analysis (FA) ratings.\nThe rating of 1 to 100",
    #             "iscal 2027 (ended April 26), revenue grew 85% from year-ago levels. AMD's fiscal quarters follow\nthe calendar year, so it is not a perfect comparison. Its revenue increased by a robust but\nsignificantly slower 38%.\n\nInvestors should also remember that the data center segment, which includes AI accelerators,\naccounts for 92% of overall revenue, compared to 56% for AMD. However, AMD's market capitalization\nis $850 billion, a small fraction of Nvidia's $4.9 trillion, causing possible headwinds from the law\nof large numbers.\n\nStory Continues\n\nEven if one considers [Nvidia stock a buy][214], investors should remember that it is the only\ncompany to grow to a market cap above $4.9 trillion, and a tenfold increase in the stock price would\ntake the market cap to $49 trillion! That may be difficult to imagine in a world where no company\nhas yet reached $6 trillion in market capitalization.\n\nIn comparison, a tenfold increase in AMD takes the market cap to $8.5 trillion. AMD's current size\nwill also make it somewhat difficult for its stock to rise by that much, though the market is on\ntrack to set new market cap records as more investors turn to these larger AI stocks.\n\nAdmittedly, neither stock is subject to a formal limit on market cap growth. Still, that difference\nstrongly",
    #             ": Rel. Strength: 98\nRelative Valuation: Rel. Valuation: 15\nRelative Profitability: Rel. Profitability: 80\n\nNVDA Sector: Technology Industry: Semiconductors\n\nAMD Sector: Technology Industry: Semiconductors\n\n## Key stats\n\nWhat are the main financial stats of NVDA vs AMD\nMarket\nName\nCurrent\nShares outstanding\n24.2B\n1.63B\nMarket cap\n$4.74T\n$868.41B\nEnterprise value\n$4.74T\n$866.69B\nEarnings\nName\nCurrent\nRevenue\n$253.49B\n$37.45B\nGross profit\n$187.95B\n$18.83B\nOperating income\n$162.29B\n$4.36B\nNet income\n$159.61B\n$5.01B\nEBIT\n$189.74B\n$5.17B\nEBITDA\n$192.97B\n$8.19B\nFree cash flow\n$119.08B\n$8.54B\nPer share\nName\nCurrent\nEPS\n$6.56\n$3.08\nEPS diluted\n$6.53\n$3.05\nFree cash flow per share\n$4.9\n$5.23\nBook value per share\n$8.04",
    #             "\n* [PEG Ratio][30]\n* [Earnings][31]\n* [Dividend][32]\n* [Charting][33]\n\n# NVDA Nvidia vs AMD Advanced Micro Devices\n\nNVDA\n\nAMD\n\nPrice (delayed)\n\n$195.74\n\n$532.57\n\nMarket cap\n\n$4.74T\n\n$868.41B\n\nP/E Ratio\n\n29.84\n\n172.91\n\nNVDA\n\nAMD\n\nDividend/share\n\n$0.28\n\nN/A\n\nEPS\n\n$6.56\n\n$3.08\n\nEnterprise value\n\n$4.74T\n\n$866.69B\n\n### NVDA Score\n\n[?][34]\nRelative Growth: Rel. Growth: 100\nRelative Strength: Rel. Strength: 55\nRelative Valuation: Rel. Valuation: 51\nRelative Profitability: Rel. Profitability: 99\n\n### AMD Score\n\n[?][35]\nRelative Growth: Rel. Growth: 99\nRelative Strength: Rel. Strength: 98\nRelative Valuation: Rel. Valuation: 15\nRelative Profitability: Rel. Profitability: 80\n\nNVDA Sector: Technology Industry: Semiconductors\n\nAMD Sector: Technology Industry",
    #             "5.087). NVDA has higher\nannual earnings (EBITDA): 193B vs. AMD (8.09B). AMD has less debt than NVDA: AMD (3.87B) vs NVDA\n(12.3B). NVDA has higher revenues than AMD: NVDA (253B) vs AMD (37.5B).\n\n──────────────┬───────┬─────┬──────────\n              │AMD    │NVDA │AMD / NVDA\n──────────────┼───────┼─────┼──────────\nCapitalization│868B   │4.74T│18%       \n──────────────┼───────┼─────┼──────────\nEBITDA        │8.09B  │193B │4%        \n──────────────┼───────┼─────┼──────────\nGain YTD      │148.679│5.087│2,923%    \n──────────────┼───────┼─────┼──────────\nP/E Ratio     │183.88 │31.95│575%  ",
    #         ],
    #         [
    #             "equity markets. NVIDIA, the company that made AI infrastructure possible, is the worst-performing\nmajor semiconductor stock this year.\n\nThe explanation is partly valuation math. NVDA entered 2026 already priced at a significant premium\nto its peers after its extraordinary 2023 and 2024 returns. When the broader sector re-rated upward\non AI enthusiasm, NVDA had less room to appreciate because the market had already given it credit\nfor a lot of its future growth. This is not a red flag; it is a feature of any stock that has run as\nfar and fast as NVIDIA has.\n\n## **The Dual-50 Shift: NVIDIA’s Customer Base Is Getting More Diversified**\n\nTwelve months ago, the most serious structural concern around NVIDIA was customer concentration. A\nhandful of hyperscalers, primarily [Microsoft][20], [Google][21], [Amazon][22], and [Meta][23],\naccounted for more than half of NVIDIA's data center revenue. The fear was straightforward: if any\none of those companies paused its AI infrastructure build-out, NVIDIA would feel it in the next\nquarter's numbers.\n\nThat risk profile has changed in a way that is not being discussed enough.\n\nIn Q1 FY2027, NVIDIA disclosed that hyperscaler revenue settled at approximately 50% of total data\ncenter revenue. The other 50% came from what NVIDIA now calls ACIE: AI",
    #             ' the forces pulling NVIDIA lower, whether they represent a structural threat or a\ntemporary headwind, how the current valuation compares to history, and what a realistic range of\noutcomes looks like from here.\n\n## **Why NVIDIA Stock Fell After Record Q1 FY2027 Earnings**\n\nThe numbers from Q1 FY2027 are not ambiguous. Revenue hit $81.6 billion, up 85% year-over-year. Data\ncenter revenue reached $75.2 billion, up 92%. Non-GAAP earnings per share came in at $1.87 against a\n[Wall Street][14] consensus of $1.76. Free cash flow for the quarter was $48.6 billion. These are\nnot metrics from a company in trouble.\n\nAnd yet NVIDIA\'s stock fell on every single trading day between May 20, when the results were\nannounced, and May 27. Then on June 5, a broader sector sell-off knocked it further.\n\nThis is not unusual behavior for NVIDIA. The phenomenon has a name in market analysis: "buy the\nrumor, sell the news." NVDA had already risen roughly 13.7% from its February earnings through its\nMay 14 all-time high. By the time the actual results confirmed what the market had already priced\nin, there was limited marginal buying left. The bar had been raised so high that even a genuine beat\nregistered as "expected."\n\nWhat',
    #             "[[INDmoney Logo]][1]\n\nINDmoney\n\n[US Stocks][2][Indian Stocks][3][ETFs][4][Mutual Funds][5][IPO][6][Features][7][Pricing][8]\nLogin\nLoginOpen Demat Account\n1. [Home][9]>\n2. [Blog][10]>\n3. [US Stocks][11]>\n4. NVIDIA Stock Down 17%. Is The AI Leader Still Expensive Or Undervalued?\n\n# NVIDIA Stock Down 17%. Is The AI Leader Still Expensive Or Undervalued?\n\n[Aadi Bihani Image]\n[\n\nAadi Bihani\n\n][12]\nLast updated:June 26, 2026\n12 min read\n[Why Is Nvidia Stock Falling?]\nTable Of Contents\n* Why NVIDIA Stock Fell After Record Q1 FY2027 Earnings\n* Why Is NVIDIA Stock Down 17%? Key Reasons Explained\n* The Dual-50 Shift: NVIDIA’s Customer Base Is Getting More Diversified\n* NVIDIA Q1 FY2027 Results: Key Numbers Investors Should Know\n* NVIDIA Analyst Price Targets: What Wall Street Expects\n* A Quick Valuation Check: Is NVIDIA Stock Cheap? NVDA Forward P/E And PEG Ratio Explained\n* NVIDIA Bear Case: Risks That Could Push The Stock Lower\n* Our Take: What NVIDIA",
    #             ' Wall Street Expects\n* A Quick Valuation Check: Is NVIDIA Stock Cheap? NVDA Forward P/E And PEG Ratio Explained\n* NVIDIA Bear Case: Risks That Could Push The Stock Lower\n* Our Take: What NVIDIA’s Stock Fall Means For Investors\n\nOn May 14, 2026, [NVIDIA stock][13] touched $236.54, a new all-time high. Six weeks later, it sits\nat $195.74, down roughly 17%, despite the company reporting what may be the most staggering single\nquarter in semiconductor history: $81.6 billion in revenue, a 25-fold dividend increase, and an $80\nbillion buyback authorization. \n\nJensen Huang himself called the stock\'s decline a "mystery" in public. That word choice says a lot.\nA CEO who built one of the most valuable companies on earth does not usually reach for "mystery"\nwhen he is calm about the situation. This article unpacks what is actually happening, why good news\nis not lifting NVIDIA right now, and whether the math at $195 supports the argument that this dip is\nworth paying attention to.\n\nLet\'s break down the forces pulling NVIDIA lower, whether they represent a structural threat or a\ntemporary headwind, how the current valuation compares to history, and what a realistic range of\noutcomes looks like from here.\n\n## **Why NVIDIA Stock Fell',
    #             " is wrong. At 22x forward\nearnings with a PEG under 0.5, NVIDIA is the cheaper company by any growth-adjusted measure.\n\nOne important caveat. The PEG ratio assumes the high earnings growth rate continues. That assumption\nis what the China headwinds, the compute price decline, and the export control tightening are all\nputting pressure on. This framework supports the dip thesis mathematically, but it does not\neliminate the downside scenario.\n\n## **NVIDIA Bear Case: Risks That Could Push The Stock Lower**\n\nEvery strong bull thesis needs a credible counter-scenario. Here is when the current analysis breaks\ndown:\n* If B200 GPU rental prices on cloud platforms continue declining toward $3 or below per hour in H2\n  2026, it would suggest AI model training activity has genuinely plateaued rather than paused for\n  the Vera Rubin upgrade cycle. That would directly challenge the revenue trajectory that underpins\n  NVIDIA's $91 billion Q2 guidance.\n* If one or more major hyperscalers, Microsoft, Google, Amazon, or Meta, meaningfully reduces AI\n  infrastructure capex in its next earnings report, the non-China demand story weakens. All four\n  have guided substantial AI spending increases. Any reversal would likely be felt in NVIDIA's next\n  results.\n* If the Vera Rubin production ramp encounters supply issues similar to the early Blackwell delays\n ",
    #         ],
    #         [
    #             "\n32°\n28°\nAverage\n31°\n26°\nLast Year\n31°\n26°\n\n## Current Weather Conditions in Mumbai\n\nMumbai, Maharashtra is currently Rain with a temperature of 31°. The current [RealFeel®\nTemperature][57] is 33° (Hot) based on observed conditions including 67% humidity, 76% cloud cover,\na UV index of 1, and winds of 24 km/h. In shade, the RealFeel Shade™ is 33°. AccuWeather's\n[MinuteCast®][58] provides minute-by-minute precipitation forecasts for Mumbai, powered by\nforecasting systems and proprietary technology developed by AccuWeather meteorologists since 1962.\n\n## Further Ahead\n\n[\n\n### Hourly\n\n][59] [\n\n### 10-Day\n\n][60] [\n\n### Monthly\n\n][61]\n\n## Around the Globe\n\n[\n\n### Hurricane Tracker\n\n][62] [\n\n### Severe Weather\n\n][63] [\n\n### Radar & Maps\n\n][64] [\n\n### News\n\n][65] [\n\n### Video\n\n][66]\n\nTop Stories\n\n[\n\nWeather News\n\nVenezuela death toll rises after back-to-back earthquakes\n\n15 hours ago\n\n][67] [\n\nWeather News\n\nFrance has hottest day ever recorded",
    #             "\n[sunny]Sunny\n\n[Humidity Icon]16%\n\n[UV Icon]1\n\n11.9 kmph /  W\n\n][108][\n\n9.\n\n[India flag]Jaunpur, Uttar Pradesh\n\n46 °C\n\n[sunny]Sunny\n\n[Humidity Icon]16%\n\n[UV Icon]1\n\n11.9 kmph /  W\n\n][109][\n\n10.\n\n[India flag]Rampur, Uttar Pradesh\n\n45 °C\n\n[sunny]Sunny\n\n[Humidity Icon]15%\n\n[UV Icon]1.9\n\n26.6 kmph /  W\n\n][110]\n\nLast Updated: 26 Jun 2026, 11:42 AM\n\n[City Weather Rankings][111]\n\n## FAQs About Mumbai Weather Conditions\n\n1. What is the current temperature in Mumbai today - 26 June 2026?\n\nThe current temperature in Mumbai is 31°C, with a feels-like temperature of 38°C. Today's forecast\nshows a high of 30°C and a low of 28°C, giving a 2 degree swing through the day as of 05:00 PM 26\nJune 2026.\n\n2. What are the current weather parameters in Mumbai today?\n\nHere is a complete overview of Mumbai's current",
    #             "�30\n────────────────┼──\n2026-06-27 16:00│30\n────────────────┴──\n\nLight rain may develop in some areas.\n\nLast Updated: 2026-06-26 17:00 (Local Time)\n\n## **Mumbai** Weather Parameters\n\n[Get AQI App][19]\nSW\n\n231°SW\n\nDirection\n\n18.4km/h\n\nWind Speed\n\n[Gust Speed Image]\n\nGust Speed\n\n7.08m/s\n\nCloud Cover\n\n75%\n\nVisibility\n\n3.5km\n\nPrecipitation\n\nPrecipitation\n\n0.07mm\n\nCurrent precipitation chances sit at 0.07mm\n\n996\n997\n998\n999\n1000\n1001\n1002\n1003\n1004\n1005\n1006\n1007\n\nPressure\n\n1005mb\n\nCurrent pressure level is a 1005mb.\n\nUV Index\n\n1.1Low\n\nThe present UV index is 1.1, consider suggestions for the same!\n\n## **Mumbai** 10-Days Weather Forecast\n\n[See Monthly][20]\n\nToday\n\n[moderate rain]\n29 °C\n73%\n\nSaturday\n\n[moderate rain]\n29 °C\n75%\n\nSunday\n\n",
    #             " of 28°C, giving a 2 degree swing through the day as of 05:00 PM 26\nJune 2026.\n\n2. What are the current weather parameters in Mumbai today?\n\nHere is a complete overview of Mumbai's current weather conditions as of 05:00 PM 26 June 2026:\nCondition : Mist Temperature : 31°C Hot (Feels like 38°C) Humidity : 71% Wind Speed : 18.4 km/h from\nSW (231 degrees), Gust 7.1 m/s UV Index : 1.1 (Low Air Pressure : 1005 mb Cloud Cover : 75%\nVisibility : 3.5 km Precipitation : 0.07 mm (74% chance of rain)\n\n3. What is the 10-day weather forecast for Mumbai from 05:00 PM 26 June 2026?\n\nFrom 05:00 PM 26 June 2026, Mumbai's 10-day forecast shows this trend: Today (Jun. 26) : Temp. 29°C,\nHum. 73% and Rain condition. Saturday (Jun. 27) : Temp. 29°C, Hum. 75% and Rain condition. Sunday\n(Jun. 28) : Temp. 29°C, Hum. 76% and Rain condition. Monday (Jun. 29) : Temp. 29°C, Hum. 76% and\nRain condition. Tuesday (Jun. 30) : Temp. 28°C, Hum.",
    #             "in/mumbai/204842/hourly-weather-forecast/204842\n[42]: /en/in/mumbai/204842/10-day-weather-forecast/204842\n[43]: /en/in/mumbai/204842/weather-radar/204842\n[44]: /en/in/mumbai/204842/minute-weather-forecast/204842\n[45]: /en/in/mumbai/204842/june-weather/204842\n[46]: /en/in/mumbai/204842/air-quality-index/204842\n[47]: /en/in/mumbai/204842/health-activities/204842\n[48]: /en/in/mumbai/204842/weather-today/204842\n[49]: /en/in/mumbai/204842/current-weather/204842\n[50]: /en/in/mumbai/204842/weather-radar/204842\n[51]: /en/in/mumbai/204842/satellite/204842\n[52]: /en/in/maharashtra/current-weather-maps\n[53]: /en/in/mumbai/204842/hourly-",
    #         ],
    #     ],
    #     "ground_truth": [
    #         (
    #             "As of June 26, 2026, NVIDIA Corp (NASDAQ: NVDA) is trading at approximately "
    #             "$194.80 USD intraday, down from a previous close of $198.91. "
    #             "Today's range is $192.13 - $200.79. Market cap is approximately $4.72 trillion. "
    #             "52-week range is $151.49 - $236.54. Next earnings date is August 26, 2026."
    #         ),
    #         (
    #             "Jensen Huang (Jen-Hsun Huang) is the CEO, President, and co-founder of NVIDIA Corporation. "
    #             "He has held the role since co-founding the company in April 1993. "
    #             "As of 2026, Forbes estimates his net worth at over $200 billion, "
    #             "making him one of the wealthiest individuals in the world."
    #         ),
    #         (
    #             "As of June 26, 2026, NVIDIA (NVDA) has a market cap of approximately $4.74 trillion USD, "
    #             "remaining the world's most valuable company, ahead of Apple and Alphabet. "
    #             "AMD (Advanced Micro Devices) has a market cap of approximately $760 billion USD as of early June 2026. "
    #             "NVIDIA's market cap is roughly 6.2x larger than AMD's. "
    #             "NVIDIA's market cap has grown ~50% year-over-year. "
    #             "Both companies compete in AI data center chips, but NVIDIA dominates with "
    #             "71% gross margins and $253B annual revenue vs AMD's $37.5B."
    #         ),
    #         (
    #             "As of June 26, 2026, the following major announcements have impacted NVIDIA's stock price: "
    #             "POSITIVE — Computex (June 1, 2026): Jensen Huang announced RTX Spark (NVIDIA's first "
    #             "Windows SoC with MediaTek, targeting laptops) and confirmed Vera Rubin full production "
    #             "with customers including OpenAI, Anthropic, xAI, and CoreWeave. Stock gained 6.3% that day. "
    #             "POSITIVE — Q1 FY2027 Earnings (May 22, 2026): Data center revenue of $75.2B, up 92% YoY. "
    #             "Record $20B capital return to shareholders. Additional $80B share repurchase authorized. "
    #             "Free cash flow of $48.6B. Q2 guidance of $91B revenue (+95% YoY). "
    #             "POSITIVE — Annual Shareholder Meeting (June 24, 2026): Focus on Blackwell and Vera Rubin "
    #             "production ramp. Jensen Huang projected Blackwell and Rubin to generate $1 trillion in "
    #             "revenue across 2026 and 2027. "
    #             "POSITIVE — Partnerships (June 2026): SK Hynix multiyear memory partnership announced. "
    #             "NAVER expanding AI infrastructure with NVIDIA. Europe unveiled 35 new NVIDIA AI supercomputers. "
    #             "Nebius Group committed $2.275B to Vera Rubin-powered facilities in the UK. "
    #             "NEGATIVE — Export controls and regulatory risk: Senator Elizabeth Warren invited Jensen Huang "
    #             "to testify before the Senate Banking Committee on China business and export compliance. "
    #             "Stock fell 6.5% on June 15. "
    #             "NEGATIVE — Competition: Qualcomm launched new AI data center chips, increasing competitive pressure. "
    #             "As of June 26, NVDA trades at ~$194, down from a 52-week high of $236.54 in May 2026. "
    #             "Wall Street consensus price target is $295, implying ~42% upside. "
    #             "Next earnings date: August 26, 2026."
    #         ),
    #         (
    #             "As of June 26, 2026, Mumbai is mostly cloudy with a current temperature of "
    #             "approximately 86.6°F (30.3°C). Today's high is expected to reach 88.3°F with a 35% "
    #             "chance of rain. Mumbai is in active monsoon season — rain probability increases "
    #             "through the week: 50% Saturday, 60% Sunday, 70% Monday, 65% Tuesday."
    #         ),
    #     ],
    # }
    return


@app.cell
def _():
    # from ragas.llms import llm_factory

    # llm_factory
    return


@app.cell
def _():
    # client = OpenAI(
    #     api_key="ollama",  # Ollama doesn't require a real key
    #     base_url="http://localhost:11434/v1",
    # )
    # local_llm_judge = llm_factory(
    #     "gemma4:e4b",
    #     provider="openai",
    #     client=client,
    #     mode=instructor.Mode.JSON,
    # )
    return


@app.cell
def _():
    # results
    return


@app.cell
def _():
    # df = results.to_pandas()
    # df.loc[
    #     :,
    #     [
    #         "user_input",
    #         "faithfulness",
    #         # "answer_relevancy",
    #         "context_precision",
    #         "context_recall",
    #     ],
    # ]
    return


if __name__ == "__main__":
    app.run()
