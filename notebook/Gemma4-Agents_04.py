import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **01. Gemma4 Test Note 03**
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
    # Standard Imports
    import re
    import logging
    import operator
    from itertools import chain
    from functools import wraps
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
    )

    # Third party Imports
    import marimo as mo
    import numpy as np
    from ollama import chat
    from pydantic import BaseModel
    from ddgs.ddgs import DDGS, DDGSException
    from langgraph.graph import START, END
    from langgraph.graph.state import StateGraph
    from rank_bm25 import BM25Okapi
    from fastembed import TextEmbedding
    from langchain_text_splitters import TokenTextSplitter

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
        START,
        Sequence,
        StateGraph,
        TextEmbedding,
        TokenTextSplitter,
        TypeVar,
        TypedDict,
        chat,
        dataclass,
        field,
        logging,
        mo,
        np,
        operator,
        perf_counter,
        re,
        sleep,
        wraps,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **02. Start Up processes**
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
    # Logging Set Up
    logging.basicConfig(
        level=logging.INFO, filename=r"D:\Codebase\Gemma4-Test\logs\logs04.log"
    )
    # Get Loggers
    logger = logging.getLogger(name="notebook04")
    return (logger,)


@app.cell
def _(Any, Callable, logger, perf_counter, wraps):
    # the logging decorator
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
    # Create Encoder
    fe_encoder_model = TextEmbedding(
        model_name="nomic-ai/nomic-embed-text-v1.5-Q",
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
    # Compile the pattern
    BM25_TOKENIZER = re.compile(r"\b\w+\b")
    return (BM25_TOKENIZER,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **03. LLM Functions**
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
    @dataclass(frozen=True)
    class Message:
        role: Literal["system", "user", "assistant"]
        content: str
        text: str = field(init=False, repr=False)

        def __post_init__(self: Self) -> None:
            object.__setattr__(self, "text", f"[{self.role}]: {self.content}")

        def __str__(self: Self) -> str:
            return self.text

        def __eq__(self: Self, other: Message) -> bool:
            return (self.role == other.role) & (self.content == other.content)

    return (Message,)


@app.cell
def _():
    # Models
    gemma_lite = "gemma4:e2b"
    return (gemma_lite,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *3.2 Use Web Decion Function*
    """)
    return


@app.cell
def _(chat, gemma_lite):
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
        # Response from user
        result = chat(
            model=gemma_lite,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            format="json",
            think=True,
        )
        # retur ndata
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
def _(Message, chat, gemma_lite):
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
        result = chat(
            model=gemma_lite,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            format=None,
            think=True,
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
def _(chat, gemma_lite):
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
        result = chat(
            model=gemma_lite,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=None,
            think=True,
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
    # **04. Web Search, Embedding, BM25, MMR & RRF**
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
    logging_decorator,
):
    @logging_decorator
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
                        WebSearchResults(counter, None, url, title, snippet, data)
                    )

                except DDGSException as e:
                    print(f"func: perform_web_search, url extraction failed{e}")
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
    @logging_decorator
    def context_scorer_fe(query: str, docs: Sequence[str], batch_size: int = 32):
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
def _(logging_decorator, nd, np):
    @logging_decorator
    def mmr(
        scores: nd.ndarray,
        doc_encoding: nd.ndarray,
        top_k: int = 10,
        lambda_param: float = 0.5,
        score_threshold: float = -np.inf,
    ):
        # Get the number of samples
        n_docs = len(scores)

        # Tracking IDs
        selected_ids = []
        remaining_ids = list(range(n_docs))
        remaining_ids = [i for i in remaining_ids if (scores[i] > score_threshold)]
        red_n_docs = len(remaining_ids)

        if red_n_docs == 0:
            raise ValueError(
                "No docs selected deduce the threshold or chedk if you are sdingi in empty data"
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

            # Update the racking indices
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


@app.cell
def _(logging_decorator):
    @logging_decorator
    def rrf(ranked_lists: list[list[int]], k: int = 60, top_n: int = 10):
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

    return (rrf,)


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
    # **05. The Agent Graph**
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
    # The Class type
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
    # Build the app
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
def _(BaseModel):
    class UseWebTemplate(BaseModel):
        use_web: bool


    # Test the model output vlaidation with json
    UseWebTemplate.model_validate_json('{"use_web": true}').model_dump()
    return (UseWebTemplate,)


@app.cell
def _(
    AgentState01,
    Message,
    Sequence,
    UseWebTemplate,
    logging_decorator,
    summarizer,
    sync_retry,
    update_query,
    use_web,
):
    @sync_retry()
    @logging_decorator
    def summarizer_node(state: AgentState01) -> AgentState01:
        # Extrac messages
        messsages = state["messages"]
        # Pass it though the summarizer
        result = summarizer(messages=messsages)
        # Content summarized
        return {"summary": result.message.content}


    @sync_retry()
    @logging_decorator
    def smart_query_node(state: AgentState01) -> AgentState01:
        # Extract the query
        query = state["messages"][-1].content
        # Extract context
        context = state["summary"]
        # Return
        result = update_query(context=context, query=query)
        # Update the query
        return {"prime_query": result.message.content}


    @sync_retry()
    @logging_decorator
    def web_use_node(state: AgentState01) -> AgentState01:
        # Message extraction
        messages: Sequence[Message] = state["messages"]

        # Extract the last query from the node
        query: str = messages[-1].content

        # Does the node need web
        resonse = use_web(user_query=query)

        # Gather the JSON String result and parse it into dict
        json_response = UseWebTemplate.model_validate_json(
            resonse.message.content
        ).model_dump()

        # Structured Response
        return json_response

    return smart_query_node, summarizer_node, web_use_node


@app.cell
def _(
    AgentState01,
    RetrivedDocs,
    context_scorer_fe,
    logging_decorator,
    mmr,
    np,
    perform_web_search,
    rrf,
    semantic_scorer,
    sync_retry,
):
    @sync_retry()
    @logging_decorator
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

        # Return the data
        return {
            "web_search_result": search_results,
            "retrived_result": final_retrived_docs,
        }

    return (web_search_node,)


@app.cell
def _(AgentState01, Message, chat, gemma_lite, logging_decorator, sync_retry):
    @sync_retry()
    @logging_decorator
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
        result = chat(
            model=gemma_lite,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=None,
            think=True,
        )

        # Message
        return {
            "messages": [Message(role="assistant", content=result.message.content)]
        }

    return (answering_node,)


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
    # The State of the
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
    # Show Graph
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
    # app01.invoke({"messages": [Message(role="user", content="Who are You?")]})
    result4 = app01.invoke(
        {
            "messages": [
                Message(role="user", content="Who are You?"),
                Message(role="assistant", content="I am Gemma4 an LLM by google!"),
                Message(
                    role="user", content="What is the Nvidia Stock Price Today?"
                ),
            ]
        }
    )

    # Show reslults
    result4
    return (result4,)


@app.cell
def _(mo, result4):
    # Show Final Answer
    mo.md(result4["messages"][-1].content)
    return


if __name__ == "__main__":
    app.run()
