import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    # Standard Imports
    import json
    import logging
    import operator
    from itertools import chain
    from dataclasses import dataclass, field
    from typing import TypedDict, Literal, Self, Sequence, Annotated, NamedTuple

    # Third party Imports
    import marimo as mo
    import numpy as np
    from rank_bm25 import BM25Okapi
    from openai import OpenAI
    from ollama import chat, generate
    from ddgs.ddgs import DDGS, DDGSException
    from langgraph.graph import START, END
    from langgraph.graph.state import StateGraph
    from sentence_transformers import SentenceTransformer, SparseEncoder
    from langchain_core.runnables.graph import MermaidDrawMethod
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
    )

    return (
        Annotated,
        BM25Okapi,
        DDGS,
        DDGSException,
        END,
        Literal,
        RecursiveCharacterTextSplitter,
        START,
        Self,
        SentenceTransformer,
        Sequence,
        StateGraph,
        TokenTextSplitter,
        TypedDict,
        chain,
        chat,
        dataclass,
        field,
        json,
        mo,
        np,
        operator,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **LLM Functions**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Message Layout**
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
    gemma_mid = "gemma4:e4b"
    return (gemma_lite,)


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
def _(use_web):
    # result = use_web("What is the Nvidia Stock prices today?")
    result = use_web(
        "What are the questions asked in coupa interviews for data scientist?"
    )
    return (result,)


@app.cell
def _(json, result):
    # Convert the string to JSON
    json.loads(result.message.content)
    return


@app.cell
def _(mo, result):
    # The Thinking Source
    mo.md(result.message.thinking)
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
def _(Message, summarizer):
    # This the test message
    test_messages_01 = [
        Message("user", "Who are you?"),
        Message("assistant", "Hi I am Gemini?"),
        Message("user", "I have an interview lined up with Coupa."),
        Message("assistant", "Congratulation!!!! How can I help you?"),
        Message(
            "user",
            "I want to know what kind of questions will be asked in the first round of Data Scientist interview",
        ),
        Message(
            "assistant",
            """
    After that recruiter screen, there may be an online technical assessment covering coding problems and data structures, typically 3 questions within 45–60 minutes. GeeksforGeeks

    Round 1 — Technical Questions Actually Asked
    ML Concepts

    You are tasked with building a decision tree to predict if a borrower will repay a loan — how would you evaluate if it's the right choice, and how would you assess performance before and after deployment? PR Newswire
    Explain how a random forest generates its ensemble of trees, and why you might choose it over logistic regression. PR Newswire
    Compare bagging vs boosting — describe scenarios where you'd prefer one over the other and discuss the tradeoffs. PR Newswire
    Your manager asks you to build a neural network model — how would you justify its complexity and explain its predictions to non-technical stakeholders? PR Newswire
    Calculating the number of trainable parameters for a CNN. ArcWeb

    Data Engineering / SQL

    ETL pipeline design questions, Snowflake data warehouse tools, PySpark coding, and data transformation. iMocha

    Coding / DSA

    DSA questions at easy to medium LeetCode level — basic LeetCode 75 is sufficient. Live coding or pseudo code expected. iMocha
    """,
        ),
        Message("user", "I am nervous!"),
    ]

    result2 = summarizer(messages=test_messages_01)
    return (result2,)


@app.cell
def _(mo, result2):
    # Result
    mo.md(result2.message.content)
    return


@app.cell
def _(mo, result2):
    # The Thinking Source
    mo.md(result2.message.thinking)
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
def _(result2, update_query):
    result3 = update_query(
        result2.message.content,
        "Where this companies office situated?",
    )
    return (result3,)


@app.cell
def _(mo, result3):
    #  Show the updated query
    mo.md(result3.message.content)
    return


@app.cell
def _(mo, result3):
    # Show the Thinking
    mo.md(result3.message.thinking)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **SetUp Actions On App Start up**
    """)
    return


@app.cell
def _(SentenceTransformer):
    # Create Encoder
    encoder_model = SentenceTransformer(
        # model_name_or_path="all-mpnet-base-v2",
        model_name_or_path="nomic-ai/nomic-embed-text-v1.5",
        cache_folder=r"D:\Codebase\Gemma4-Test\model",
        local_files_only=True,
    )
    return (encoder_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Web Search**
    """)
    return


@app.cell
def _(
    DDGS,
    DDGSException,
    RecursiveCharacterTextSplitter,
    Sequence,
    TokenTextSplitter,
    chain,
):
    def perform_web_search(query: str, max_results: int = 5) -> Sequence[str]:
        # Show web results
        web_results = []

        # Search the web
        with DDGS() as d:
            # Get the urls from the search
            for url_dict in d.text(query=query, max_results=max_results):
                try:
                    # Get url form this
                    url = url_dict.get("href")
                    # Extract the data
                    data = d.extract(url, fmt="text_markdown").get("content")
                    # Add data to container
                    web_results.append(data)

                except DDGSException as e:
                    print(f"DDGSException: {e}")

        # The Recursice Character text split
        rcts = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )
        tts = TokenTextSplitter(chunk_size=300, chunk_overlap=50)
        # The Splitter splits web results
        split_text = chain.from_iterable(
            tts.split_text(txt) for txt in web_results
        )

        # Return the data
        return list(split_text)

    return (perform_web_search,)


@app.cell
def _(BM25Okapi, Sequence, encoder_model, np):
    def context_scorer(query: str, docs: Sequence[str]):
        # Create Document Encoding
        doc_encoding = encoder_model.encode(
            inputs=[f"search_document: {sent}" for sent in docs],
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Create query Embedding
        query_encoding = encoder_model.encode_query(
            f"search_query: {query}",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Similarity
        sim_score = np.einsum("ne, e -> n", doc_encoding, query_encoding)

        # return
        return (sim_score, doc_encoding, query_encoding)


    def semantic_scorer(query: str, docs: Sequence[str]):
        # Tokenize the query
        tokenized_docs = [doc.lower().strip().split() for doc in docs]
        tokenized_query = query.lower().strip().split()

        # Build BM25 index
        bm25 = BM25Okapi(corpus=tokenized_docs)

        # Score all docs against query
        scores = bm25.get_scores(tokenized_query)

        # return
        return scores

    return context_scorer, semantic_scorer


@app.cell
def _(nd, np):
    def mmr(
        scores: nd.arrays,
        doc_encoding: nd.arrays,
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


@app.function
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
    return list(rrf_ranks)[:top_n], sorted(scores.values())


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
    # **The Agent Graph**
    """)
    return


@app.cell
def _(Annotated, Message, Sequence, TypedDict, operator):
    # Build the app
    class AgentState01(TypedDict):
        messages: Annotated[Sequence[Message], operator.add]
        summary: None | str
        prime_query: None | str
        use_web: bool
        web_search_result: Sequence[str]

    return (AgentState01,)


@app.cell
def _(
    AgentState01,
    Message,
    Sequence,
    json,
    summarizer,
    update_query,
    use_web,
):
    def summarizer_node(state: AgentState01) -> AgentState01:
        # Extrac messages
        messsages = state["messages"]
        # Pass it though the summarizer
        result = summarizer(messages=messsages)
        # Content summarized
        return {"summary": result.message.content}


    def smart_query_node(state: AgentState01) -> AgentState01:
        # Extract the query
        query = state["messages"][-1].content
        # Extract context
        context = state["summary"]
        # Return
        result = update_query(context=context, query=query)
        # Update the query
        return {"prime_query": result.message.content}


    def web_use_node(state: AgentState01) -> AgentState01:
        # Message extraction
        messages: Sequence[Message] = state["messages"]

        # Extract the last query from the node
        query: str = messages[-1].content

        # Does the node need
        resonse = use_web(user_query=query)
        json_return = json.loads(resonse.message.content)

        # message
        return {"use_web": json_return["use_web"]}

    return smart_query_node, summarizer_node, web_use_node


@app.cell
def _(AgentState01, context_scorer, mmr, perform_web_search, semantic_scorer):
    def web_search_node(state: AgentState01) -> AgentState01:
        # Extract Query from sate
        query = state["prime_query"]

        # Search The web
        search_results = perform_web_search(query)

        # Rank the documents Context Wise
        doc_context_score, doc_encoding, query_encoding = context_scorer(
            query, search_results
        )

        # Rank the documents Context Wise
        doc_semantic_score = semantic_scorer(query, search_results)

        # Get MMR IDs based on
        mmr_dense = mmr(doc_context_score, doc_encoding, score_threshold=0.5)
        # Get MMR IDs based on Sparsity
        mmr_sparse = mmr(
            doc_semantic_score / doc_semantic_score.max(),
            doc_encoding,
            score_threshold=0.5,
        )

        # Get the Final results
        final_docs_index, _ = rrf([mmr_dense, mmr_sparse], top_n=5)
        # Final Docs
        final_docs = [search_results[i] for i in final_docs_index]

        # Return the data
        return {"web_search_result": final_docs}

    return (web_search_node,)


@app.cell
def _(AgentState01, Message, chat, gemma_lite):
    def answering_node(state: AgentState01) -> AgentState01:
        # Get enghances query
        user_prompt = state["prime_query"]

        # If not None
        if state["web_search_result"] is not None:
            # Convert this to string
            consolidates_results = "\n-----\n".join(state["web_search_result"])
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


@app.cell
def _(Message, app01):
    # app01.invoke({"messages": [Message(role="user", content="Who are You?")]})
    app01.invoke(
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
    return


@app.cell
def _(mo):
    mo.md("""
    [assistant]: The provided search results offer several figures related to Nvidia's stock price:

    *   The current price of Nvidia Corporation stock is listed as **$878.08**.
    *   One section of the context states that the Nvidia stock price today is **$208.19**.
    *   Trading activity data shows an approximate price for the NVDA ticker was **Around $182–$185**.
    """)
    return


if __name__ == "__main__":
    app.run()
