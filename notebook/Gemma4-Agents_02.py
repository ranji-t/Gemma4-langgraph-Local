import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    # Standard Imports
    import json
    import operator
    from dataclasses import dataclass, field
    from typing import TypedDict, Literal, Self, Sequence, Annotated

    # Third party Imports
    import marimo as mo
    from openai import OpenAI
    from ollama import chat, generate
    from langgraph.graph import START, END
    from langgraph.graph.state import StateGraph

    return (
        Annotated,
        Literal,
        OpenAI,
        Self,
        Sequence,
        StateGraph,
        TypedDict,
        chat,
        dataclass,
        field,
        json,
        mo,
        operator,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **LLM Functions**
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
def _(Message):
    # Tresting Repr
    Message(role="system", content="You are good Boy")
    return


@app.cell
def _(Message):
    # The Message
    Message(role="system", content="You are good Boy").text
    return


@app.cell
def _():
    # Models
    gemma_lite = "gemma4:e2b"
    gemma_mid = "gemma4:e4b"
    return (gemma_lite,)


@app.cell
def _(OpenAI):
    # Creat a Open AI clinet
    client = OpenAI(
        api_key="ollama",  # Required but ignored by ollama
        base_url="http://localhost:11434/v1",
    )
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
    # **The Agent Graph**
    """)
    return


@app.cell
def _(Annotated, Message, Sequence, TypedDict, operator):
    # Build the app
    class AgentState01(TypedDict):
        messages: Annotated[Sequence[Message], operator]
        web_query: None | str
        search_web: bool

    return (AgentState01,)


@app.cell
def _(AgentState01, Message, Sequence, json, use_web):
    def web_use_node(state: AgentState01) -> AgentState01:
        # Message extraction
        messages: Sequence[Message] = state["messages"]

        # Extract the last query from the node
        query: str = messages[-1].content

        # Does the node need
        resonse = use_web(user_query=query)
        json_return = json.load(resonse.message.content)

        # message
        return {"use_web": json_return["use_web"]}

    return (web_use_node,)


@app.cell
def _(AgentState01, StateGraph, web_use_node):
    # The State of the
    builder01 = StateGraph(AgentState01)

    # Build Node & Edges of the app
    builder01.add_node("Web Search?", web_use_node)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
