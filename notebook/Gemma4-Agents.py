import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ***1. Testing Ollama+Gemma4 Chat & Generate Capability***
    """)
    return


@app.cell
def _():
    # Standard Imports
    import pytz
    from datetime import datetime
    from typing import Callable

    # Third party Imports
    import marimo as mo
    from ddgs import DDGS
    from ollama import chat, generate, ChatResponse, GenerateResponse, Message

    return (
        Callable,
        ChatResponse,
        DDGS,
        GenerateResponse,
        Message,
        chat,
        datetime,
        generate,
        mo,
        pytz,
    )


@app.cell
def _():
    # Models
    gemma_light = "gemma4:e2b"
    gemma_mid = "gemma4:e4b"
    return (gemma_light,)


@app.cell
def _(GenerateResponse, gemma_light, generate):
    # Generator
    gen_res: GenerateResponse = generate(
        model=gemma_light,
        prompt="How can we prove Pythogoras theorem with triangle similarity?",
        think=True,
        stream=False,
    )
    return (gen_res,)


@app.cell
def _(gen_res: "GenerateResponse", mo):
    # Display the output
    mo.md(gen_res["response"])
    return


@app.cell
def _(gen_res: "GenerateResponse", mo):
    # Display the thinking
    mo.md(gen_res["thinking"])
    return


@app.cell
def _(ChatResponse, chat, gemma_light):
    # Generator
    chat_res: ChatResponse = chat(
        model=gemma_light,
        messages=[
            {
                "role": "system",
                "content": "You are a Senior Analyst and you answer to matematical, scientiefinc & Analytic questions only any other queres decline them politely.",
            },
            {
                "role": "user",
                "content": "How can we prove Pythogoras theorem with triangle similarity?",
            },
        ],
        think=True,
        stream=False,
    )
    return (chat_res,)


@app.cell
def _(chat_res: "ChatResponse", mo):
    # Display Chat Response
    mo.md(chat_res["message"]["content"])
    return


@app.cell
def _(chat_res: "ChatResponse", mo):
    # Display Chat Response
    mo.md(chat_res["message"]["thinking"])
    return


@app.cell
def _(chat_res: "ChatResponse"):
    chat_res
    return


@app.cell
def _(ChatResponse, chat, gemma_light):
    # Generator
    chat_res2: ChatResponse = chat(
        model=gemma_light,
        messages=[
            {
                "role": "system",
                "content": "You are a Senior Analyst and you answer to matematical, scientiefinc & Analytic questions only any other queres decline them politely.",
            },
            {
                "role": "user",
                "content": "How can we prove Pythogoras theorem with triangle similarity?",
            },
        ],
        think=True,
        stream=False,
    )
    return (chat_res2,)


@app.cell
def _(chat_res2: "ChatResponse", mo):
    # Display Chat Response
    mo.md(chat_res2["message"]["content"])
    return


@app.cell
def _(chat_res2: "ChatResponse", mo):
    # Display Chat Response
    mo.md(chat_res2["message"]["thinking"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #  ***2. Testing Tool Calling in Ollama+Gemma4***
    """)
    return


@app.cell
def _():
    sys_prompt2 = """
    You are a Senior Analyst.

    - Use tools for any computation or real-time data
    - Do NOT guess or approximate
    - If a tool exists, ALWAYS prefer it
    - Keep answers short and direct"""
    return (sys_prompt2,)


@app.cell
def _(Callable, Message, datetime, pytz):
    # Tools to be used
    def add_two_numbers(a: str, b: str) -> int | float:
        """Sums two numbers a & b"""
        return int(a) + int(b)

    def multiply_two_numbers(a: str, b: str) -> int | float:
        """Multiply two numbers a & b"""
        return int(a) * int(b)

    def call_tools(
        tool_call: Message.ToolCall, tool_dict: dict[str, Callable]
    ) -> dict[str:str]:
        # Get the func form the tool call
        func = tool_call.function

        # Get function name and params
        func_name = func.get("name")
        params = func.get("arguments")

        # get the actual function
        func_obj = tool_dict[func_name]

        # Get result
        result = func_obj(**params)

        # Return value
        return {
            "role": "tool",
            "name": func_name,
            "content": str(result),
        }

    def date_time_rightnow() -> str:
        """Get The date time now in IST as String. No arguments required."""
        # get the datetime of the present
        dt = datetime.now()

        # Adding a timezone for asia /kolkate
        timezone = pytz.timezone("Asia/Kolkata")

        # getting the timezone using localize method
        mydt = timezone.localize(dt)

        # The date as str
        return mydt.strftime("%Y-%m-%d %H:%M:%S %Z")

    # The Tool dict is genrated
    tools_dict = {
        func.__name__: func
        for func in [add_two_numbers, multiply_two_numbers, date_time_rightnow]
    }
    return call_tools, tools_dict


@app.cell
def _(ChatResponse, chat, gemma_light, sys_prompt2):
    # Generator
    chat_res3: ChatResponse = chat(
        model=gemma_light,
        messages=[
            {
                "role": "system",
                "content": sys_prompt2,
            },
            {
                "role": "user",
                "content": "What are the values of 2342324423472893234 + 32423422342343424 \n 2342324423472893234 * 32423422342343424",
            },
        ],
        think=True,
        stream=False,
    )
    return (chat_res3,)


@app.cell
def _(chat_res3: "ChatResponse", mo):
    mo.md(chat_res3["message"]["content"])
    return


@app.cell
def _():
    (
        2342324423472893234 + 32423422342343424,
        2342324423472893234 * 32423422342343424,
    )
    return


@app.cell
def _(ChatResponse, chat, gemma_light, sys_prompt2, tools_dict):
    # Messages
    messages = [
        {
            "role": "system",
            "content": sys_prompt2,
        },
        {
            "role": "user",
            "content": "Perfomr Addition and Multiplication on these two numbers 2342324423472893234 & 32423422342343424?",
        },
    ]

    # Generator
    chat_res4: ChatResponse = chat(
        model=gemma_light,
        messages=messages,
        think=False,
        stream=False,
        tools=list(tools_dict.values()),
    )
    return chat_res4, messages


@app.cell
def _(
    ChatResponse,
    call_tools,
    chat,
    chat_res4: "ChatResponse",
    gemma_light,
    messages,
    tools_dict,
):
    if tools_list := chat_res4["message"].get("tool_calls"):
        # get Results:
        tool_resluts = [call_tools(res, tool_dict=tools_dict) for res in tools_list]

        # Extended messages
        extened_messages = [*messages, chat_res4["message"], *tool_resluts]

        # Final results:
        final_result: ChatResponse = chat(
            model=gemma_light,
            messages=extened_messages,
            think=False,
            stream=False,
            tools=list(tools_dict.values()),
        )
    return final_result, tools_list


@app.cell
def _(tools_list):
    [_.function for _ in tools_list]
    return


@app.cell
def _(final_result: "ChatResponse", mo):
    mo.md(final_result["message"].get("content"))
    return


@app.cell
def _():
    (
        2342324423472893234 + 32423422342343424,
        2342324423472893234 * 32423422342343424,
    )
    return


@app.cell
def _():
    2342324423472893234.0 + 32423422342343424.0
    return


@app.cell
def _(ChatResponse, call_tools, chat, gemma_light, sys_prompt2, tools_dict):
    # Messages
    messages2 = [
        {
            "role": "system",
            "content": sys_prompt2,
        },
        {
            "role": "user",
            "content": "What is the time right now?",
        },
    ]

    # Generator
    chat_res5: ChatResponse = chat(
        model=gemma_light,
        messages=messages2,
        think=False,
        stream=False,
        tools=list(tools_dict.values()),
    )

    if tools_list2 := chat_res5["message"].get("tool_calls"):
        # get Results:
        tool_resluts2 = [call_tools(res, tool_dict=tools_dict) for res in tools_list2]

        # Extended messages
        extened_messages2 = [*messages2, chat_res5["message"], *tool_resluts2]

        # Final results:
        final_result2: ChatResponse = chat(
            model=gemma_light,
            messages=extened_messages2,
            think=False,
            stream=False,
            tools=list(tools_dict.values()),
        )
    return final_result2, tool_resluts2, tools_list2


@app.cell
def _(final_result2: "ChatResponse", mo):
    # Display the data
    mo.md(final_result2["message"].get("content"))
    return


@app.cell
def _(tool_resluts2):
    tool_resluts2
    return


@app.cell
def _(tools_list2):
    tools_list2
    return


@app.cell
def _(DDGS):
    results = DDGS().text(query="Python classes?", max_results=5)
    results
    return (results,)


@app.cell
def _(results):
    print(results[1]["body"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Using LangSmith Evluation**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *Creating / Reading  Dataset & Creating / listing Exmaples*
    """)
    return


@app.cell
def _():
    # Standrd Imports

    # Third Party Imports
    from dotenv import load_dotenv
    from langsmith import Client

    return Client, load_dotenv


@app.cell
def _(load_dotenv):
    # Load dotenv file
    load_dotenv()
    return


@app.cell
def _(Client):
    # Create Clinet
    client = Client()
    # Name the data set
    dataset_name = "Simple ChatBot QnA Dataset"
    # Create Dataset / Read Dataset
    # dataset = client.create_dataset(dataset_name=dataset_name)
    dataset = client.read_dataset(dataset_name=dataset_name)
    return client, dataset_name


@app.cell
def _():
    # Tere DataSet ID is stored
    dataset_id = "dd31d4ce-a954-4682-8b63-19b8b36d55e9"
    return (dataset_id,)


@app.cell
def _():
    # # My examples
    # examples = [
    #     {
    #         "inputs": {"question": "What is LangGraph?"},
    #         "outputs": {
    #             "answer": "LangGraph is a framework for building stateful, multi-step workflows with LLMs using graph-based execution."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "How is LangGraph different from LangChain?"},
    #         "outputs": {
    #             "answer": "LangGraph focuses on stateful, graph-based workflows with cycles and control flow, while LangChain primarily provides linear chains and utilities for LLM applications."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "What problem does LangGraph solve?"},
    #         "outputs": {
    #             "answer": "LangGraph solves the problem of managing complex, stateful LLM workflows that require branching, loops, and memory across multiple steps."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "What is a node in LangGraph?"},
    #         "outputs": {
    #             "answer": "A node in LangGraph represents a unit of computation, typically a function that processes input state and returns updated state."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "What is an edge in LangGraph?"},
    #         "outputs": {
    #             "answer": "An edge defines the transition between nodes in the graph and determines how execution flows from one node to another."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "What is state in LangGraph?"},
    #         "outputs": {
    #             "answer": "State in LangGraph is a shared data structure that is passed between nodes and updated as the workflow executes."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "Can LangGraph handle loops?"},
    #         "outputs": {
    #             "answer": "Yes, LangGraph supports cyclic graphs, allowing workflows to include loops and iterative processes."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "Why is LangGraph useful for agents?"},
    #         "outputs": {
    #             "answer": "LangGraph is useful for agents because it enables controlled multi-step reasoning, tool usage, and state persistence across iterations."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "What is a conditional edge in LangGraph?"},
    #         "outputs": {
    #             "answer": "A conditional edge determines the next node based on logic applied to the current state, enabling branching workflows."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "How does LangGraph manage memory?"},
    #         "outputs": {
    #             "answer": "LangGraph manages memory through the shared state object, which persists and evolves as nodes process data."
    #         },
    #     },
    #     {
    #         "inputs": {
    #             "question": "What is the role of the START node in LangGraph?"
    #         },
    #         "outputs": {
    #             "answer": "The START node defines the entry point of the graph where execution begins."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "What is the END node in LangGraph?"},
    #         "outputs": {
    #             "answer": "The END node signifies the termination of the workflow when execution completes."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "Can LangGraph be used for RAG pipelines?"},
    #         "outputs": {
    #             "answer": "Yes, LangGraph is well suited for RAG pipelines where retrieval, reasoning, and generation steps need to be orchestrated in a structured workflow."
    #         },
    #     },
    #     {
    #         "inputs": {
    #             "question": "How does LangGraph help prevent infinite loops?"
    #         },
    #         "outputs": {
    #             "answer": "LangGraph allows developers to control loop conditions and termination logic explicitly through state and conditional edges."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "What is a practical use case of LangGraph?"},
    #         "outputs": {
    #             "answer": "A practical use case is building an agent that retrieves documents, evaluates them, refines queries, and iterates until a satisfactory answer is produced."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "Explain LangGraph in simple terms."},
    #         "outputs": {
    #             "answer": "LangGraph lets you design LLM workflows like a flowchart where each step updates shared data and decides what to do next."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "Does LangGraph support parallel execution?"},
    #         "outputs": {
    #             "answer": "LangGraph can support parallel execution depending on how nodes and edges are defined, though it is primarily designed for controlled workflows."
    #         },
    #     },
    #     {
    #         "inputs": {
    #             "question": "What makes LangGraph better than simple chains for complex workflows?"
    #         },
    #         "outputs": {
    #             "answer": "LangGraph supports loops, branching, and persistent state, which are difficult or impossible to implement cleanly with simple linear chains."
    #         },
    #     },
    #     {
    #         "inputs": {"question": "How is state updated in LangGraph?"},
    #         "outputs": {
    #             "answer": "Each node receives the current state, processes it, and returns an updated version that is passed to the next node."
    #         },
    #     },
    #     {
    #         "inputs": {
    #             "question": "Why is LangGraph important for production LLM systems?"
    #         },
    #         "outputs": {
    #             "answer": "LangGraph enables structured, debuggable, and reproducible workflows, which are essential for reliability in production systems."
    #         },
    #     },
    # ]

    # # Add / Create Examples
    # client.create_examples(dataset_id=dataset_id, examples=examples)
    return


@app.cell
def _(client, dataset_id):
    # Rest of Example IDs
    examples = list(client.list_examples(dataset_id=dataset_id))
    list(examples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # *LLM as Judge*
    """)
    return


@app.cell
def _():
    from openai import OpenAI
    from langsmith.wrappers import wrap_openai

    return OpenAI, wrap_openai


@app.cell
def _(OpenAI, wrap_openai):
    # Create Open AI clinet Wrapper
    openai_client = wrap_openai(OpenAI())

    # System prompt
    eval_instructions = "You are an expert professor specialized in grading student's answer to questions."

    # The Evauation metic Correctness
    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        # The Prompt
        user_content = f"""You are grading the following question:
        {inputs["question"]}
        Here is the real answer
        {reference_outputs["answer"]}
        You are grading the following predicted answer:
        {outputs["response"]}
        Respond with CORRECT or INCORRECT:
        Grade:
        """
        # Get response
        response = (
            openai_client.chat.completions.create(
                model="gpt-5.4-nano-2026-03-17",
                temperature=0,
                messages=[
                    {"role": "system", "content": eval_instructions},
                    {"role": "user", "content": user_content},
                ],
            )
            .choices[0]
            .message.content
        )
        return response == "CORRECT"

    return correctness, openai_client


@app.cell
def _(openai_client):
    default_instructions = (
        "Respond to the users question in a short, concise manner (one short sentence)."
    )

    def my_app(
        inputs: dict,
    ) -> str:
        return {
            "response": (
                openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": default_instructions},
                        {"role": "user", "content": inputs["question"]},
                    ],
                )
                .choices[0]
                .message.content
            )
        }

    return (my_app,)


@app.cell
def _(client, correctness, dataset_name, my_app):
    client.evaluate(
        my_app,
        data=dataset_name,
        evaluators=[correctness],
        experiment_prefix="openai-4o-mini-chatbot",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
