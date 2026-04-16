import marimo

__generated_with = "0.23.1"
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
    from typing import TypedDict, Any, Callable

    # Third party Imports
    import marimo as mo
    from ddgs import DDGS
    from langgraph.graph import StateGraph, START, END
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
        tool_resluts = [
            call_tools(res, tool_dict=tools_dict) for res in tools_list
        ]

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
        tool_resluts2 = [
            call_tools(res, tool_dict=tools_dict) for res in tools_list2
        ]

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
