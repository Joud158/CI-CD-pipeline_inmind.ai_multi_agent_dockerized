# Assignment 1

from __future__ import annotations
import ast # to parse math exps
import operator as op # python functions for opertaions
import os # to read env variables
from typing import Any
from langchain_core.tools import tool # to convert python functions into agent tools
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage # message types
from langgraph.graph import StateGraph, MessagesState, START # define the LangGraph workflow
from langgraph.prebuilt import ToolNode, tools_condition # manage tool execution
from rag_core import DigitalAgronomistRAG, make_rag_tool # RAG pipeline, make_rag_tool is to wrap the RAG pipeline as a tool

# here i listed the allowed operations for the calculator
# so this is a restricted operation dictionary so the calcultor supports only these instead of arbitrary code
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

# this is to safely evaluate a math exp
def safe_eval(expr: str) -> float: # type-hinting
    node = ast.parse(expr, mode="eval") # parse into ast
    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.operand))
        raise ValueError("Only basic arithmetic is allowed.")
    return _eval(node)

# calls the above function
# this is an agent tool
@tool("calculator")
def calculator(expression: str) -> str: # we return as a string for the agent to use directly in the response
    """Evaluate a basic arithmetic expression such as 12+8 or 5*3."""
    try:
        return str(safe_eval(expression))
    except Exception as exc:
        return f"Calculator error: {exc}"

# another agent tool
# here we return the names of text files used in RAG 
@tool("list_sources")
def list_sources() -> str:
    """List the names of the indexed agronomy source files."""
    return "\n".join(
        [
            "01_crop_diagnostics.txt",
            "02_irrigation_water.txt",
            "03_soil_nutrition.txt",
            "04_postharvest_supplychain.txt",
            "05_farm_business_pricing.txt",
        ]
    )

SYSTEM_PROMPT = """
You are a practical digital agronomist assistant.

Rules:
1. Use agronomy_rag_search for agronomy, irrigation, soil, post-harvest, or farm business questions.
2. Use calculator only for explicit arithmetic.
3. Use list_sources only if the user asks what documents are available.
4. Never invent facts outside the indexed documents.
5. If a tool result is already sufficient, answer directly and do not call more tools.
6. For normal agronomy questions, use at most one tool call.
7. Keep answers short, practical, and preserve chunk citations when available.
""".strip()

def build_graph() -> Any:
    # builds the RAG pipeline and wraps it as a callable tool
    rag = DigitalAgronomistRAG().build()
    rag_tool = make_rag_tool(rag)
    tools = [rag_tool, calculator, list_sources] # the toolset available to the model
    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0, # makes the model more deterministic
    )
    model_with_tools = model.bind_tools(tools) # tells the model what tools it is allowed to call
    tool_node = ToolNode(tools) # creates the execution node that can actually run those tools inside the graph
    # receives the conversation state and decides what to do next
    def call_model(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        # if the last message is a ToolMessage, force a final answer without tools
        if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
            final_prompt = messages + [
                SystemMessage(content="You have the tool result. Answer the user directly. Do not call any tool.")
            ]
            response = model.invoke(final_prompt)
            return {"messages": [response]}
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")
    return builder.compile()

def sample_chat() -> None:
    print("Building graph...")
    graph = build_graph()
    print("Graph built. Running first question...")
    messages = [
        {"role": "user", "content": "My zucchini leaves have white powdery spots. What should I do first?"}
    ]
    result = graph.invoke({"messages": messages}, config={"recursion_limit": 10})
    first_answer = result["messages"][-1].content
    print("\nFirst response:")
    print(first_answer)
    messages.append({"role": "assistant", "content": first_answer})
    messages.append({"role": "user", "content": "Calculate 12 + 8."})
    result = graph.invoke({"messages": messages}, config={"recursion_limit": 10})
    print("\nSecond response:")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    sample_chat()