# Assignment 2

from __future__ import annotations
import os # env variables
import re # pattern matching in the guardrails
import sqlite3 # for local database
from pathlib import Path # for database path
from typing import Any, Dict, List, Tuple
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool # converts python functions into tools
from langchain_groq import ChatGroq # llm backend
from langgraph.graph import END, START, MessagesState, StateGraph # build the graph of langgraph
from langgraph.prebuilt import ToolNode, tools_condition # handle tool execution
from rag_core import DigitalAgronomistRAG

# the local database (sqlite)
DB_PATH = Path("agronomy_agent.db")

# Guardrails
BLOCKED_USER_PATTERNS = [
    r"ignore previous instructions",
    r"reveal your system prompt",
    r"developer message",
    r"system prompt",
    r"rm -rf",
    r"drop table",
    r"delete from",
]

# blocks dangerous sql operations
BLOCKED_SQL_PATTERNS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bALTER\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bPRAGMA\b",
    r"\bVACUUM\b",
    r"\bTRIGGER\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
]

# checks if the user message contains any blocked thing
def validate_user_message(text: str) -> str | None:
    lowered = text.lower()
    for pattern in BLOCKED_USER_PATTERNS:
        if re.search(pattern, lowered):
            return f"Blocked by guardrail: the message matched '{pattern}'."
    return None

# validation of sql query
def validate_sql(sql_query: str) -> str | None:
    cleaned = sql_query.strip()
    # empty are not accepted
    if not cleaned:
        return "Blocked: empty SQL query."
    # Allow only one statement
    if ";" in cleaned[:-1]:
        return "Blocked: only one SQL statement is allowed."
    upper = cleaned.upper()
    # check if blocked
    for pattern in BLOCKED_SQL_PATTERNS:
        if re.search(pattern, upper):
            return f"Blocked SQL pattern: {pattern}"
    # should start with SELECT
    if not upper.startswith("SELECT"):
        return "Blocked: the SQL executor is read-only and only allows SELECT."
    if "FARMER_NOTES" not in upper and "SQLITE_MASTER" not in upper:
        return "Blocked: query only farmer_notes or sqlite_master."
    return None

# database initialization
def init_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS farmer_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT NOT NULL,
            tag TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

# DB tools
@tool("save_note")
def save_note(note: str, tag: str = "general") -> str:
    """Save a short farming note into SQLite if it does not already exist."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT 1 FROM farmer_notes WHERE note = ? AND tag = ? LIMIT 1",
        (note, tag),
    ).fetchone()
    if existing:
        conn.close()
        return f"Note already exists under tag='{tag}'."
    conn.execute("INSERT INTO farmer_notes(note, tag) VALUES(?, ?)", (note, tag))
    conn.commit()
    conn.close()
    return f"Saved note under tag='{tag}'."

# validates read only sql
@tool("execute_safe_sql")
def execute_safe_sql(sql_query: str) -> str:
    """Execute a read-only SQL SELECT query on the agronomy SQLite database."""
    init_db()
    block_reason = validate_sql(sql_query)
    if block_reason:
        return block_reason
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # fetch rows
    rows = conn.execute(sql_query).fetchall()
    conn.close()
    result = [dict(row) for row in rows[:50]]
    return str(result)

# queries sqlite_master
# allows the assistant to inspect the database structure before forming a sql query
@tool("get_db_schema")
def get_db_schema() -> str:
    """Return the SQLite schema so the assistant knows the available tables and columns."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    return "\n\n".join(f"{name}:\n{sql}" for name, sql in rows)

# Helper routing
def extract_note_and_tag(user_message: str) -> Tuple[str, str]:
    text = user_message.strip()
    lowered = text.lower()
    tag = "general"
    if "disease" in lowered or "mildew" in lowered or "aphid" in lowered:
        tag = "diseases"
    elif "soil" in lowered or "fertilizer" in lowered or "nitrogen" in lowered:
        tag = "soil"
    elif "water" in lowered or "irrigation" in lowered:
        tag = "irrigation"
    prefixes = [
        "save a note that",
        "save that",
        "save note that",
        "remember that",
        "store that",
        "store a note that",
    ]
    cleaned = text
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = text[len(prefix):].strip()
            break
    cleaned = cleaned.rstrip(".")
    return cleaned, tag

# checks whether the user intends to save something
def is_save_request(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(
        phrase in lowered
        for phrase in [
            "save a note",
            "save that",
            "remember that",
            "store that",
            "store a note",
        ]
    )

# check whether user is explicitly asking for SQL or wants saved notes listed
def is_sql_or_list_request(user_message: str) -> bool:
    lowered = user_message.lower()
    return (
        "use sql" in lowered
        or "show all saved notes" in lowered
        or "list saved notes" in lowered
    )

# Graph-based supervisor
SYSTEM_PROMPT = """
You are the supervisor for a small multi-agent-like farming assistant.

Available specialists/tools:
1. rag_specialist -> for agronomy, crop, disease, irrigation, soil, post-harvest, and farm-business questions.
2. save_note -> save a short structured farming note.
3. execute_safe_sql -> run a read-only SQL SELECT query on farmer_notes only.
4. get_db_schema -> inspect the database schema only when needed before a SQL query.

Rules:
- For any farming knowledge question, you must use rag_specialist before answering.
- When rag_specialist is used, base your final answer only on its returned content.
- Do not add outside agronomy advice that is not present in the tool result.
- Use save_note when the user asks to save/store/remember a note.
- Use execute_safe_sql only when the user explicitly asks to use SQL or list/query saved notes.
- Use get_db_schema only if needed before SQL.
- After getting a useful tool result, answer the user directly.
- Do not call the same tool repeatedly for the same user turn.
- For normal user turns, use at most one tool call unless schema inspection is necessary before SQL.
- Keep the final answer concise and clear.
""".strip()

def build_graph() -> Any:
    # we initialize the db
    init_db()
    # build rag system
    rag = DigitalAgronomistRAG().build()
    # this tool calls rag system
    @tool("rag_specialist")
    def rag_specialist(query: str) -> str:
        """Answer agronomy questions using only the indexed documents."""
        payload = rag.answer(query, return_metadata=True)
        answer = payload["answer"]
        sources = ", ".join(item["chunk_id"] for item in payload["retrieved"])
        return f"{answer}\n\nRetrieved chunks: {sources}"
    tools = [rag_specialist, save_note, execute_safe_sql, get_db_schema]
    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
    )
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)
    # llm node in the graph
    def call_model(state: MessagesState) -> Dict[str, List[Any]]:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        # If a tool was just used, force a final direct answer
        if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
            final_messages = messages + [
                SystemMessage(
                    content=(
                        "You already have the tool result. "
                        "Answer the user directly now. "
                        "Do not call any more tools."
                    )
                )
            ]
            response = model.invoke(final_messages)
            return {"messages": [response]}
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    builder.add_edge("tools", "llm")
    return builder.compile()

# takes the graph result and extracts readable text
def extract_text(agent_result: Dict[str, Any]) -> str:
    message = agent_result["messages"][-1]
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content)
    return str(message)

# In-memory chat app
class InMemoryChatApp:
    def __init__(self) -> None:
        self.graph = build_graph()
        self.chat_history: List[Dict[str, str]] = []
    def ask(self, user_message: str) -> str:
        block_reason = validate_user_message(user_message)
        if block_reason:
            return block_reason
        if is_save_request(user_message):
            note, tag = extract_note_and_tag(user_message)
            result = save_note.invoke({"note": note, "tag": tag})
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": result})
            return result
        if is_sql_or_list_request(user_message):
            sql = "SELECT id, note, tag, created_at FROM farmer_notes ORDER BY id"
            result = execute_safe_sql.invoke({"sql_query": sql})
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": result})
            return result
        self.chat_history.append({"role": "user", "content": user_message})
        result = self.graph.invoke(
            {"messages": self.chat_history},
            config={"recursion_limit": 10},
        )
        assistant_text = extract_text(result)
        self.chat_history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

if __name__ == "__main__":
    app = InMemoryChatApp()
    print(app.ask("Save a note that zucchini plot A likely has powdery mildew."))
    print(app.ask("What should I do first for powdery mildew on zucchini?"))
    print(app.ask("Use SQL to show all saved notes."))