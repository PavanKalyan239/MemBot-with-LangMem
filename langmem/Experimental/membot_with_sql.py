# """
# MemBot: A context-aware, persistent chatbot using LangGraph and LangMem.
# - Uses Azure ChatGPT for responses.
# - Stores every query and response in InMemoryStore with SQLite backup and all-MiniLM-L12-v2 embeddings.
# - Maintains full conversation context via messages state.
# """

# import os
# import sqlite3
# from langgraph.prebuilt import create_react_agent
# from langgraph.store.memory import InMemoryStore
# from langmem import create_manage_memory_tool, create_search_memory_tool
# from sentence_transformers import SentenceTransformer
# from azure_openai_llm import get_llm
# from uuid import uuid4

# # Embedding model setup
# embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

# def embed_text(text: str) -> list:
#     """Convert text to embeddings using all-MiniLM-L12-v2."""
#     return embedding_model.encode(text, convert_to_numpy=True).tolist()

# # SQLite persistence setup
# DB_PATH = "membot_memories.db"
# NAMESPACE = ("user_1",)

# def init_db():
#     """Initialize SQLite database for memory backup."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS memories (
#             id TEXT PRIMARY KEY,
#             namespace TEXT,
#             value TEXT UNIQUE  -- Prevent duplicates
#         )
#     """)
#     conn.commit()
#     conn.close()

# def load_from_sqlite(store: InMemoryStore):
#     """Load memories from SQLite into InMemoryStore at startup."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT id, namespace, value FROM memories WHERE namespace = ?", (NAMESPACE[0],))
#     for key, ns, value in cursor.fetchall():
#         store._data[(ns,)] = store._data.get((ns,), {})
#         store._data[(ns,)][key] = type('Memory', (), {'value': value})()
#     conn.close()

# def save_to_sqlite(memory_entry: str):
#     """Save a memory entry to SQLite, avoiding duplicates."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     key = str(uuid4())
#     cursor.execute(
#         "INSERT OR IGNORE INTO memories (id, namespace, value) VALUES (?, ?, ?)",
#         (key, NAMESPACE[0], memory_entry)
#     )
#     conn.commit()
#     conn.close()

# # Memory store setup
# memory_store = InMemoryStore(index={"dims": 384, "embed": embed_text})
# init_db()
# load_from_sqlite(memory_store)

# # Azure ChatGPT model
# llm = get_llm()

# # Memory tools
# manage_memory_tool = create_manage_memory_tool(namespace=NAMESPACE)
# search_memory_tool = create_search_memory_tool(namespace=NAMESPACE)

# # System prompt
# SYSTEM_PROMPT = """
# You are MemBot, a helpful assistant with memory. Your goals:
# 1. Assist users conversationally.
# 2. Use `manage_memory_tool` to store EVERY user query and assistant response as a single memory entry (e.g., "User: I like Python | Bot: Noted, you like Python").
# 3. Use `search_memory_tool` to retrieve relevant memories when answering questions or providing context.
# Keep responses natural and use the full conversation history (passed in messages) for coherence.
# """

# # Agent setup
# agent = create_react_agent(
#     model=llm,
#     tools=[manage_memory_tool, search_memory_tool],
#     store=memory_store,
#     prompt=SYSTEM_PROMPT
# )

# def print_stored_memories() -> None:
#     """Print all memories stored in InMemoryStore, extracting content."""
#     print("\n--- Stored Memories ---")
#     try:
#         all_memories = memory_store._data.get(NAMESPACE, {})
#         if not all_memories:
#             print("No memories stored yet.")
#         else:
#             for i, (key, item) in enumerate(all_memories.items(), 1):
#                 value = getattr(item, "value", "None") if item else "None"
#                 if isinstance(value, dict) and "content" in value:
#                     value = value["content"]
#                 print(f"Memory {i}: Key={key}, Value={value}")
#     except Exception as e:
#         print(f"Error retrieving memories: {e}")
#     print("----------------------\n")

# def chat_with_membot() -> None:
#     """Run an interactive chat loop with MemBot."""
#     print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
#     conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    
#     try:
#         while True:
#             user_input = input("You: ").strip()
#             if user_input.lower() == "exit":
#                 print("MemBot: Goodbye!")
#                 break

#             # Normalize input to avoid duplicates (case-insensitive)
#             normalized_input = user_input.lower()

#             # Add user input to history
#             conversation_history.append({"role": "user", "content": user_input})

#             # Invoke agent with full history
#             response = agent.invoke({"messages": conversation_history})

#             # Extract response
#             ai_response = (
#                 response["messages"][-1].content
#                 if isinstance(response, dict) and "messages" in response
#                 else str(response)
#             )
#             print(f"MemBot: {ai_response}")

#             # Add bot response to history
#             conversation_history.append({"role": "assistant", "content": ai_response})

#             # Store normalized query and response
#             memory_entry = f"User: {normalized_input} | Bot: {ai_response}"
#             agent.invoke({
#                 "messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]
#             })
#             save_to_sqlite(memory_entry)

#             # Show stored memories
#             print_stored_memories()

#     except Exception as e:
#         print(f"Error occurred: {e}")
#     except KeyboardInterrupt:
#         print("\nMemBot: Goodbye!")

# if __name__ == "__main__":
#     chat_with_membot()


"""
MemBot: A context-aware, persistent chatbot using LangGraph and LangMem.
- Uses Azure ChatGPT for responses.
- Stores every query and response in InMemoryStore with SQLite backup and all-MiniLM-L12-v2 embeddings.
- Maintains full conversation context via persistent messages state.
"""

import os
import sqlite3
import json
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from sentence_transformers import SentenceTransformer
from azure_openai_llm import get_llm
from uuid import uuid4

# Embedding model setup
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

def embed_text(text: str) -> list:
    """Convert text to embeddings using all-MiniLM-L12-v2."""
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

# SQLite persistence setup
DB_PATH = "membot_memories.db"
NAMESPACE = ("user_1",)

# System prompt (moved up)
SYSTEM_PROMPT = """
You are MemBot, a helpful assistant with persistent memory. Your goals:
1. Assist users conversationally.
2. Use `manage_memory_tool` to store EVERY user query and assistant response as a single memory entry (e.g., "User: I like Python | Bot: Noted, you like Python").
3. Use `search_memory_tool` to retrieve relevant memories when answering questions about past interactions (e.g., "What was my last question?"). If asked about prior queries, search memories and provide the most recent relevant user input.
Keep responses natural and use the full conversation history (passed in messages) for coherence. If search fails, say so explicitly.
"""

def init_db():
    """Initialize SQLite database for memory and history backup."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            namespace TEXT,
            value TEXT UNIQUE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            messages TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_from_sqlite(store: InMemoryStore) -> list:
    """Load memories and conversation history from SQLite at startup."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Load memories
    cursor.execute("SELECT id, namespace, value FROM memories WHERE namespace = ?", (NAMESPACE[0],))
    for key, ns, value in cursor.fetchall():
        store._data[(ns,)] = store._data.get((ns,), {})
        store._data[(ns,)][key] = type('Memory', (), {'value': value})()
    # Load history
    cursor.execute("SELECT messages FROM history ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    history = json.loads(row[0]) if row else [{"role": "system", "content": SYSTEM_PROMPT}]
    conn.close()
    return history

def save_to_sqlite(memory_entry: str, history: list):
    """Save memory and conversation history to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Save memory
    key = str(uuid4())
    cursor.execute(
        "INSERT OR IGNORE INTO memories (id, namespace, value) VALUES (?, ?, ?)",
        (key, NAMESPACE[0], memory_entry)
    )
    # Save history
    cursor.execute("INSERT INTO history (messages) VALUES (?)", (json.dumps(history),))
    conn.commit()
    conn.close()

# Memory store setup
memory_store = InMemoryStore(index={"dims": 384, "embed": embed_text})
init_db()
conversation_history = load_from_sqlite(memory_store)

# Azure ChatGPT model
llm = get_llm()

# Memory tools
manage_memory_tool = create_manage_memory_tool(namespace=NAMESPACE)
search_memory_tool = create_search_memory_tool(namespace=NAMESPACE)

# Agent setup
agent = create_react_agent(
    model=llm,
    tools=[manage_memory_tool, search_memory_tool],
    store=memory_store,
    prompt=SYSTEM_PROMPT
)

def print_stored_memories() -> None:
    """Print all memories stored in InMemoryStore, extracting content."""
    print("\n--- Stored Memories ---")
    try:
        all_memories = memory_store._data.get(NAMESPACE, {})
        if not all_memories:
            print("No memories stored yet.")
        else:
            for i, (key, item) in enumerate(all_memories.items(), 1):
                value = getattr(item, "value", "None") if item else "None"
                if isinstance(value, dict) and "content" in value:
                    value = value["content"]
                print(f"Memory {i}: Key={key}, Value={value}")
    except Exception as e:
        print(f"Error retrieving memories: {e}")
    print("----------------------\n")

def chat_with_membot() -> None:
    """Run an interactive chat loop with MemBot."""
    global conversation_history
    print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
    
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("MemBot: Goodbye!")
                break

            # Normalize input
            normalized_input = user_input.lower()

            # Add user input to history
            conversation_history.append({"role": "user", "content": user_input})

            # Invoke agent with full history
            response = agent.invoke({"messages": conversation_history})

            # Extract response
            ai_response = (
                response["messages"][-1].content
                if isinstance(response, dict) and "messages" in response
                else str(response)
            )
            print(f"MemBot: {ai_response}")

            # Add bot response to history
            conversation_history.append({"role": "assistant", "content": ai_response})

            # Store normalized query and response
            memory_entry = f"User: {normalized_input} | Bot: {ai_response}"
            agent.invoke({
                "messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]
            })
            save_to_sqlite(memory_entry, conversation_history)

            # Show stored memories
            print_stored_memories()

    except Exception as e:
        print(f"Error occurred: {e}")
    except KeyboardInterrupt:
        print("\nMemBot: Goodbye!")

if __name__ == "__main__":
    chat_with_membot()