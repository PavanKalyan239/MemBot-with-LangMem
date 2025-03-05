# """
# MemBot: A context-aware, persistent chatbot using LangGraph and LangMem.
# - Uses Azure ChatGPT for responses.
# - Stores up to 3 conversations in InMemoryStore, batches to SQLite every 3, with all-MiniLM-L12-v2 embeddings.
# - Maintains full conversation context via persistent messages state.
# """

# import os
# import sqlite3
# import json
# import asyncio
# from collections import deque
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
# MAX_IN_MEMORY = 3  # Limit to 3 conversations in InMemoryStore

# # System prompt
# SYSTEM_PROMPT = """
# You are MemBot, a helpful assistant with persistent memory. Your goals:
# 1. Assist users conversationally.
# 2. Use `manage_memory_tool` to store EVERY user query and assistant response as a single memory entry.
# 3. For questions about past interactions, ALWAYS use `search_memory_tool` to retrieve relevant memories from InMemoryStore. If no relevant memory is found, indicate it might be in older records and rely on conversation history if available. Return the EXACT user input from the most relevant memory.
# Keep responses natural and use the full conversation history (passed in messages) for coherence.
# """

# def init_db():
#     """Initialize SQLite database for memory and history backup."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS memories (
#             id TEXT PRIMARY KEY,
#             namespace TEXT,
#             value TEXT UNIQUE
#         )
#     """)
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS history (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             messages TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()

# def load_from_sqlite(store: InMemoryStore) -> list:
#     """Load latest conversation history from SQLite at startup (memories loaded on demand)."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT messages FROM history ORDER BY id DESC LIMIT 1")
#     row = cursor.fetchone()
#     history = json.loads(row[0]) if row else [{"role": "system", "content": SYSTEM_PROMPT}]
#     conn.close()
#     return history

# async def save_to_sqlite(memory_queue: deque, history: list):
#     """Save batched memories and conversation history to SQLite."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     for memory_entry in memory_queue:
#         key = str(uuid4())
#         cursor.execute(
#             "INSERT OR IGNORE INTO memories (id, namespace, value) VALUES (?, ?, ?)",
#             (key, NAMESPACE[0], memory_entry)
#         )
#     cursor.execute("INSERT INTO history (messages) VALUES (?)", (json.dumps(history),))
#     conn.commit()
#     conn.close()

# async def memory_batcher(agent, config: dict, memory_queue: deque, history: list):
#     """Batch and save memories to SQLite every 3 conversations."""
#     while True:
#         await asyncio.sleep(1)  # Check every second
#         if len(memory_queue) >= MAX_IN_MEMORY:
#             await save_to_sqlite(memory_queue, history)
#             memory_queue.clear()  # Clear the queue after saving
#             # Clear InMemoryStore and reload only the latest 3 (already handled by queue limit)

# def search_sqlite(query: str) -> str | None:
#     """Search SQLite for a memory matching the query."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT value FROM memories WHERE value LIKE ? ORDER BY id ASC LIMIT 1", (f"%{query}%",))
#     row = cursor.fetchone()
#     conn.close()
#     return row[0] if row else None

# # Memory store setup
# memory_store = InMemoryStore(index={"dims": 384, "embed": embed_text})
# init_db()
# conversation_history = load_from_sqlite(memory_store)
# memory_queue = deque(maxlen=MAX_IN_MEMORY)  # Store only 3 conversations in memory

# # Azure ChatGPT model (assumed async-compatible)
# llm = get_llm()

# # Memory tools
# manage_memory_tool = create_manage_memory_tool(namespace=NAMESPACE)
# search_memory_tool = create_search_memory_tool(namespace=NAMESPACE)

# # Agent setup
# agent = create_react_agent(
#     model=llm,
#     tools=[manage_memory_tool, search_memory_tool],
#     store=memory_store,
#     prompt=SYSTEM_PROMPT
# )

# async def print_stored_memories():
#     """Print memories from InMemoryStore."""
#     print("\n--- Stored Memories (InMemoryStore) ---")
#     try:
#         all_memories = memory_store._data.get(NAMESPACE, {})
#         if not all_memories:
#             print("No memories in InMemoryStore yet.")
#         else:
#             for i, (key, item) in enumerate(all_memories.items(), 1):
#                 value = getattr(item, "value", "None") if item else "None"
#                 if isinstance(value, dict) and "content" in value:
#                     value = value["content"]
#                 print(f"Memory {i}: Key={key}, Value={value}")
#     except Exception as e:
#         print(f"Error retrieving memories: {e}")
#     print("----------------------\n")

# async def chat_with_membot():
#     """Run an interactive async chat loop with MemBot."""
#     global conversation_history
#     print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
#     config = {"configurable": {"thread_id": "user_1_thread"}}

#     # Start background memory batcher
#     asyncio.create_task(memory_batcher(agent, config, memory_queue, conversation_history))

#     try:
#         while True:
#             user_input = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
#             user_input = user_input.strip()
#             if user_input.lower() == "exit":
#                 # Save any remaining memories before exiting
#                 if memory_queue:
#                     await save_to_sqlite(memory_queue, conversation_history)
#                 print("MemBot: Goodbye!")
#                 break

#             normalized_input = user_input.lower()
#             conversation_history.append({"role": "user", "content": user_input})

#             response = await agent.ainvoke({"messages": conversation_history}, config=config)
#             ai_response = (
#                 response["messages"][-1].content
#                 if isinstance(response, dict) and "messages" in response
#                 else str(response)
#             )
#             print(f"MemBot: {ai_response}")
#             conversation_history.append({"role": "assistant", "content": ai_response})

#             # Store in memory queue and InMemoryStore
#             memory_entry = f"User: {normalized_input} | Bot: {ai_response}"
#             memory_queue.append(memory_entry)
#             await agent.ainvoke({
#                 "messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]
#             }, config=config)

#             # Keep InMemoryStore limited to 3
#             if len(memory_store._data.get(NAMESPACE, {})) > MAX_IN_MEMORY:
#                 oldest_key = min(memory_store._data[NAMESPACE].keys())
#                 del memory_store._data[NAMESPACE][oldest_key]

#             await print_stored_memories()

#             # Check for memory retrieval from SQLite if needed
#             if "what was my first message" in normalized_input:
#                 memory = await agent.ainvoke({
#                     "messages": [{"role": "user", "content": user_input}]
#                 }, config=config)
#                 if "don’t have that information" in memory["messages"][-1].content:
#                     sqlite_result = search_sqlite("User:")
#                     if sqlite_result:
#                         print(f"MemBot (from SQLite): Your first message was: {sqlite_result.split('|')[0].replace('User: ', '')}")

#     except Exception as e:
#         print(f"Error occurred: {e}")
#     except KeyboardInterrupt:
#         if memory_queue:
#             await save_to_sqlite(memory_queue, conversation_history)
#         print("\nMemBot: Goodbye!")

# if __name__ == "__main__":
#     asyncio.run(chat_with_membot())


#----------------------Printing the SQLite database-----------------------#

"""
MemBot: A context-aware, persistent chatbot using LangGraph and LangMem.
- Uses Azure ChatGPT for responses.
- Stores up to 3 conversations in InMemoryStore, batches to SQLite every 3, with all-MiniLM-L12-v2 embeddings.
- Maintains full conversation context via persistent messages state.
"""

import os
import sqlite3
import json
import asyncio
from collections import deque
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from sentence_transformers import SentenceTransformer
from azure_openai_llm import get_llm
from uuid import uuid4

# Embedding model setup
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

def embed_text(text: str) -> list:
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

# SQLite persistence setup
DB_PATH = "membot_memories.db"
NAMESPACE = ("user_1",)
MAX_IN_MEMORY = 3

# System prompt
SYSTEM_PROMPT = """
You are MemBot, a helpful assistant with persistent memory. Your goals:
1. Assist users conversationally.
2. Use `manage_memory_tool` to store EVERY user query and assistant response as a single memory entry.
3. For questions about past interactions, ALWAYS use `search_memory_tool` to retrieve relevant memories from InMemoryStore. If no relevant memory is found, indicate it might be in older records and rely on conversation history if available. Return the EXACT user input from the most relevant memory.
Keep responses natural and use the full conversation history (passed in messages) for coherence.
"""

def init_db():
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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT messages FROM history ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    history = json.loads(row[0]) if row else [{"role": "system", "content": SYSTEM_PROMPT}]
    conn.close()
    return history

async def save_to_sqlite(memory_queue: deque, history: list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for memory_entry in memory_queue:
        key = str(uuid4())
        cursor.execute(
            "INSERT OR IGNORE INTO memories (id, namespace, value) VALUES (?, ?, ?)",
            (key, NAMESPACE[0], memory_entry)
        )
    cursor.execute("INSERT INTO history (messages) VALUES (?)", (json.dumps(history),))
    conn.commit()
    conn.close()
    # Log and verify save
    print(f"\n--- Saved {len(memory_queue)} memories to SQLite ---")
    await print_sqlite_memories()

async def print_sqlite_memories():
    """Print all memories stored in SQLite for verification."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, value FROM memories WHERE namespace = ?", (NAMESPACE[0],))
    rows = cursor.fetchall()
    if not rows:
        print("SQLite: No memories stored yet.")
    else:
        print("SQLite Memories:")
        for i, (key, value) in enumerate(rows, 1):
            print(f"  {i}: Key={key}, Value={value}")
    conn.close()

async def memory_batcher(agent, config: dict, memory_queue: deque, history: list):
    while True:
        await asyncio.sleep(1)  # Check every second
        if len(memory_queue) >= MAX_IN_MEMORY:
            print(f"\nBatching {len(memory_queue)} conversations to SQLite...")
            await save_to_sqlite(memory_queue, history)
            memory_queue.clear()

def search_sqlite(query: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM memories WHERE value LIKE ? ORDER BY id ASC LIMIT 1", (f"%{query}%",))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# Memory store setup
memory_store = InMemoryStore(index={"dims": 384, "embed": embed_text})
init_db()
conversation_history = load_from_sqlite(memory_store)
memory_queue = deque(maxlen=MAX_IN_MEMORY)

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

async def print_stored_memories():
    print("\n--- Stored Memories (InMemoryStore) ---")
    try:
        all_memories = memory_store._data.get(NAMESPACE, {})
        if not all_memories:
            print("No memories in InMemoryStore yet.")
        else:
            for i, (key, item) in enumerate(all_memories.items(), 1):
                value = getattr(item, "value", "None") if item else "None"
                if isinstance(value, dict) and "content" in value:
                    value = value["content"]
                print(f"Memory {i}: Key={key}, Value={value}")
    except Exception as e:
        print(f"Error retrieving memories: {e}")
    print("----------------------\n")

async def chat_with_membot():
    global conversation_history
    print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
    config = {"configurable": {"thread_id": "user_1_thread"}}
    conversation_count = 0  # Track number of conversations

    asyncio.create_task(memory_batcher(agent, config, memory_queue, conversation_history))

    try:
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
            user_input = user_input.strip()
            if user_input.lower() == "exit":
                if memory_queue:
                    await save_to_sqlite(memory_queue, conversation_history)
                print("MemBot: Goodbye!")
                break

            normalized_input = user_input.lower()
            conversation_history.append({"role": "user", "content": user_input})

            response = await agent.ainvoke({"messages": conversation_history}, config=config)
            ai_response = (
                response["messages"][-1].content
                if isinstance(response, dict) and "messages" in response
                else str(response)
            )
            print(f"MemBot: {ai_response}")
            conversation_history.append({"role": "assistant", "content": ai_response})

            memory_entry = f"User: {normalized_input} | Bot: {ai_response}"
            memory_queue.append(memory_entry)
            await agent.ainvoke({
                "messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]
            }, config=config)

            if len(memory_store._data.get(NAMESPACE, {})) > MAX_IN_MEMORY:
                oldest_key = min(memory_store._data[NAMESPACE].keys())
                del memory_store._data[NAMESPACE][oldest_key]

            conversation_count += 1
            print(f"Conversation #{conversation_count}")

            await print_stored_memories()

            if "what was my first message" in normalized_input:
                memory = await agent.ainvoke({
                    "messages": [{"role": "user", "content": user_input}]
                }, config=config)
                if "don’t have that information" in memory["messages"][-1].content:
                    sqlite_result = search_sqlite("User:")
                    if sqlite_result:
                        print(f"MemBot (from SQLite): Your first message was: {sqlite_result.split('|')[0].replace('User: ', '')}")

    except Exception as e:
        print(f"Error occurred: {e}")
    except KeyboardInterrupt:
        if memory_queue:
            await save_to_sqlite(memory_queue, conversation_history)
        print("\nMemBot: Goodbye!")

if __name__ == "__main__":
    asyncio.run(chat_with_membot())