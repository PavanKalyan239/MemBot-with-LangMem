# MemBot: A Context-Aware Chatbot with LangMem(Experimental)

MemBot is a chatbot leveraging **LangGraph** and **LangMem** to provide conversational assistance with memory capabilities for multiple users. Built with **Azure ChatGPT** as the language model, it uses **LangMem’s `InMemoryStore`** to store episodic memories and **`MemorySaver`** for conversation state persistence, supporting a synchronous Hot Path Quickstart approach with multi-user functionality.

---

## Project Overview

- **Objective**: Create a chatbot that remembers user interactions across sessions, scalable for multiple users.
- **Tech Stack**:
  - **LangGraph**: Workflow orchestration.
  - **LangMem**: Memory management.
  - **Azure ChatGPT**: LLM via `azure_openai_llm.py`.
  - **Sentence Transformers**: Embeddings (`all-MiniLM-L12-v2`).
- **File**: `multi_user_inmemory.py`—current implementation.

---

## LangMem Features and Functionalities

### 1. Memory Storage with `InMemoryStore`
- **Description**: Stores conversation data as key-value pairs in memory, indexed with embeddings for similarity search.
- **Current Use**:
  - **Episodic Memories**: Stores full query-response pairs (e.g., `"User: hi | Bot: Hello! How can I assist you today?"`) in `InMemoryStore`.
  - **Setup**:
    ```python
    memory_store = InMemoryStore(index={"dims": 384, "embed": embed_text})
    ```
    - `dims=384`: Matches all-MiniLM-L12-v2 embedding size.
    - `embed=embed_text`: Custom function for SentenceTransformer embeddings.
    - **Multi-User**: Isolated via `NAMESPACE` (e.g., `("user_pavan",)`).
- **Pros**:
  - Fast, in-memory storage—ideal for real-time chats.
  - Supports multi-user separation with `NAMESPACE`.
- **Cons**:
  - Session-only—resets on script exit.
  - No explicit `k` (nearest neighbors) or length limits exposed in 0.0.14y.
- **Potential**:
  - **SQLite Extension**: Add persistence (not implemented—see "Next Steps").
    ```python
    # Pseudo-code
    def save_to_sqlite(user_id, memory_entry):
        # Store in SQLite
    ```

---

### 2. Hot Path Quickstart
- **Description**: Agent actively manages memory during the chat using `manage_memory_tool` and `search_memory_tool`.
- **Current Use**:
  - **Storage**:
    ```python
    memory_entry = f"User: {normalized_input} | Bot: {ai_response}"
    agent.invoke({"messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]}, config=config)
    ```
  - **Retrieval**:
    ```python
    search_memory_tool = create_search_memory_tool(namespace=namespace)
    ```
  - **Multi-User**: Each user has unique tools via `NAMESPACE`.
- **Pros**:
  - Works with 0.0.14y—no upgrade needed.
  - Immediate storage—reliable for short sessions.
- **Cons**:
  - Synchronous—blocks chat flow slightly.
  - `search_memory_tool` may lag (assumed `k=1`, no config).

---

### 3. Background Quickstart (Not Implemented)
- **Description**: Memories extracted asynchronously via `MemoryManager`—decoupled from chat flow.
- **Features**:
  - **Episodic Memory**: Stores full interactions (like Hot Path).
  - **Semantic Memory**: Extracts facts (e.g., "Pavan likes Python")—requires `semantic_memory_function`.
  - **Setup**: Uses `create_memory_store_manager` and `MemoryManager` (e.g., `memory_manager.ainvoke(messages)`).
- **Why Not Used**:
  - Requires `langmem ≥0.2.x`—0.0.14y lacks `langmem.core` (`ModuleNotFoundError`).
  - Attempted pseudo-background with `asyncio.Queue`—reverted to sync Hot Path.
- **Pros**:
  - Non-blocking—faster chat responses.
  - Richer memory (episodic + semantic).
- **Cons**:
  - Version-limited—needs upgrade.

---

### 4. Multi-User Support
- **Description**: Isolates user data using `thread_id` and `NAMESPACE`.
- **Current Use**:
  - **Thread ID**:
    ```python
    config = {"configurable": {"thread_id": f"{user_id}_thread"}}
    ```
  - **Namespace**:
    ```python
    namespace = (f"user_{user_id}",)
    ```
  - **Execution**: Multiple CMD instances—each runs `multi_user_inmemory.py` for a user.
- **Pros**:
  - Simple scaling—no shared state conflicts.
  - Independent histories and memories.
- **Cons**:
  - In-memory only—needs SQLite for persistence.

---

### 5. Checkpointer with `MemorySaver`
- **Description**: Persists conversation state (messages) in RAM per `thread_id`.
- **Current Use**:
  - Replaces static prompt instructions—LLM sees full history.
    ```python
    checkpointer = MemorySaver()
    ```
  - Multi-user via `config`—each user’s state is isolated.
- **Pros**:
  - Lightweight—keeps context without disk I/O.
  - Seamless multi-user support.
- **Cons**:
  - Session-only—resets on exit.

---

## Setup

### Prerequisites
- **Python**: 3.8+ (tested on Windows CMD).
- **Virtual Environment**: `venv` recommended.

### Installation
1. **Clone Repository**:
   ```bash
   git clone https://github.com/PavanKalyan239/MemBot-with-LangMem.git
   cd langmem
   ```

2. **Create Virtual Environment**:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install Requirements**:
   ```
   pip install -r requirements.txt1
   ```

5. **Update `.env` file**.
6. **Running MemBot**:
   ```
   python membot_with_memory.py
   ```

### Official Documentation
- LangGraph : https://langchain-ai.github.io/langgraph/
- LangMem : https://langchain-ai.github.io/langmem/
- Sentence Transformers : https://www.sbert.net/

