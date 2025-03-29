"""
MemBot: A context-aware chatbot using LangGraph and LangMem with InMemorySaver.
- Uses Azure ChatGPT for responses.
- Stores every query and response in InMemoryStore with all-MiniLM-L12-v2 embeddings.
- Persists messages state in-memory via thread_id and InMemorySaver.
"""

import os
import asyncio
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from sentence_transformers import SentenceTransformer
from azure_openai_llm import get_llm  # Assuming this provides an async-compatible LLM

# Embedding model setup
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

def embed_text(text: str) -> list:
    """Convert text to embeddings using all-MiniLM-L12-v2."""
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

# Memory store and checkpointer setup
NAMESPACE = ("user_1",)
memory_store = InMemoryStore(index={"dims": 384, "embed": embed_text})
checkpointer = MemorySaver()  # In-memory persistence for messages state

# Azure ChatGPT model (assumed to support async)
llm = get_llm()

# Memory tools (assumed to be sync; we'll wrap them if needed)
manage_memory_tool = create_manage_memory_tool(namespace=NAMESPACE)
search_memory_tool = create_search_memory_tool(namespace=NAMESPACE)

# System prompt
SYSTEM_PROMPT = """
You are MemBot, a helpful assistant with memory. Your goals:
1. Assist users conversationally.
2. Use `manage_memory_tool` to store EVERY user query and assistant response as a single memory entry.
3. For questions about past interactions, ALWAYS use `search_memory_tool` to retrieve relevant memories. Sort memories by order (earliest first) and return the EXACT user input from the first relevant memory. If the tool fails, use the conversation history (messages) to find the first user input.
Keep responses natural and use the full conversation history (passed in messages) for coherence.
"""

# Agent setup
agent = create_react_agent(
    model=llm,
    tools=[manage_memory_tool, search_memory_tool],
    store=memory_store,
    checkpointer=checkpointer,
    prompt=SYSTEM_PROMPT
)

async def print_stored_memories() -> None:
    """Print all memories stored in InMemoryStore, extracting content."""
    print("\n--- Stored Memories ---")
    try:
        all_memories = memory_store._data.get(NAMESPACE, {})
        if not all_memories:
            print("No memories stored yet.")
        else:
            for i, (key, item) in enumerate(sorted(all_memories.items(), key=lambda x: x[0]), 1):
                value = getattr(item, "value", "None") if item else "None"
                if isinstance(value, dict) and "content" in value:
                    value = value["content"]
                print(f"Memory {i}: Key={key}, Value={value}")
    except Exception as e:
        print(f"Error retrieving memories: {e}")
    print("----------------------\n")

async def store_memory_in_background(agent, memory_entry: str, config: dict, delay: float = 4.0) -> None:
    """Store a memory entry asynchronously in the background."""
    await asyncio.sleep(delay)
    try:
        await agent.ainvoke(
            {"messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]},
            config=config
        )
    except Exception as e:
        print(f"Error storing memory in background: {e}")

async def chat_with_membot() -> None:
    """Run an interactive async chat loop with MemBot."""
    print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    config = {"configurable": {"thread_id": "user_1_thread"}}  # Thread-specific state

    try:
        while True:
            # Async input (using a simple workaround since `input()` is sync)
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
            user_input = user_input.strip()
            if user_input.lower() == "exit":
                print("MemBot: Goodbye!")
                break

            # Normalize input
            normalized_input = user_input.lower()

            # Add user input to history
            conversation_history.append({"role": "user", "content": user_input})

            # Invoke agent asynchronously
            response = await agent.ainvoke({"messages": conversation_history}, config=config)

            # Extract response
            ai_response = (
                response["messages"][-1].content
                if isinstance(response, dict) and "messages" in response
                else str(response)
            )
            print(f"MemBot: {ai_response}")

            # Add bot response to history
            conversation_history.append({"role": "assistant", "content": ai_response})

            # Store memory in the background using a task
            memory_entry = f"User: {normalized_input} | Bot: {ai_response}"
            asyncio.create_task(store_memory_in_background(agent, memory_entry, config))

            # Show stored memories (still synchronous for simplicity)
            await print_stored_memories()

    except Exception as e:
        print(f"Error occurred: {e}")
    except KeyboardInterrupt:
        print("\nMemBot: Goodbye!")

if __name__ == "__main__":
    asyncio.run(chat_with_membot())