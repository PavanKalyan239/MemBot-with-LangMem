"""
MemBot: A context-aware chatbot using LangGraph and LangMem with InMemorySaver.
- Uses Azure ChatGPT for responses.
- Stores every query and response in InMemoryStore with all-MiniLM-L12-v2 embeddings.
- Persists messages state in-memory via thread_id and InMemorySaver.
"""

import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from sentence_transformers import SentenceTransformer
from azure_openai_llm import get_llm

# Embedding model setup
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

def embed_text(text: str) -> list:
    """Convert text to embeddings using all-MiniLM-L12-v2."""
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

# Memory store and checkpointer setup
NAMESPACE = ("user_1",)
memory_store = InMemoryStore(index={"dims": 384, "embed": embed_text})
checkpointer = MemorySaver()  # In-memory persistence for messages state

# Azure ChatGPT model
llm = get_llm()

# Memory tools
manage_memory_tool = create_manage_memory_tool(namespace=NAMESPACE)
search_memory_tool = create_search_memory_tool(namespace=NAMESPACE)

# System prompt
SYSTEM_PROMPT = """
You are MemBot, a helpful assistant with memory. Your goals:
1. Assist users conversationally.
2. Use `manage_memory_tool` to store EVERY user query and assistant response as a single memory entry (e.g., "User: I like Python | Bot: Noted, you like Python").
3. For questions about past interactions (e.g., "What was my first message?"), ALWAYS use `search_memory_tool` to retrieve relevant memories. Sort memories by order (earliest first) and return the EXACT user input from the first relevant memory. If the tool fails, use the conversation history (messages) to find the first user input.
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

def print_stored_memories() -> None:
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

def chat_with_membot() -> None:
    """Run an interactive chat loop with MemBot."""
    print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    config = {"configurable": {"thread_id": "user_1_thread"}}  # Thread-specific state
    
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

            # Invoke agent with thread config
            response = agent.invoke({"messages": conversation_history}, config=config)

            # Extract response
            ai_response = (
                response["messages"][-1].content
                if isinstance(response, dict) and "messages" in response
                else str(response)
            )
            print(f"MemBot: {ai_response}")

            # Add bot response to history
            conversation_history.append({"role": "assistant", "content": ai_response})

            # Store in LangMem with config
            memory_entry = f"User: {normalized_input} | Bot: {ai_response}"
            agent.invoke(
                {"messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]},
                config=config  # Added here to fix the error
            )

            # Show stored memories
            print_stored_memories()

    except Exception as e:
        print(f"Error occurred: {e}")
    except KeyboardInterrupt:
        print("\nMemBot: Goodbye!")

if __name__ == "__main__":
    chat_with_membot()