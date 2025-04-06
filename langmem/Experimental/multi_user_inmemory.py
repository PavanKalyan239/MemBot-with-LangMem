"""
MemBot: A context-aware chatbot using LangGraph and LangMem with InMemoryStore.
- Uses Azure ChatGPT for responses.
- Stores every query and response in InMemoryStore with all-MiniLM-L12-v2 embeddings.
- Persists conversation state via MemorySaver checkpointer with thread_id, multi-user support.
"""

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

# Global memory store and checkpointer
memory_store = InMemoryStore(index={"dims": 384, "embed": embed_text})
checkpointer = MemorySaver()

# Azure ChatGPT model
llm = get_llm()

# System prompt (minimal)
SYSTEM_PROMPT = "You are MemBot, a helpful assistant with memory."

def print_stored_memories(user_id: str) -> None:
    """Print all memories stored for a specific user."""
    namespace = (f"user_{user_id}",)
    print(f"\n--- Stored Memories for {user_id} ---")
    try:
        all_memories = memory_store._data.get(namespace, {})
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

def chat_with_membot(user_id: str) -> None:
    """Run a synchronous interactive chat loop for a specific user."""
    print(f"MemBot: Hi {user_id}! Ask me anything. (Type 'exit' to stop)")
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    config = {"configurable": {"thread_id": f"{user_id}_thread"}}
    namespace = (f"user_{user_id}",)

    # Create user-specific tools
    manage_memory_tool = create_manage_memory_tool(namespace=namespace)
    search_memory_tool = create_search_memory_tool(namespace=namespace)

    # Agent setup for this user
    agent = create_react_agent(
        model=llm,
        tools=[manage_memory_tool, search_memory_tool],
        store=memory_store,
        checkpointer=checkpointer,
        prompt=SYSTEM_PROMPT
    )

    try:
        while True:
            user_input = input(f"{user_id}: ").strip()
            if user_input.lower() == "exit":
                print(f"MemBot: Goodbye {user_id}!")
                break

            normalized_input = user_input.lower()
            conversation_history.append({"role": "user", "content": user_input})
            response = agent.invoke({"messages": conversation_history}, config=config)
            ai_response = response["messages"][-1].content if isinstance(response, dict) else str(response)
            print(f"MemBot: {ai_response}")

            conversation_history.append({"role": "assistant", "content": ai_response})
            memory_entry = f"User: {normalized_input} | Bot: {ai_response}"
            # Store via agent.invoke to avoid __pregel_store error
            agent.invoke(
                {"messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]},
                config=config
            )

            print_stored_memories(user_id)

    except Exception as e:
        print(f"Error occurred for {user_id}: {e}")
    except KeyboardInterrupt:
        print(f"\nMemBot: Goodbye {user_id}!")

def main():
    """Main function to start the chatbot for a user."""
    user_id = input("Enter your username: ").strip().lower()
    if not user_id:
        user_id = "default_user"
    chat_with_membot(user_id)

if __name__ == "__main__":
    main()