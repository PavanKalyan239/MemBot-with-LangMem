# import os
# from langgraph.prebuilt import create_react_agent
# from langgraph.store.memory import InMemoryStore
# from langmem import create_manage_memory_tool, create_search_memory_tool
# from sentence_transformers import SentenceTransformer
# from azure_openai_llm import get_llm

# # Load local embedding model
# model = SentenceTransformer('all-MiniLM-L12-v2')

# # Embedding function for LangMem
# def embed_func(text):
#     return model.encode(text, convert_to_numpy=True)

# # Setup memory store
# store = InMemoryStore(index={"dims": 384, "embed": embed_func})

# # Get Azure ChatGPT model
# llm = get_llm()

# # Create memory tools with namespace
# manage_memory = create_manage_memory_tool(namespace=("user_1",))
# search_memory = create_search_memory_tool(namespace=("user_1",))

# # Create agent with memory tools
# agent = create_react_agent(
#     model=llm, 
#     tools=[manage_memory, search_memory],
#     store=store
# )

# def print_stored_memories():
#     print("\n--- Stored Memories ---")
#     if hasattr(store, "_data") and isinstance(store._data, dict):
#         all_namespaces = list(store._data.keys())
#     else:
#         print("Error: Cannot retrieve stored keys.")
#         return

#     if not all_namespaces:
#         print("No memories stored yet.")
#     else:
#         for namespace in all_namespaces:
#             for key, item in store._data[namespace].items():
#                 try:
#                     value = item.value if item else None
#                     print(f"Memory: Namespace={namespace}, Key={key}, Value={value}")
#                 except Exception as e:
#                     print(f"Error retrieving memory for namespace {namespace}, key {key}: {e}")
#     print("----------------------\n")

# # Chat loop with memory
# def chat_with_membot():
#     print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
#     try:
#         while True:
#             user_input = input("You: ")
#             if user_input.lower() == 'exit':
#                 print("MemBot: Goodbye!")
#                 break
            
#             # Run the agent with memory
#             response = agent.invoke({'messages': [{"role": "user", "content": user_input}]})
            
#             # Extract AI response (FIXED)
#             if isinstance(response, dict) and "messages" in response:
#                 ai_response = response["messages"][-1].content  # FIXED
#             else:
#                 ai_response = str(response)
            
#             print(f"MemBot: {ai_response}")
            
#             # Print what’s stored after each turn
#             print_stored_memories()

#     except Exception as e:
#         print("Error occurred: ", e)
#     except KeyboardInterrupt:
#         print("MemBot: Goodbye!")

# if __name__ == "__main__":
#     chat_with_membot()


import os
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from sentence_transformers import SentenceTransformer
from azure_openai_llm import get_llm

# Load local embedding model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Embedding function for LangMem
def embed_func(text):
    return model.encode(text, convert_to_numpy=True)

# Setup memory store
store = InMemoryStore(index={"dims": 384, "embed": embed_func})

# Get Azure ChatGPT model
llm = get_llm()

# Create memory tools with namespace
manage_memory = create_manage_memory_tool(namespace=("user_1",))
search_memory = create_search_memory_tool(namespace=("user_1",))

# Create agent with memory tools
agent = create_react_agent(
    model=llm,
    tools=[manage_memory, search_memory],
    store=store
)

def print_stored_memories():
    print("\n--- Stored Memories ---")
    if hasattr(store, "_data") and isinstance(store._data, dict):
        all_namespaces = list(store._data.keys())
    else:
        print("Error: Cannot retrieve stored keys.")
        return

    if not all_namespaces:
        print("No memories stored yet.")
    else:
        for namespace in all_namespaces:
            for key, item in store._data[namespace].items():
                try:
                    value = item.value if item else None
                    print(f"Memory: Namespace={namespace}, Key={key}, Value={value}")
                except Exception as e:
                    print(f"Error retrieving memory for namespace {namespace}, key {key}: {e}")
    print("----------------------\n")

# Chat loop with memory
def chat_with_membot():
    print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
    
    # Initialize conversation history
    conversation_history = [
        {"role": "assistant", "content": "Hi! Ask me anything. (Type 'exit' to stop)"}
    ]

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("MemBot: Goodbye!")
                break

            # Add user input to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # System prompt to ensure memory is checked for every response
            system_prompt = (
                "You are MemBot, a helpful AI assistant. For every user input, "
                "Store all the query and response in the InMemoryStore. "
                "use the search_memory tool to check the InMemoryStore for relevant information "
                "Incorporate any relevant memory into your response. If the user states a preference "
                "use manage_memory to store it"
                "For all the queries check the relevant respone in the InMemoryStore. "
            )

            # Combine system prompt with conversation history
            messages = [{"role": "system", "content": system_prompt}] + conversation_history

            # Run the agent with memory tools
            response = agent.invoke({'messages': messages})
            
            # Extract AI response
            if isinstance(response, dict) and "messages" in response:
                ai_response = response["messages"][-1].content
            else:
                ai_response = str(response)
            
            # Add agent response to conversation history
            conversation_history.append({"role": "assistant", "content": ai_response})

            print(f"MemBot: {ai_response}")
            print_stored_memories()

    except Exception as e:
        print("Error occurred: ", e)
    except KeyboardInterrupt:
        print("MemBot: Goodbye!")

if __name__ == "__main__":
    chat_with_membot()