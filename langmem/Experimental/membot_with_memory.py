import os
from re import search
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from sentence_transformers import SentenceTransformer
from azure_openai_llm import get_llm

# Load Local Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embedding function for LangMem
def embed_func(text):
    return model.encode(text, convert_to_numpy=True)

# Setup Memory
store = InMemoryStore(index={"dims": 384, "embed": embed_func})

# Get Azure OpenAI LLM
llm = get_llm()

# Create LangMem tools
manage_memory = create_manage_memory_tool(namespace=('user_1',))
search_memory = create_search_memory_tool(namespace=('user_1',))


agent = create_react_agent(model=llm, tools=[manage_memory, search_memory], store=store)

# simple chat loop
def chat_with_membot():
    print("MemBot: Hi! Ask me anything. (Type 'exit' to stop)")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("MemBot: Goodbye!")
                break
            response = agent.invoke({'messages': [{"role": "user", "content": user_input}]})
            
            # Extract AI response (handling different return types)
            if isinstance(response, dict) and "messages" in response:
                ai_response = response["messages"][-1].content
            else:
                ai_response = str(response)  # Fallback

            print(f"MemBot: {ai_response}")
    except Exception as e:
        print("Error occurred: ", e)
    except KeyboardInterrupt:
        print("MemBot: Goodbye!")

if __name__ == "__main__":
    chat_with_membot()