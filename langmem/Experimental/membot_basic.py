from http import client
import os
from langgraph.prebuilt import create_react_agent
from azure_openai_llm import get_llm

model = get_llm()

agent = create_react_agent(model, tools=[])

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