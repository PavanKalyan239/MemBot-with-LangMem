# import streamlit as st
# from inmemory_membot import agent, SYSTEM_PROMPT

# # Streamlit UI
# st.title("MemBot: Context-Aware Chatbot")

# # Initialize conversation history in session state
# if "conversation_history" not in st.session_state:
#     st.session_state.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# # Display conversation history
# for message in st.session_state.conversation_history:
#     if message["role"] == "user":
#         with st.chat_message("user"):
#             st.write(message["content"])
#     elif message["role"] == "assistant":
#         with st.chat_message("assistant"):
#             st.write(message["content"])

# # Chat input and logic
# def chat_with_membot():
#     config = {"configurable": {"thread_id": "default_thread"}}  # Simplified thread ID

#     user_input = st.chat_input("Ask MemBot something...")
#     if user_input:
#         # Add user input to history
#         st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
#         # Invoke the agent with the full conversation history
#         response = agent.invoke({"messages": st.session_state.conversation_history}, config=config)
#         ai_response = response["messages"][-1].content if isinstance(response, dict) else str(response)
        
#         # Add assistant response to history
#         st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
        
#         # Display the new user message and response
#         with st.chat_message("user"):
#             st.write(user_input)
#         with st.chat_message("assistant"):
#             st.write(ai_response)
        
#         # Store memory
#         memory_entry = f"User: {user_input.lower()} | Bot: {ai_response}"
#         agent.invoke(
#             {"messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]},
#             config=config
#         )

# # Run the chat function
# chat_with_membot()

import streamlit as st
from inmemory_membot import agent, SYSTEM_PROMPT

# Custom CSS for left (bot) and right (user) alignment
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
    }
    .user-message-container {
        display: flex;
        justify-content: flex-end; /* Push to the right */
        margin: 5px 0;
    }
    .user-message {
        padding: 10px;
        background-color: #DCF8C6;
        border-radius: 10px;
        max-width: 70%;
        color: black;
    }
    .bot-message-container {
        display: flex;
        justify-content: flex-start; /* Push to the left */
        margin: 5px 0;
    }
    .bot-message {
        padding: 10px;
        background-color: #E5E5EA;
        border-radius: 10px;
        max-width: 70%;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("MemBot: Context-Aware Chatbot")

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# Container for chat messages
chat_container = st.container()

# Display conversation history
with chat_container:
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message-container"><div class="user-message">{message["content"]}</div></div>', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f'<div class="bot-message-container"><div class="bot-message">{message["content"]}</div></div>', unsafe_allow_html=True)

# Chat input and logic
def chat_with_membot():
    config = {"configurable": {"thread_id": "default_thread"}}  # Simplified thread ID

    user_input = st.chat_input("Ask MemBot something...")
    if user_input:
        # Add user input to history and display it instantly
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        with chat_container:
            st.markdown(f'<div class="user-message-container"><div class="user-message">{user_input}</div></div>', unsafe_allow_html=True)
        
        # Process bot response separately
        response = agent.invoke({"messages": st.session_state.conversation_history}, config=config)
        ai_response = response["messages"][-1].content if isinstance(response, dict) else str(response)
        
        # Add assistant response to history and display it
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
        with chat_container:
            st.markdown(f'<div class="bot-message-container"><div class="bot-message">{ai_response}</div></div>', unsafe_allow_html=True)
        
        # Store memory
        memory_entry = f"User: {user_input.lower()} | Bot: {ai_response}"
        agent.invoke(
            {"messages": [{"role": "system", "content": f"Use manage_memory_tool to store: {memory_entry}"}]},
            config=config
        )

# Run the chat function
chat_with_membot()
