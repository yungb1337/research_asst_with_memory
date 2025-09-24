import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
api_key = st.secrets["api"]["key"]
)

st.title("Research Assistant with Memory")
#history
if "history" not in st.session_state:
    st.session_state.history = []



#Chat Prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Assistant and you have to mimic like {assistant} for the rest of the conversation."),
    ("human", "Tell me about {topic}.")
])


#User input
assistantP = st.text_input("Which person you want me to act like(My favourite is Cortana)")
topic = st.text_input("What is your Question")


#Button Logic
if st.button("Click Me!") and topic:
    messages = chat_prompt.format_messages(
    assistant = assistantP,
    topic= topic
    )

    full_conv = st.session_state.history + messages
    result = llm.invoke(full_conv) 

    # Save both user message and assistant reply to history
    st.session_state.history.append({"role": "human", "content": topic})
    if hasattr(result, "content"):
        reply = result.content
    else:
        reply = str(result)
    st.session_state.history.append({"role": "assistant", "content": reply})

for msg in reversed(st.session_state.history):
    display_role = assistantP if msg["role"] == "assistant" else "You"
    st.markdown(f"**{display_role}:** {msg['content']}")









