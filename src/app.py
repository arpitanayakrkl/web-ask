import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
def get_response(user_input):
  return "I dont know"

st.set_page_config(page_title="ask any website", page_icon="blue_book")
st.title("WEB-ASK")
if "chat_history" not in st.session_state:
  st.session_state.chat_history=[
  AIMessage(content="Hello,I am weba!! Ask right away"),
  ]
with st.sidebar:
  st.header("settings")
  website_url = st.text_input("website url")

user_query=st.chat_input("Ask me!!")
if user_query is not None and user_query !="":
  response=get_response(user_query)
  st.session_state.chat_history.append(HumanMessage(content=user_query))
  st.session_state.chat_history.append(AIMessage(content=response))
  with st.chat_message("Human"):
   st.write(user_query)
  with st.chat_message("AI"):
   st.write(response)
with st.sidebar:
  st.write(st.session_state.chat_history)
