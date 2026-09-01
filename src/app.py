import streamlit as st
st.set_page_config(page_title="ask any website", page_icon="blue_book")
st.title("WEB-ASK")
with st.sidebar:
  st.header("settings")
  website_url = st.text_input("website url")

user_query=st.chat_input("Ask me!!")
if user_query is not None and user_query !="":
  with st.chat_message("Human"):
   st.write(user_query)
  with st.chat_message("AI"):
   st.write("Sorry,I don't know")
