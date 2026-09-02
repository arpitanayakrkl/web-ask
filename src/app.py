import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader

def get_response(user_input):
  return "I dont know"

def get_vectorstore_from_url(url):
  #get text from website and convert it into vectorstore 
  loader = WebBaseLoader(url)
  documents = loader.load()
  return documents


st.set_page_config(page_title="ask any website", page_icon="blue_book")
st.title("WEB-ASK")
if "chat_history" not in st.session_state:
  st.session_state.chat_history=[
  AIMessage(content="Hello,I am weba!! Ask right away"),
  ]
with st.sidebar:
  st.header("settings")
  website_url = st.text_input("website url")
if website_url is None or  website_url == "":
  st.info("please enter a website url")
else:
  documents = get_vectorstore_from_url(website_url)
  with st.sidebar:
    st.write(documents)

  #user input here 
  user_query=st.chat_input("Ask me!!")
  if user_query is not None and user_query !="":
    response=get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
  
  #conversation 
  for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
      with st.chat_message("AI"):
       st.write(message.content)
    elif isinstance(message,HumanMessage):
      with st.chat_message("You"):
        st.write(message.content)
