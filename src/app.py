import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
import os
load_dotenv()

def get_response(user_input):
  return "I dont know"

def get_vectorstore_from_url(url):
  #get text from website and convert it into vectorstore 
  loader = WebBaseLoader(url)
  document = loader.load()
  #document splitting 
  text_splitter = RecursiveCharacterTextSplitter()
  document_chunks = text_splitter.split_documents(document)
  #create vectorstore from document chunks
  embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
  )

  vector_store = Chroma.from_documents(
    documents=document_chunks,
    embedding=embeddings
  )
  return vector_store

def get_context_retriever_chain(vector_store):
  llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)
  retriever=vector_store.as_retriever()
  prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest follow-up question, generate a standalone search query to retrieve relevant information. If the follow-up question is unrelated to the chat history, use the latest question as the search query."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
  retriever_chain=create_history_aware_retriever(llm=llm,retriever=retriever,prompt=prompt)
  return retriever_chain
  
#app configuration
st.set_page_config(page_title="ask any website", page_icon="blue_book")
st.title("WEB-ASK")
if "chat_history" not in st.session_state:
  st.session_state.chat_history=[
  AIMessage(content="Hello,I am webask! Ask right away"),
  ]
with st.sidebar:
  st.header("settings")
  website_url = st.text_input("website url")
if website_url is None or  website_url == "":
  st.info("please enter a website url")
else:
  vector_store = get_vectorstore_from_url(website_url)
  retriever_chain = get_context_retriever_chain(vector_store)

  user_query = st.chat_input("Ask me!!")

  if user_query is not None and user_query != "":
      retrieved_docs = retriever_chain.invoke({
          "chat_history": st.session_state.chat_history,
          "input": user_query
      })

      st.write("Retrieved docs:", retrieved_docs)

      response = get_response(user_query)

      st.session_state.chat_history.append(
          HumanMessage(content=user_query)
      )
      st.session_state.chat_history.append(
          AIMessage(content=response)
      )

  #conversation 
  for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
      with st.chat_message("AI"):
       st.write(message.content)
    elif isinstance(message,HumanMessage):
      with st.chat_message("You"):
        st.write(message.content)
