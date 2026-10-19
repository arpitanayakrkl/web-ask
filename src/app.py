import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_groq import ChatGroq
import os
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
load_dotenv()


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

def get_conversational_rag_chain(vector_store):
  llm=ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
  )
  prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses the following pieces of context to answer the users question. If you don't know the answer, just say you don't know, don't try to make up an answer. focus on the context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
  
  stuff_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
  return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
  retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
  conversation_rag_chain= get_conversational_rag_chain(retriever_chain)
  response =conversation_rag_chain.invoke({
          "chat_history":st.session_state.chat_history,
          "input":user_query
        })
  return response['answer']
#app configuration
st.set_page_config(page_title="ask any website", page_icon="blue_book")
st.title("WEB-ASK")

with st.sidebar:
  st.header("settings")
  website_url = st.text_input("website url")
if website_url is None or  website_url == "":
  st.info("please enter a website url")
else:
  #session state
  if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
    AIMessage(content="Hello,I am webask! Ask right away"),
    ]
 
  if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url(website_url)

  retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
  conversation_rag_chain= get_conversational_rag_chain(retriever_chain)
  user_query = st.chat_input("Ask me!!")

  if user_query is not None and user_query != "":
      response=get_response(user_query)

      st.write(response)

  #conversation 
  for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
      with st.chat_message("AI"):
       st.write(message.content)
    elif isinstance(message,HumanMessage):
      with st.chat_message("You"):
        st.write(message.content)
