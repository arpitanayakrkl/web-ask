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
from urllib.parse import urlparse
from supabase import create_client

load_dotenv()
def get_supabase_client():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
def is_valid_url(url):
    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed_url = urlparse(url)

    if parsed_url.scheme not in ["http", "https"]:
        return None

    if not parsed_url.netloc or "." not in parsed_url.netloc:
        return None

    return url
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    if not document:
        raise ValueError(
            "No content could be extracted from this website."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    document_chunks = text_splitter.split_documents(document)

    if not document_chunks:
        raise ValueError(
            "The website does not contain enough readable content."
        )

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

def get_conversational_rag_chain(retriever_chain):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question using the context below. If you don't know the answer, say so. Use only the provided context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    return create_retrieval_chain(
        retriever_chain,
        stuff_documents_chain
    )

def get_response(user_input):
    response = st.session_state.conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response

def reset_url():
    st.session_state.website_url = ""

    st.session_state.chat_history = [
        AIMessage(content="Hello, I am WEB-ASK! Ask right away.")
    ]

    st.session_state.loaded_url = None
    st.session_state.vector_store = None
    st.session_state.conversation_rag_chain = None

def clear_chat():
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am WEB-ASK! Ask right away.")
    ]
#app configuration
st.set_page_config(page_title="ask any website", page_icon="blue_book")
if "supabase" not in st.session_state:
    st.session_state.supabase = get_supabase_client()
st.sidebar.success("Supabase connected!")
st.title("WEB-ASK")
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "conversation_rag_chain" not in st.session_state:
    st.session_state.conversation_rag_chain = None
with st.sidebar:
    st.header("Enter Website URL")

    website_url = st.text_input(
        "Website URL",
        key="website_url"
    )

    st.button(
        "Reset URL",
        on_click=reset_url,
        use_container_width=True
    )
    st.button(
    "Clear Chat",
    on_click=clear_chat,
    use_container_width=True
)
if not website_url.strip():
    st.info("Enter a website URL to get started.")

else:
    validated_url = is_valid_url(website_url)

    if validated_url is None:
        st.error("Please enter a valid website URL.")

    else:
      #session state
      if "chat_history" not in st.session_state:
        st.session_state.chat_history=[
        AIMessage(content="Hello,I am webask! Ask right away"),
        ]
    
      if (
    "loaded_url" not in st.session_state
      or st.session_state.loaded_url != validated_url):
        try:
          with st.spinner("Processing website..."):
              new_vector_store = get_vectorstore_from_url(validated_url)

          st.session_state.vector_store = new_vector_store
          st.session_state.loaded_url = validated_url

          st.session_state.chat_history = [
              AIMessage(content="Hello, I am WEB-ASK! Ask right away.")
          ]

        except Exception as e:
            st.error(f"Unable to process website: {str(e)}")
            st.exception(e)
            st.stop()

      retriever_chain = get_context_retriever_chain(
          st.session_state.vector_store
      )

      conversation_rag_chain = get_conversational_rag_chain(
          retriever_chain
      )

      st.session_state.conversation_rag_chain = conversation_rag_chain

      # Display previous messages
      for message in st.session_state.chat_history:
          if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

                sources = message.additional_kwargs.get("sources", [])

                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.markdown(f"- [{source}]({source})")
          elif isinstance(message, HumanMessage):
              with st.chat_message("user"):
                  st.write(message.content)

      user_query = st.chat_input("Ask me!!")

      if user_query:
        response = get_response(user_query)

        answer = response["answer"]

        # Extract source URLs
        sources = list(dict.fromkeys(
            doc.metadata.get("source")
            for doc in response["context"]
            if doc.metadata.get("source")
        ))

        st.session_state.chat_history.append(
            HumanMessage(content=user_query)
        )

        st.session_state.chat_history.append(
            AIMessage(
                content=answer,
                additional_kwargs={"sources": sources}
            )
        )

        st.rerun()