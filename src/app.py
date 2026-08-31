import streamlit as st
st.set_page_config(page_title="ask any website", page_icon="blue_book")
st.title("WEB-ASK")
with st.sidebar:
  st.header("settings")
  website_url = st.text_input("website url")