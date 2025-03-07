import streamlit as st
import requests, http

st.title("Rags' Rag")
st.caption("Enter the next apparel you want to make and our engineers will give you advice to make the best apparel possible.")
with st.form("my_form"):
    prompt = st.text_input("Enter Prompt", placeholder="Your next idea", key="prompt_text")
    form_submit_btn = st.form_submit_button("Send")

if form_submit_btn:
    try:
        response = requests.post(url="http://127.0.0.1:8000/ask_llm/", json={"prompt": prompt})

        if response.status_code == 200:
            st.balloons()
            st.markdown(response.json())
    except:
        st.write("Error when querying the API")
    