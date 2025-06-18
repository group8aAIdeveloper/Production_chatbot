import streamlit as st
import requests

API_URL = "http://localhost:8000/api/message"

st.title("Ad Performance Chatbot")

st.markdown("**Top Priority Questions:**")
st.markdown("- What are the top 10 evergreen ads?")
st.markdown("- What are the top 10 microdata ads?")
st.markdown("- What are the trends?")
st.markdown("- Give me a summary of ads")
st.markdown("- Summary of ad_id 123456 (replace with your ad_id)")

question = st.text_input("Ask any question (or click a suggestion above):")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(API_URL, json={"query": question}, headers=headers)
                response.raise_for_status()
                result = response.json()
                st.success("Result:")
                st.markdown(f"**Response:** {result['text']}")
                with st.expander("View Detailed Data"):
                    st.json(result['data'])
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP Error: {http_err}, Response: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
