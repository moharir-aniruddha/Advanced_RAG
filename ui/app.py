import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Corporate AI Assistant", page_icon="🤖")

# Initialize Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Advanced RAG Bot")
st.markdown("Interact with your web documentation in real-time.")

# Sidebar for URL Ingestion
with st.sidebar:
    st.header("Data Ingestion")
    url_input = st.text_area("Enter URLs (one per line)")
    if st.button("Train on URLs"):
        urls = [u.strip() for u in url_input.split("\n") if u.strip()]
        if urls:
            with st.spinner("Analyzing documents..."):
                resp = requests.post("http://localhost:8000/ingest", json={"urls": urls})
                if resp.status_code == 200:
                    st.success("Knowledge Base Updated!")
                else:
                    st.error("Ingestion failed.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me anything about the ingested docs..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {"session_id": st.session_state.session_id, "query": prompt}
            response = requests.post("http://localhost:8000/chat", json=payload)

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]

                # Display Answer
                st.markdown(answer)
                if sources:
                    with st.expander("View Sources"):
                        for s in sources:
                            st.write(f"- {s}")

                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Failed to get response from backend.")

# In ui/app.py
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stMarkdown p {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)