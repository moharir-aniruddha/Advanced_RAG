import streamlit as st
import requests
import uuid
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Corporate AI Assistant", page_icon="🤖")

# --- SESSION INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")

    # 1. Ingestion Section
    st.header("Data Ingestion")
    url_input = st.text_area("Enter URLs (one per line)", placeholder="https://example.com")

    if st.button("🚀 Train on URLs", use_container_width=True):
        urls = [u.strip() for u in url_input.split("\n") if u.strip()]
        if urls:
            with st.spinner("Analyzing documents..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/ingest",
                        json={"urls": urls}
                    )
                    if resp.status_code == 200:
                        st.success("Knowledge Base Updated!")
                    else:
                        st.error("Ingestion failed.")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.divider()

    # 2. System Management (The Clear Feature)
    st.header("System Management")
    if st.button("🗑️ Wipe Knowledge Base", help="Deletes all ingested data and chat history", use_container_width=True):
        with st.spinner("Clearing system..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/clear")
                if resp.status_code == 200:
                    # Clear local UI state immediately
                    st.session_state.messages = []
                    # Generate a new session ID for a fresh start
                    st.session_state.session_id = str(uuid.uuid4())
                    st.success("System Reset Complete!")
                    # Refresh the page to show empty chat
                    st.rerun()
                else:
                    st.error("Failed to clear database.")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

# --- MAIN CHAT INTERFACE ---
st.title("🤖 Advanced RAG Bot")
st.markdown("Interact with your web documentation in real-time.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Logic
if prompt := st.chat_input("Ask me anything about the ingested docs..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"session_id": st.session_state.session_id, "query": prompt}
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    sources = data.get("sources", [])

                    # Display Answer
                    st.markdown(answer)

                    # Display Sources if they exist
                    if sources:
                        with st.expander("View Sources"):
                            for s in sources:
                                st.write(f"- {s}")

                    # Update history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Backend Error (Status {response.status_code})")
            except Exception as e:
                st.error(f"Failed to reach backend: {e}")

# Custom Styling
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .stMarkdown p { font-size: 1.1rem; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)