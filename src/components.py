import os
from pathlib import Path

import streamlit as st


def file_upload_component() -> dict | None:
    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file to upload",
    )

    if uploaded_file is not None:
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size} bytes")
        st.write(f"**Type:** {uploaded_file.type}")

        st.subheader("Document Metadata")

        col1, col2 = st.columns(2)

        with col1:
            department = st.text_input(
                "Department", placeholder="e.g., finance, hr, legal"
            )
            year = st.number_input("Year", min_value=1900, max_value=2100, value=2025)

        with col2:
            tags = st.text_input(
                "Tags (comma-separated)", placeholder="e.g., tax, income, regulation"
            )
            priority = st.selectbox("Priority", ["low", "medium", "high"], index=1)

        tag_list = (
            [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        )

        metadata = {
            "department": department,
            "year": year,
            "tags": tag_list,
            "priority": priority,
            "filename": uploaded_file.name,
            "file_type": uploaded_file.type,
            "file_size": uploaded_file.size,
        }

        # Show metadata preview
        st.subheader("Metadata Preview")
        st.json(metadata)

        return {"file": uploaded_file, "metadata": metadata}

    return None


def display_sources(sources):
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for i, source in enumerate(sources):
            with st.expander(f"Source {i+1} - Score: {source['score']:.3f}"):
                if source.get("metadata"):
                    st.write("**Metadata:**")
                    st.json(source["metadata"])

                if source.get("content"):
                    st.write("**Content:**")
                    st.write(source["content"])


def display_chat_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                display_sources(message["sources"])


def chat_component() -> dict | None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown(
        """
        <style>
        [data-testid="stChatMessageContent"] p {
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }
        [data-testid="stChatMessageContent"] {
            font-size: 1rem !important;
        }
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if prompt := st.chat_input("Ask me anything about your uploaded documents..."):
        return {"text": prompt, "files": [], "images": []}

    display_chat_messages()

    return None


def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:
    Path(upload_dir).mkdir(exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def clear_chat_history():
    if "messages" in st.session_state:
        st.session_state.messages = []
