import os
from pathlib import Path
from typing import Any

import streamlit as st


def file_upload_component() -> dict[str, Any] | None:
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
            "file_size": uploaded_file.size,
        }

        # Show metadata preview
        st.subheader("Metadata Preview")
        st.json(metadata)

        return {"file": uploaded_file, "metadata": metadata}

    return None


def display_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for i, source in enumerate(sources):
            metadata = source.get("metadata", {})
            content = source.get("content", "")

            # Create a more descriptive title for the source
            source_title = f"Source {i+1} - Score: {source['score']:.3f}"

            # Add content type indicators
            content_types = []
            if metadata.get("contains_table"):
                content_types.append("📊 Table")
            if metadata.get("contains_figure") or metadata.get("contains_picture"):
                content_types.append("🖼️ Image")
            if content_types:
                source_title += f" ({', '.join(content_types)})"

            with st.expander(source_title):
                # Display content type badges
                if content_types:
                    cols = st.columns(len(content_types))
                    for j, content_type in enumerate(content_types):
                        with cols[j]:
                            st.info(content_type)

                # Display page information
                if metadata.get("page"):
                    st.write(f"**Page:** {metadata['page']}")

                # Display captions if available
                captions = []
                if metadata.get("table_caption"):
                    captions.append(f"**Table Caption:** {metadata['table_caption']}")
                if metadata.get("figure_caption"):
                    captions.append(f"**Figure Caption:** {metadata['figure_caption']}")
                if metadata.get("picture_caption"):
                    captions.append(
                        f"**Picture Caption:** {metadata['picture_caption']}"
                    )

                if captions:
                    for caption in captions:
                        st.write(caption)

                # Display section headings
                if metadata.get("headings"):
                    st.write(f"**Section:** {', '.join(metadata['headings'])}")

                # Display content with special formatting for tables
                if content:
                    st.write("**Content:**")
                    if metadata.get("contains_table"):
                        # Try to format table content nicely
                        st.code(content, language="text")
                    else:
                        st.write(content)

                # Show full metadata in collapsible section
                with st.expander("Full Metadata"):
                    st.json(metadata)


def display_chat_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                display_sources(message["sources"])


def chat_component() -> dict[str, Any] | None:
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


def save_uploaded_file(uploaded_file: Any, upload_dir: str = "uploads") -> str:
    Path(upload_dir).mkdir(exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def clear_chat_history() -> None:
    if "messages" in st.session_state:
        st.session_state.messages = []
