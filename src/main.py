import streamlit as st

from components import (
    chat_component,
    clear_chat_history,
    file_upload_component,
    save_uploaded_file,
)
from config import settings
from file_loader import FileLoader
from rag import RAGParams, RAGSystem


def initialize_rag_system():
    """Initialize the RAG system"""
    return RAGSystem(
        RAGParams(
            llm_model=settings.rag.llm_model,
            embedding_model=settings.rag.embedding_model,
            collection_name=settings.rag.collection_name,
            llm_api_key=settings.rag.openrouter_key,
        )
    )


def main():
    st.set_page_config(
        page_title="RAG Document Chat",
        page_icon=":robot:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(":robot: RAG Document Chat System")
    st.markdown("Upload documents and chat with your knowledge base!")

    # Initialize RAG system in session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = initialize_rag_system()

    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")

        if st.button("Clear Chat History"):
            clear_chat_history()
            st.session_state.rag_system.clear_history()
            st.success("Chat history cleared!")
            st.rerun()

        if st.button("Clear Knowledge Base"):
            st.warning("Knowledge base clearing not implemented yet")

        st.divider()

        # System info
        st.subheader("System Info")
        st.write(f"**LLM Model:** {settings.rag.llm_model}")
        st.write(f"**Embedding Model:** {settings.rag.embedding_model}")
        st.write(f"**Collection:** {settings.rag.collection_name}")

    tab1, tab2 = st.tabs(["Upload Documents", "Chat"])

    with tab1:
        st.header("Document Upload")

        # File upload component
        upload_result = file_upload_component()

        if upload_result:
            file_data = upload_result["file"]
            metadata = upload_result["metadata"]

            # Upload button
            if st.button("Upload to Knowledge Base", type="primary"):
                try:
                    with st.spinner("Processing document..."):
                        # Save file
                        file_path = save_uploaded_file(file_data)

                        # Load and split document
                        documents = FileLoader().load_and_split(
                            file_path=file_path,
                            custom_metadata=metadata,
                        )

                        # Add to RAG system
                        st.session_state.rag_system.add_documents(documents)

                        st.success(
                            f"Document '{file_data.name}' uploaded successfully!"
                        )
                        st.info(
                            f"Added {len(documents)} document chunks to knowledge base"
                        )

                        # Show document info
                        with st.expander("Document Details"):
                            st.json(metadata)

                except Exception as e:
                    st.error(f"Error uploading document: {str(e)}")

    with tab2:
        st.header("Chat with Your Documents")

        if (
            not hasattr(st.session_state.rag_system, "vector_store")
            or not st.session_state.rag_system.vector_store
        ):
            st.warning("No documents uploaded yet. Please upload some documents first!")
            return

        chat_result = chat_component()

        if chat_result and chat_result.get("text"):
            user_query = chat_result["text"]

            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_system.search(
                        query=user_query, score_threshold=0.1
                    )

                    sources = []
                    if response and response.get("source_documents"):
                        for doc in response["source_documents"][:3]:
                            try:
                                if hasattr(doc, "payload") and hasattr(doc, "score"):
                                    source_data = {
                                        "score": getattr(doc, "score", 0.0),
                                        "metadata": (
                                            doc.payload.get("metadata", {})
                                            if doc.payload
                                            else {}
                                        ),
                                        "content": (
                                            doc.payload.get("text", "")
                                            if doc.payload
                                            else ""
                                        ),
                                    }
                                    sources.append(source_data)
                            except Exception as source_error:
                                print(
                                    f"Error processing source document: {source_error}"
                                )
                                continue

                    result_content = "Sorry, I couldn't process your query."
                    if response and response.get("result"):
                        llm_result = response["result"]
                        try:
                            if hasattr(llm_result, "content"):
                                result_content = llm_result.content
                            elif hasattr(llm_result, "text"):
                                result_content = llm_result.text
                            else:
                                result_content = str(llm_result)
                        except Exception as content_error:
                            print(f"Error extracting content: {content_error}")
                            result_content = "Error extracting response content."

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result_content,
                            "sources": sources,
                        }
                    )

                except Exception as e:
                    import traceback

                    error_msg = f"Error processing query: {str(e)}"
                    print(f"Full error traceback: {traceback.format_exc()}")
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "sources": []}
                    )

            st.rerun()


if __name__ == "__main__":
    main()
