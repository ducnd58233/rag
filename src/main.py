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


def initialize_rag_system() -> RAGSystem:
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
                            doc_info = {
                                "custom_metadata": metadata,
                                "document_stats": {
                                    "chunks_created": len(documents),
                                    "total_content_length": sum(
                                        len(doc.page_content) for doc in documents
                                    ),
                                },
                            }
                            st.json(doc_info)

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
                    if response.get("source_documents"):
                        sources = [
                            {
                                "score": doc.score,
                                "metadata": doc.payload.get("metadata", {}),
                                "content": doc.payload.get("text", ""),
                            }
                            for doc in response["source_documents"][:3]
                        ]

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["result"].content,
                            "sources": sources,
                        }
                    )

                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "sources": []}
                    )

            st.rerun()


if __name__ == "__main__":
    main()
