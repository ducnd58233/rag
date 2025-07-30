from .config import settings
from .file_loader import load_and_split
from .rag import RAGParams, RAGSystem

if __name__ == "__main__":
    rag1 = RAGSystem(
        RAGParams(
            llm_model=settings.rag.llm_model,
            embedding_model=settings.rag.embedding_model,
            collection_name=settings.rag.collection_name,
            llm_api_key=settings.rag.openrouter_key,
        )
    )
    documents = load_and_split(
        file_path="uploads/thue_tncn.pdf",
        custom_metadata={
            "department": "finance",
            "year": 2013,
            "tags": ["tax", "income", "regulation"],
        },
    )
    rag1.add_documents(documents)

    result1 = rag1.search(
        query="tiền thưởng kèm theo kỷ niệm chương được tính như thế nào",
        metadata_filter={
            "department": "finance",
            "year": 2013,
        },
        score_threshold=0.1,
    )

    rag2 = RAGSystem(
        RAGParams(
            llm_model=settings.rag.llm_model,
            embedding_model=settings.rag.embedding_model,
            collection_name=settings.rag.collection_name,
            llm_api_key=settings.rag.openrouter_key,
        )
    )
    result2 = rag2.search(
        query="thông tin các khoản chịu thuế",
        metadata_filter={
            "department": "finance",
            "year": {"lte": 2010},
        },
        score_threshold=0.1,
    )

    print("RAG result 1: ", result1["result"].content)
    print("RAG result 2: ", result2["result"].content)
