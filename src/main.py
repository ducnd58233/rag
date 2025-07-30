from .config import settings
from .file_loader import load_and_split
from .rag import RAGParams, RAGSystem

if __name__ == "__main__":
    rag = RAGSystem(
        RAGParams(
            llm_model=settings.rag.llm_model,
            embedding_model=settings.rag.embedding_model,
            collection_name=settings.rag.collection_name,
            llm_api_key=settings.rag.openrouter_key,
        )
    )
    documents = load_and_split(file_path="uploads/thue_tncn.pdf")
    rag.add_documents(documents)
    question = "tiền thưởng kèm theo kỷ niệm chương được tính như thế nào"
    result = rag.query(question)
    print(result)
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
