from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from .vector_store import QdrantVectorStore


@dataclass
class RAGParams:
    llm_model: str = "meta-llama/llama-3.1-8b-instruct"
    embedding_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    collection_name: str = "uploaded_documents"
    llm_api_key: str = ""
    temperature: float = 0.1
    request_timeout_seconds: int = 30


class RAGSystem:
    def __init__(self, params: RAGParams):
        self.llm = ChatOpenAI(
            model=params.llm_model,
            api_key=params.llm_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=params.temperature,
            request_timeout=params.request_timeout_seconds,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=params.embedding_model,
        )
        self.vector_store = QdrantVectorStore(
            collection_name=params.collection_name,
            embeddings=self.embeddings,
        )
        self.history = ""

    def add_documents(self, documents: list[Document]):
        self.vector_store.add_documents(documents)

    def _create_prompt(self) -> PromptTemplate:
        template = """
You are helpfull assistant that was create by "ducnd58233".
Use the following history of this conversation and pieces of context to answer the question about the story at the end.
If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer.
Pay attention to the context of the question rather than just looking for similar keywords in the corpus.
Always say "thanks for asking!" at the end of the answer. Generate answer by only Vietnamese.
Please reranking following context given query as question before answer the question. each context was separated by "---"
\n---\n
History: {history}
\n---\n
Context: {context}
\n---\n
Question: {question}
Helpful Answer:
"""
        return PromptTemplate(
            template=template,
            input_variables=["history", "context", "question"],
        )

    def query(self, query: str):
        results = self.vector_store.get_relevant_documents(query, k=10)

        context = "\n---\n".join([doc["text"] for doc in results])

        prompt = self._create_prompt().format(
            history=self.history,
            context=context,
            question=query,
        )

        answer = self.llm.invoke(prompt)
        self.history += f"query: {query} \n answer: {answer.content}"

        return {
            "result": answer,
            "source_documents": results,
        }

    def clear_history(self):
        self.history = ""
