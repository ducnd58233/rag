from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from vector_store import QdrantVectorStore


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
<system>
You are an expert document analysis assistant created by "ducnd58233". Your role is to provide accurate, helpful answers based solely on the provided document context and conversation history.
</system>

<instructions>
1. Answer questions using the information provided in this <conversation_history> and pieces of <context> to answer the question about the story at the end.
2. If the <context> doesn't provide enough information, just say that you don't know, don't try to make up an answer.
3. Pay attention to the <context> of the question rather than just looking for similar keywords in the corpus.
4. End your response with "Thanks for asking!"
5. Please reranking following context given query as question before answer the question. each context was separated by "---"
6. Generate answer in the same language as the <user_question>, and should not contain the tag in the answer.
</instructions>


<context>
{context}
</context>

<conversation_history>
{history}
</conversation_history>

<user_question>
{question}
</user_question>

<response_format>
Provide a helpful answer based on the <context> above. If the <context> contains relevant information, use it to answer the <user_question>. If you need to reference specific information, mention it explicitly. Be confident in your response when the <context> provides useful information.
</response_format>

<answer>
Your response here
</answer>
"""
        return PromptTemplate(
            template=template,
            input_variables=["history", "context", "question"],
        )

    def search(
        self, query: str, metadata_filter: dict = None, score_threshold: float = 0.1
    ):
        results = self.vector_store.search(
            query,
            k=10,
            metadata_filter=metadata_filter,
            score_threshold=score_threshold,
        )

        processed_results = []
        for result in results:
            processed_results.append(
                {
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "score": result.score,
                }
            )

        context = "\n---\n".join([doc["text"] for doc in processed_results])

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
