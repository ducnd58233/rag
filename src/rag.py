from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from llm import OpenRouterClient, OpenRouterParams
from vector_store import QdrantVectorStore


@dataclass
class RAGParams:
    llm_model: str
    embedding_model: str
    collection_name: str
    llm_api_key: str
    temperature: float | None = 0.1
    request_timeout_seconds: int | None = 30


class RAGSystem:
    def __init__(self, params: RAGParams):
        self.llm_client = OpenRouterClient(
            OpenRouterParams(
                llm_model=params.llm_model,
                embedding_model=params.embedding_model,
                collection_name=params.collection_name,
                llm_api_key=params.llm_api_key,
                temperature=params.temperature,
            )
        )
        self.vector_store = QdrantVectorStore(
            collection_name=params.collection_name,
            embeddings=self.llm_client.get_embeddings(),
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
        try:
            results = self.vector_store.search(
                query,
                k=10,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
            )

            processed_results = []
            images_base64 = []

            for result in results:
                processed_results.append(
                    {
                        "text": result.payload.get("text", ""),
                        "metadata": result.payload.get("metadata", {}),
                        "score": result.score,
                    }
                )
                if result.payload.get("metadata", {}).get("image_base64"):
                    images_base64.append(
                        result.payload.get("metadata", {}).get("image_base64")
                    )

            context = "\n---\n".join([doc["text"] for doc in processed_results])

            prompt = self._create_prompt().format(
                history=self.history,
                context=context,
                question=query,
            )

            answer = self.llm_client.generate(prompt, images_base64)
            self.history += f"query: {query} \n answer: {answer.content}"

            return {
                "result": answer,
                "source_documents": results,
            }
        except Exception as e:
            raise Exception(f"Error in RAG search: {str(e)}")

    def clear_history(self):
        self.history = ""
