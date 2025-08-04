from dataclasses import dataclass
from typing import Any

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
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={
                "normalize_embeddings": True,
                "truncate_input": True,
            },
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
1. Answer questions using the information provided in this <conversation_history> and pieces of <context> to answer the question.
2. If the <context> doesn't provide enough information, just say that you don't know, don't try to make up an answer.
3. Pay attention to the <context> of the question rather than just looking for similar keywords.
4. End your response with "\nThanks for asking!"
5. Please reranking following context given query as question before answer the question. each context was separated by "---"
6. Generate answer in the same language as the <user_question>, and should not contain the tag in the answer.

For tables and images in the context:
- Format tables clearly when referencing them
- Describe images and their content when mentioned
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

    def _process_context(self, processed_results: list[dict[str, Any]]) -> str:
        contexts = []

        for doc in processed_results:
            text = doc["text"]
            metadata = doc["metadata"]
            print(metadata)

            context_parts = [text]

            content_info = []
            if metadata.get("content_type"):
                content_type = metadata["content_type"]
                page_info = f"Page {metadata.get('page', 'unknown')}"
                content_info.append(f"[{content_type.upper()} - {page_info}]")

                if content_type == "table" and metadata.get("table_caption"):
                    content_info.append(f"Table Caption: {metadata['table_caption']}")
                elif content_type == "figure" and metadata.get("figure_caption"):
                    content_info.append(f"Figure Caption: {metadata['figure_caption']}")
                elif content_type == "picture" and metadata.get("picture_caption"):
                    content_info.append(
                        f"Picture Caption: {metadata['picture_caption']}"
                    )

                    if metadata.get("image_format"):
                        content_info.append(f"Image Format: {metadata['image_format']}")

            # Add general document metadata flags
            flags = []
            if metadata.get("contains_table"):
                flags.append("📊 Contains Table")
            if metadata.get("contains_figure"):
                flags.append("📈 Contains Figure")
            if metadata.get("contains_picture"):
                flags.append("🖼️ Contains Picture")

            if flags:
                content_info.extend(flags)

            if content_info:
                context_parts.extend(content_info)

            contexts.append("\n".join(context_parts))

        return "\n---\n".join(contexts)

    def search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.1,
    ) -> dict[str, Any]:
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

        context = self._process_context(processed_results)

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
