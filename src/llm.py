from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


@dataclass
class OpenRouterParams:
    llm_model: str = "meta-llama/llama-3.1-8b-instruct"
    embedding_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    collection_name: str = "uploaded_documents"
    llm_api_key: str = ""
    temperature: float = 0.1
    request_timeout_seconds: int = 30
    system_prompt: str = ""


class OpenRouterClient:
    def __init__(self, params: OpenRouterParams):
        self._llm = ChatOpenAI(
            model=params.llm_model,
            api_key=params.llm_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=params.temperature,
            request_timeout=params.request_timeout_seconds,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=params.embedding_model,
        )
        self._system_prompt = params.system_prompt

    def get_embeddings(self) -> Embeddings:
        return self._embeddings

    def get_llm(self) -> BaseChatModel:
        return self._llm

    def generate(self, prompt: str, images_base64: list[str] = None) -> BaseMessage:
        try:
            messages: list[BaseMessage] = []

            if self._system_prompt:
                messages.append(SystemMessage(content=self._system_prompt))

            if not images_base64:
                messages.append(HumanMessage(content=prompt))
            else:
                content = [{"type": "text", "text": prompt}]

                for image_base64 in images_base64:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        }
                    )

                messages.append(HumanMessage(content=content))

            return self._llm.invoke(messages)
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
