import hashlib
import json

from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    ScoredPoint,
    VectorParams,
)

from config import settings


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
    ):
        """
        Initialize Qdrant vector store manager.

        Args:
            collection_name (str): Name of the collection
            vector_size (int): Size of the embedding vectors
            qdrant_url (str): URL of the Qdrant server
        """
        self.collection_name = collection_name
        self.client = QdrantClient(url=settings.qdrant.url)
        self.embeddings = embeddings

    def create_collection(self) -> bool:
        """
        Create a new collection if it doesn't exist.

        Returns:
            bool: True if collection was created or already exists
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = {collection.name for collection in collections}

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self._get_embedding_dim(),
                        distance=Distance.COSINE,
                    ),
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
            return True
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            return False

    def _get_embedding_dim(self) -> int:
        return len(self.embeddings.embed_query("test"))

    def delete_collection(self) -> bool:
        """
        Delete the collection if it exists.

        Returns:
            bool: True if collection was deleted or didn't exist
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = {collection.name for collection in collections}

            if self.collection_name in collection_names:
                self.client.delete_collection(collection_name=self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} doesn't exist")
            return True
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            return False

    def _generate_id(self, document: Document) -> str:
        content_hash = hashlib.md5(
            (
                document.page_content + json.dumps(document.metadata, sort_keys=True)
            ).encode()
        ).hexdigest()
        return content_hash

    def add_documents(self, documents: list[Document]) -> bool:
        """
        Add documents to the collection.

        Args:
            documents (list[Document]): List of documents to add

        Returns:
            bool: True if documents were added successfully
        """
        try:
            if not self.create_collection():
                return False

            points = []
            for doc in documents:
                if not doc.page_content or not doc.page_content.strip():
                    continue

                embedding = self.embeddings.embed_query(doc.page_content)

                point = PointStruct(
                    id=self._generate_id(doc),
                    vector=embedding,
                    payload={
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"Added {len(points)} documents to collection {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False

    def _build_filter(self, metadata_filter: dict) -> Filter | None:
        if not metadata_filter:
            return None

        conditions = []
        for key, value in metadata_filter.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "gte":
                        conditions.append(
                            FieldCondition(key=f"metadata.{key}", range=Range(gte=val))
                        )
                    elif op == "lte":
                        conditions.append(
                            FieldCondition(key=f"metadata.{key}", range=Range(lte=val))
                        )
                    elif op == "gt":
                        conditions.append(
                            FieldCondition(key=f"metadata.{key}", range=Range(gt=val))
                        )
                    elif op == "lt":
                        conditions.append(
                            FieldCondition(key=f"metadata.{key}", range=Range(lt=val))
                        )
            elif isinstance(value, list):
                conditions.append(
                    FieldCondition(key=f"metadata.{key}", match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None

    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict = None,
        score_threshold: float = 0.1,
    ) -> list[ScoredPoint]:
        """
        Search the collection for documents similar to the query.

        Args:
            query (str): The query to search for
            k (int): The number of documents to return
            metadata_filter (dict): The metadata filter to apply to the search
            score_threshold (float): The score threshold to apply to the search

        Returns:
            list[Document]: List of documents similar to the query
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            qdrant_filter = self._build_filter(metadata_filter)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            )
            return results
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            dict: Collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {}
