import uuid

from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .config import settings


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        vector_size: int = settings.qdrant.vector_size,
    ):
        """
        Initialize Qdrant vector store manager.

        Args:
            collection_name (str): Name of the collection
            vector_size (int): Size of the embedding vectors
            qdrant_url (str): URL of the Qdrant server
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
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
                        size=self.vector_size,
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
                embedding = self.embeddings.embed_query(doc.page_content)
                point = PointStruct(
                    id=str(uuid.uuid4()),
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
            print(
                f"Added {len(documents)} documents to collection {self.collection_name}"
            )
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False

    def search(self, query: str, k: int = 5) -> list[Document]:
        """
        Search the collection for documents similar to the query.

        Args:
            query (str): The query to search for
            k (int): The number of documents to return

        Returns:
            list[Document]: List of documents similar to the query
        """
        query_embedding = self.embeddings.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
        )
        return results

    def get_relevant_documents(self, query: str, k: int = 5) -> list[Document]:
        """
        Search for similar documents using vector similarity.

        Args:
            query (str): Search query
            k (int): Number of results to return

        Returns:
            list[Document]: List of similar documents with scores
        """
        try:
            results = self.search(query, k)
            processed_results = []

            for scored_point in results:
                processed_results.append(
                    {
                        "text": scored_point.payload["text"],
                        "metadata": scored_point.payload.get("metadata", {}),
                        "score": scored_point.score,
                    }
                )

            return processed_results
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
