from pathlib import Path

from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition


class FileLoader:
    def load_and_split(self, file_path: str) -> list[Document]:
        """
        Load a file and split it into chunks.

        Args:
            file_path (str): Path to the file

        Returns:
            list[Document]: List of document chunks
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            elements = partition(
                filename=str(file_path),
                detect_language_per_element=True,
            )

            chunks = chunk_by_title(
                elements,
                max_characters=524,
                combine_text_under_n_chars=64,
            )
            documents = []

            for element in chunks:
                text = element.text
                metadata = element.metadata
                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata.to_dict(),
                    )
                )
            return documents

        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
