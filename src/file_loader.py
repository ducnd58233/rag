from pathlib import Path

from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from .config import settings


def load_and_split(file_path: str) -> list[Document]:
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
            max_characters=settings.file_loader.chunk_size,
            combine_text_under_n_chars=settings.file_loader.chunk_overlap,
        )
        documents = []

        for idx, element in enumerate(chunks):
            text = element.text
            metadata = element.metadata.to_dict()
            if (
                len(text) < settings.file_loader.chunk_overlap
                and idx + 1 < len(chunks)
                and metadata.get("page", -1) + 1
                == chunks[idx + 1].metadata.to_dict().get("page", -1)
            ):
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )
        return documents

    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")
