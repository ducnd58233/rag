from pathlib import Path

from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, ElementType
from unstructured.partition.auto import partition
from unstructured.partition.utils.constants import PartitionStrategy

from config import settings


class FileLoader:
    def load_and_split(
        self,
        file_path: str,
        custom_metadata: dict = {},
    ) -> list[Document]:
        """
        Load a file and split it into chunks.

        Args:
            file_path (str): Path to the file
            custom_metadata (dict): Custom metadata to add to the document

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
                pdf_infer_table_structure=True,
                strategy=PartitionStrategy.HI_RES,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True,
            )

            if not elements:
                raise ValueError("No content could be extracted from the file")

            chunks = chunk_by_title(
                elements,
                max_characters=settings.file_loader.chunk_size,
                new_after_n_chars=256,
                combine_text_under_n_chars=settings.file_loader.chunk_overlap,
            )
            self._extract_images(chunks)
            documents = []

            for idx, element in enumerate(chunks):
                text = element.text
                if not text or not text.strip():
                    continue

                metadata = element.metadata.to_dict()
                if (
                    len(text) < settings.file_loader.chunk_overlap
                    and idx + 1 < len(chunks)
                    and metadata.get("page", -1) + 1
                    == chunks[idx + 1].metadata.to_dict().get("page", -1)
                ):
                    continue

                metadata.update(custom_metadata)

                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

            if not documents:
                raise ValueError(
                    "No valid document chunks could be created from the file"
                )

            return documents

        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    def _extract_images(self, chunks: list[Element]):
        images = []
        for chunk in chunks:
            if ElementType.COMPOSITE_ELEMENT in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if ElementType.IMAGE in str(type(el)):
                        images.append(el)

        chunks.extend(images)
