from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    AsciiDocFormatOption,
    DocumentConverter,
    ExcelFormatOption,
    HTMLFormatOption,
    ImageFormatOption,
    MarkdownFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types import DoclingDocument
from langchain_core.documents import Document
from transformers import AutoTokenizer

from config import settings


class FileLoader:
    def __init__(self):
        self.chunker = self._create_chunker()
        self.converter = DocumentConverter(format_options=self._create_format_options())

    def _create_chunker(self) -> HybridChunker:
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(
                settings.rag.embedding_model, clean_up_tokenization_spaces=True
            ),
            max_tokens=settings.file_loader.max_chunk_length,
        )
        return HybridChunker(
            tokenizer=tokenizer, merge_peers=settings.file_loader.merge_peers
        )

    def _create_format_options(self) -> dict[InputFormat, Any]:
        accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice[settings.file_loader.accelerator_device],
            num_threads=settings.file_loader.num_threads,
        )

        pdf_pipeline_options = PdfPipelineOptions(
            images_scale=settings.file_loader.images_scale,
            generate_page_images=settings.file_loader.generate_page_images,
            generate_picture_images=settings.file_loader.generate_picture_images,
            do_ocr=settings.file_loader.do_ocr,
            do_table_structure=settings.file_loader.do_table_structure,
            accelerator_options=accelerator_options,
        )

        return {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.PPTX: PowerpointFormatOption(
                pipeline_options=pdf_pipeline_options
            ),
            InputFormat.XLSX: ExcelFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.HTML: HTMLFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.ASCIIDOC: AsciiDocFormatOption(),
            InputFormat.MD: MarkdownFormatOption(),
        }

    def _convert_chunks_to_documents(
        self,
        chunks: list[Any],
        custom_metadata: dict[str, Any],
    ) -> list[Document]:
        documents = []

        for chunk in chunks:
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue

            try:
                chunk_metadata = chunk.meta.export_json_dict()
            except Exception:
                chunk_metadata = {}

            if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                for item in chunk.meta.doc_items:
                    if hasattr(item, "label"):
                        label = str(item.label).lower()
                        if "table" in label:
                            chunk_metadata["has_table"] = True
                            if hasattr(item, "caption"):
                                chunk_metadata["table_caption"] = item.caption
                        elif "figure" in label:
                            chunk_metadata["has_figure"] = True
                            if hasattr(item, "caption"):
                                chunk_metadata["figure_caption"] = item.caption
                        elif "picture" in label:
                            chunk_metadata["has_picture"] = True
                            if hasattr(item, "caption"):
                                chunk_metadata["picture_caption"] = item.caption
                            # Add image metadata
                            if hasattr(item, "image") and item.image:
                                chunk_metadata["image_format"] = getattr(
                                    item.image, "format", None
                                )
                                chunk_metadata["image_size"] = getattr(
                                    item.image, "size", None
                                )

            final_metadata = {**chunk_metadata, **custom_metadata}

            documents.append(Document(page_content=chunk_text, metadata=final_metadata))

        return documents

    def load_and_split(
        self,
        file_path: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Load a file and split it into chunks using Docling.

        Supports all Docling input formats: PDF, DOCX, XLSX, PPTX, HTML,
        Images (PNG, JPEG, TIFF, BMP, WEBP), Markdown, AsciiDoc, CSV.

        Args:
            file_path: Path to the file
            custom_metadata: Custom metadata to add to documents

        Returns:
            list[Document]: List of document chunks
        """
        if custom_metadata is None:
            custom_metadata = {}

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            conversion_result = self.converter.convert(file_path)
            docling_doc: DoclingDocument = conversion_result.document

            chunks = list(self.chunker.chunk(docling_doc))
            documents = self._convert_chunks_to_documents(
                chunks,
                custom_metadata,
            )

            return documents

        except Exception as e:
            raise Exception(f"Error processing file '{file_path}': {str(e)}")
