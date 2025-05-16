import asyncio
# import tempfile # No longer needed
import io
import logging
import sys
from typing import Optional, List, Any, BinaryIO

from markitdown import (
    MarkItDown,
    DocumentConverter,
    DocumentConverterResult,
    StreamInfo,
    MissingDependencyException,
    FailedConversionAttempt,
)

# Import pymupdf here for use in convert method, already checked for dependency
import pymupdf

from pdf_to_markdown.pdf_handling.pdf_to_markdown import pdf_to_markdown
from pdf_to_markdown.ocr_services.ocr_service_interface import OCRInterface

from pdf_to_markdown.ocr_services.llm_based_ocr_service import LLMBasedOCRService

logger = logging.getLogger(__name__)

__plugin_interface_version__ = 1

# --- Dependency Checking --- 
_dependency_exc_info_pymupdf = None
_dependency_exc_info_pymupdf4llm = None

try:
    if 'pymupdf' not in sys.modules: # Check if already imported by the top-level import
        import pymupdf # Try to import if not already (should be, but defensive)
except ImportError:
    _dependency_exc_info_pymupdf = sys.exc_info()

try:
    import pymupdf4llm
except ImportError:
    _dependency_exc_info_pymupdf4llm = sys.exc_info()
# --- End Dependency Checking ---

# Define accepted MIME types and extensions for PDFs
ACCEPTED_MIME_TYPES = [
    "application/pdf",
]
ACCEPTED_FILE_EXTENSIONS = [".pdf"]


def register_converters(markitdown: MarkItDown, **kwargs_from_markitdown_init):
    """
    Called during construction of MarkItDown instances to register converters provided by plugins.
    kwargs_from_markitdown_init contains arguments passed to MarkItDown's constructor.
    """
    # Extract potential OCR configurations from kwargs
    # These keys are based on what MarkItDown seems to use internally or pass around
    ocr_service_instance = kwargs_from_markitdown_init.get("custom_pdf_ocr_service") # Explicit override
    llm_client = kwargs_from_markitdown_init.get("llm_client")
    llm_model = kwargs_from_markitdown_init.get("llm_model")
    docintel_endpoint = kwargs_from_markitdown_init.get("docintel_endpoint")
    docintel_credential = kwargs_from_markitdown_init.get("docintel_credential")

    converter = MuPdfMarkitdownConverter(
        ocr_service=ocr_service_instance,
        llm_client=llm_client,
        llm_model=llm_model,
        docintel_endpoint=docintel_endpoint,
        docintel_credential=docintel_credential
        # Pass any other relevant kwargs if your converter's __init__ needs them
    )
    markitdown.register_converter(converter, priority=-1.0)  # Lower than built-in PDF converter

class MuPdfMarkitdownConverter(DocumentConverter):
    """
    A custom MarkItDown converter for PDF files using the project's pdf_to_markdown logic.
    Relies on pymupdf, pymupdf4llm, and a configured OCRInterface implementation.
    """

    def __init__(
        self,
        ocr_service: Optional[OCRInterface] = None,
        llm_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
        **other_config # For future expansion or other direct configs
    ):
        super().__init__()
        self.ocr_service: Optional[OCRInterface] = None
        self._initialize_ocr_service(
            direct_ocr_service=ocr_service,
            llm_client=llm_client,
            llm_model=llm_model,
            **other_config
        )

        if not self.ocr_service:
            logger.error(
                "MuPdfMarkitdownConverter: OCR Service COULD NOT BE INITIALIZED. "
                "PDF text extraction will be severely limited or fail. "
                "Provide 'custom_pdf_ocr_service', or 'llm_client'/'llm_model', or 'docintel_endpoint'."
            )
            # Depending on strictness, you might raise an error here
            # raise ValueError("OCR Service could not be initialized for CustomPdfMarkitdownConverter")

    def _initialize_ocr_service(
        self,
        direct_ocr_service: Optional[OCRInterface] = None,
        llm_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
        **other_config
    ):
        if direct_ocr_service is not None and isinstance(direct_ocr_service, OCRInterface):
            logger.info("Using directly provided OCR service instance.")
            self.ocr_service = direct_ocr_service
            return
        
        # Attempt to initialize based on general LLM client/model config
        if llm_client and llm_model:
            try:
                # Extract show_progress from other_config or default to False
                show_progress = other_config.pop('show_progress', False)
                self.ocr_service = LLMBasedOCRService(
                    llm_client=llm_client,
                    llm_model=llm_model,
                    show_progress=show_progress,
                    **other_config
                )
                logger.info(f"Initialized OCR service using LLMBasedOCRService with model: {llm_model}, show_progress={show_progress}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize LLMBasedOCRService: {e}. OCR will not be available.")
        
        logger.warning("No suitable configuration found to initialize an OCR service for CustomPdfMarkitdownConverter.")

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        """
        Checks if this converter can handle the given file stream.
        """
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in ACCEPTED_FILE_EXTENSIONS:
            return True

        for accepted_mimetype in ACCEPTED_MIME_TYPES:
            if mimetype.startswith(accepted_mimetype): # Use startswith for flexibility e.g. application/pdf;charset=...
                return True

        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        """
        Converts a PDF file stream to Markdown.
        MarkItDown calls this method with a file-like object (stream).
        """
        # --- Dependency Check --- 
        if _dependency_exc_info_pymupdf is not None:
            raise MissingDependencyException(
                f"Missing dependency for {type(self).__name__} (.pdf via custom plugin): pymupdf for custom PDF processing is required."
            ) from _dependency_exc_info_pymupdf[1].with_traceback( # type: ignore[union-attr]
                _dependency_exc_info_pymupdf[2]
            )
        if _dependency_exc_info_pymupdf4llm is not None:
            raise MissingDependencyException(
                f"Missing dependency for {type(self).__name__} (.pdf via custom plugin): pymupdf4llm for custom PDF to Markdown conversion is required."
            ) from _dependency_exc_info_pymupdf4llm[1].with_traceback( # type: ignore[union-attr]
                _dependency_exc_info_pymupdf4llm[2]
            )
        # --- End Dependency Check ---

        if not self.ocr_service:
            logger.error(
                "CustomPdfMarkitdownConverter: OCR Service was not initialized. "
                "Create a ocr service that implements OCRInterface and provide it to the markdown constructor with argument: 'ocr_service'. "
                "Or provide an (openai api supporting) llm_client and llm_model to the constructor with arguments: 'llm_client' and 'llm_model'. "
                "Cannot process PDF effectively."
            )
            raise FailedConversionAttempt("OCR Service not initialized for this PDF converter plugin.")

        pdf_bytes = file_stream.read()
        pdf_bytes_io = io.BytesIO(pdf_bytes)
        
        markdown_content = ""
        title_from_pdf: Optional[str] = None

        doc_for_title: Optional[pymupdf.Document] = None
        try:
            if 'pymupdf' not in sys.modules:
                 raise ImportError("PyMuPDF was not properly imported despite passing initial check for title extraction.")
            doc_for_title = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") # type: ignore
            title_from_pdf = doc_for_title.metadata.get('title')
        except Exception as e:
            logger.warning(f"Could not extract title from PDF: {e}")
        finally:
            if doc_for_title is not None:
                doc_for_title.close()

        try:
            # kwargs here are from the MarkItDown.convert() call, not from __init__.
            # They could contain per-conversion overrides if needed.
            pages_to_process: Optional[List[int]] = kwargs.get("pages")
            force_ocr_flag: bool = kwargs.get("force_ocr", False)
            show_progress_flag: bool = kwargs.get("show_progress", False)
            
            self.ocr_service.show_progress = show_progress_flag
            
            raw_results_dict = asyncio.run(
                pdf_to_markdown(
                    pdf_source=pdf_bytes_io, 
                    image_ocr_service=self.ocr_service, # Use the initialized service
                    pages=pages_to_process,
                    force_ocr=force_ocr_flag,
                    show_progress=show_progress_flag,
                )
            )
            
             # Add page number as a heading for each page
            sorted_page_numbers = sorted(raw_results_dict.keys())
            combined_md_parts = [
                f"# Page {pn+1}\n\n{raw_results_dict[pn]}" for pn in sorted_page_numbers
            ]
            markdown_content = "\n\n-----\n\n".join(combined_md_parts)


        except Exception as e:
            logger.error(f"Error during PDF to Markdown conversion: {e}", exc_info=True)
            markdown_content = f"# Error during PDF conversion\n\nAn error occurred: {e}"
        
        return DocumentConverterResult(
            title=title_from_pdf,
            markdown=markdown_content
        )
