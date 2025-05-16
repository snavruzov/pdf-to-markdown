from .pdf_handling.pdf_to_markdown import pdf_to_markdown, pdf_to_markdown_sync
from .ocr_services.ocr_service_interface import OCRInterface, OCRProcessingError
from .ocr_services.llm_based_ocr_service import LLMBasedOCRService
from .pdf_handling.ocr_needed import ocr_needed

__all__ = [
    "pdf_to_markdown",
    "pdf_to_markdown_sync",
    "OCRInterface",
    "OCRProcessingError",
    "LLMBasedOCRService",
    "ocr_needed",
]

# Optional: Define a package version
__version__ = "0.1.0"
