import asyncio
import os
import tempfile
import io
import logging
import pathlib
from typing import List, Optional, Union, Dict

# Third-party imports
import pymupdf
import pymupdf4llm
import re
from PIL import Image
from tqdm import tqdm

# Local imports
from pdf_to_markdown.ocr_services.ocr_service_interface import OCRInterface, OCRProcessingError
from pdf_to_markdown.pdf_handling.ocr_needed import ocr_needed
from pdf_to_markdown.table_detection_services.table_detection_service_interface import TableDetectionInterface, TableDetectionError
from pdf_to_markdown.pdf_handling.page_renderer import render_page_to_pil_image

pymupdf4llm_img_ref_pattern = re.compile(r"!\[\]\((.*)\)")

logger = logging.getLogger(__name__)

# path variable seems to be for local testing, can be kept or removed as per project standards
# path = "/Users/dries/Downloads/ARBEDISREGLEMENT BEDIENDE BKSI PACKING DEFINITIEF 02112023.pdf"


def create_pymupdf4llm_img_ref(img_path: str) -> str:
    """
    Create a markdown image reference for pymupdf4llm.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Markdown image reference string
    """
    img_path = os.path.realpath(img_path)
    return f'![]({img_path})'


async def ocr_pages(
    doc: pymupdf.Document, 
    image_ocr_service: OCRInterface,
    show_progress: bool,
    pages: Optional[List[int]] = None) -> Dict[int, str]:
    """
    Batch process PDF pages with OCR.
    
    Args:
        doc: The PyMuPDF Document object.
        image_ocr_service: OCR service to use.
        show_progress: Whether to show a progress bar.
        pages: List of page indices to process, or None for all pages.
        
    Returns:
        Dictionary mapping page numbers to extracted text.
    """
    results = {}
    pil_images_for_ocr: List[Image.Image] = []
    page_indices: List[int] = []
    
    to_process = list(range(len(doc))) if pages is None else pages
        
    logger.info(f"Preparing {len(to_process)} pages for OCR by rendering them.")
    
    DEFAULT_DPI_FOR_OCR = 300 # Default if service gives no specific config

    iterator = tqdm(to_process, total=len(to_process), desc="Rendering pages for OCR") if show_progress else to_process
        
    for page_num in iterator:
        page = doc[page_num]
        try:
            render_config = image_ocr_service.get_best_size()
            pil_image = render_page_to_pil_image(
                page=page, 
                render_config=render_config, 
                default_dpi=DEFAULT_DPI_FOR_OCR, 
                purpose="OCR"
            )
            # pil_image.show() # Uncomment for debugging to see the rendered image
            pil_images_for_ocr.append(pil_image)
            page_indices.append(page_num)
        except Exception as e:
            logger.error(f"Error rendering page {page_num} for OCR: {e}. Skipping this page for OCR.", exc_info=True)
            # Optionally, add a placeholder or specific error message for this page in results
            # results[page_num] = f"Error rendering page: {e}"
            continue # Skip to the next page
    
    if pil_images_for_ocr:
        try:
            logger.info(f"Sending {len(pil_images_for_ocr)} rendered pages to OCR service.")
            batch_results = await image_ocr_service.aocr_image(pil_images_for_ocr)
            results.update({page_nr: batch_results[i] for i, page_nr in enumerate(page_indices) if i < len(batch_results)})
        except OCRProcessingError as e:
            logger.error(f"OCR processing failed for a batch of pages: {e}. These pages might be missing from results.", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during batch OCR call: {e}. These pages might be missing from results.", exc_info=True)
        finally:
            # Close PIL images after OCR processing to free resources
            for img in pil_images_for_ocr:
                try:
                    img.close()
                except Exception as e_close:
                    logger.warning(f"Error closing PIL image after OCR: {e_close}", exc_info=True)
    
    return results


async def process_readable_pages(
    doc: pymupdf.Document,
    image_ocr_service: OCRInterface, 
    pages: List[int],
    show_progress: bool = False,
    table_strategy: Optional[str] = None) -> Dict[int, str]:
    """
    Process readable pages and return the results.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # this method is quite computationally intensive, so we use asyncio to run it in a separate thread
        # TODO: what about complex tables? How do we detect and extract them?
        md_text = await asyncio.to_thread(
            pymupdf4llm.to_markdown,
            doc,
            pages=pages,
            write_images=True,
            image_path=tmpdir,
            image_format="jpg",
            show_progress=show_progress,
            embed_images=False, # images are stored as references in the markdown
            table_strategy=table_strategy, # "lines_strict", # Other options are: lines, lines_strict and text -> handled by OCR
            image_size_limit=0.02, # default = 0.05
            dpi=300, # high resolution for better OCR
        )
        
        images = []
        image_paths = []
        
        # Collect all images from the temporary directory
        for image_file in os.listdir(tmpdir):
            img_path = os.path.join(tmpdir, image_file)
            image_paths.append(img_path)
            image_obj = Image.open(img_path)
            images.append(image_obj)
        
        parsed_contents = await image_ocr_service.aocr_image(images)
        
        for image in images:
            image.close()
        parsed_images = {path: content for path, content in zip(image_paths, parsed_contents)}

        # Replace image references with OCR content
        new_md_text_parts = []
        last_original_md_span_end = 0
        
        for reference in pymupdf4llm_img_ref_pattern.finditer(md_text):
            reference_span = reference.span()
            begin_span, end_span = reference_span[0], reference_span[1]
            img_ref = reference.group(1)
            img_content = parsed_images.get(img_ref, "")
            
            first_part = md_text[last_original_md_span_end:begin_span]
            new_md_text_parts.extend([
                first_part,
                "\n----start image description----\n",
                img_content,
                "\n----end image description----\n"
            ])
            last_original_md_span_end = end_span + 1
            
        new_md_text_parts.append(md_text[last_original_md_span_end:])
        md_text = "".join(new_md_text_parts)
    
    # split long markdown text into pages
    page_list = md_text.split("\n-----\n")
    if not pages:
        return {page_nr: page for page_nr, page in enumerate(page_list)}
    else:
        return {page_nr: page for page_nr, page in zip(pages, page_list)}
    

async def pdf_to_markdown(
    pdf_source: Union[str, pathlib.Path, bytes, io.BytesIO, pymupdf.Document],
    image_ocr_service: OCRInterface, 
    table_detection_service: Optional[TableDetectionInterface] = None,
    pages: Optional[List[int]] = None,
    force_ocr: bool = False,
    show_progress: bool = False) -> dict[int, str]:
    """
    Convert a PDF file to markdown text.
    
    Args:
        pdf_source: PDF source, can be a file path (str or pathlib.Path),
                    bytes, an io.BytesIO stream, or a pymupdf.Document object.
        image_ocr_service: Service to use for OCR on images.
        table_detection_service: Service to use for table detection
        pages: List of page indices to process, or None for all pages.
        force_ocr: Whether to force OCR on all pages.
        show_progress: Whether to show progress bars.
        
    Returns:
        A dictionary mapping page numbers to their markdown content.
    """
    
    doc_to_process: Optional[pymupdf.Document] = None
    opened_internally = False

    if isinstance(pdf_source, pymupdf.Document):
        doc_to_process = pdf_source
    elif isinstance(pdf_source, (str, pathlib.Path)):
        doc_to_process = pymupdf.open(filename=str(pdf_source)) # type: ignore
        opened_internally = True
    elif isinstance(pdf_source, (bytes, io.BytesIO)):
        doc_to_process = pymupdf.open(stream=pdf_source, filetype="pdf") # type: ignore
        opened_internally = True
    else:
        raise TypeError(
            f"Unsupported pdf_source type: {type(pdf_source)}. "
            "Expected str, pathlib.Path, bytes, io.BytesIO, or pymupdf.Document."
        )

    if doc_to_process is None: # Should be caught by TypeError, but as a safeguard
        raise ValueError("Failed to load PDF document from the provided source.")

    try:
        total_pages = len(doc_to_process)

        if isinstance(pages, list):
            page_numbers = [p for p in pages if 0 <= p < total_pages]
        else:
            page_numbers = list(range(total_pages))
        
        pages_needing_ocr = []
        readable_pages = []
        if not force_ocr:
            page_iterator = tqdm(page_numbers, desc="Analyzing pages for initial OCR needs") if show_progress else page_numbers
            for page_nr in page_iterator:
                if ocr_needed(doc_to_process[page_nr], table_detection_service=table_detection_service):
                    pages_needing_ocr.append(page_nr)
                else:
                    readable_pages.append(page_nr)
        else:
            logger.info("Force OCR is enabled. All pages initially marked for OCR.")
            pages_needing_ocr = list(page_numbers) # Ensure it's a copy if page_numbers is an iterator/range
            readable_pages = []

        # Log final classification counts
        logger.info(f"Final page classification: {len(pages_needing_ocr)} pages for OCR, {len(readable_pages)} pages for direct Markdown conversion.")
            
        tasks = []
        if pages_needing_ocr:
            tasks.append(ocr_pages(
                doc=doc_to_process,
                image_ocr_service=image_ocr_service,
                show_progress=show_progress,
                pages=pages_needing_ocr
            ))
        else:
            # Create an empty future or result for OCR pages if none need OCR
            # This ensures asyncio.gather gets two results as expected by subsequent merging logic.
            ocr_future = asyncio.Future()
            ocr_future.set_result({})
            tasks.append(ocr_future)

        if readable_pages:
            table_strategy = "lines_strict" if table_detection_service is None else None
            tasks.append(process_readable_pages(doc_to_process, image_ocr_service, readable_pages, show_progress, table_strategy))
        else:
            readable_future = asyncio.Future()
            readable_future.set_result({})
            tasks.append(readable_future)
            
        # Results will be a list of two dictionaries, one from ocr_pages, one from process_readable_pages
        task_results = await asyncio.gather(*tasks)
        
        # Merge the results from OCRed pages and readable pages
        final_results: Dict[int, str] = {}
        for res_dict in task_results:
            if isinstance(res_dict, dict):
                final_results.update(res_dict)
        return final_results
    finally:
        if opened_internally and doc_to_process:
            doc_to_process.close()
            logger.info("Internally opened PDF document closed.")

def pdf_to_markdown_sync(
    pdf_source: Union[str, pathlib.Path, bytes, io.BytesIO, pymupdf.Document],
    image_ocr_service: OCRInterface, 
    table_detection_service: Optional[TableDetectionInterface] = None,
    pages: Optional[List[int]] = None,
    force_ocr: bool = False,
    show_progress: bool = False) -> dict[int, str]:
    """
    Synchronously convert a PDF file to markdown text.
    This is a wrapper around `pdf_to_markdown`.

    Args:
        pdf_source: PDF source (file path, bytes, stream, or pymupdf.Document).
        image_ocr_service: Service for OCR on images.
        table_detection_service: Service for table detection.
        pages: Optional list of page indices to process.
        force_ocr: Whether to force OCR on all pages.
        show_progress: Whether to show progress bars.

    Returns:
        A dictionary mapping page numbers to markdown content.
    """
    # Ensure nest_asyncio is applied if we are in a context (like Jupyter) where an event loop might already be running.
    # This is a common pattern for libraries that want to offer a sync API by running an async backend.
    try:
        import nest_asyncio
        nest_asyncio.apply()
        logger.debug("nest_asyncio applied for pdf_to_markdown_sync.")
    except ImportError:
        logger.debug("nest_asyncio not found, proceeding without it. This might cause issues in certain environments like Jupyter notebooks if an event loop is already running.")
    except RuntimeError as e: # a RuntimeError may be raised if an event loop is already running and nest_asyncio is not installed or cannot apply
        logger.warning(f"Could not apply nest_asyncio in pdf_to_markdown_sync: {e}. This might cause issues in some environments.")

    # Helper to run the async function in a new event loop or existing one if patched by nest_asyncio
    def run_async_in_sync_context():
        return asyncio.run(pdf_to_markdown(
            pdf_source=pdf_source,
            image_ocr_service=image_ocr_service,
            table_detection_service=table_detection_service,
            pages=pages,
            force_ocr=force_ocr,
            show_progress=show_progress
        ))
    
    # Attempt to run, handling potential event loop issues
    try:
        return run_async_in_sync_context()
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e) or \
           "Event loop is closed" in str(e) or \
           "There is no current event loop in thread" in str(e): # Also handle case where no loop exists
            
            logger.warning(f"RuntimeError with event loop in pdf_to_markdown_sync: {e}. Attempting to manage loop explicitly.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(pdf_to_markdown(
                    pdf_source=pdf_source,
                    image_ocr_service=image_ocr_service,
                    table_detection_service=table_detection_service,
                    pages=pages,
                    force_ocr=force_ocr,
                    show_progress=show_progress
                ))
                return result
            finally:
                loop.close()
                asyncio.set_event_loop(None) # Clean up the event loop from the current thread
        else:
            raise # Re-raise other runtime errors
