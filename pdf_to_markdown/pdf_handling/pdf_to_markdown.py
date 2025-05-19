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
from pdf_to_markdown.table_services.table_service_interface import TableInterface, TableDetectionError
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
    show_progress: bool = False) -> Dict[int, str]:
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
            table_strategy=None, # "lines_strict", # Other options are: lines, lines_strict and text -> handled by OCR
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
    table_service: Optional[TableInterface] = None,
    pages: Optional[List[int]] = None,
    force_ocr: bool = False,
    show_progress: bool = False) -> dict[int, str]:
    """
    Convert a PDF file to markdown text.
    
    Args:
        pdf_source: PDF source, can be a file path (str or pathlib.Path),
                    bytes, an io.BytesIO stream, or a pymupdf.Document object.
        image_ocr_service: Service to use for OCR on images.
        table_service: Service to use for table detection
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
                if ocr_needed(doc_to_process[page_nr], table_service=table_service):
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
            tasks.append(process_readable_pages(doc_to_process, image_ocr_service, readable_pages, show_progress))
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
    table_service: Optional[TableInterface] = None,
    pages: Optional[List[int]] = None,
    force_ocr: bool = False,
    show_progress: bool = False) -> dict[int, str]:
    """
    Synchronous wrapper for the pdf_to_markdown function.
    Converts a PDF file to markdown text.
    
    Args:
        pdf_source: PDF source, can be a file path (str or pathlib.Path),
                    bytes, an io.BytesIO stream, or a pymupdf.Document object.
        image_ocr_service: Service to use for OCR on images.
        table_service: Service to use for table detection
        pages: List of page indices to process, or None for all pages.
        force_ocr: Whether to force OCR on all pages.
        show_progress: Whether to show progress bars.
        
    Returns:
        A dictionary mapping page numbers to their markdown content.
    """
    # Ensure nest_asyncio is applied if an event loop is already running,
    # which can happen in environments like Jupyter notebooks.
    # Users might need to install it: pip install nest_asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If a loop is running, asyncio.run() cannot be called directly.
            # nest_asyncio allows reentrant usage of asyncio.run().
            # We attempt to import and apply it. If not available, users might face issues.
            try:
                import nest_asyncio
                nest_asyncio.apply()
                logger.info("nest_asyncio applied for synchronous call within a running event loop.")
            except ImportError:
                logger.warning(
                    "Running in an active event loop. Consider installing 'nest_asyncio' "
                    "or use the asynchronous 'pdf_to_markdown' function directly."
                )
    except RuntimeError:
        # No event loop is currently set, asyncio.run() will create a new one.
        pass

    return asyncio.run(pdf_to_markdown(
        pdf_source=pdf_source,
        image_ocr_service=image_ocr_service,
        table_service=table_service,
        pages=pages,
        force_ocr=force_ocr,
        show_progress=show_progress
    ))
