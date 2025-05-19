import pymupdf
import math
import logging
from typing import Optional

from pdf_to_markdown.table_detection_services.table_detection_service_interface import TableDetectionInterface, TableDetectionError
from pdf_to_markdown.pdf_handling.page_renderer import render_page_to_pil_image

pymupdf.TOOLS.mupdf_display_warnings(False)
logger = logging.getLogger(__name__)


# ----- heuristic functions -----

def has_no_text(page: pymupdf.Page, min_alphanum_ratio: float = 0.15) -> bool:
    """
    Checks if the page contains any meaningful text.
    Meaningful text is defined as having at least a certain ratio of alphanumeric characters
    to the total number of characters.
    Args:
        page: The pymupdf Page object.
        min_alphanum_ratio: The minimum ratio of alphanumeric characters to total characters
                            for the text to be considered meaningful.
    """
    text = page.get_text(flags=0).strip()
    if not text:
        return True
    
    text_length = len(text)
    if text_length == 0: # Should be caught by "if not text" but good for safety
        return True

    # Calculate the minimum number of alphanumeric characters required based on the ratio.
    # math.ceil ensures that if ratio > 0 and text exists, at least 1 alphanum char is needed.
    target_alphanum_needed = math.ceil(text_length * min_alphanum_ratio)
    # we increase the threshold to 40% of the text length if the text is very short
    if target_alphanum_needed <= 1:
        target_alphanum_needed = math.ceil(0.4*text_length)

    alphanum_count = 0
    for char in text:
        if char.isalnum():
            alphanum_count += 1
            if alphanum_count >= target_alphanum_needed:
                return False  # Found enough alphanumeric characters
    
    return True  # Not enough alphanumeric characters


def has_weird_whitespace_ratio(page: pymupdf.Page, max_allowed_whitespace_ratio: float = 0.36) -> bool:
    """
    Checks if the page has a large number of small vector graphics.
    This can indicate simulated text where each character is a small path.
    """
    text = page.get_text().strip()
    if not text:
        return False
    whitespace_count = sum(1 for char in text if char.isspace())
    total_chars = len(text)
    whitespace_ratio = whitespace_count / total_chars
    to_much = whitespace_ratio > max_allowed_whitespace_ratio
    # todo: maybe add a check to look at difference when setting get_text(flags=pymupdf.TEXT_INHIBIT_SPACES (setting this flag also removes pymupdf.TEXT_PRESERVE_WHITESPACE))
    return to_much


def has_many_small_vector_graphics(page: pymupdf.Page, path_threshold: int = 1000, avg_area_threshold: float = 10.0) -> bool:
    """
    Checks if the page has a large number of small vector graphics.
    This can indicate simulated text where each character is a small path.
    Args:
        page: The pymupdf Page object.
        path_threshold: The minimum number of paths to be considered "many".
        avg_area_threshold: The maximum average area for paths to be considered "small".
    """
    paths = page.get_drawings()
    if len(paths) < path_threshold:
        return False

    total_area = 0
    valid_paths_count = 0
    for path in paths:
        if path["rect"]:  # Ensure the path has a valid bounding box
            rect = path["rect"]
            total_area += rect.width * rect.height
            valid_paths_count +=1
    
    if valid_paths_count == 0: # Avoid division by zero
        return False

    average_area = total_area / valid_paths_count
    return average_area < avg_area_threshold


def is_page_covered_by_image(page: pymupdf.Page) -> bool:
    """
    Checks if the page is almost completely covered by images.
    An arbitrary threshold of 95% of the page area is used.
    """
    image_area = 0.0
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return False  # Avoid division by zero for empty pages

    for img_info in page.get_image_info(hashes=False, xrefs=False):
        img_w = img_info["width"]
        img_h = img_info["height"]
        image_area += img_w * img_h

    return (image_area / page_area) >= 0.95


# ----- Helper for table detection -----
def _check_for_tables_with_service(page: pymupdf.Page, table_detection_service: TableDetectionInterface) -> bool:
    """
    Renders a page and uses the provided table_detection_service to detect tables.
    Returns True if tables are detected, False otherwise or on error.
    """
    logger.debug(f"Page {page.number}: Checking for tables with TableDetectionService.")
    pil_image_for_table_detection = None
    DEFAULT_DPI_FOR_TABLE_DETECTION = 150 # Default if service gives no specific config
    try:
        render_config = table_detection_service.get_best_size()
        pil_image_for_table_detection = render_page_to_pil_image(
            page=page, 
            render_config=render_config, 
            default_dpi=DEFAULT_DPI_FOR_TABLE_DETECTION, 
            purpose="table detection"
        )
        
        if table_detection_service.detect_tables_on_page(pil_image_for_table_detection):
            logger.info(f"Page {page.number}: Tables DETECTED by TableDetectionService.")
            return True
        else:
            logger.debug(f"Page {page.number}: No tables detected by TableDetectionService.")
            return False
    except TableDetectionError as e:
        logger.error(f"Page {page.number}: TableDetectionService detection failed: {e}. Assuming no tables for this check.", exc_info=True)
    except Exception as e:
        logger.error(f"Page {page.number}: Unexpected error during TableDetectionService image rendering or detection: {e}. Assuming no tables.", exc_info=True)
    finally:
        if pil_image_for_table_detection:
            try:
                pil_image_for_table_detection.close()
            except Exception as e:
                logger.error(f"Page {page.number}: Error closing temporary PIL image for table detection: {e}", exc_info=True)
    return False # Default to False in case of any error or if no tables found


# ----- ocr needed -----
def ocr_needed(page: pymupdf.Page, table_detection_service: Optional[TableDetectionInterface] = None) -> bool:
    """
    Determines if OCR is needed for a given page based on a set of criteria,
    including an optional table detection service.
    """
    if has_no_text(page):
        # if is_page_covered_by_image(page):
        #     logger.debug(f"Page {page.number}: OCR needed (no text, covered by image).")
        #     return True
        logger.debug(f"Page {page.number}: OCR needed (no text, not primarily image).")
        return True
    
    if has_weird_whitespace_ratio(page):
        logger.debug(f"Page {page.number}: OCR needed (weird whitespace ratio).")
        return True
    
    if has_many_small_vector_graphics(page):
        logger.debug(f"Page {page.number}: OCR needed (many small vector graphics).")
        return True
    
    if is_page_covered_by_image(page):
        # This check is important. If it's mostly an image, even if some text exists,
        # OCR might be better to capture everything, including text within the image.
        logger.debug(f"Page {page.number}: OCR needed (covered by image, implies text might be part of image).")
        return True

    # If none of the above heuristics triggered, and a table_detection_service is provided, delegate to it.
    if table_detection_service:
        logger.debug(f"Page {page.number}: Standard heuristics passed. Delegating to TableDetectionService for table check.")
        if _check_for_tables_with_service(page, table_detection_service):
            # The helper function already logs its findings (detection or errors)
            # ocr_needed just uses the boolean result here.
            return True # OCR is needed if tables are found by the service

    logger.debug(f"Page {page.number}: OCR not needed by any heuristic or TableDetectionService (if active).")
    return False