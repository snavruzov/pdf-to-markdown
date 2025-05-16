import pymupdf
import math

pymupdf.TOOLS.mupdf_display_warnings(False)


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


# ----- ocr needed -----
def ocr_needed(page: pymupdf.Page) -> bool:
    """
    Determines if OCR is needed for a given page based on a set of criteria.
    """
    if has_no_text(page):
        # If there's no text, OCR is likely needed,
        # but check if it's because it's covered by an image.
        if is_page_covered_by_image(page):
            return True
        # If not covered by an image and no text, could be a blank page or scanned
        # Further checks could be added here if needed, e.g. for blank pages.
        # For now, assume OCR is beneficial if no text is found.
        return True
    
    if has_weird_whitespace_ratio(page):
        # if there is more whitespace than text, it's probably a scanned page
        return True
    
    if has_many_small_vector_graphics(page):
        return True
    
    # Even if there is some text, it might be very sparse and the page
    # is mostly an image.
    if is_page_covered_by_image(page):
        # If the page is mostly an image, but some text was found,
        # it's a judgement call.
        # To prioritize OCR on image-heavy pages with little text, this is set to True.
        return True # Let other checks decide

    return False