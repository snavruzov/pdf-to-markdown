import pymupdf
import io
import logging
from typing import Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

def render_page_to_pil_image(
    page: pymupdf.Page, 
    render_config: Optional[Union[tuple[int, int], int]], 
    default_dpi: int,
    purpose: str = "processing" # e.g., "OCR", "table detection" for logging
) -> Image.Image:
    """
    Renders a PDF page to a PIL Image object based on the provided configuration.

    Args:
        page: The pymupdf.Page object to render.
        render_config: The desired rendering configuration:
                       - A tuple (width, height) for specific dimensions.
                       - An int for a specific DPI.
                       - None to use the default_dpi.
        default_dpi: The DPI to use if render_config is None or if dimensions are zero.
        purpose: A string describing the purpose of rendering, for logging.

    Returns:
        A PIL.Image.Image object.
        
    Raises:
        ValueError: If the page dimensions are invalid for zoom calculation.
    """
    page_rect = page.rect
    current_w, current_h = page_rect.width, page_rect.height
    pix = None

    if isinstance(render_config, tuple):  # (width, height)
        w, h = render_config
        if current_w > 0 and current_h > 0:
            zoom_w = w / current_w
            zoom_h = h / current_h
            zoom = min(zoom_w, zoom_h)  # Maintain aspect ratio, fit within target dimensions
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
            logger.debug(f"Page {page.number}: Rendering for {purpose} with zoom {zoom:.2f} (target: {w}x{h}).")
        else:
            logger.warning(
                f"Page {page.number} has zero width or height. "
                f"Using default DPI {default_dpi} for {purpose} rendering."
            )
            # Fallback to default DPI
            pix = page.get_pixmap(dpi=default_dpi, colorspace=pymupdf.csRGB)
            logger.debug(f"Page {page.number}: Rendering for {purpose} with fallback DPI {default_dpi} due to zero page dimensions.")
    elif isinstance(render_config, int):  # DPI
        dpi_to_use = render_config
        pix = page.get_pixmap(dpi=dpi_to_use, colorspace=pymupdf.csRGB)
        logger.debug(f"Page {page.number}: Rendering for {purpose} with DPI {dpi_to_use}.")
    else:  # None or unexpected
        if render_config is not None:
            logger.warning(
                f"Page {page.number}: Unexpected render_config type '{type(render_config)}' "
                f"for {purpose}. Using default DPI {default_dpi}."
            )
        pix = page.get_pixmap(dpi=default_dpi, colorspace=pymupdf.csRGB)
        logger.debug(f"Page {page.number}: Rendering for {purpose} with default DPI {default_dpi} (no specific config).")

    if pix is None: # Should not happen if logic above is correct, but as a safeguard
        logger.error(f"Page {page.number}: Pixmap creation failed for {purpose}. Using emergency fallback rendering.")
        pix = page.get_pixmap(dpi=default_dpi, colorspace=pymupdf.csRGB)

    img_bytes = pix.pil_tobytes(format="PNG")
    pil_image = Image.open(io.BytesIO(img_bytes))
    return pil_image 