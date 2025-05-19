import logging
from typing import List, Any, Optional
import asyncio
import base64
import io
import re # Import re module
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception, wait_random

from PIL import Image # Pillow's Image

from pdf_to_markdown.ocr_services.ocr_service_interface import OCRInterface, OCRProcessingError

# Add tqdm import
try:
    from tqdm import tqdm
    import tqdm.asyncio as tqdm_asyncio
except ImportError:
    tqdm = None  # fallback if tqdm is not installed
    tqdm_asyncio = None

logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = """\
Parse this image in structured markdown format.
-   Extract all text and tables from this image in structured markdown format.
-   Prioritize the logical relationships between data elements. Structure the output to clearly represent these relationships, even if it deviates from the original layout. Focus on grouping related information (e.g., field labels with their values, table headers with corresponding data).
-   Use whitespace, indentation, and line breaks to enhance readability and indicate hierarchy. Avoid strictly mirroring the visual layout if it hinders clarity.
-   For complex (nested) tables, ensure that the logical structure is paramount. Individual cell values and nested tables should be clearly separated and grouped to maintain their semantic connections.
-   Use [x] or [ ] syntax for checkboxes.
-   If the image contains no discernible text or tables, then give a detailed caption of the image.
-   Please, only return the structured markdown text, no other text, confirmation or comments.
-   Don't surround the text with markdown code fences (e.g. ```markdown).
"""

# Helper function for tenacity to identify rate limit errors
def _is_rate_limit_error(exception: Exception) -> bool:
    """Checks if the exception is a rate limit error for tenacity in a client-agnostic way."""
    # Check by status code (common for HTTP-based clients)
    if hasattr(exception, 'status_code') and getattr(exception, 'status_code') == 429:
        logger.warning(f"Rate limit error (status 429) detected: {exception}. Retrying...")
        return True
    # Check by exception class name (common pattern, e.g., openai.RateLimitError)
    if exception.__class__.__name__ == 'RateLimitError':
        logger.warning(f"RateLimitError (by class name) detected: {exception}. Retrying...")
        return True

    return False

class LLMBasedOCRService(OCRInterface):
    """
    An OCR service implementation that uses a general-purpose Large Language Model (LLM)
    capable of performing OCR tasks (e.g., describing images with text content).
    Assumes an OpenAI-compatible client for chat completions with image input.
    """

    def __init__(self, llm_client: Any, llm_model: str, show_progress: bool, **kwargs):
        """
        Initializes the LLM-based OCR service.

        Args:
            llm_client: The language model client (e.g., OpenAI client instance).
                       It must have a `chat.completions.create` method.
            llm_model: The specific model name to use (e.g., "gpt-4o").
            **kwargs: Additional keyword arguments for configuration.
                      Can include 'ocr_prompt' to customize the prompt.
        """
        if llm_client is None or llm_model is None:
            raise ValueError("llm_client and llm_model must be provided for LLMBasedOCRService.")

        # Defensive check for OpenAI client compatibility
        if not hasattr(llm_client, 'chat') or \
           not hasattr(llm_client.chat, 'completions') or \
           not hasattr(llm_client.chat.completions, 'create'):
            raise ValueError(
                "llm_client does not appear to be OpenAI-compatible. "
                "It must have a `chat.completions.create` method."
            )

        self._create_is_async = asyncio.iscoroutinefunction(llm_client.chat.completions.create)
        
        self.show_progress = show_progress
        self.llm_client = llm_client
        self.llm_model = llm_model
        # Default prompt focuses on text extraction, but can be overridden
        self.ocr_prompt = kwargs.get("ocr_prompt", DEFAULT_OCR_PROMPT)
        #self.max_tokens = kwargs.get("max_tokens", 2000) # Max tokens for the LLM response

        logger.info(f"LLMBasedOCRService initialized with model: {self.llm_model}, client type: {type(llm_client).__name__}, create_is_async: {self._create_is_async}, prompt: '{self.ocr_prompt[:50]}...'")

    def get_best_size(self) -> Optional[tuple[int, int]]:
        """
        Returns the optimal image size (width, height) for this OCR model.
        GPT-4o (and similar models) typically handle various resolutions but have limits.
        Images are often downscaled by the API if too large. Common advice is to keep images
        under certain pixel limits or aspect ratios for best performance/cost.
        Returning None implies flexibility.
        """
        return None # Example: (2048, 2048)

    def _pil_to_base64_data_uri(self, image: Image.Image, format="PNG") -> str:
        """
        Converts a PIL Image object to a base64 encoded data URI.
        """
        buffered = io.BytesIO()
        # Ensure image is in RGB or RGBA for PNG/JPEG if it's P mode or other
        if image.mode == 'P' or image.mode == 'L':
            image = image.convert('RGBA') if format == "PNG" else image.convert('RGB')
        elif image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB') # Default conversion for other modes

        image.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_base64}"

    def _clean_llm_response(self, text: str) -> str:
        """Cleans common markdown code fences from LLM responses."""
        if not text: # Handle empty or None input gracefully
            return ""
        
        # Pattern to match leading ``` optionally followed by a language specifier and a newline,
        # and trailing ```. Case-insensitive and multiline.
        # It handles cases like: ```markdown\nTEXT\n``` or ```\nTEXT\n``` or ```python\nTEXT\n```
        # It also handles if the LLM just puts ```TEXT``` on one line (though less common for blocks)
        
        # Remove leading fence with optional language and newline
        cleaned_text = re.sub(r"^\s*```(?:[a-zA-Z0-9_\-]+)?\s*\n?", "", text, flags=re.IGNORECASE)
        # Remove trailing fence
        cleaned_text = re.sub(r"\n?\s*```\s*$", "", cleaned_text, flags=re.IGNORECASE)
        
        # Fallback for single-line fences like ```text``` if the above didn't catch it fully
        # This is more aggressive if the text *is* just a language specifier, but unlikely for OCR content.
        if cleaned_text.startswith("```") and cleaned_text.endswith("```") and "\n" not in cleaned_text[3:-3]:
             # Check if what's inside looks like a language specifier or is very short (likely not the actual content)
            content_inside = cleaned_text[3:-3].strip()
            if not (len(content_inside) < 15 and content_inside.isalnum() and not content_inside.isspace()):
                 # If it's not a short alphanumeric string (likely a language specifier), then strip the fences
                 cleaned_text = content_inside

        return cleaned_text.strip() # Final strip for any residual whitespace

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 3), # Add up to 3s random jitter
        stop=stop_after_delay(600), # Stop retrying after 10 minutes
        retry=retry_if_exception(_is_rate_limit_error),
        reraise=True # Reraise the last exception if all retries fail
    )
    async def _call_llm_for_ocr(self, image_data_url: str) -> str:
        """
        Makes an individual asynchronous call to the LLM for OCR.
        Assumes an OpenAI-compatible client.
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url, "detail": "high"}, # Use high detail for OCR
                        },
                        {"type": "text", "text": self.ocr_prompt},
                    ],
                }
            ]
            
            # This is an example for OpenAI client. Adjust if your client API differs.
            # Ensure your client's create method is awaitable if it's an async client.
            if self._create_is_async:
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_model, 
                    messages=messages,
                    #max_tokens=self.max_tokens, 
                    temperature=0
                )
            else: # Synchronous client called in thread
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=self.llm_model, 
                    messages=messages,
                    #max_tokens=self.max_tokens,
                    temperature=0
                )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                raw_content = response.choices[0].message.content
                cleaned_content = self._clean_llm_response(raw_content)
                return cleaned_content
            else:
                logger.warning("LLM response was empty or not in expected format.")
                raise OCRProcessingError("LLM OCR: No content in response. The image may not contain text or the model could not process it.")
        except Exception as e:
            if _is_rate_limit_error(e):
                # Propagate rate limit errors so tenacity can retry
                raise
            logger.error(f"Error during LLM API call for OCR (final attempt or non-retriable): {e}", exc_info=True)
            raise OCRProcessingError(f"LLM OCR failed: {str(e)}. Please try again later or check your API settings.") from e

    async def aocr_image(self, images: List[Image.Image]) -> List[str]:
        """
        Asynchronously performs OCR on a list of PIL Image objects using the configured LLM.
        """
        if not images:
            return []

        logger.info(f"Performing async OCR on {len(images)} images with LLM model {self.llm_model}.")

        async def process_image(image_obj):
            try:
                image_data_url = self._pil_to_base64_data_uri(image_obj, format="PNG")
                return await self._call_llm_for_ocr(image_data_url)
            except Exception as e:
                logger.error(f"Error converting image to base64 data URI: {e}", exc_info=True)
                raise OCRProcessingError(f"Failed to process image: {str(e)}. Please check the image format or try again.") from e

        tasks = [process_image(image_obj) for image_obj in images]
        if self.show_progress and tqdm_asyncio is not None:
            # Use tqdm.asyncio.tqdm for async progress bar
            results = await tqdm_asyncio.tqdm.gather(*tasks, desc="LLM OCR (async)")
        else:
            if self.show_progress and tqdm_asyncio is None:
                logger.warning("tqdm.asyncio is not available. Falling back to asyncio.gather without a progress bar.")
            results = await asyncio.gather(*tasks)
        return results

    def ocr_image(self, images: List[Image.Image]) -> List[str]:
        """
        Performs OCR on a list of PIL Image objects using the configured LLM.
        This is a synchronous wrapper around the asynchronous version.
        Note: Running multiple LLM calls synchronously like this can be very slow.
        It's highly recommended to use the async `aocr_image` method in an async context.
        """
        if not images:
            return []
            
        logger.warning(
            f"Executing LLMBasedOCRService.ocr_image ({len(images)} images) synchronously. "
            "This can be slow. Prefer using aocr_image in an async context."
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.info("ocr_image called within a running asyncio loop. Relying on existing loop or nest_asyncio.")
                # Progress bar logic for sync context
                if self.show_progress and tqdm is not None:
                    # Use a wrapper coroutine to yield results one by one
                    async def gather_with_progress():
                        results = []
                        for coro in tqdm([self.aocr_image([img]) for img in images], desc="LLM OCR (sync)"):
                            # Each call returns a list of one result
                            res = await coro
                            results.append(res[0] if res else None)
                        return results
                    results = loop.run_until_complete(gather_with_progress())
                else:
                    results = loop.run_until_complete(self.aocr_image(images))
            else:
                if self.show_progress and tqdm is not None:
                    async def gather_with_progress():
                        results = []
                        for coro in tqdm([self.aocr_image([img]) for img in images], desc="LLM OCR (sync)"):
                            res = await coro
                            results.append(res[0] if res else None)
                        return results
                    results = asyncio.run(gather_with_progress())
                else:
                    results = asyncio.run(self.aocr_image(images))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e) or \
               "Event loop is closed" in str(e):
                logger.error(
                    "ocr_image cannot run a new event loop as one is already running or closed. "
                    "Consider using aocr_image or ensure nest_asyncio is correctly applied. "
                    f"Error: {e}"
                )
                raise OCRProcessingError("Event loop conflict in sync ocr_image. Please use the async method or check your event loop setup.") from e
            else:
                raise # Re-raise other runtime errors
        except Exception as e:
            logger.error(f"General error in synchronous ocr_image: {e}", exc_info=True)
            raise OCRProcessingError(f"General error in OCR: {str(e)}") from e
        # If any result is None, raise an error for that image
        if any(r is None for r in results):
            raise OCRProcessingError("One or more images could not be processed. Check logs for details.")
        return results

    def __str__(self):
        return f"LLMBasedOCRService (model={self.llm_model})" 