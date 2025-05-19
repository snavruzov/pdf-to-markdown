import asyncio
import base64
import io
import logging
import time
from typing import Any, Optional, Union

from PIL import Image
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception, wait_random

# Updated import path and class name
from pdf_to_markdown.table_detection_services.table_detection_service_interface import TableDetectionInterface, TableDetectionError

logger = logging.getLogger(__name__)

DEFAULT_TABLE_DETECTION_PROMPT = '''\
Does this image contain any simple, complex, borderless, or otherwise clearly defined tables? 
Respond ONLY with "yes" or "no". 
Do not include any other text, explanations, or formatting. Only the word "yes" or "no" is allowed.'''
MAX_RESPONSE_VALIDATION_ATTEMPTS = 3 # Number of times to retry if LLM response is not "yes" or "no"

def _is_rate_limit_error(exception: Exception) -> bool:
    """Checks if the exception is a rate limit error for tenacity in a client-agnostic way."""
    if hasattr(exception, 'status_code') and getattr(exception, 'status_code') == 429:
        logger.warning(f"Rate limit error (status 429) detected: {exception}. Retrying...")
        return True
    if exception.__class__.__name__ == 'RateLimitError': # e.g., openai.RateLimitError
        logger.warning(f"RateLimitError (by class name) detected: {exception}. Retrying...")
        return True
    return False

class LLMBasedTableDetector(TableDetectionInterface): # Updated inheritance
    """
    An implementation of TableDetectionInterface that uses a Large Language Model (LLM)
    to detect tables in images.
    Assumes an OpenAI-compatible client for chat completions with image input.
    """

    def __init__(self, llm_client: Any, llm_model: str, **kwargs):
        if llm_client is None or llm_model is None:
            raise ValueError("llm_client and llm_model must be provided for LLMBasedTableDetector.")
        if not hasattr(llm_client, 'chat') or \
           not hasattr(llm_client.chat, 'completions') or \
           not hasattr(llm_client.chat.completions, 'create'):
            raise ValueError(
                "llm_client does not appear to be OpenAI-compatible. "
                "It must have a `chat.completions.create` method."
            )
        self._create_is_async = asyncio.iscoroutinefunction(llm_client.chat.completions.create)
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.table_detection_prompt = kwargs.get("table_detection_prompt", DEFAULT_TABLE_DETECTION_PROMPT)
        self.max_response_validation_attempts = kwargs.get("max_response_validation_attempts", MAX_RESPONSE_VALIDATION_ATTEMPTS)

        logger.info(
            f"LLMBasedTableDetector initialized with model: {self.llm_model}, "
            f"client type: {type(llm_client).__name__}, "
            f"create_is_async: {self._create_is_async}, "
            f"prompt: '{self.table_detection_prompt[:70]}...', "
            f"max_validation_attempts: {self.max_response_validation_attempts}"
        )

    def get_best_size(self) -> Optional[Union[tuple[int, int], int]]:
        return 150

    def _pil_to_base64_data_uri(self, image: Image.Image, format="PNG") -> str:
        buffered = io.BytesIO()
        if image.mode == 'P' or image.mode == 'L':
            image = image.convert('RGBA') if format.upper() == "PNG" else image.convert('RGB')
        elif image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        image.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_base64}"

    def _parse_llm_response(self, content: Optional[str]) -> tuple[Optional[bool], str]:
        """
        Parses the LLM response content.
        Returns a tuple: (parsed_boolean, response_type_string).
        parsed_boolean is True for "yes", False for "no", None for invalid/empty.
        response_type_string is "yes", "no", "empty", or "invalid_format".
        """
        if not content:
            return None, "empty"
        
        cleaned_content = content.strip().lower()
        if cleaned_content == "yes":
            return True, "yes"
        elif cleaned_content == "no":
            return False, "no"
        else:
            return None, "invalid_format"

    async def _execute_llm_call_async(self, messages: list) -> Any:
        """Executes the LLM call asynchronously using a direct await."""
        if self._create_is_async:
            return await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=3
            )
        else: # Synchronous client
            return await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=3
            )

    def _execute_llm_call_sync(self, messages: list) -> Any:
        """
        Synchronously executes the LLM chat completion call.
        Handles both synchronous and asynchronous LLM clients.
        If the client is asynchronous, its 'create' coroutine is run to completion.
        """
        llm_create_func = self.llm_client.chat.completions.create
        kwargs = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 3  # Assuming small response for "yes"/"no"
        }

        if self._create_is_async:
            # Asynchronous client: need to run its coroutine to completion.
            coro = llm_create_func(**kwargs)
            try:
                # Try to use an existing event loop if available and running.
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    logger.debug(
                        "LLMTableDetector._execute_llm_call_sync: Async client in sync context, "
                        "loop running. Using loop.run_until_complete."
                    )
                    return loop.run_until_complete(coro)
                else:
                    # Loop exists but is not running, or we want asyncio.run's management.
                    logger.debug(
                        "LLMTableDetector._execute_llm_call_sync: Async client in sync context, "
                        "loop not running. Using asyncio.run."
                    )
                    return asyncio.run(coro)
            except RuntimeError as e:
                # Handles cases like:
                # - "asyncio.run() cannot be called from a running event loop" (if nest_asyncio not effective)
                # - "There is no current event loop in thread"
                # - Event loop is closed.
                logger.warning(
                    f"LLMTableDetector._execute_llm_call_sync: RuntimeError ('{e}') with event loop. "
                    "Attempting to run async client call in a new dedicated event loop."
                )
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)  # Clean up to avoid interference
        else:
            # Synchronous client: direct call.
            logger.debug("LLMTableDetector._execute_llm_call_sync: Sync client in sync context. Direct call.")
            return llm_create_func(**kwargs)

    @retry( # This retry is for API errors (e.g. rate limits) for the ASYNC flow
        wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
        stop=stop_after_delay(180),
        retry=retry_if_exception(_is_rate_limit_error),
        reraise=True
    )
    async def _perform_async_llm_detection_with_validation_loop(
        self, 
        initial_prompt_text: str, 
        image_data_url: str,
    ) -> bool:
        """
        Manages the LLM call attempts, including response validation and history building.
        This function does NOT handle API-level retries (e.g., rate limits), 
        its callers are expected to use tenacity for that.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}},
                    {"type": "text", "text": initial_prompt_text},
                ],
            }
        ]

        for attempt in range(self.max_response_validation_attempts):
            try:
                # Make the actual LLM call using the provided executor
                response_obj = await self._execute_llm_call_async(messages)
                llm_response_content = response_obj.choices[0].message.content if response_obj.choices and response_obj.choices[0].message else None

                parsed_result, response_type = self._parse_llm_response(llm_response_content)

                if parsed_result is not None: # Handles "yes" or "no"
                    if parsed_result:
                        logger.debug(f"LLM detected table. Raw response: '{llm_response_content}'")
                    else:
                        logger.debug(f"LLM did not detect table. Raw response: '{llm_response_content}'")
                    return parsed_result
                
                # If parsed_result is None, it means response was "empty" or "invalid_format"
                logger.warning(
                    f"LLM response was '{response_type}'. Raw: '{llm_response_content}'. "
                    f"(Attempt {attempt + 1}/{self.max_response_validation_attempts})"
                )

                if attempt < self.max_response_validation_attempts - 1:
                    messages.append({"role": "assistant", "content": llm_response_content or ""}) # Use raw content or empty string
                    if response_type == "empty":
                        error_feedback = "Your response was empty. Please respond with only 'yes' or 'no'."
                    else: # "invalid_format"
                        error_feedback = f"Your response '{llm_response_content}' was not 'yes' or 'no'. Please respond with only 'yes' or 'no'."
                    messages.append({"role": "user", "content": error_feedback})
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(
                        f"LLM table detection: Exhausted {self.max_response_validation_attempts} attempts. "
                        f"Last response type was '{response_type}', raw: '{llm_response_content}'"
                    )
                    return False

            except Exception as e:
                # This catches errors from llm_executor_func or parsing.
                # API specific errors (like rate limits) should ideally be handled by tenacity on the caller of this function.
                # If an error occurs here that tenacity on the caller *doesn't* catch and retry, it will propagate.
                logger.error(
                    f"Exception during LLM detection attempt {attempt + 1}/{self.max_response_validation_attempts}: {e}",
                    exc_info=True
                )
                # If it's the last attempt, or a non-retriable error by the outer tenacity, let it propagate
                # or decide to return False. For now, let it propagate if not handled by outer tenacity.
                # However, the prompt asks for this function NOT to handle API level retries.
                # So, if an exception reaches here, it means it's either a non-API error from the executor,
                # or an API error that the outer tenacity has given up on.
                # We should re-raise to ensure the outer tenacity (if any) or caller sees it.
                # Or, more simply, this loop is for *validation*. If the API call itself fails hard, it should raise.
                raise TableDetectionError(f"LLM API call failed during validation loop: {str(e)}") from e


        logger.error(f"LLM table detection failed after all {self.max_response_validation_attempts} validation attempts.")
        return False
    
    @retry( # This retry is for API errors (e.g. rate limits) for the SYNC flow
        wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
        stop=stop_after_delay(180),
        retry=retry_if_exception(_is_rate_limit_error),
        reraise=True
    )
    def _perform_sync_llm_detection_with_validation_loop(
        self, 
        initial_prompt_text: str, 
        image_data_url: str,
    ) -> bool:
        """
        Manages the LLM call attempts, including response validation and history building.
        This function does NOT handle API-level retries (e.g., rate limits), 
        its callers are expected to use tenacity for that.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}},
                    {"type": "text", "text": initial_prompt_text},
                ],
            }
        ]

        for attempt in range(self.max_response_validation_attempts):
            try:
                # Make the actual LLM call using the provided executor
                response_obj = self._execute_llm_call_sync(messages)
                llm_response_content = response_obj.choices[0].message.content if response_obj.choices and response_obj.choices[0].message else None

                parsed_result, response_type = self._parse_llm_response(llm_response_content)

                if parsed_result is not None: # Handles "yes" or "no"
                    if parsed_result:
                        logger.debug(f"LLM detected table. Raw response: '{llm_response_content}'")
                    else:
                        logger.debug(f"LLM did not detect table. Raw response: '{llm_response_content}'")
                    return parsed_result
                
                # If parsed_result is None, it means response was "empty" or "invalid_format"
                logger.warning(
                    f"LLM response was '{response_type}'. Raw: '{llm_response_content}'. "
                    f"(Attempt {attempt + 1}/{self.max_response_validation_attempts})"
                )

                if attempt < self.max_response_validation_attempts - 1:
                    messages.append({"role": "assistant", "content": llm_response_content or ""}) # Use raw content or empty string
                    if response_type == "empty":
                        error_feedback = "Your response was empty. Please respond with only 'yes' or 'no'."
                    else: # "invalid_format"
                        error_feedback = f"Your response '{llm_response_content}' was not 'yes' or 'no'. Please respond with only 'yes' or 'no'."
                    messages.append({"role": "user", "content": error_feedback})
                    time.sleep(1)
                    continue
                else:
                    logger.error(
                        f"LLM table detection: Exhausted {self.max_response_validation_attempts} attempts. "
                        f"Last response type was '{response_type}', raw: '{llm_response_content}'"
                    )
                    return False # Default to False if all validation attempts fail

            except Exception as e:
                logger.error(
                    f"Exception during sync LLM detection attempt {attempt + 1}/{self.max_response_validation_attempts}: {e}",
                    exc_info=True
                )
                # Similar to async, re-raise to let outer tenacity handle API errors if it's configured for this sync path.
                # Or if it's a fundamental error in the sync call itself.
                raise TableDetectionError(f"Sync LLM API call failed during validation loop: {str(e)}") from e
        
        logger.error(f"Sync LLM table detection failed after all {self.max_response_validation_attempts} validation attempts.")
        return False

    async def adetect_tables_on_page(self, image: Image.Image) -> bool:
        """
        Asynchronously detects if the given image contains a table using LLM.
        Handles API-level retries (e.g., rate limits) using tenacity.
        Also includes an internal loop for validating the "yes"/"no" response format.
        """
        if not isinstance(image, Image.Image):
            raise TableDetectionError("Invalid input: image must be a PIL.Image.Image object.")
        try:
            image_data_url = self._pil_to_base64_data_uri(image)
            return await self._perform_async_llm_detection_with_validation_loop(
                initial_prompt_text=self.table_detection_prompt,
                image_data_url=image_data_url
            )
        except Exception as e:
            if isinstance(e, TableDetectionError): # Re-raise if it's already our specific error
                raise
            logger.error(f"Async LLM Table detection failed: {e}", exc_info=True)
            # Wrap other exceptions in TableDetectionError
            raise TableDetectionError(f"Async LLM table detection failed: {str(e)}") from e

    def detect_tables_on_page(self, image: Image.Image) -> bool:
        """
        Synchronously detects if the given image contains a table using LLM.
        Handles API-level retries (e.g., rate limits) using tenacity.
        Also includes an internal loop for validating the "yes"/"no" response format.
        """
        if not isinstance(image, Image.Image):
            raise TableDetectionError("Invalid input: image must be a PIL.Image.Image object.")
        try:
            image_data_url = self._pil_to_base64_data_uri(image)
            return self._perform_sync_llm_detection_with_validation_loop(
                initial_prompt_text=self.table_detection_prompt, 
                image_data_url=image_data_url
            )
        except Exception as e:
            if isinstance(e, TableDetectionError): # Re-raise if it's already our specific error
                raise
            logger.error(f"Sync LLM Table detection failed: {e}", exc_info=True)
            raise TableDetectionError(f"Sync LLM table detection failed: {str(e)}") from e

    def __str__(self):
        return f"LLMBasedTableDetector (model={self.llm_model})" 