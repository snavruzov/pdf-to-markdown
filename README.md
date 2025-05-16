# MarkItDown PDF Converter Plugin

This project provides a **MarkItDown plugin** for converting PDF files to Markdown with an emphasis on extracting _all_ content, including text, tables, and images—even from challenging or scanned documents. The plugin is designed for **correctness over speed**, ensuring that as much information as possible is preserved in the Markdown output.

## Features

- **Text Extraction**: Uses [PyMuPDF](https://pymupdf.readthedocs.io/) and [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html) to extract text, tables, and images from PDFs when possible.
- **Fallback to LLM-based OCR**: If text extraction is incomplete (e.g., for scanned pages, complex tables, or images), the plugin automatically falls back to an LLM-based OCR service. This ensures that even non-searchable or image-based content is converted to Markdown.
- **Table and Image Handling**: Tables and images are extracted and, if necessary, described or transcribed using the LLM OCR backend.
- **Correctness First**: The plugin prioritizes extracting _everything_ from the PDF, even if this means slower processing due to LLM calls for OCR or table/image understanding.

## How It Works

1. **Text Extraction**: For each page, the plugin first attempts to extract text, tables, and images using PyMuPDF and pymupdf4llm.
2. **Heuristics for OCR**: If a page is detected as non-readable (e.g., scanned, mostly images, or simulated text), it is sent to the LLM-based OCR service for processing.
3. **LLM OCR Fallback**: The plugin uses a configurable LLM client (such as OpenAI's GPT-4o) to perform OCR and generate Markdown for images, tables, and non-readable content.
4. **Combining Results**: All extracted and OCR'd content is merged into a single Markdown output, with clear page and section boundaries.

## Dependencies

- Python 3.11+
- [PyMuPDF](https://pymupdf.readthedocs.io/) (`pymupdf`)
- [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html)
- [Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- [tqdm](https://tqdm.github.io/) (for progress bars)
- [markitdown](https://github.com/microsoft/markitdown/tree/main)
- An LLM client (e.g., OpenAI Python SDK) for OCR fallback

All dependencies are listed in `pyproject.toml`.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <this-repo-url>
   cd pdf-to-markdown
   ```
2. **Install dependencies** (using [Poetry](https://python-poetry.org/) or pip):
   
   **Using Poetry (recommended):**
   ```bash
   poetry install
   ```

   **Using pip:**
   If you are not using Poetry, you can install the package and its dependencies directly from the project directory (which reads `pyproject.toml`):
   ```bash
   pip install .
   ```
   (If this package were published on PyPI, you would typically use `pip install pdf-to-markdown`.)

## Using as a Standalone Library

Besides its use as a MarkItDown plugin, `pdf-to-markdown` can also be used as a standalone Python library to convert PDF content directly.

Here's how you can use the `pdf_to_markdown` function with the `LLMBasedOCRService`:

```python
import asyncio
from pdf_to_markdown import pdf_to_markdown, pdf_to_markdown_sync, LLMBasedOCRService
import os # For accessing environment variables

# --- Option 1: Setup LLM Client (Example using OpenAI) ---
# Make sure to install it: pip install openai
from openai import OpenAI

# It's recommended to set your API key as an environment variable for security.
# e.g., export OPENAI_API_KEY='your_api_key_here'
# If the OPENAI_API_KEY environment variable is set, the client will use it automatically.
# Otherwise, you can pass it directly: OpenAI(api_key="YOUR_API_KEY")
openai_client = OpenAI()
# print("OpenAI client initialized. Ensure your OPENAI_API_KEY is set or passed directly.")
openai_model = "gpt-4o" # Or your preferred OpenAI model, e.g., "gpt-4-turbo"

# --- Option 2: Setup LLM Client (Example using Google Gemini via OpenAI-compatible API) ---
# Make sure to install openai: pip install openai
# You would still use the OpenAI library, but configure it for Google's endpoint.

# It's recommended to set your Gemini API key as an environment variable.
# e.g., export GEMINI_API_KEY='your_google_api_key'
# gemini_api_key = os.environ.get("GEMINI_API_KEY")
# if gemini_api_key:
#     google_client_openai_compat = OpenAI(
#         api_key=gemini_api_key,
#         base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
#     )
#     google_model_openai_compat = "gemini-1.5-flash-latest" # Or your preferred compatible Gemini model
#     print("Google Gemini client (OpenAI compatible) configured. Ensure your GEMINI_API_KEY is set.")
# else:
#     print("GEMINI_API_KEY not found in environment variables. Skipping Google client setup.")
#     google_client_openai_compat = None
#     google_model_openai_compat = None

# For the example, we'll default to using the standard OpenAI client.
# To use the Google client, uncomment the section above and set:
# llm_client_to_use = google_client_openai_compat
# llm_model_to_use = google_model_openai_compat

llm_client_to_use = openai_client # Choose which client to use (e.g., openai_client or google_client_openai_compat)
llm_model_to_use = openai_model   # Choose the corresponding model

# --- Initialize OCR Service ---
# show_progress=True requires tqdm to be installed
ocr_service = LLMBasedOCRService(
    llm_client=llm_client_to_use,
    llm_model=llm_model_to_use,
    show_progress=True # Set to False if tqdm is not available or not desired
)

# --- Path to your PDF ---
pdf_file_path = "path/to/your/example.pdf" # Replace with your PDF file path

# --- Asynchronous Usage Example ---
async def run_async_conversion():
    print("--- Running Asynchronous Conversion ---")
    markdown_results = await pdf_to_markdown(
        pdf_source=pdf_file_path,
        image_ocr_service=ocr_service,
        show_progress=True
    )
    for page_num, md_content in sorted(markdown_results.items()):
        print(f"\n--- Page (Async) {page_num + 1} ---\n{md_content}")

# --- Synchronous Usage Example ---
def run_sync_conversion():
    print("\n--- Running Synchronous Conversion ---")
    markdown_results = pdf_to_markdown_sync(
        pdf_source=pdf_file_path,
        image_ocr_service=ocr_service,
        show_progress=True
    )
    for page_num, md_content in sorted(markdown_results.items()):
        print(f"\n--- Page (Sync) {page_num + 1} ---\n{md_content}")

if __name__ == "__main__":
    # To run the async version:
    # asyncio.run(run_async_conversion())
    
    # To run the sync version:
    run_sync_conversion()
```

**Explanation:**

1.  **Import necessary modules**: `asyncio`, `pdf_to_markdown`, `pdf_to_markdown_sync`, `LLMBasedOCRService`, and your chosen LLM client library (`openai` or `google.generativeai`).
2.  **Set up LLM Client**: 
    *   **OpenAI**: 
    *   **Google Gemini (OpenAI-compatible API)**: You can also use Google's Gemini models through their OpenAI-compatible endpoint. Install `openai` and initialize the `OpenAI` client with `base_url="https://generativelanguage.googleapis.com/v1beta/openai/"` and your `GEMINI_API_KEY`. This allows using Gemini models (like `"gemini-1.5-flash-latest"`) with the existing `LLMBasedOCRService` without needing a separate client library or wrapper. The example defaults to using the standard OpenAI client but shows how to configure for Gemini.
    * Every provider that can accessed through the openai client can be used out of the box for the LLMBasedOCRService.
3.  **Initialize `LLMBasedOCRService`**: Pass the chosen LLM client and model name.
4.  **Call Conversion Functions**:
    *   `pdf_to_markdown` (async): Use with `await` inside an `async` function (e.g., `run_async_conversion`).
    *   `pdf_to_markdown_sync` (sync): Call directly (e.g., `run_sync_conversion`).
5.  **Process Results**: Both return a dictionary mapping 0-indexed page numbers to Markdown content.

This setup allows you to leverage the PDF parsing and OCR capabilities of the package in any Python application, choosing between asynchronous or synchronous execution and different LLM backends based on your needs.

### Model Choice Considerations

When selecting an LLM for OCR, consider factors like accuracy, speed, and cost. According to benchmark results from the [Omni OCR Benchmark](https://github.com/getomni-ai/benchmark) (as of the time of writing), Google's `gemini-2.0-flash` (or newer flash versions) is often highlighted as a strong contender for its balance of good performance, speed, and cost-effectiveness for OCR tasks. However, always refer to the latest benchmark data and your specific project requirements when making a decision.

## Usage

This plugin is designed to be used as a MarkItDown converter. You can use it programmatically:

```python
from markitdown import MarkItDown
from pdf_to_markdown.markitdown_mupdf_converter import register_converters

# Optionally, provide your LLM client and model for OCR fallback
llm_client = ...  # e.g., OpenAI client
llm_model = "gpt-4o"

mid = MarkItDown(
    custom_pdf_ocr_service=None,  # or your own OCR service that implements the OCRInterface 
    llm_client=llm_client,
    llm_model=llm_model,
)
register_converters(mid)

with open("example.pdf", "rb") as f:
    result = mid.convert(f, extension=".pdf")
    print(result.markdown)
```

## Configuration
- You can provide your own OCR service or use the built-in LLM-based OCR (requires an OpenAI-compatible client).
- The plugin will automatically decide when to use OCR based on page content heuristics.

## Notes
- **Performance**: Because the plugin prioritizes correctness and completeness, processing may be slower than pure text extractors, especially for scanned or image-heavy PDFs.
- **LLM Costs**: Using LLM-based OCR may incur API costs depending on your provider and the number of pages/images processed.

## Optional Arguments

You can customize the behavior of the plugin by providing the following optional arguments when constructing the `MarkItDown` instance or when calling `convert`:

### Constructor Arguments (for `MarkItDown`)
- **`ocr_service`**: An instance of a custom OCR service implementing the `OCRInterface`. If provided, this will be used for all OCR tasks instead of the built-in LLM-based OCR.
- **`llm_client`**: An OpenAI-compatible client instance (or similar) for LLM-based OCR. Required if you want to use the built-in LLM OCR fallback.
- **`llm_model`**: The model name to use with the LLM client (e.g., `"gpt-4o"`).
- **`docintel_endpoint`**: (Optional) Endpoint for Azure Document Intelligence OCR, if you want to use Azure's OCR service.
- **`docintel_credential`**: (Optional) Credentials for Azure Document Intelligence OCR.
- **`show_progress`**: (bool, default `False`) If `True`, shows progress bars during processing (requires `tqdm`).

### Per-Conversion Arguments (for `convert`)
- **`pages`**: A list of page indices to process. If not provided, all pages are processed.
- **`force_ocr`**: (bool, default `False`) If `True`, forces OCR on all pages, even if text extraction is possible.
- **`show_progress`**: (bool, default `False`) If `True`, shows progress bars for this conversion.

### Example Usage

```python
from markitdown import MarkItDown
from openai import OpenAI

llm_client = OpenAI(api_key="sk-...")
llm_model = "gpt-4o"

mid = MarkItDown(
    enable_plugins=True,
    ocr_service=None,  # or your own OCR service
    llm_client=llm_client,
    llm_model=llm_model,
    show_progress=True,
)

with open("example.pdf", "rb") as f:
    result = mid.convert(f, extension=".pdf", force_ocr=True, show_progress=True, pages=[0, 1, 2])
    print(result.markdown)
```

**Notes:**
- If you provide both `ocr_service` and `llm_client`/`llm_model`, the custom `ocr_service` takes precedence.
- If neither is provided, the plugin will not be able to perform OCR and will return an error for non-readable pages.
- `show_progress` can be set globally (in the constructor) or per conversion.
- `pages` allows you to process only a subset of pages.
- `force_ocr` is useful for scanned PDFs or when you want to ensure all content is processed via OCR.

## License

MIT License 