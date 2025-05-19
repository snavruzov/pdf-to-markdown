# MarkItDown PDF Converter Plugin

This project provides a **MarkItDown plugin** for converting PDF files to Markdown with an emphasis on extracting _all_ content, including text, tables, and images—even from challenging or scanned documents. The plugin is designed for **correctness over speed**, ensuring that as much information as possible is preserved in the Markdown output.

## Features

- **Text Extraction**: Uses [PyMuPDF](https://pymupdf.readthedocs.io/) and [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html) to extract text, tables, and images from PDFs when possible.
- **Fallback to LLM-based OCR**: If text extraction is incomplete (e.g., for scanned pages, complex tables, or images), the plugin automatically falls back to an LLM-based OCR service. This ensures that even non-searchable or image-based content is converted to Markdown.
- **Enhanced Table Detection**: Incorporates a `TableDetectionInterface` allowing for sophisticated table detection (e.g., using an LLM via `LLMBasedTableDetector`). If tables are detected on a page initially deemed readable by heuristics, that page is re-routed for full OCR to ensure accurate table extraction.
- **Table and Image Handling**: Tables and images are extracted and, if necessary, described or transcribed using the LLM OCR backend.
- **Correctness First**: The plugin prioritizes extracting _everything_ from the PDF, even if this means slower processing due to LLM calls for OCR or table/image understanding.

## How It Works

1. **Text Extraction**: For each page, the plugin first attempts to extract text, tables, and images using PyMuPDF and pymupdf4llm.
2. **Heuristics and Table Detection for OCR**: A page is sent for OCR if:
    a. It's detected as non-readable by standard heuristics (e.g., scanned, mostly images, simulated text).
    b. Or, if standard heuristics pass, but a configured `TableDetectionInterface` (like the default `LLMBasedTableDetector` if an LLM is set up) detects the presence of tables on the page. This helps catch subtle text-based tables that might otherwise be missed.
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

### Optional Dependency for Synchronous APIs in Asynchronous Environments

If you plan to use the synchronous API functions (like `pdf_to_markdown_sync` or `LLMBasedTableDetector.detect_tables_on_page` when it internally runs an async client) in an environment that already has an active asyncio event loop (such as Jupyter notebooks or certain GUI frameworks), you might encounter `RuntimeError`s related to nested event loops.

To enable smoother operation in these environments, you can install an optional dependency when installing this package:

```bash
pip install .[async_sync_env]
```
Or, if installing from PyPI (once published):
```bash
pip install pdf-to-markdown[async_sync_env]
```

This installs `nest_asyncio`, which helps the library manage nested event loops gracefully. The library will attempt to use `nest_asyncio` if it's available, falling back to standard asyncio behavior otherwise (which may lead to the aforementioned `RuntimeError`s in nested loop scenarios).

## Using as a Standalone Library

Besides its use as a MarkItDown plugin, `pdf-to-markdown` can also be used as a standalone Python library to convert PDF content directly.

Here's how you can use the `pdf_to_markdown` function with the `LLMBasedOCRService` and `LLMBasedTableDetector`:

```python
import asyncio
from pdf_to_markdown import (
    pdf_to_markdown, 
    pdf_to_markdown_sync, 
    LLMBasedOCRService # Assuming LLMBasedTableDetector is also in pdf_to_markdown.__init__ or imported directly
)
# If LLMBasedTableDetector is in a submodule, import it like this:
from pdf_to_markdown.table_detection_services import LLMBasedTableDetector

import os # For accessing environment variables

# --- Option 1: Setup LLM Client (Example using OpenAI) ---
# Make sure to install it: pip install openai
from openai import OpenAI, AsyncOpenAI

# It's recommended to set your API key as an environment variable for security.
# e.g., export OPENAI_API_KEY='your_api_key_here'
# If the OPENAI_API_KEY environment variable is set, the client will use it automatically.
# Otherwise, you can pass it directly: OpenAI(api_key="YOUR_API_KEY")

# Synchronous OpenAI Client
sync_openai_client = OpenAI()

# Asynchronous OpenAI Client
async_openai_client = AsyncOpenAI()

# print("OpenAI clients initialized. Ensure your OPENAI_API_KEY is set or passed directly.")
openai_model = "gpt-4o-mini" # Or your preferred OpenAI model, e.g., "gpt-4-turbo"

# --- Option 2: Setup LLM Client (Example using Google Gemini via OpenAI-compatible API) ---
# (Code for Google Gemini client setup omitted for brevity, same as before)

llm_client_to_use = async_openai_client # Choose which client to use (e.g., async_openai_client or sync_openai_client)
                                        # For best performance with the library's async core, AsyncOpenAI is generally preferred.
llm_model_to_use = openai_model   # Choose the corresponding model

# --- Initialize OCR Service ---
ocr_service = LLMBasedOCRService(
    llm_client=llm_client_to_use,
    llm_model=llm_model_to_use,
    show_progress=True
)

# --- Initialize Table Detection Service ---
table_detection_svc = LLMBasedTableDetector(
    llm_client=llm_client_to_use,
    llm_model=llm_model_to_use
    # You can customize prompt, retries etc. via kwargs if needed
)


# --- Path to your PDF ---
pdf_file_path = "path/to/your/example.pdf" # Replace with your PDF file path

# --- Asynchronous Usage Example ---
async def run_async_conversion():
    print("--- Running Asynchronous Conversion ---")
    markdown_results = await pdf_to_markdown(
        pdf_source=pdf_file_path,
        image_ocr_service=ocr_service,
        table_detection_service=table_detection_svc, # Pass the table detection service
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
        table_detection_service=table_detection_svc, # Pass the table detection service
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

1.  **Import necessary modules**: `asyncio`, `pdf_to_markdown`, `pdf_to_markdown_sync`, `LLMBasedOCRService`, `LLMBasedTableDetector` (from `pdf_to_markdown.table_detection_services`), and your chosen LLM client library.
2.  **Set up LLM Client**: (As before)
3.  **Initialize `LLMBasedOCRService`**: (As before)
4.  **Initialize `LLMBasedTableDetector`**: Instantiate the table detector, also using the LLM client and model. This service is then passed to the conversion functions using the `table_detection_service` parameter. If you don't provide a `table_detection_service`, but an LLM client/model is configured, the converter will try to use `LLMBasedTableDetector` by default.
5.  **Call Conversion Functions**:
    *   `pdf_to_markdown` (async): Pass `table_detection_service=table_detection_svc`.
    *   `pdf_to_markdown_sync` (sync): Pass `table_detection_service=table_detection_svc`.
6.  **Process Results**: (As before)

This setup allows you to leverage the PDF parsing and OCR capabilities of the package in any Python application, choosing between asynchronous or synchronous execution and different LLM backends based on your needs.

### Model Choice Considerations

When selecting an LLM for OCR, consider factors like accuracy, speed, and cost. According to benchmark results from the [Omni OCR Benchmark](https://github.com/getomni-ai/benchmark) (as of the time of writing), Google's `gemini-2.0-flash` (or newer flash versions) is often highlighted as a strong contender for its balance of good performance, speed, and cost-effectiveness for OCR tasks. However, always refer to the latest benchmark data and your specific project requirements when making a decision.

## Usage

This plugin is designed to be used as a MarkItDown converter. You can use it programmatically:

```python
from markitdown import MarkItDown
# Assuming register_converters is in pdf_to_markdown.markitdown_mupdf_converter
from pdf_to_markdown.markitdown_mupdf_converter import register_converters 

# Optionally, provide your LLM client and model for OCR fallback and default table detection
llm_client = ...  # e.g., OpenAI client
llm_model = "gpt-4o"

mid = MarkItDown(
    custom_pdf_ocr_service=None,  # or your own OCR service that implements the OCRInterface 
    table_detection_service=None, # or your own Table Detection service that implements TableDetectionInterface
    llm_client=llm_client,
    llm_model=llm_model,
)
register_converters(mid) # Ensure this is called to register your custom converter

with open("example.pdf", "rb") as f:
    result = mid.convert(f, extension=".pdf")
    print(result.markdown)
```

## Configuration
- You can provide your own OCR service (`custom_pdf_ocr_service`) or use the built-in LLM-based OCR (requires an OpenAI-compatible client and model).
- Similarly, you can provide your own table detection service (`table_detection_service`) or use the built-in `LLMBasedTableDetector` (which also uses the configured LLM client and model if provided).
- The plugin will automatically decide when to use OCR based on page content heuristics and table detection results.

## Notes
- **Performance**: Because the plugin prioritizes correctness and completeness, processing may be slower than pure text extractors, especially for scanned or image-heavy PDFs.
- **LLM Costs**: Using LLM-based OCR may incur API costs depending on your provider and the number of pages/images processed.

## Optional Arguments

You can customize the behavior of the plugin by providing the following optional arguments when constructing the `MarkItDown` instance or when calling `convert`:

### Constructor Arguments (for `MarkItDown`)
- **`custom_pdf_ocr_service`**: An instance of a custom OCR service implementing the `OCRInterface`. If provided, this will be used for all OCR tasks instead of the built-in LLM-based OCR.
- **`table_detection_service`**: An instance of a custom table detection service implementing the `TableDetectionInterface`. If provided, this will be used for table detection.
- **`llm_client`**: An OpenAI-compatible client instance (or similar). If provided (along with `llm_model`), it will be used for:
    - The default `LLMBasedOCRService` (if `custom_pdf_ocr_service` is not given).
    - The default `LLMBasedTableDetector` (if `table_detection_service` is not given).
- **`llm_model`**: The model name to use with the LLM client (e.g., `\"gpt-4o\"`).
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
    enable_plugins=True,   # mandatory
    llm_client=llm_client, # mandatory
    llm_model=llm_model,   # mandatory
    show_progress=True,    # not mandatory
    ocr_service=None,      # not mandatory (your own OCR service, maps to custom_pdf_ocr_service)
    table_detection_service=None, # not mandatory (your own Table detection service)
)

with open("example.pdf", "rb") as f:
    result = mid.convert(f, extension=".pdf", force_ocr=True, show_progress=True, pages=[0, 1, 2])
    print(result.markdown)
```

**Notes:**
- If you provide both `custom_pdf_ocr_service` and `llm_client`/`llm_model`, the custom `custom_pdf_ocr_service` takes precedence for OCR.
- If you provide both `table_detection_service` and `llm_client`/`llm_model`, the custom `table_detection_service` takes precedence for table detection.
- If an `llm_client` and `llm_model` are provided:
    - And no `custom_pdf_ocr_service` is given, `LLMBasedOCRService` will be used.
    - And no `table_detection_service` is given, `LLMBasedTableDetector` will be used.
- If neither OCR configuration (custom or LLM-based) is provided, the plugin will not be able to perform OCR and will return an error for non-readable pages.
- If no table detection configuration is provided (custom or LLM-based), table detection will rely solely on `pymupdf4llm`'s capabilities without the enhanced LLM check.
- `show_progress` can be set globally (in the constructor) or per conversion.
- `pages` allows you to process only a subset of pages.
- `force_ocr` is useful for scanned PDFs or when you want to ensure all content is processed via OCR.

## License

MIT License 