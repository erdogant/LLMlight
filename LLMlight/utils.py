""" Utils."""
import json
from json_repair import repair_json
import re
import pymupdf
import logging
import os
import unicodedata

logger = logging.getLogger(__name__)


def chunk_text(text, method='chars', chunk_size=512, overlap=0):
    """
    Split text into non-overlapping chunks.
    For RSE, we typically want non-overlapping chunks so we can reconstruct segments properly.

    Args:
        text (str): Input text to chunk
        method (str): Chunk on either 'chars' or 'words'
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[str]: List of text chunks
    """
    chunks = []

    # Chunk in chars
    if method == 'chars':
        if chunk_size is not None and chunk_size > 0:
            if overlap >= chunk_size:
                raise ValueError(
                    f"overlap ({overlap}) must be less than chunk_size ({chunk_size}); "
                    "otherwise chunking cannot advance through the text."
                )
            # Simple character-based chunking
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk:  # Ensure we don't add empty chunks
                    chunks.append(chunk)
            return chunks
        else:
            return [text]

    elif method == 'words':
        # Splits text into chunks of approximately chunk_size words.
        if chunk_size is not None:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        else:
            return [text]

    else:
        raise ValueError(f'Chunking method {method} is not supported.')


# Try to load the string as JSON
def is_valid_json(string):
    try:
        # json.loads(string)
        response_text = repair_json(string)
        response_text = json.loads(response_text)
        if response_text == '':
            response_text = string
        return response_text
    except json.JSONDecodeError:
        return string

def clean_text(text: str) -> str:
    """Cleans up the text by removing unnecessary spaces between characters and excessive whitespace."""
    # Normalize unicode (e.g., é → e)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    # text = str(text).replace('\n', ' ').replace('"', '')
    text = str(text).replace('{', '').replace('}', '')

    # Clean excessive spaces between words (multiple spaces turned into one)
    text = re.sub(r'\s+', ' ', text.strip())  # Clean up multiple spaces

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters (keep alphanumeric, punctuation, whitespace)
    text = re.sub(r'[^a-zA-Z0-9\s\.,;:\?!\'"-]', ' ', text)

    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Return
    return text

# %%
def read_pdf(file_path, title_pages=None, body_pages=None, reference_pages=None, return_type='dict'):
    """
    Reads a PDF file and extracts its text content.

    Args:
        file_path (str): Path to the PDF file.
        title_pages (list of int): 1-based page numbers treated as title. Default [1, 2].
        body_pages (list of int): 1-based page numbers treated as body. Default [] (all non-title/reference pages).
        reference_pages (list of int): 1-based page numbers treated as references. Negative indices count from the end. Default [-1].
        return_type (str): 'dict' returns {'title', 'body', 'references'}; any other value returns a plain string.

    Returns:
        dict or str: Extracted text, or None if the file is missing or not a PDF.
    """
    logger.info('Reading pdf')
    if not os.path.isfile(file_path):
        logger.error(f'File not found on disk: {file_path}')
        return None
    if not file_path.lower().endswith('.pdf'):
        logger.error("The provided file path is not a valid PDF file.")
        return None
    if title_pages is None: title_pages = [1, 2]
    if body_pages is None: body_pages = []
    if reference_pages is None: reference_pages = [-1]
    if isinstance(title_pages, (str, int)): title_pages = [title_pages]
    if isinstance(body_pages, (str, int)): body_pages = [body_pages]
    if isinstance(reference_pages, (str, int)): reference_pages = [reference_pages]
    # Coerce all elements to int so string inputs like "-1" don't cause TypeError
    # during negative-index resolution or membership tests.
    try:
        title_pages     = [int(p) for p in title_pages]
        body_pages      = [int(p) for p in body_pages]
        reference_pages = [int(p) for p in reference_pages]
    except (ValueError, TypeError) as e:
        logger.error(f"Page list contains non-integer value: {e}")
        return {"title": "", "body": "", "references": ""} if return_type == 'dict' else ""

    try:
        # Open pdf
        doc = pymupdf.open(file_path)
        # Get the total number of pages
        num_pages = len(doc)
        # Resolve negative indices to 1-based page numbers.
        # e.g. -1 on a 10-page doc -> 10, not 9.
        reference_pages = [num_pages + page + 1 if page < 0 else page for page in reference_pages]

        title_text = ""
        body_text = []
        references_text = ""
        context = {}

        for page_num in range(0, len(doc)):
            # Get page text
            page_text = doc[page_num].get_text("text")
            # text cleaning
            page_text = clean_text(page_text)

            # Set title text
            if (page_num + 1) in title_pages:
                title_text += "\n" + page_text
            elif (page_num + 1) in reference_pages:
                references_text += "\n" + page_text
            elif (page_num + 1) in body_pages:
                body_text.append(page_text)
            elif len(body_pages) == 0:
                body_text.append(page_text)

    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        if return_type == 'dict':
            return {"title": "", "body": "", "references": ""}
        return ""

    # Return
    if return_type=='dict':
        context = {"title": title_text.strip(),
                   "body": "\n".join(body_text).strip(),
                   "references": references_text.strip(),
                   }
    else:
        parts = [title_text.strip()] + body_text + [references_text.strip()]
        context = "\n---\n".join(p for p in parts if p)

    # Return
    return context


def read_document(file_path, return_type='str'):
    """
    Generic document reader powered by Microsoft's ``markitdown`` library.

    Unlike :func:`read_pdf`, which is PDF-specific, this function can extract
    text from (almost) any file type -- Word (.doc/.docx), PowerPoint
    (.ppt/.pptx), Excel (.xls/.xlsx), PDF, HTML, CSV/JSON/XML, plain text,
    images (via EXIF metadata / OCR), audio (via speech transcription), ZIP
    archives (recursively), and more -- converting the content into clean
    Markdown text that is well suited as LLM context (headings, lists, and
    tables are preserved using Markdown syntax).

    Args:
        file_path (str): Path to a local file on disk.
        return_type (str): 'str' (default) returns the extracted Markdown
            text as a plain string. 'dict' returns {'title', 'body'} for
            consistency with :func:`read_pdf`.

    Returns:
        str or dict: Extracted markdown text (or {'title': '', 'body': ''}
        / '' on failure -- missing file, unsupported/corrupt file, or the
        optional ``markitdown`` dependency not being installed).
    """
    logger.info('Reading document with markitdown')

    empty = {"title": "", "body": ""} if return_type == 'dict' else ""

    if not file_path or not os.path.isfile(file_path):
        logger.error(f'File not found on disk: {file_path}')
        return empty

    try:
        from markitdown import MarkItDown
    except ImportError:
        logger.error(
            "The 'markitdown' package is required for read_document(). "
            "Install it with: pip install \"markitdown[all]\""
        )
        return empty

    try:
        converter = MarkItDown(enable_plugins=False)
        result = converter.convert(file_path)
        text = (getattr(result, 'text_content', '') or '').strip()
        title = (getattr(result, 'title', None) or '').strip()
    except Exception as e:
        logger.error(f"Error converting document with markitdown: {e}")
        return empty

    # Light normalisation only. Note: clean_text() is deliberately NOT used
    # here, because it strips characters such as '#', '*', '|', '_' that
    # markitdown uses to represent Markdown headings, lists, and tables --
    # applying it would destroy the structure markitdown just extracted.
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if not text:
        logger.warning(f'markitdown produced no text content for: {file_path}')

    if return_type == 'dict':
        return {"title": title, "body": text}

    return text


def count_words(string):
    if string.strip() != '':
        words = string.strip().split()
        words = [word.strip() for word in words if word.strip() and not word.strip().isdigit()]
        logger.info(f"Word count: {len(words)}, Number of characters: {len(string)}")
        return len(words)
