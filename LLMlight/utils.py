""" Utils."""
import json
from json_repair import repair_json
import re
import pymupdf
import logging
import os
import numpy as np

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

def clean_text(text):
    """Cleans up the text by removing unnecessary spaces between characters and excessive whitespace."""
    # text = str(text).replace('\n', ' ').replace('"', '')
    text = str(text).replace('{', '').replace('}', '')
    # Clean excessive spaces between words (multiple spaces turned into one)
    text = re.sub(r'\s+', ' ', text.strip())  # Clean up multiple spaces
    # Return
    return text

def read_pdf(file_path, title_pages=[1, 2], body_pages=[], reference_pages=[-1], return_type='dict'):
    logger.info('Reading pdf')
    """
    Reads a PDF file and extracts its text content as a string.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.

    """
    if not os.path.isfile(file_path):
        logger.error(f'File not found on disk: {file_path}')
        return None
    if not file_path.lower().endswith('.pdf'):
        logger.error("The provided file path is not a valid PDF file.")
        None
    if title_pages is None: title_pages = []
    if body_pages is None: body_pages = []
    if reference_pages is None: reference_pages = []
    if isinstance(title_pages, (str, int)): title_pages = [title_pages]
    if isinstance(body_pages, (str, int)): body_pages = [body_pages]
    if isinstance(reference_pages, (str, int)): reference_pages = [reference_pages]

    try:
        # Open pdf
        doc = pymupdf.open(file_path)
        # Get the total number of pages
        num_pages = len(doc)
        # Replace negative indices with corresponding positive indices
        reference_pages = [num_pages + page if page < 0 else page for page in reference_pages]

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
            if np.isin(page_num + 1, title_pages):
                title_text += "\n" + page_text
            elif np.isin(page_num + 1, reference_pages):
                references_text += "\n" + page_text
            elif np.isin(page_num + 1, body_pages):
                body_text.append(page_text)
            elif len(body_pages)==0:
                body_text.append(page_text)

    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return {"title": "", "body": "", "references": ""}

    # Return
    if return_type=='dict':
        context = {"title": title_text.strip(),
                   "body": "\n".join(body_text).strip(),
                   "references": references_text.strip(),
                   }
    else:
        context = title_text.strip() + "\n---\n".join(body_text).strip() + '\n---\n' + references_text.strip()

    # Return
    return context


def count_words(string):
    if string.strip() != '':
        words = string.strip().split(' ')
        words = [word.strip() for word in words if word.strip() and not word.strip().isdigit()]
        logger.info(f"Word count: {len(words)}, Number of characters: {len(string)}")
        return len(words)
