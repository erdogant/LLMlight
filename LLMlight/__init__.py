from LLMlight.LLMlight import LLMlight

import LLMlight.RAG as RAG
import LLMlight.utils as utils
import LLMlight.memory as memory
import LLMlight.extract as extract
from LLMlight.sqlite_hnsw import SqliteHnswLLM

from LLMlight.LLMlight import (
    convert_messages_to_model,
    compute_tokens,
    get_embeddings,
    _set_embedding,
    _resolve_embedding,
    _resolve_chunks,
    _validate_params,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.2.0'

# Setup root logger
# Setup package-level logger
import logging
_logger = logging.getLogger('LLMlight')
_log_handler = logging.StreamHandler()
_formatter = logging.Formatter(fmt='[{asctime}] [{name:<12.12}] [{levelname:<8}] {message}', style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
# avoid duplicate handlers if re-imported
if not _logger.hasHandlers(): _logger.addHandler(_log_handler)
_logger.setLevel(logging.DEBUG)
_logger.propagate = False  # prevent duplicate log lines

# module level doc-string
__doc__ = """
LLMlight
=====================================================================

LLMlight is a Python package for running Large Language Models (LLMs) locally with minimal dependencies. It provides a simple interface to interact with various LLM models, including support for GGUF models and local API endpoints.

Example
-------
>>> from LLMlight import LLMlight
>>> # Initialize with endpoint
>>> client = LLMlight(model='mistralai/mistral-small-3.2', endpoint="http://localhost:1234/v1/chat/completions")
>>> # Run queries
>>> response = client.prompt('Explain quantum computing in simple terms')
>>> print(response)

Example
-------
>>> # Use the entire context without RAG or embeddings
>>> from LLMlight import LLMlight
>>> # Initialize with endpoint
>>> client = LLMlight(model='mistralai/mistral-small-3.2', endpoint="http://localhost:1234/v1/chat/completions", context_strategy=None, embedding=None, retrieval_method=None)
>>> # Run query with user-context
>>> response = client.prompt('What is the capital of France?', context='The capital of France is Amsterdam.', instructions='Do not argue with the information in the context. Only return the information from the context.')
>>> print(response)

Example
-------
>>> # Import library
>>> from LLMlight import LLMlight
>>> # Initialize model
>>> client = LLMlight()
>>> # List all available models at endpoint
>>> modelnames = client.get_available_models(validate=False)
>>> print(modelnames)
>>> # Check whether models really work
>>> modelnames = client.get_available_models(validate=True)
>>> print(modelnames)

Example
-------
>>> # Import library
>>> from LLMlight import LLMlight
>>> # Initialize model
>>> client = LLMlight(model='mistralai/mistral-small-3.2', context_strategy='global_reasoning')
>>> # Read pdf
>>> context = client.read_pdf(r'c://path_to_your_files//article_1.pdf', return_type='string')
>>> # Create response
>>> response = client.prompt('Summarize the main points of this document.', context=context)
>>> print(response)

Example
-------
>>> # Example to use video memory for local storing of information
>>> # Import library
>>> from LLMlight import LLMlight
>>> # Initialize with default settings
>>> client = LLMlight(model='mistralai/mistral-small-3.2', retrieval_method=None)
>>> # Create new video memory
>>> client.memory_init()
>>> # Add pdf file
>>> filepaths = [r'c://path_to_your_files//article_1.pdf', r'c://path_to_your_files//my_file.txt']
>>> client.memory_add(input_files=filepaths)
>>> # Add text chunks
>>> client.memory_add(text=['Apes like USB sticks', 'Trees are mainly yellow'])
>>> # Build memory
>>> client.memory_save(filepath="knowledge_base.mp4", overwrite=False)
>>> response = client.prompt('What do apes like?', instructions='Only return the information from the context. Answer with maximum of 3 words, and starts with "Apes like: "')
>>> print(response)
>>> # Run a simple query
>>> response = client.prompt('What is the capital of France?', context='The capital of France is Amsterdam.', instructions='Do not argue with the information in the context. Only return the information from the context.')
>>> print(response)
>>> response = client.prompt('Provide a summary of HyperSpectral from the pdf or text file.', instructions='Do not argue with the information in the context. Only return the information from the context.')
>>> print(response)

Example
-------
>>> # Example to use existing video memory
>>> # Import library
>>> from LLMlight import LLMlight
>>> # Initialize with default settings
>>> client = LLMlight(model='mistralai/mistral-small-3.2', retrieval_method="knowledge_base.mp4")
>>> # Create queries
>>> response = client.prompt('What do apes like?', instructions='Only return the information from the context. Answer with maximum of 3 words, and starts with "Apes like: "')
>>> print(response)

Example
-------
>>> # Example to use existing video memory
>>> # Import library
>>> from LLMlight import LLMlight
>>> # Initialize with default settings
>>> client = LLMlight(retrieval_method="knowledge_base.mp4", embedding={'memory': 'tifidf', 'context': 'bert'})
>>> # Create queries
>>> response = client.prompt('What do apes like?', instructions='Only return the information from the context. Answer with maximum of 3 words, and starts with "Apes like: "')
>>> print(response)

Example
-------
>>> client = LLMlight(model='mistralai/mistral-small-3.2')
>>> response = client.prompt('What is the capital of France?')
>>> print(response)

Example
-------
>>> # With context and instructions
>>> response = client.prompt('What do apes like?', context='Apes like USB sticks.', instructions='Answer in one sentence using only the context.')
>>> print(response)

Example
-------
>>> # Override temperature for a single call
>>> response = client.prompt('Write a short poem.', temperature=1.0)
>>> print(response)

Example
-------
>>> # Structured extraction with source grounding, via Google's LangExtract
>>> # https://github.com/google/langextract
>>> from LLMlight import LLMlight
>>> from LLMlight import extract as lx_extract
>>> client = LLMlight(model='mistralai/mistral-small-3.2')
>>> examples = [lx_extract.build_example(
...     text="ROMEO. But soft! What light through yonder window breaks?",
...     extractions=[{"extraction_class": "character",
...                    "extraction_text": "ROMEO",
...                    "attributes": {"emotional_state": "wonder"}}],
... )]
>>> result = client.extract(
...     text_or_documents="Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
...     prompt_description="Extract characters, emotions, and relationships in order of appearance.",
...     examples=examples,
... )
>>> # Instantly visualize every extraction, highlighted in its original context
>>> html = client.visualize_extractions(result, output_html="visualization.html")

References
----------
https://github.com/erdogant/LLMlight

"""
