from LLMlight.LLMlight import LLMlight

import LLMlight.RAG as RAG
import LLMlight.utils as utils

from LLMlight.LLMlight import (
    convert_messages_to_model,
    compute_tokens,
    )

import logging

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.3.0'

# Setup root logger
_logger = logging.getLogger('LLMlight')
# Remove only our specific handler if it exists
if _logger.name == 'LLMlight':
    for handler in _logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.formatter and handler.formatter._fmt == '[{asctime}] [{name}] [{levelname}] {msg}':
            _logger.removeHandler(handler)

_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_logger.addHandler(_log_handler)
_log_handler.setLevel(logging.DEBUG)
_logger.propagate = False


# module level doc-string
__doc__ = """
LLMlight
=====================================================================

LLMlight is a Python package for running Large Language Models (LLMs) locally with minimal dependencies. It provides a simple interface to interact with various LLM models, including support for GGUF models and local API endpoints.

Example
-------
>>> from LLMlight import LLMlight
>>> # Initialize with endpoint
>>> client = LLMlight(endpoint="http://localhost:1234/v1/chat/completions")
>>> # Run queries
>>> response = client.prompt('Explain quantum computing in simple terms')
>>> print(response)

Example
-------
>>> # Use the entire context without RAG or embeddings
>>> from LLMlight import LLMlight
>>> # Initialize with endpoint
>>> client = LLMlight(endpoint="http://localhost:1234/v1/chat/completions", preprocessing=None, embedding=None, retrieval_method=None)
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
>>> client = LLMlight(preprocessing='global_reasoning')
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
>>> client = LLMlight(embedding=None, chunks=None)
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
>>> client = LLMlight(embedding=None, chunks=None, path_to_memory="knowledge_base.mp4")
>>> # Create queries
>>> response = client.prompt('What do apes like?', instructions='Only return the information from the context. Answer with maximum of 3 words, and starts with "Apes like: "')
>>> print(response)

References
----------
https://github.com/erdogant/LLMlight

"""
