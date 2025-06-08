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
>>> # Initialize with LM Studio endpoint
>>> model = LLMlight(endpoint="http://localhost:1234/v1/chat/completions")
>>> # Run queries
>>> response = model.prompt('Explain quantum computing in simple terms')
>>> print(response)

Example
-------
>>> # Import library
>>> from LLMlight import LLMlight
>>> # Initialize model
>>> model = LLMlight()
>>> # List all available models at endpoint
>>> modelnames = model.get_available_models(validate=False)
>>> print(modelnames)
>>> # Check whether models really work
>>> modelnames = model.get_available_models(validate=True)
>>> print(modelnames)

Example
-------
>>> # Import library
>>> from LLMlight import LLMlight
>>> # Initialize with default settings
>>> client = LLMlight(embedding=None, chunks=None)
>>> # Create new video memory
>>> client.memory_init(filepath="knowledge_base.mp4")
>>> # Add pdf file
>>> filepaths = [r'c://path_to_your_files//article_1.pdf', r'c://path_to_your_files//my_file.txt']
>>> client.memory_add(input_files=filepaths)
>>> # Add text chunks
>>> client.memory_add(text=['Apes like USB sticks', 'Trees are mainly yellow'])
>>> # Build memory
>>> client.memory_save(overwrite=False)
>>> response = client.prompt('What do apes like?', instructions='Only return the information from the context. Answer with maximum of 3 words, and starts with "Apes like: "')
>>> print(response)
>>> # Run a simple query
>>> response = client.prompt('What is the capital of France?', context='The capital of France is Amsterdam.', instructions='Do not argue with the information in the context. Only return the information from the context.')
>>> print(response)
>>> response = client.prompt('Provide a summary of HyperSpectral from the pdf or text file.', instructions='Do not argue with the information in the context. Only return the information from the context.')
>>> print(response)

References
----------
https://github.com/erdogant/LLMlight

"""
