import logging
from LLMlight.LLMlight import LLMlight

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.2.0'

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

Example
-------
>>> from LLMlight import LLMlight
>>> # Initialize model
>>> model = LLMlight(verbose='info')
>>> modelnames = model.get_available_models(validate=True)
>>> print(modelnames)


References
----------
https://github.com/erdogant/LLMlight

"""
