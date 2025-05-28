import logging
from LLMlight.LLMlight import LLMlight

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'

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

LLMlight is for...

Example
-------
>>> import LLMlight as LLMlight
>>> model = LLMlight.fit_transform(X)
>>> fig,ax = LLMlight.plot(model)

References
----------
https://github.com/erdogant/LLMlight

"""
