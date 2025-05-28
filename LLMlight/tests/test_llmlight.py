import unittest
from LLMlight import LLMlight

class TestLLMlight(unittest.TestCase):

    model = LLMlight(verbose='debug')
    model.check_logger()
