import unittest
from LLMlight import LLMlight

class TestLLMlight(unittest.TestCase):
    
    def readme_examples():
        from LLMlight import LLMlight
        # Initialize with LM Studio endpoint
        client = LLMlight(model='mistralai/mistral-small-3.2',
                          endpoint="http://localhost:1234/v1/chat/completions")
        # Run queries
        response = client.prompt('Explain quantum computing in simple terms')
        
        
        # Initialize with LM Studio endpoint
        client = LLMlight(model='mistralai/mistral-small-3.2',
                          endpoint="http://localhost:1234/v1/chat/completions")
        
        modelnames = client.get_available_models(validate=False)
        print(modelnames)


    def append_to_memory():
        # Initialize with default settings
        client = LLMlight(model='microsoft/phi-4', file_path="saved_memory_3.mp4", context_strategy=None)

        # Add memory
        client.memory_add(text=['The capital of France is Amsterdam.', 'Some other chunk of text about sandwhiches'], overwrite=True)
        # Check chunks: must be none because it is not saved!
        chunks = client.memory_chunks()
        assert chunks is None
        # Save
        client.memory_save()
        # Get chunks
        chunks = client.memory_chunks()
        # Check
        assert len(chunks)==2

        # Add
        client.memory_add(text=['1234', '56789', '9101112'], overwrite=True)
        # Save
        client.memory_save()
        # Get chunks
        chunks = client.memory_chunks()
        # Test
        assert len(chunks)==5

        # Add
        client.memory_add(text=['1234', 'new', 'new2'], overwrite=True)
        # Save
        client.memory_save()
        # Get chunks
        chunks = client.memory_chunks()
        # Test
        assert len(chunks)==7
        
        
        # Initialize with default settings
        client = LLMlight(model='microsoft/phi-4', file_path="saved_memory_3.mp4", context_strategy=None)
        # Get chunks
        chunks = client.memory_chunks()
        # Test
        assert len(chunks)==7
        # Add
        client.memory_add(text=['new3'], overwrite=True)
        # Save
        client.memory_save()
        # Get chunks
        chunks = client.memory_chunks()
        # Test
        assert len(chunks)==8


    def only_text():
        # Initialize with default settings
        client = LLMlight(model='microsoft/phi-4', file_path="not_saved_memory.mp4", context_strategy=None)
        # Run with context
        response = client.prompt('What is the capital of France?', context='The capital of France is Amsterdam.', instructions='Do not argue with the information in the context. Only return the information from the context.')
        print(response)

    def start_with_memory_and_context():
        # Initialize with default settings
        client = LLMlight(model='microsoft/phi-4', file_path="saved_memory.mp4", context_strategy=None)

        # Add memory
        client.memory_add(text=['The capital of France is Amsterdam.',
                                'Some other chunk of text about sandwhiches'], overwrite=True)

        # Check chunks: must be none because it is not saved!
        chunks = client.memory_chunks()
        assert chunks is None

        # Save
        client.memory_save()
        chunks = client.memory_chunks()
        assert len(chunks)==2

        # Run with context
        response = client.prompt('What is the capital of France?',
                                 context='Monkeys like USB sticks.',
                                 instructions='Do not argue with the information in the context. Return the answer with at most 2 words.',
                                 )
        print(response)
        assert 'amsterdam' in response.lower()

        response = client.prompt('What do monkeys like?',
                                 context='Monkeys like USB sticks.',
                                 instructions='Do not argue with the information in the context. Return the answer with at most 2 words.',
                                 )
        print(response)
        assert 'usb' in response.lower()
        

        # Run with context
        response = client.prompt('What is the capital of France?', context='The capital of France is Amsterdam.', instructions='Do not argue with the information in the context. Only return the information from the context.')


    def start_with_only_memory():
        # Initialize with default settings
        client = LLMlight(model='microsoft/phi-4', file_path="saved_memory_2.mp4", context_strategy=None)

        # Add memory
        client.memory_add(text=['The capital of France is Amsterdam.',
                                'Some other chunk of text about sandwhiches'], overwrite=True)

        # Check chunks: must be none because it is not saved!
        chunks = client.memory_chunks()
        assert chunks is None

        # Save
        client.memory_save()
        chunks = client.memory_chunks()
        assert len(chunks)==2

        # Run with context
        response = client.prompt('What is the capital of France?',
                                 instructions='Do not argue with the information in the context. Return the answer with at most 2 words.',
                                 )
        print(response)
        assert 'amsterdam' in response.lower()

        response = client.prompt('What do monkeys like?',
                                 instructions='Do not argue with the information in the context. Return the answer with at most 2 words.',
                                 )
        print(response)
        assert not 'usb' in response.lower()
        

        # Run with context
        response = client.prompt('What is the capital of France?', context='The capital of France is Amsterdam.', instructions='Do not argue with the information in the context. Only return the information from the context.')

    def load_memory():
        # Initialize with default settings
        client = LLMlight(model='microsoft/phi-4')

    def append_to_memory():
        # Initialize with default settings
        client = LLMlight(model='microsoft/phi-4')

    def start_with_pdf():
        # Initialize with default settings
        client = LLMlight(model='microsoft/phi-4')

