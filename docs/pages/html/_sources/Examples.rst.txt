Quickstart
==========

LLMlight is a library for lightweight, modular and efficient use of LLM and RAG workflows. Below are quick examples using the main functions of the library.

Installation
--------------

.. code-block:: bash

    pip install LLMlight

Basic Usage
-----------

Initialize and Simple Prompting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from LLMlight import LLMlight

    # Initialize an LLMlight client (default settings)
    client = LLMlight(model='mistralai/mistral-small-3.2')

    # Ask a question using a language model
    response = client.prompt('What is the capital of France?')
    print(response)

Memory Management (Video Memory)
---------------------------------

Create, add to, and query a persistent video memory:

.. code-block:: python

    from LLMlight import LLMlight

    client = LLMlight()
    # Initialize a new video memory file
    client.memory_init(file_path='knowledge_base.mp4')

    # Add knowledge (as chunks of text)
    client.memory_add(text=['Apes like USB sticks', 'The capital of France is Paris.'], overwrite=True)

    # Store memory to disk
    client.memory_save()

    # Query from the memory
    print(client.prompt('What do apes like?'))

    # Show memory stats
    client.memory.show_stats()

Working with Files (PDFs)
--------------------------

Add the content of a PDF to memory:

.. code-block:: python

    from LLMlight import LLMlight

    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(file_path='knowledge_base.mp4')

    # Add a PDF file to the memory (extracts and chunks text automatically)
    client.memory_add(files='https://erdogant.github.io/publications/papers/2020%20-%20Taskesen%20et%20al%20-%20HNet%20Hypergeometric%20Networks.pdf')
    # Store memory to disk
    client.memory_save(overwrite=True)

    # Query on the new knowledge
    response = client.prompt('Summarize the document.')
    print(response)


Advanced: Load Existing Memory and Continue
--------------------------------------------

.. code-block:: python

    from LLMlight import LLMlight

    # Load previously saved video memory
    client = LLMlight(model='mistralai/mistral-small-3.2', retrieval_method='knowledge_base.mp4')
    # Query from loaded memory
    print(client.prompt('What is the capital of France?'))



.. include:: add_bottom.add