Basic Prompting
################

LLMlight is a library for lightweight, modular and efficient use of LLM and RAG workflows. Below are quick examples using the main functions of the library.

.. code-block:: python

    from LLMlight import LLMlight

    # Initialize an LLMlight client
    client = LLMlight(model='mistralai/mistral-small-3.2')

    # Ask a question using a language model
    response = client.prompt('What is the capital of France?')
    print(response)


Working with Files (PDFs)
################################

Add the content of a PDF to memory using the default SQLite backend:

.. code-block:: python

    # Import library
    from LLMlight import LLMlight

    # Initialize model and memory
    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(store_path='knowledge_base.db')

    # Add a PDF file to the memory (extracts and chunks text automatically)
    client.memory_add(files='https://erdogant.github.io/publications/papers/2020%20-%20Taskesen%20et%20al%20-%20HNet%20Hypergeometric%20Networks.pdf')

    # Store the ANN index to disk (SQLite DB is persisted automatically)
    client.memory_save()

    # Query on the new knowledge
    response = client.prompt('Summarize the document.')
    print(response)


Loading an Existing Knowledge Base
####################################

Reload a previously built SQLite knowledge base and query it:

.. code-block:: python

    from LLMlight import LLMlight

    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(store_path='knowledge_base.db')

    response = client.prompt('What are Graphical Hypergeometric Networks?',
                             instructions='Answer using only the context.')
    print(response)


Create Summaries
###################################

Creating summaries can be done using the summary functionality. In this example, a sliding window with the last 5 chunks is kept in memory and expanded.

.. code-block:: python

    # Import library
    from LLMlight import LLMlight
    
    # Initialize    
    client = LLMlight(model='mistralai/mistral-small-3.2', top_chunks=5)
    
    # Add multiple PDF files to the database
    url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
    pdf_text = client.read_pdf(url)

    # Create summary
    text_summary = client.summarize(context=pdf_text)
    print(text_summary)


Adding and Removing Memory Chunks
####################################

.. code-block:: python

    from LLMlight import LLMlight

    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(store_path='knowledge_store.db')

    # Add text chunks
    client.memory_add(text=['Apes like USB sticks.', 'The capital of France is Paris.'])

    # Inspect stored chunks
    client.memory_chunks(n=10)

    # Search memory
    results = client.memory.search('apes', top_k=3)
    # results: list of (id, score, metadata) tuples

    # Remove by id
    client.memory_remove(ids=results[0][0])

    # Or remove by query (top-1 match by default)
    client.memory_remove(query='capital of France')

    # Prompt using stored knowledge
    response = client.prompt('What do apes like?')
    print(response)


.. include:: add_bottom.add
