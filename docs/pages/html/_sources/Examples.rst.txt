Basic Prompting
################

LLMlight is a library for lightweight, modular and efficient use of LLM and RAG workflows. Below are quick examples using the main functions of the library.

.. code-block:: python

    from LLMlight import LLMlight

    # Initialize an LLMlight client (default settings)
    client = LLMlight(model='mistralai/mistral-small-3.2')

    # Ask a question using a language model
    response = client.prompt('What is the capital of France?')
    print(response)


Working with Files (PDFs)
################################

Add the content of a PDF to memory:

.. code-block:: python

    # Import library
    from LLMlight import LLMlight

    # Initialize model and memory
    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(file_path='knowledge_base.mp4')

    # Add a PDF file to the memory (extracts and chunks text automatically)
    client.memory_add(files='https://erdogant.github.io/publications/papers/2020%20-%20Taskesen%20et%20al%20-%20HNet%20Hypergeometric%20Networks.pdf')

    # Store memory to disk
    client.memory_save(overwrite=True)

    # Query on the new knowledge
    response = client.prompt('Summarize the document.')
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


.. include:: add_bottom.add