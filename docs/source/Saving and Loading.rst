
Saving
################

LLMlight allows you to persist both your learned models and local knowledge databases, enabling reproducibility and easy sharing across projects. Saving a model ensures that all learned parameters, preprocessing settings, embeddings, and context strategies are stored, so that you can later reload the model without retraining. This is particularly useful when working with large datasets or when fine-tuning local models for specialized tasks.

Saving a learned model can be done using the function :func:`LLMlight.LLMlight.LLMlight.memory_save`. This stores the model on disk under a specified file name or path. It is recommended to include versioning in the filename to track updates over time.

.. code:: python

    # Load library
    from LLMlight import LLMlight
    
    # Initialize with default settings
    client = LLMlight(model='mistralai/mistral-small-3.2', file_path='local_database.mp4')
    
    url1 = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
    url2 = 'https://erdogant.github.io/publications/papers/2020%20-%20Taskesen%20et%20al%20-%20HNet%20Hypergeometric%20Networks.pdf'
    
    # Add multiple PDF files to the database
    client.memory_add(files=[url1, url2])
    
    # Add more chunks of information
    client.memory_add(text=['Small chunk that is also added to the database.',
                            'The capital of France is Amsterdam.'],
                      overwrite=True)
    
    # Add all file types from a directory
    client.memory_add(dirpath='c:/my_documents/',
                      filetypes = ['.pdf', '.txt', '.epub', '.md', '.doc', '.docx', '.rtf', '.html', '.htm'],
                      )
    
    # Store to disk
    client.memory_save()


.. note::
    - The `save` function preserves all model parameters, embeddings, and preprocessing configurations.
    - Saved models can be shared with colleagues or used on another machine, provided LLMlight is installed.

Loading
################

Loading a previously saved model can be done using the function :func:`LLMlight.LLMlight.LLMlight.memory_load`. This restores the model into memory, ready for predictions or further analysis.

.. code:: python

    # Import
    from LLMlight import LLMlight

    # Initialize with local database
    client = LLMlight(model='mistralai/mistral-small-3.2', file_path='local_database.mp4')
    
    # Get the top 5 chunks
    client.memory_chunks(n=5)
    
    # Search through the chunks using a query
    out1 = client.memory.retriever.search('Attention Is All You Need', top_k=3)
    out2 = client.memory.retriever.search('Enrichment analysis, Hypergeometric Networks', top_k=3)
    out3 = client.memory.retriever.search('Capital of Amsterdam', top_k=3)

.. note::
    - Ensure that the version of LLMlight used for loading is compatible with the version used to save the model.
    - Loading a model does not require retraining, which can save substantial time for large or complex models.
    - Models saved with additional local knowledge (MemVid databases) can be loaded with their memory intact, allowing immediate retrieval-augmented tasks.


Memory Management
################################

Create, add to, and query a persistent memory:

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


Advanced: Load Existing Memory and Continue
################################################

.. code-block:: python

    from LLMlight import LLMlight

    # Load previously saved video memory
    client = LLMlight(model='mistralai/mistral-small-3.2', retrieval_method='knowledge_base.mp4')

    # Query from loaded memory
    print(client.prompt('What is the capital of France?'))


.. include:: add_bottom.add