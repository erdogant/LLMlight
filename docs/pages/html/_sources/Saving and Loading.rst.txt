Saving
################

LLMlight allows you to persist your local knowledge databases, enabling reproducibility and easy sharing across projects. The default SQLite backend writes rows to disk immediately on every ``memory_add`` call. Calling ``memory_save`` additionally persists the HNSW ANN index so that fast approximate nearest-neighbour search is available on reload.

Saving a knowledge base can be done using :func:`LLMlight.LLMlight.LLMlight.memory_save`.

.. code:: python

    # Load library
    from LLMlight import LLMlight
    
    # Initialize with default SQLite backend
    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(store_path='local_database.db')
    
    url1 = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
    url2 = 'https://erdogant.github.io/publications/papers/2020%20-%20Taskesen%20et%20al%20-%20HNet%20Hypergeometric%20Networks.pdf'
    
    # Add multiple PDF files to the database
    client.memory_add(files=[url1, url2])
    
    # Add more chunks of information
    client.memory_add(text=['Small chunk that is also added to the database.',
                            'The capital of France is Amsterdam.'])
    
    # Add all file types from a directory
    client.memory_add(dirpath='c:/my_documents/',
                      filetypes=['.pdf', '.txt', '.epub', '.md', '.doc', '.docx', '.rtf', '.html', '.htm'])
    
    # Persist the ANN index to disk (SQLite rows are already saved)
    client.memory_save()


.. note::
    - SQLite rows are written to disk immediately on ``memory_add``; ``memory_save`` persists the HNSW index alongside the database.
    - Saved knowledge bases can be shared with colleagues or used on another machine, provided LLMlight is installed.

Loading
################

Loading a previously saved knowledge base can be done using :func:`LLMlight.LLMlight.LLMlight.memory_init` (which calls ``load`` internally when the store file exists) or the explicit :func:`LLMlight.LLMlight.LLMlight.memory_load`.

.. code:: python

    from LLMlight import LLMlight

    # Initialize and load existing knowledge base
    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(store_path='local_database.db')

    # Inspect top 5 chunks
    client.memory_chunks(n=5)

    # Search through chunks
    out1 = client.memory.search('Attention Is All You Need', top_k=3)
    out2 = client.memory.search('Enrichment analysis, Hypergeometric Networks', top_k=3)
    out3 = client.memory.search('Capital of Amsterdam', top_k=3)

    # Query using stored knowledge
    response = client.prompt('What is an attention network?')
    print(response)

.. note::
    - Ensure the version of LLMlight used for loading is compatible with the version used to save the database.
    - Loading does not require reprocessing documents, which saves substantial time for large collections.


Rebuilding the ANN Index
##########################

If the HNSW index file is missing or out of date (e.g. after manually editing the SQLite database), rebuild it with:

.. code:: python

    from LLMlight import LLMlight

    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(store_path='local_database.db')

    # Rebuild and save the ANN index
    client.memory_reindex(batch_size=64, save_index=True)

    # Query as normal
    response = client.prompt('What is the capital of France?')
    print(response)


Memory Management
################################

Create, add to, and query a persistent memory:

.. code-block:: python

    from LLMlight import LLMlight

    client = LLMlight(model='mistralai/mistral-small-3.2')
    # Initialize a new SQLite knowledge store
    client.memory_init(store_path='knowledge_base.db')

    # Add knowledge (as chunks of text)
    client.memory_add(text=['Apes like USB sticks', 'The capital of France is Paris.'])

    # Persist the ANN index
    client.memory_save()

    # Query from the memory
    print(client.prompt('What do apes like?'))

    # Show memory stats
    client.memory.show_stats()


Advanced: Remove Chunks and Re-query
######################################

.. code-block:: python

    from LLMlight import LLMlight

    client = LLMlight(model='mistralai/mistral-small-3.2')
    client.memory_init(store_path='knowledge_base.db')

    # Find chunks matching a query
    results = client.memory.search('capital of France', top_k=3)
    # results: [(id, score, metadata), ...]

    # Remove by id
    client.memory_remove(ids=results[0][0])

    # Or remove top-3 matches for a query
    client.memory_remove(query='USB sticks', top_k=3)

    # Verify removal
    client.memory_chunks(n=10)


.. include:: add_bottom.add
