Basic Prompting
################

LLMlight is a library for lightweight, modular and efficient use of LLM and RAG workflows. Below are quick examples using the main functions of the library.

.. code-block:: python

    from LLMlight import LLMlight
    
    # Initialize an LLMlight client
    client = LLMlight(model='liquid/lfm2-24b-a2b')
    
    # Ask a question using a language model
    response = client.prompt('What is the capital of France?')
    print(response)


Loading an Existing Knowledge Base
####################################

Reload a previously built SQLite knowledge base and query it:

.. code-block:: python

    from LLMlight import LLMlight
    
    # Initialize an LLMlight client
    client = LLMlight(model='liquid/lfm2-24b-a2b')
    client.memory_load(store_path='local_database.db')
    
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



Create Agent Discussion
################################

One of the most interesting applications is creating autonomous discussions between multiple AI agents.
In this example, you will learn to setup two different agents, each with its own knowledge base, retrieval settings, and personality.
During the conversation, every agent uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from its memory before generating a response. 
This allows the agents to reason from different perspectives, challenge each other's assumptions, and build upon previously exchanged ideas. By assigning distinct expertise domains, such as a Data Scientist, and an Farmer, you can simulate debates, brainstorming sessions, or expert panel discussions. 

See the code block below to combine memory, retrieval, and language generation to create dynamic multi-agent workflows that go far beyond simple question-answer interactions.


.. code-block:: python

    from LLMlight import LLMlight
    
    # ====================================================
    # Agent A: Data Scientist
    # ====================================================
    agent_a = LLMlight(
        model="google/gemma-4-26b-a4b-qat",
        retrieval_method="naive_rag",
        context_strategy=None,
        top_chunks=5,
        temperature=0.7,
    )
    agent_a.memory_init(store_path="agent_a.db")
    
    # ====================================================
    # Agent B: Farmer
    # ====================================================
    agent_b = LLMlight(
        model="google/gemma-4-26b-a4b-qat",
        retrieval_method="naive_rag",
        context_strategy=None,
        top_chunks=5,
        temperature=0.7,
    )
    agent_b.memory_init(store_path="agent_b.db")
    
    # ====================================================
    # Moderator Agent (keeps discussion structured)
    # ====================================================
    moderator = LLMlight(
        model="google/gemma-4-26b-a4b-qat",
        retrieval_method="naive_rag",
        context_strategy=None,
        top_chunks=5,
        temperature=0.3,
    )
    moderator.memory_init(store_path="moderator.db")
    
    # ====================================================
    # Scoring Agent (decides convergence / stopping)
    # ====================================================
    scoring_agent = LLMlight(
        model="google/gemma-4-26b-a4b-qat",
        retrieval_method="naive_rag",
        context_strategy=None,
        top_chunks=5,
        temperature=0.0,  # deterministic scoring
    )
    scoring_agent.memory_init(store_path="scoring.db")
    
    
    # ====================================================
    # Shared memory (optional logging)
    # ====================================================
    shared_memory = LLMlight(model="google/gemma-4-26b-a4b-qat")
    shared_memory.memory_init(store_path="discussion.db")
    
    
    # ====================================================
    # Discussion Loop with early stopping
    # ====================================================
    topic = "Discuss the importance of attention mechanisms in modern AI."
    message = topic
    
    MAX_ROUNDS = 5
    AGREEMENT_THRESHOLD = 0.85  # stop if convergence is high enough
    
    for turn in range(MAX_ROUNDS):
    
        print(f"\n{'='*80}")
        print(f"ROUND {turn+1}")
        print(f"{'='*80}")
    
        # --------------------------
        # Agent A
        # --------------------------
        response_a = agent_a.prompt(f"""
        You are a Data Scientist.
    
        Topic:
        {message}
    
        Respond in 1-2 paragraphs and ask a question.
        """)
    
        print("\nAgent A:")
        print(response_a)
    
        # --------------------------
        # Agent B
        # --------------------------
        response_b = agent_b.prompt(f"""
        You are a Farmer.
    
        Data Scientist said:
        {response_a}
    
        Respond in 1-2 paragraphs and continue the discussion.
        """)
    
        print("\nAgent B:")
        print(response_b)
    
        # --------------------------
        # Moderator summary
        # --------------------------
        moderator_summary = moderator.prompt(f"""
        You are a neutral moderator.
    
        Data Scientist:
        {response_a}
    
        Farmer:
        {response_b}
    
        Summarize:
        - agreements
        - disagreements
        - next question toward consensus
        """)
    
        print("\nModerator:")
        print(moderator_summary)
    
        # --------------------------
        # Scoring Agent (convergence check)
        # --------------------------
        score_output = scoring_agent.prompt(f"""
        You are a scoring system.
    
        Evaluate agreement between the two agents.
    
        Data Scientist:
        {response_a}
    
        Farmer:
        {response_b}
    
        Moderator summary:
        {moderator_summary}
    
        Return ONLY a number between 0 and 1:
        - 1.0 = full agreement / consensus reached
        - 0.0 = complete disagreement
        """)
    
        try:
            score = float(score_output.strip())
        except:
            score = 0.0
    
        print("\nAgreement Score:", score)
    
        # --------------------------
        # Store memory
        # --------------------------
        shared_memory.memory_add(response_a)
        shared_memory.memory_add(response_b)
        shared_memory.memory_add(moderator_summary)
    
        # --------------------------
        # Early stopping condition
        # --------------------------
        if score >= AGREEMENT_THRESHOLD:
            print("\nConsensus reached early. Stopping discussion.")
            break
    
        # Next round context
        message = moderator_summary
    
    
    # ====================================================
    # Final summary
    # ====================================================
    final_summary = shared_memory.prompt("""
    Summarize the full discussion:
    
    - final consensus
    - key arguments
    - remaining open points (if any)
    """)
    
    print("\nFINAL SUMMARY")
    print("=" * 80)
    print(final_summary)


.. include:: add_bottom.add
