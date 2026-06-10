# %%
from LLMlight import LLMlight

# Initialize LLMlight client
client = LLMlight(verbose='info')

# Get a list of available models
modelnames = client.get_available_models(validate=False)

# Print available models
print("Available models:", modelnames)

# %%

from LLMlight import LLMlight

# Initialize with default settings and only use the top 5 chunks in the analysis
client = LLMlight(model='liquid/lfm2-24b-a2b', top_chunks=5, chunks={'method': 'words', 'size': 1000, 'overlap': 200}, alpha=0.05)
client.memory_init(store_path='local_database.db', overwrite=True)

# Add multiple PDF files to the database
url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
pdf_text = client.read_pdf(url)
client.memory_add(text=pdf_text)
# Ad another pdf
url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
pdf_text = client.read_pdf(url)
client.memory_add(text=pdf_text)

# Get all chunks
out = client.memory.get_all_chunks()
print(f'Number of chunks stored: {len(out)}')

client.prompt('Explain the working of HNet - hypergeometric networks', response_format='Response with a maximum of 4 sentences.')

# %%
client = LLMlight(model='liquid/lfm2-24b-a2b', alpha=0.05)
client.memory_init(store_path='local_database.db', overwrite=False)
# Get all chunks
out = client.memory.get_all_chunks()
print(f'Number of chunks stored: {len(out)}')

# Make the prompt with only the significant chunks
client.prompt('Explain the working of HNet hypergeometric networks', response_format='Response with a maximum of 4 sentences.')


# %%
summary_text = client.summarize(context=pdf_text)
print(summary_text)

# [WARNING] Prompt length (XXXX tokens) exceeds the model context window (4096 tokens).
# The model will truncate the input and important context may be lost.

#   How to fix:
#   * Reduce chunk size:    LLMlight(..., chunks={'size': 500})
#     Smaller chunks = fewer tokens per prompt.
#   * Reduce chunk overlap: LLMlight(..., chunks={'overlap': 50})
#   * Reduce top_chunks:    LLMlight(..., top_chunks=3)
#     Fewer chunks combined into a single prompt.
#   * Use summarize():      client.summarize(context=text)
#     Splits the document automatically, chunk by chunk.
#   * Increase n_ctx:       LLMlight(..., n_ctx=8192)
#     Only works if your model actually supports a larger window.
    
# %% Preprocessing & Chunking
client.memory_add(text=['Apes like USB sticks.', 'The capital of France is Amsterdam.'])
out = client.memory.get_all_chunks()
print(out)
len(out)

client.prompt('What is the capital of france?', instructions='Your response must be the truth.')
client.prompt('What is the capital of france?', instructions='only use the context.', response_format='only output 1 word.')

# Store the database to disk (SQLite is auto-persisted; this also saves the ANN index)
# client.memory_save()

# %% RAG with Statistical Validation

# Load database for inference.
client = LLMlight(model='liquid/lfm2-24b-a2b', alpha=0.1)
client.memory_init(store_path='local_database1.db')

# Inspect top 2 chunks
client.memory_chunks(n=2)

# Search through chunks using queries
out1a = client.relevant_memory_retrieval('Attention Is All You Need')
out2a = client.relevant_memory_retrieval('Enrichment analysis, Hypergeometric Networks')

# RAG only
out1b = client.memory.search('Attention Is All You Need')
out2b = client.memory.search('Enrichment analysis, Hypergeometric Networks', top_k=3)


# %%
from LLMlight import LLMlight

# Initialize an LLMlight client
client = LLMlight(model='liquid/lfm2-24b-a2b')

# Ask a question using a language model
response = client.prompt('What is the capital of France?')
print(response)

# %%
from LLMlight import LLMlight

# Initialize an LLMlight client
client = LLMlight(model='liquid/lfm2-24b-a2b')
client.memory_load(store_path='local_database.db')

# Ask a question using a language model
response = client.prompt('What is the capital of France?')
print(response)


# %%
# Import library
from LLMlight import LLMlight

# Initialize model and memory
client = LLMlight(model='liquid/lfm2-24b-a2b')
client.memory_init(store_path='knowledge_base.db')

# Add a PDF file to the memory (extracts and chunks text automatically)
client.memory_add(files='https://erdogant.github.io/publications/papers/2020%20-%20Taskesen%20et%20al%20-%20HNet%20Hypergeometric%20Networks.pdf')

# Store the ANN index to disk (SQLite DB is persisted automatically)
client.memory_save()

# Query on the new knowledge
response = client.prompt('Summarize the document.')
print(response)


# %%
# Import library
from LLMlight import LLMlight

# Initialize
client = LLMlight(model='liquid/lfm2-24b-a2b', top_chunks=5)

# Add multiple PDF files to the database
url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
pdf_text = client.read_pdf(url)

# Create summary
text_summary = client.summarize(context=pdf_text)
print(text_summary)

# %%
from LLMlight import LLMlight

client = LLMlight(model='liquid/lfm2-24b-a2b')
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

# %%
# Import library
from LLMlight import LLMlight

# Initialize model and memory
client = LLMlight(model='liquid/lfm2-24b-a2b')
client.memory_init(store_path='knowledge_base.db')

# Add a PDF file to the memory (extracts and chunks text automatically)
client.memory_add(files='https://erdogant.github.io/publications/papers/2020%20-%20Taskesen%20et%20al%20-%20HNet%20Hypergeometric%20Networks.pdf')

# Store the ANN index to disk (SQLite DB is persisted automatically)
client.memory_save()

# Query on the new knowledge
response = client.prompt('Summarize the document.')
print(response)


# %%
from LLMlight import LLMlight

client = LLMlight(model='liquid/lfm2-24b-a2b')
client.memory_init(store_path='knowledge_base.db')
response = client.prompt('What are Graphical Hypergeometric Networks?', instructions='Answer using only the context.')
print(response)

# %%
from LLMlight import LLMlight

client = LLMlight(model='mistralai/mistral-small-3.2')
client.memory_init(store_path='local_database.db')

# Rebuild and save the ANN index
client.memory_reindex(batch_size=64, save_index=True)

# Query as normal
response = client.prompt('What is the capital of France?')
print(response)


# %%
from LLMlight import LLMlight

# ====================================================
# Agent A: Data Scientist
# ====================================================
agent_a = LLMlight(
    model="google/gemma-4-26b-a4b-qat",
    retrieval_method="naive_rag",
    embedding="bert",
    context_strategy=None,
    top_chunks=5,
    temperature=0.7,
)

agent_a.memory_init(store_path="agent_a.db")

agent_a.memory_add("""
Large Language Models are one of the most important step we did in the field of AI
It helps the workload and the work easier and faster.
""")

agent_a.memory_add("""
Large Language Models use transformer architectures and are trained on
massive text corpora using self-supervised learning.
""")


# ====================================================
# Agent B: Farmer
# ====================================================
agent_b = LLMlight(
    model="google/gemma-4-26b-a4b-qat",
    retrieval_method="naive_rag",
    embedding="bert",
    context_strategy=None,
    top_chunks=5,
    temperature=0.7,
)

agent_b.memory_init(store_path="agent_b.db")

agent_b.memory_add("""
The use of AI and machine learning consumes to much power and there is no need for this
new technology. The human work was good enough and there is no need to change that.
Recent research shows that LLMs hallucinate and do not solve real world applications.
""")


# ====================================================
# Discussion Loop
# ====================================================

topic = "Discuss the importance of the use of Large Language Models and AI."
message = topic

for turn in range(5):

    print(f"\n{'='*80}")
    print(f"ROUND {turn+1}")
    print(f"{'='*80}")

    response_a = agent_a.prompt(
        system="You are a Data Scientist.",
        query=
        f"""

        Topic:
        {message}

        """,

        response_format="Give your opinion in 1-2 paragraphs and ask a question to the farmer.",
    )

    print("\nAgent A:")
    print(response_a)

    response_b = agent_b.prompt(
        system="You are a farmer.",
        query=
        f"""

        The Data Scientist said:

        {response_a}
        
        """,
        response_format="Respond to the discussion in 1-2 paragraphs and ask a follow-up question to the data scientist.",
    )

    print("\nAgent B:")
    print(response_b)

    message = response_b


# %%

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

agent_a.memory_add("""
Large Language Models are one of the most important step we did in the field of AI
It helps the workload and the work easier and faster.
Large Language Models use transformer architectures and are trained on
massive text corpora using self-supervised learning.
""")

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

agent_b.memory_add("""
The use of AI and machine learning consumes to much power and there is no need for this
new technology. The human work was good enough and there is no need to change that.
Recent research shows that LLMs hallucinate and do not solve real world applications.
""")


# ====================================================
# Agent C: Moderator
# ====================================================
moderator = LLMlight(
    model="google/gemma-4-26b-a4b-qat",
    retrieval_method="naive_rag",
    context_strategy=None,
    top_chunks=5,
    temperature=0.3,  # lower temperature for objective summaries
)

moderator.memory_init(store_path="moderator.db")


# ====================================================
# Shared discussion memory
# ====================================================
shared_memory = LLMlight(model="google/gemma-4-26b-a4b-qat")

shared_memory.memory_init(store_path="discussion.db")


# ====================================================
# Discussion Loop
# ====================================================
topic = "Discuss the importance of the use of Large Language Models and AI."
message = topic

for turn in range(5):

    print(f"\n{'='*80}")
    print(f"ROUND {turn+1}")
    print(f"{'='*80}")

    # --------------------------------------------
    # Agent A responds
    # --------------------------------------------
    response_a = agent_a.prompt(
        f"""
        You are a Data Scientist.

        Current discussion:
        {message}

        Provide your opinion and ask a question to the Farmer in 1-2 paragraphs.
        """
    )

    print("\nData Scientist:")
    print(response_a)

    # --------------------------------------------
    # Agent B responds
    # --------------------------------------------
    response_b = agent_b.prompt(
        f"""
        You are a Farmer.

        The Data Scientist said:

        {response_a}

        Respond and ask a follow-up question in 1-2 paragraphs.
        """
    )

    print("\nFarmer:")
    print(response_b)

    # --------------------------------------------
    # Moderator summarizes
    # --------------------------------------------
    moderator_summary = moderator.prompt(
        f"""
        You are a neutral moderator.

        Data Scientist:
        {response_a}

        Farmer:
        {response_b}

        Perform the following tasks:
        1. Summarize the key arguments.
        2. Identify agreements.
        3. Identify disagreements.
        4. Propose one question that helps both agents move toward consensus.

        Keep the output concise.
        """
    )

    print("\nModerator:")
    print(moderator_summary)

    # Store discussion history
    shared_memory.memory_add(response_a)
    shared_memory.memory_add(response_b)
    shared_memory.memory_add(moderator_summary)

    # Next round starts from moderator guidance
    message = moderator_summary


# ====================================================
# Final consensus
# ====================================================
consensus = shared_memory.prompt(
    """
    Review the discussion and provide:

    - Main conclusions
    - Remaining disagreements
    - Final consensus statement

    Keep it under 200 words.
    """
)

print("\nFINAL CONSENSUS")
print("=" * 80)
print(consensus)

# %%

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

# %%


# Import library
from LLMlight import LLMlight

# Initialize model and memory
client = LLMlight(model='google/gemma-4-26b-a4b-qat', retrieval_method='naive_rag', context_strategy='global-reasoning', top_chunks=6)
client = LLMlight(model='google/gemma-4-26b-a4b-qat', retrieval_method='naive_rag', context_strategy='chunk-wise', top_chunks=6)
client = LLMlight(model='google/gemma-4-26b-a4b-qat', retrieval_method='naive_rag', context_strategy=None, top_chunks=6)

# Create (or load) database
client.memory_init(store_path='knowledge_base.db')

# Add a PDF file to the database (extracts and chunks text automatically)
url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
pdf_text = client.read_pdf(url)

# Write to db
client.memory_add(text=pdf_text)
len(client.memory_chunks())
client.memory_chunks(1)

# Store to disk (SQLite DB is persisted automatically)
client.memory_save()

# Query on the new knowledge
response = client.prompt('What are attention networks? Summarize in 2 sentence')
print(response)

# %% Context strategy : global-reasoning
from LLMlight import LLMlight

# Initialize model and memory
client = LLMlight(model='google/gemma-4-26b-a4b-qat', retrieval_method='naive_rag', context_strategy='global-reasoning', top_chunks=6)
# client = LLMlight(model='google/gemma-4-26b-a4b-qat', retrieval_method='naive_rag', context_strategy='chunk-wise', top_chunks=6)
# client = LLMlight(model='google/gemma-4-26b-a4b-qat', retrieval_method='naive_rag', context_strategy=None, top_chunks=6)


# or: overwrite in one go
client.memory_init('knowledge_base.db', overwrite=True)


# Create (or load) database
client.memory_init(store_path='knowledge_base.db')
len(client.memory_chunks())

# Query on the new knowledge
response = client.prompt('What are attention networks? Summarize in 2 sentence')
print(response)

# release the file lock
client.memory_close()


# %%
from LLMlight import LLMlight

# Initialize an LLMlight client
client = LLMlight(model='google/gemma-4-26b-a4b-qat')

# Ask a question using a language model
response = client.prompt('What is the capital of France?')
print(response)

# [LLMlight.LLM] [INFO    ] Model            : google/gemma-4-26b-a4b-qat
# [LLMlight.LLM] [INFO    ] Context strategy : disabled
# [LLMlight.LLM] [INFO    ] Retrieval method : naive_rag
# [LLMlight.LLM] [INFO    ] Embedding        : {'memory': 'bert', 'context': 'bert'}
# [LLMlight.LLM] [INFO    ] Alpha (sig. test): None
# [LLMlight.LLM] [INFO    ] Chunk config     : {'method': 'chars', 'size': 1000, 'overlap': 200}
# [LLMlight.LLM] [INFO    ] LLMlight initialised.
# [LLMlight.LLM] [INFO    ] Creating response with google/gemma-4-26b-a4b-qat..
# [LLMlight.LLM] [INFO    ] No context strategy applied.
# [LLMlight.LLM] [INFO    ] No context is provided into the prompt.
# [LLMlight.LLM] [INFO    ] Running model: google/gemma-4-26b-a4b-qat 
# The capital of France is Paris.

# %%


# %% Get models
from LLMlight import LLMlight
client = LLMlight()


# %%

dbname = 'knowledge_store.db'

from LLMlight import LLMlight
client = LLMlight(model='gemma-4-e2b-it')
client.memory_init(store_path=dbname)
client.memory_add(text=['BMC test1'])
client.memory.get_all_chunks()
results = client.memory.search('bmc')
print(results)
client.memory_add(text=['Apes like USB sticks.', 'The capital of France is Amsterdam.'])
client.memory.get_all_chunks()

# Remove
# Load library
from LLMlight import LLMlight
# normal init
client = LLMlight(model='gemma-4-e2b-it')
# Init or load when exists
client.memory_init(store_path=dbname)
client.memory_chunks()

# Add text
client.memory_add(text=['BMC test3'])

results = client.memory.search('bmc')
# [(31, 0.23, {'text': 'BMC test', 'id': 31})]
client.memory_chunks()
client.prompt('what do you know about BMC?')

client.memory_remove(ids=results[0][0])
client.memory_remove(query='BMC')     # by query (top-1 match)
client.memory_remove(query='BMC', top_k=3)  # top-3 matches
client.prompt('what do you know about BMC?')


# %%


# Load library
from LLMlight import LLMlight
# normal init
client = LLMlight(model='gemma-4-e2b-it', embedding='bert')
# Initialize a local SQLite+HNSW memory (default backend)
client.memory_init(store_path='knowledge_store.db')  # creates 'knowledge_store.db'
# Initialize memvid backend (video-memory)
# client.memory_init(file_path='my_video_memory.mp4', backend='memvid')

# Add chunks
client.memory_add(text=['Apes like USB sticks.', 'The capital of France is Amsterdam.'])

# should return the inserted chunks
client.memory_chunks()  
                  
# Save index (saves ANN index if hnswlib is present)
# client.memory_save()
# Query memory
results = client.relevant_memory_retrieval('Tell me about Apes.')  # will use sqlite+hnsw by default

print(results[0][2]['text'])
print(results[0][1])

results = client.relevant_memory_retrieval('What is the capital of France?')  # will use sqlite+hnsw by default
print(results[0][2]['text'])
print(results[0][1])

# %%


from LLMlight import LLMlight
# normal init
client = LLMlight()
client.get_available_models()

# %%

# Load library
from LLMlight import LLMlight
# normal init
client = LLMlight(model='gemma-4-e2b-it')

# prompt
out = client.prompt('who are you?')
print(out)

# %%


from LLMlight import LLMlight
client = LLMlight()
client.memory_init(store_path='knowledge_store.db')
client.memory_add(text=['BMC test1'])
client.memory.get_all_chunks()


# Rebuild the ANN index (requires sentence-transformers + hnswlib)
client.memory.retriever.index_manager.backend.reindex(batch_size=64, save_index=True)
results = client.relevant_memory_retrieval('What do apes like?', return_type='list')
print(results)



# %%

# =============================================================================
# Make a summary
# =============================================================================

from LLMlight import LLMlight
client = LLMlight(model='gemma-4-e2b-it', top_chunks=5)
client.memory_init(store_path='neurips_store10.db')

# Add multiple PDF files to the database
url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
pdf_text = client.read_pdf(url)

client.memory_add(text=pdf_text)
out = client.memory.get_all_chunks()
len(out)

client.prompt('When to get BLEU score of 41.0?')

summary_text = client.summarize(context=pdf_text)
print(summary_text)

# Add chunks
client.memory_add(text=['Apes like USB sticks.', 'The capital of France is Amsterdam.'])
out = client.memory.get_all_chunks()
print(out)
len(out)
client.prompt('What is the capital of france?', instructions='Your response must be the truth.')
client.prompt('What is the capital of france?', instructions='only use the context')

# %%

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# Load library
from LLMlight import LLMlight

# Initialize with default settings
# client = LLMlight(model='unsloth/gemma-4-26b-a4b-it')
client = LLMlight(model='mistralai/mistral-small-3.2')

# Add multiple PDF files to the database
url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
pdf_text = client.read_pdf(url)

# Create response
responses = []
for temp in [0.99, 0.99, 0.99, 0.99, 0.99, 0.1, 0.1, 0.1, 0.1, 0.1]:
    response = client.prompt('Summarize how layers are used in an attention network in combination to the increasing complexity.',
                             context=pdf_text,
                             instructions='You are a helpfull assistant. Keep your answer brief.',
                             temperature=temp,
                             )
    
    responses.append(response)

