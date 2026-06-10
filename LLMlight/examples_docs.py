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
client = LLMlight(model='liquid/lfm2-24b-a2b', top_chunks=5, chunks={'method': 'words', 'size': 1000, 'overlap': 200})
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

client.prompt('Explain the working of HNet - hypergeometric networks in 4 sentences.')

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

# %%

# %%

# %%

# %%

# %%

# %%

# %%


