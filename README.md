# LLMlight

[![Python](https://img.shields.io/pypi/pyversions/LLMlight)](https://img.shields.io/pypi/pyversions/LLMlight)
[![Pypi](https://img.shields.io/pypi/v/LLMlight)](https://pypi.org/project/LLMlight/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/LLMlight/)
[![LOC](https://sloc.xyz/github/erdogant/LLMlight/?category=code)](https://github.com/erdogant/LLMlight/)
[![Downloads](https://static.pepy.tech/personalized-badge/LLMlight?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/LLMlight)
[![Downloads](https://static.pepy.tech/personalized-badge/LLMlight?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/LLMlight)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/LLMlight/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/LLMlight.svg)](https://github.com/erdogant/LLMlight/network)
[![Issues](https://img.shields.io/github/issues/erdogant/LLMlight.svg)](https://github.com/erdogant/LLMlight/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Medium](https://img.shields.io/badge/Medium-Blog-black)](https://erdogant.github.io/LLMlight/pages/html/Documentation.html#medium-blog)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://erdogant.github.io/LLMlight/pages/html/Documentation.html#colab-notebook)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/LLMlight/pages/html/Documentation.html#)

<div>
LLMlight is a Python package for running Large Language Models (LLMs) locally with minimal dependencies. It provides a simple interface to interact with various LLM models, including support for GGUF models and local API endpoints. ⭐️Star it if you like it⭐️
</div>

---
<p align="left">
  <a href="https://erdogant.github.io/llmlight/">
    <img src="https://raw.githubusercontent.com/erdogant/llmlight/master/docs/figs/schematic_overview.png" width="600" alt="Schematic Overview" />
  </a>
</p>


### Key Features

| Feature | Description |
|---------|-------------|
| [**Local LLM Support**](https://erdogant.github.io/LLMlight/pages/html/Algorithm.html#get-available-llm-models) | Run LLMs locally with minimal dependencies. |
| [**Full Prompt Control**]() | Fine-grained control over prompts including Query, Instructions, System, Context, Response Format, Automatic formatting, Temperature, and Top P. |
| [**Single Endpoint for All Local Models**](https://erdogant.github.io/LLMlight/pages/html/Algorithm.html#get-available-llm-models) | One unified endpoint to connect different local models, supporting Hermes-3-Llama-3.2-3B, Mistral-7B-Grok, OpenHermes-2.5-Mistral-7B, Gemma-2-9B-IT, and more. |
| [**Flexible Embedding Methods**](https://erdogant.github.io/LLMlight/pages/html/Algorithm.html#preprocessing-chunking) | Multiple embedding strategies: TF-IDF for structured documents, Bag of Words (BOW), BERT for free text, BGE-Small. |
| [**Advanced Retrieval Methods**](https://erdogant.github.io/LLMlight/pages/html/Algorithm.html#rag-with-statistical-validation) | Supports Naive RAG with fixed chunking and RSE (Relevant Segment Extraction). |
| [**Context Strategies**](https://erdogant.github.io/LLMlight/pages/html/Algorithm.html#rag-with-statistical-validation) | Advanced reasoning for complex queries using Global-reasoning and Chunk-wise approaches. |
| [**Local Memory**](https://erdogant.github.io/LLMlight/pages/html/Saving%20and%20Loading.html) | Video memory storage for efficient local use. |
| [**PDF Processing**](https://erdogant.github.io/LLMlight/pages/html/Saving%20and%20Loading.html) | Native support for reading and processing PDF documents. |

---

## Documentation & Resources

- [Documentation](https://erdogant.github.io/LLMlight)
- [Blog Posts](https://erdogant.github.io/LLMlight/pages/html/Documentation.html#medium-blog)
- [GitHub Issues](https://github.com/erdogant/LLMlight/issues)

## Quick Start

### Installation

```bash
# Install from PyPI
pip install LLMlight

# Install from GitHub
pip install git+https://github.com/erdogant/LLMlight
```

### Basic Usage with Endpoint

```python
from LLMlight import LLMlight

# Initialize with default settings
client = LLMlight(endpoint='http://localhost:1234/v1/chat/completions')

# Run a simple query
response = client.prompt('What is the capital of France?',
                         context='The capital of France is Amsterdam.',
                         instructions='Do not argue with the information in the context. Only return the information from the context.')
print(response)
# According to the provided context, the capital of France is Amsterdam.

```

## Examples


### 1. Check Available Models at Endpoint

```python
from LLMlight import LLMlight

# Initialize client
from LLMlight import LLMlight
# Initialize with LM Studio endpoint
client = LLMlight(model='mistralai/mistral-small-3.2',
                  endpoint="http://localhost:1234/v1/chat/completions")

modelnames = client.get_available_models(validate=False)
print(modelnames)

```

### 2. Basic Usage with Local GGUF

```python
from LLMlight import LLMlight

# Use with a local GGUF client
client = LLMlight(endpoint='path/to/your/client.gguf')

# Run a simple query
response = client.prompt('What is the capital of France?',
                         context='The capital of France is Amsterdam.',
                         instructions='Do not argue with the information in the context. Only return the information from the context.')
print(response)
# According to the provided context, the capital of France is Amsterdam.

```

### 3. Using with LM Studio

```python
# Import library
from LLMlight import LLMlight

# Initialize with LM Studio endpoint
client = LLMlight(model='mistralai/mistral-small-3.2',
                  endpoint="http://localhost:1234/v1/chat/completions")

# Run queries
response = client.prompt('Explain quantum computing in simple terms')

```

### 3. Query against PDF files

```python

# Load library
from LLMlight import LLMlight

# Initialize with default settings
client = LLMlight(model='mistralai/mistral-small-3.2',
                  context_strategy='chunk-wise',
                  retrieval_method='naive_rag',
                  embedding={'memory': 'memvid', 'context': 'bert'},
                  top_chunks=5)

# Read pdf
path = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
pdf_text = client.read_pdf(path, return_type='text')
context = pdf_text + '\n More text can be appended in this manner'

# Make a prompt
response = client.prompt('What is an attention network?',
                         context=context,
                         instructions='Answer the question using only the information from the context. If the answer can not be found, tell that.')

print(response)

```

### 4. Global Reasoning

```python
from LLMlight import LLMlight
client = LLMlight(model='microsoft/phi-4', context_strategy='global-reasoning')

path = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
pdf_text = client.read_pdf(path, return_type='text')

# Make a prompt
response = client.prompt('What is an attention network?',
                         context=context,
                         instructions='Answer the question using only the information from the context. If the answer can not be found, tell that.')
print(response)


```


### 5. Creating Local Memory Database

```python

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


# =============================================================================
# Load from database
# =============================================================================

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

```

### 6. Load Local Memory Database

```python

# Import library
from LLMlight import LLMlight

# Initialize with default settings
client = LLMlight(preprocessing=None, retrieval_method=None, path_to_memory="knowledge_base.mp4")

# Create queries
response = client.prompt('What do apes like?', instructions='Only return the information from the context. Answer with maximum of 3 words, and starts with "Apes like: "')
print(response)

```

### Contributors
Thank the contributors!

<p align="left">
  <a href="https://github.com/erdogant/llmlight/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=erdogant/llmlight" />
  </a>
</p>

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* Yes! This library is entirely **free** but it runs on coffee! :) Feel free to support with a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a>.

[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy+me+a+coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/erdogant)
