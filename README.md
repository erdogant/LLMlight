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
  <a href="https://erdogant.github.io/LLMlight/">
    <img src="https://raw.githubusercontent.com/erdogant/llmlight/master/docs/figs/schematic_overview.png" width="600" alt="Schematic Overview" />
  </a>
</p>


### Key Features

| Feature | Description |
|---------|-------------|
| [**Local LLM Support**](https://erdogant.github.io/LLMlight/pages/html/Algorithm.html#get-available-llm-models) | Run LLMs locally with minimal dependencies. |
| [**Full Prompt Control**]() | Fine-grained control over prompts including Query, Instructions, System, Context, Response Format, Automatic formatting, Temperature, and Top P. |
| [**Single Endpoint for All Local Models**](https://erdogant.github.io/LLMlight/pages/html/Algorithm.html#get-available-llm-models) | One unified endpoint to connect different local models. |
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


### Installation

```bash
# Install from PyPI
pip install LLMlight

```

## Quick Start


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

### 2. Basic Usage with Endpoint

```python
from LLMlight import LLMlight

# Initialize with default settings
client = LLMlight(model='openai/gpt-oss-20b', endpoint='http://localhost:1234/v1/chat/completions')

# Run a simple query
response = client.prompt('What is the capital of France?',
                         context='The capital of France is Amsterdam.',
                         instructions='Do not argue with the information in the context. Only return the information from the context.')
print(response)
# According to the provided context, the capital of France is Amsterdam.

```

### 3. Using with LM Studio

https://erdogant.github.io/LLMlight/pages/html/Algorithm.html#get-available-llm-models

### 3. Query against PDF files
https://erdogant.github.io/LLMlight/pages/html/Examples.html#working-with-files-pdfs

### 4. Creating Local Memory Database

https://erdogant.github.io/LLMlight/pages/html/Saving%20and%20Loading.html#memory-management


### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* Yes! This library is entirely **free** but it runs on coffee! :) Feel free to support with a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a>.

[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy+me+a+coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/erdogant)
