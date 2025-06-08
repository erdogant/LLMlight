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

<div align="center">
  <img src="https://github.com/erdogant/LLMlight/blob/master/docs/figs/logo.png" width="350" />
</div>

LLMlight is a Python package for running Large Language Models (LLMs) locally with minimal dependencies. It provides a simple interface to interact with various LLM models, including support for GGUF models and local API endpoints.

## 🌟 Key Features

- **Local LLM Support**: Run LLMs locally with minimal dependencies
- **Full Promp Control**:
  - Query
  - Instructions
  - System
  - Context
  - Response Format
  - Automatic formatting
  - Temperature
  - Top P
- **Single Endpoint will Connect All Local Models**: Compatible with various models including:
  - Hermes-3-Llama-3.2-3B
  - Mistral-7B-Grok
  - OpenHermes-2.5-Mistral-7B
  - Gemma-2-9B-IT
- **Flexible Embedding Methods**: Support for multiple embedding approaches:
  - TF-IDF for structured documents
  - Bag of Words (BOW)
  - BERT for free text
  - BGE-Small
- **Advanced Retrieval Methods**:
  - Naive RAG with fixed chunking
  - RSE (Relevant Segment Extraction)
- **Advanced Preprocessing Methods**: Advanced reasoning capabilities for complex queries.
  - Global-reasoning
  - chunk-wise
- **Local Memory**: 
  - Video memory for storage 
- **PDF Processing**: Built-in support for reading and processing PDF documents

## 📚 Documentation & Resources

- [Documentation](https://erdogant.github.io/LLMlight)
- [Blog Posts](https://erdogant.github.io/LLMlight/pages/html/Documentation.html#medium-blog)
- [GitHub Issues](https://github.com/erdogant/LLMlight/issues)

## 🚀 Quick Start

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

## 📊 Examples


### 1. Basic Usage with Local GGUF

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

### 2. Using with LM Studio

```python
from LLMlight import LLMlight

# Initialize with LM Studio endpoint
client = LLMlight(endpoint="http://localhost:1234/v1/chat/completions")

# Run queries
response = client.prompt('Explain quantum computing in simple terms')
```

### 3. Check Available Models at Endpoint

```python
from LLMlight import LLMlight

# Initialize client
from LLMlight import LLMlight
client = LLMlight(verbose='info')

modelnames = client.get_available_models(validate=False)
print(modelnames)

```

### 3. Query against PDF files

```python
from LLMlight import LLMlight

# Initialize client
client = LLMlight()

# Read PDF
context = client.read_pdf(r'path/to/document.pdf', return_type='string')

# Query the document
response = client.prompt('Summarize the main points of this document', 
                         context=context)

print(response)

```

### 4. Global Reasoning

```python
from LLMlight import LLMlight

# Initialize client
client = LLMlight(preprocessing='global_reasoning')

# Read PDF
context = client.read_pdf(r'path/to/document.pdf', return_type='string')

# Query about the document
response = client.prompt('Summarize the main points of this document', 
                         context=context,
                         instructions='Do not argue with the information in the context. Only return the information from the context.')

print(response)

```


### 5. Creating Local Memory Database

```python

# Import library
from LLMlight import LLMlight

# Initialize with default settings
client = LLMlight(preprocessing=None, retrieval_method=None)

# Load existing video memory
client.memory_init(path_to_memory="knowledge_base.mp4")

# Append more documents: PDF/txt/etc files
filepaths = [r'c://path_to_your_files//article_1.pdf', r'c://path_to_your_files//my_file.txt']
client.memory_add(input_files=filepaths)

# Add text chunks if you like
client.memory_add(text=['Apes like USB sticks', 'Trees are mainly yellow'])

# Save Memory to disk. You can either create new one or overwite existing one.
client.memory_save(filepath="knowledge_base_with_more_data.mp4", overwrite=False)

# Run a simple query
response = client.prompt('What do apes like?', instructions='Only return the information from the context. Answer with maximum of 3 words, and starts with "Apes like: "')
print(response)

response = client.prompt('What is the capital of France?', context='The capital of France is Amsterdam.', instructions='Do not argue with the information in the context. Only return the information from the context.')
print(response)

response = client.prompt('Provide a summary of HyperSpectral from the pdf or text file.', instructions='Do not argue with the information in the context. Only return the information from the context.')
print(response)

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


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Contributors

<div align="center">
  <img src="https://contrib.rocks/image?repo=erdogant/LLMlight" />
</div>

## 👨‍💻 Maintainer

- **Erdogan Taskesen** ([@erdogant](https://github.com/erdogant))

## ☕ Support

This library is free and open source. If you find it useful, consider supporting its development:

<a href="https://www.buymeacoffee.com/erdogant"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" /></a>

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/erdogant/LLMlight/blob/master/LICENSE) file for details.
