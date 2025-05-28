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
- **Multiple Model Support**: Compatible with various models including:
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
- **PDF Processing**: Built-in support for reading and processing PDF documents
- **Global Reasoning**: Advanced reasoning capabilities for complex queries

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

### Basic Usage

```python
from LLMlight import LLMlight

# Initialize with default settings
model = LLMlight()

# Run a simple query
response = model.run('What is the capital of France?', 
                    system="You are a helpful assistant.")

# Use with a local GGUF model
model = LLMlight(endpoint='path/to/your/model.gguf')
response = model.run('Tell me about quantum computing')
```

## 📊 Examples

### 1. Using with LM Studio

```python
from LLMlight import LLMlight

# Initialize with LM Studio endpoint
model = LLMlight(endpoint="http://localhost:1234/v1/chat/completions")

# Run queries
response = model.run('Explain quantum computing in simple terms')
```

### 2. Validate Models

```python
from LLMlight import LLMlight

# Initialize model
from LLMlight import LLMlight
model = LLMlight(verbose='info')

modelnames = model.get_available_models(validate=True)
print(modelnames)

```

### 3. Processing PDF Documents and Ask Questions

```python
from LLMlight import LLMlight

# Initialize model
model = LLMlight()

# Read and process PDF
model.read_pdf('path/to/document.pdf')

# Query about the document
response = model.run('Summarize the main points of this document')

print(response)

```

### 4. Global Reasoning

```python
from LLMlight import LLMlight

# Initialize model
model = LLMlight()

# Read and process PDF
model.read_pdf('path/to/document.pdf')

# Query about the document
response = model.run('Summarize the main points of this document', global_reasoning=True)

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
