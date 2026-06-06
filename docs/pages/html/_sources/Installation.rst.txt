Installation
################

Create environment
**********************

If desired, install ``LLMlight`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_LLMlight python=3.12
    conda activate env_LLMlight


Pypi
**********************

.. code-block:: console

    # Install from Pypi:
    pip install LLMlight

    # Force update to latest version
    pip install -U LLMlight


Github source
************************************

.. code-block:: console

    # Install directly from github
    pip install git+https://github.com/erdogant/LLMlight


Optional Dependencies
************************************

LLMlight works out of the box with its core dependencies. The following optional packages unlock additional features:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``sentence-transformers``
     - Semantic (BERT-style) embeddings for memory and context retrieval
   * - ``hnswlib``
     - Fast approximate nearest-neighbour search (HNSW index); falls back to TF-IDF without it
   * - ``memvid``
     - QR-code video memory backend (alternative to SQLite)
   * - ``distfit``
     - Statistical significance testing of retrieval scores (``alpha`` parameter)
   * - ``llama-cpp-python``
     - Run local GGUF models without an API server

Install all optional extras at once:

.. code-block:: console

    pip install sentence-transformers hnswlib distfit


Uninstalling
################

Remove environment
**********************

.. code-block:: console

   # List all the active environments. LLMlight should be listed.
   conda env list

   # Remove the LLMlight environment
   conda env remove --name LLMlight

   # List all the active environments. LLMlight should be absent.
   conda env list


Remove installation
**********************

Note that the removal of the environment will also remove the ``LLMlight`` installation.

.. code-block:: console

    # Install from Pypi:
    pip uninstall LLMlight



.. include:: add_bottom.add
