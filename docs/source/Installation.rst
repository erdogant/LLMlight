Installation
################

Create environment
**********************

If desired, install ``LLMlight`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_LLMlight python=3.8
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

