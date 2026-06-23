"""LLMlight.

Name        : LLMlight.py
Author      : E.Taskesen
Contact     : erdogant@gmail.com
github      : https://github.com/erdogant/LLMlight
Licence     : See licences

"""

import requests
import logging
import os
import numpy as np
# llama-cpp-python is an optional native dependency. Import lazily and give a helpful error if missing.
try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None
import copy
import re
from tqdm import tqdm
import tempfile

from typing import List, Union

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# sentence_transformers and distfit are optional heavy dependencies; import them only when needed.
# memvid is an optional memory backend; import within memory module when used.

try:
    from . import RAG
    from . import utils
    from . import memory
except Exception as e:
    import memory
    import RAG
    import utils

# Set external loggers to ERROR
logger = logging.getLogger(__name__)
loggers = ["httpx", "httpcore", "huggingface_hub", "transformers", "sentence_transformers"]
for name in loggers:
    logging.getLogger(name).setLevel(logging.ERROR)

# %%
class LLMlight:
    """Large Language Model Light.

    Run your LLM models local and with minimum dependencies.
    1. Go to LM-studio.
    2. Go to left panel and select developers mode.
    3. On top select your model of interest.
    4. Then go to settings in the top bar.
    5. Enable "server on local network" (only use when needed).
    6. Enable Running.

    How LLMlight Works
    -------------------
    LLMlight processes text through several key stages to generate intelligent responses:

    1. Context strategy
    ---------------------
    The input context can be processed in different ways:
    - No Context strategy: Uses the raw context directly
    - Chunk-wise processing: Breaks down the context into manageable chunks, processes each chunk independently, and combines results
    - Global reasoning: Creates a global summary of the context before processing

    2. Retrieval Method Stage
    --------------------------
    Three main approaches for retrieving relevant information:
    - Naive RAG: Splits text into chunks and uses similarity scoring to find the most relevant sections
    - RSE (Relevant Segment Extraction): Identifies and extracts complete relevant text segments
    - No retrieval: Uses the entire context directly

    3. Embedding Stage
    -------------------
    Multiple embedding options for text representation:
    - TF-IDF: Best for structured documents with matching query terms
    - Bag of Words: Simple word frequency approach
    - BERT: Advanced contextual embeddings for free-form text
    - BGE-small: Efficient embedding model for general use

    4. Prompting Stage
    -------------------
    The system constructs prompts by combining:
    - System message: Defines the AI's role and behavior
    - Context: Processed and retrieved relevant information
    - User query: The specific question or request
    - Instructions: Additional guidance for response generation

    5. Response Generation
    -----------------------
    The system can be configured through various parameters to optimize for different use cases, from simple Q&A to complex document analysis.
    The model generates responses using:
    - Temperature control: Adjusts response randomness (0.8 default)
    - Top-p sampling: Controls response diversity
    - Context window management: Handles token limits efficiently

    Processing Flow
    ----------------
    The system follows a sequential processing flow where each stage builds upon the previous one. First, the input context undergoes the context strategy, where it can be either used as-is or transformed into chunks for more manageable processing. These chunks are then passed through the retrieval method stage, which determines how relevant information is extracted and organized.
    During the embedding stage, the text is converted into numerical representations that capture its semantic meaning. This is crucial for the system to understand and process the content effectively. The embedding method chosen can significantly impact the system's ability to match queries with relevant content.
    The prompting stage brings together all the processed information, combining it with the user's query and any specific instructions. This creates a comprehensive prompt that guides the model in generating an appropriate response. The final response generation stage uses this prompt to create a coherent and relevant output, with parameters like temperature and top-p sampling helping to control the response's characteristics.
    Throughout this process, the system maintains flexibility through various configuration options, allowing it to adapt to different types of queries and contexts. This modular approach enables the system to handle everything from simple questions to complex document analysis tasks efficiently.

    Parameters
    ----------
    model : str
        Model identifier served by the endpoint, e.g. ``'mistralai/mistral-small-3.2'``, ``'unsloth/gemma-4-26b-a4b-it'``, ``'qwen/qwen3-coder-30b'``.
        When ``None`` the available models are listed and ``__init__`` returns early.
    retrieval_method : str, default ``'naive_rag'``
        None          -- No chunking. The entire context is forwarded to the LLM. Use only when context fits within ``n_ctx``.
        'naive_rag'   -- Context is split into chunks; top-k chunks are selected by cosine similarity and combined into the prompt.
        'RSE'         -- Relevant Segment Extraction: contiguous high-scoring segments are identified and reconstructed. Requires ``embedding`` in ``('bert', 'bge-small')``.
    embedding : str, dict, or None, default ``None``
        Controls how text is vectorised for retrieval.

        ``None``          -- Retrieval embedding **disabled**. Memory retrieval and context retrieval both skip the similarity step and fall through to the full-context path.
        ``'automatic'``   -- Shorthand for ``{'memory': 'memvid', 'context': 'bert'}``.
        A string          -- Applies the same method to both memory and context paths.
                            Valid values: ``'tfidf'``, ``'bow'``, ``'bert'``, ``'bge-small'``, ``'memvid'``.
                            Note: ``'memvid'`` is only valid for the memory path; it is silently corrected to ``'bert'`` when used as the context embedding.
        A dict            -- Specify paths independently: ``{'memory': 'memvid', 'context': 'bert'}``.
                            Keys ``'memory'`` and ``'context'`` are both optional; omitted keys inherit the ``'automatic'`` defaults.

        Guidance by use-case:
          ``'tfidf'``    -- Structured text, query terms likely present verbatim.
          ``'bow'``      -- Simple word-frequency matching.
          ``'bert'``     -- Free-text; query wording may not match the document.
          ``'bge-small'``-- Compact retrieval-optimised model; good general default.
          ``'memvid'``   -- Use the memvid backend's built-in FAISS similarity.
    context_strategy : str or None, default ``None``
        None               -- Raw (or retrieved) context is passed directly to the LLM.
        'chunk-wise'       -- Each chunk is analysed independently against the query; per-chunk answers are combined in a final pass.
        'global-reasoning' -- Each chunk is summarised; summaries are merged into one coherent response.
    temperature : float, default ``0.8``
        Sampling temperature in ``[0, 2]``. ``0`` is deterministic; higher values increase randomness.
    top_p : float, default ``1.0``
        Nucleus sampling threshold in ``(0, 1]``. ``1.0`` disables filtering.
    chunks : dict or None, default ``None``
        Chunking configuration. Missing keys fall back to defaults.
        Keys:

        ``'method'``  -- ``'chars'`` (default) or ``'words'``.
        ``'size'``    -- Chunk length in characters or words (default ``1000``). Smaller sizes improve retrieval precision but reduce the context available to the LLM. Rough estimate: 1 000 words ≈ 3 000 tokens.
        ``'overlap'`` -- Overlap between consecutive chunks (default ``200``). Must be less than ``'size'``.

        Legacy aliases accepted: ``'type'`` -> ``'method'``,
        ``'chunk_size'`` -> ``'size'``.
    top_chunks : int, default ``5``
        Number of top-ranked chunks returned by retrieval.
    n_ctx : int, default ``None``
        Model context window in tokens. This is automatically derived from the model settings when set to None.
    file_path : str or None, default ``None``
        Path to an existing memory store to load at construction time.
        Relative paths are resolved under the LLMlight temp directory.
        The extension determines the backend: ``.db`` -> sqlite, ``.mp4`` -> memvid.
    endpoint : str, default ``'http://localhost:1234/v1/chat/completions'``
        URL of the OpenAI-compatible chat completions endpoint, or an absolute
        path to a local ``.gguf`` model file.
    verbose : str or int, default ``'info'``
        Logging verbosity. Accepts level names (``'debug'``, ``'info'``,
        ``'warning'``, ``'error'``, ``'silent'``) or integer log levels.

    Examples
    --------
    >>> from LLMlight import LLMlight
    >>> # Simple prompt -- no context
    >>> client = LLMlight(model='mistralai/mistral-small-3.2')
    >>> response = client.prompt('What is the capital of France?')
    >>> print(response)

    >>> # RAG over a PDF with bert embeddings
    >>> client = LLMlight(model='mistralai/mistral-small-3.2', retrieval_method='naive_rag', embedding='bert', top_chunks=5)
    >>> context = client.read_pdf('https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf')
    >>> response = client.prompt('Summarize the main contributions.', context=context)
    >>> print(response)

    >>> # Persistent SQLite memory store
    >>> client = LLMlight(model='mistralai/mistral-small-3.2', embedding='automatic')
    >>> client.memory_init(store_path='knowledge.db')
    >>> client.memory_add(text=['Apes like USB sticks.', 'The capital of France is Amsterdam.'])
    >>> client.memory_save()
    >>> response = client.prompt('What do apes like?')
    >>> print(response)

    >>> # Disable embedding entirely -- full context passed to LLM
    >>> client = LLMlight(model='mistralai/mistral-small-3.2', embedding=None, retrieval_method=None)
    >>> response = client.prompt('Summarise this.', context='Short enough to fit in context window.')
    >>> print(response)

    """
    def __init__(self,
                 model: str = None,
                 retrieval_method: str = 'naive_rag',
                 embedding='bert',
                 context_strategy: str = None,
                 alpha: float = None,
                 top_chunks: int = 5,
                 temperature: float = 0.8,
                 top_p: float = 1.0,
                 chunks: dict = None,
                 n_ctx: int = None,
                 file_path: str = None,
                 endpoint: str = "http://localhost:1234/v1/chat/completions",
                 timeout = 600,
                 verbose: (str, int) = 'info',
                 ):

        # Validate and normalise all parameters before storing anything
        params = _validate_params(
            model=model,
            retrieval_method=retrieval_method,
            embedding=embedding,
            context_strategy=context_strategy,
            alpha=alpha,
            top_chunks=top_chunks,
            temperature=temperature,
            top_p=top_p,
            chunks=chunks,
        )

        # Store validated/normalised values -- all set unconditionally so every
        # attribute exists even when model=None causes an early return below.
        self.model            = params['model']
        self.retrieval_method = params['retrieval_method']
        self.embedding        = params['embedding']
        self.context_strategy = params['context_strategy']
        self.alpha            = params['alpha']
        self.top_chunks       = params['top_chunks']
        self.temperature      = params['temperature']
        self.top_p            = params['top_p']
        self.chunks           = params['chunks']
        self.timeout          = timeout
        self.n_ctx            = n_ctx
        self.endpoint         = endpoint
        self.context          = None
        self.tempdir          = os.path.join(tempfile.gettempdir(), 'temp_LLMlight')
        self.store_path       = self._resolve_file_path(file_path)
        self.modelinfo        = {}

        # Get the max context length
        if self.model is not None and (n_ctx is None or n_ctx < 512):
            # Store modelinfo
            self.modelinfo = self.get_model_info(model=model)
            # self.n_ctx = modelinfo.get('max_context_length') or 8192
            loaded = self.modelinfo.get("loaded_instances") or []
            self.n_ctx = (loaded[0].get("config", {}).get("context_length", 16384) if loaded else 16384)

        # When no model is given, report available models and return early.
        # All attributes above are already set so tests can inspect them.
        if self.model is None:
            models = self.get_available_models(validate=False)
            if models is not None:
                logger.info(f'Available models: {models}')
                logger.info(
                    f'Set model before proceeding: '
                    f'Example: client = LLMlight(model="{models[0]}", endpoint="{endpoint}").'
                )
                self.models = models
            return

        # Create tempdir
        os.makedirs(self.tempdir, exist_ok=True)

        # Load memory from disk when a store path was provided
        if self.store_path:
            self.memory_load(self.store_path)

        # Load a local GGUF model when the endpoint points to a file
        if os.path.isfile(self.endpoint):
            self.llm = load_local_gguf_model(self.endpoint, n_ctx=self.n_ctx)

        logger.info(f'Model            : {self.model}')
        logger.info(f'Context strategy : {self.context_strategy or "disabled"}')
        logger.info(f'Retrieval method : {self.retrieval_method or "disabled"}')
        logger.info(f'Embedding        : {self.embedding}')
        logger.info(f'Alpha (sig. test): {self.alpha}')
        logger.info(f'Chunk config     : {self.chunks}')
        logger.info(f'Contex window    : {self.n_ctx}')
        logger.info('LLMlight initialised.')

    def _resolve_file_path(self, filepath: str):
        """Return an absolute path for *filepath*, or None when not given.

        Resolution rules (applied in order):
        1. None or empty string  -> returns None.
        2. Already absolute      -> returned unchanged.
        3. Relative / bare name  -> resolved relative to self.tempdir.
        """
        if not filepath:
            return None
        if os.path.isabs(filepath):
            return filepath
        return os.path.join(self.tempdir, filepath)

    def get_full_path(self, filepath: str):
        """Alias kept for backwards compatibility.  Use _resolve_file_path()."""
        return self._resolve_file_path(filepath)

    def prompt(self,
               query: str,
               instructions: str = None,
               system: str = None,
               context: str = None,
               response_format=None,
               temperature: (int, float) = None,
               top_p: (int, float) = None,
               stream: bool = False,
               return_type: str = 'string',
               thinking: bool = True,
               ):
        """Run the model and return its response.

        Orchestrates the full pipeline: memory retrieval -> context retrieval ->
        context strategy -> prompt assembly -> LLM call.

        Parameters
        ----------
        query : str
            The question or task, e.g. ``'What is the capital of France?'``.
        context : str, optional
            Raw text to search for relevant information.  When ``retrieval_method``
            is set the text is chunked and the top-k chunks are selected; when
            ``retrieval_method=None`` the full string is forwarded to the LLM
            (a warning is emitted when it likely exceeds ``n_ctx``).
            Defaults to ``self.context`` when ``None``.
        instructions : str, optional
            Task-specific instructions appended to the prompt, e.g.
            ``'Answer using only information from the context.'``.
        system : str, optional
            System message that sets the LLM's role.  When ``None`` a sensible
            default is used.
        response_format : str, optional
            Desired output format hint, e.g. ``'Return a JSON object.'``.
        temperature : float, optional
            Sampling temperature. Overrides the instance default for this call.
        top_p : float, optional
            Nucleus sampling threshold. Overrides the instance default for this call.
        stream : bool, default ``False``
            Whether to stream the response.
        return_type : str, default ``'string'``
            How to post-process the raw LLM output:

            ``'string'``              -- Plain text; ``<think>...</think>`` blocks removed.
            ``'string_with_thinking'``-- Plain text including any thinking blocks.
            ``'dict'``                -- Parse response as JSON and return a dict.
            ``'raw'``                 -- Return the full raw API response object.
        verbose : str or int, optional
            Override logging verbosity for this call only.
        thinking : bool, default ``True``
            Whether the model is allowed to "think" (emit reasoning, e.g.
            ``<think>...</think>`` blocks) before producing its final answer.
            When ``False``, LLMlight asks the model/backend to skip its
            reasoning step: a ``chat_template_kwargs={'enable_thinking': False}``
            hint is sent to backends that support it (e.g. vLLM/LM Studio with
            Qwen3-style models), and ``/no_think`` is appended to the system
            message as a fallback for backends that rely on that convention.
            Any ``<think>...</think>`` content still present in the raw output
            is removed when ``return_type='string'``, regardless of this flag.

        Returns
        -------
        str or dict
            The model's response, post-processed according to ``return_type``.
            Returns an error string prefixed with ``'Error:'`` on HTTP failure.

        Examples
        --------
        >>> client = LLMlight(model='mistralai/mistral-small-3.2')
        >>> response = client.prompt('What is the capital of France?')
        >>> print(response)

        >>> # With context and instructions
        >>> response = client.prompt('What do apes like?', context='Apes like USB sticks.', instructions='Answer in one sentence using only the context.')
        >>> print(response)

        >>> # Override temperature for a single call
        >>> response = client.prompt('Write a short poem.', temperature=1.0)
        >>> print(response)
        """
        logger.info(f'Creating response with {self.model}..')

        if context is None: context = self.context
        if temperature is None: temperature = self.temperature
        if top_p is None: top_p = self.top_p
        self.task = 'max'

        # Set system message
        system = set_system_message(system)

        # Toggle "thinking" mode. Some backends (e.g. vLLM/LM Studio serving
        # Qwen3-style models) support this via a chat_template_kwargs hint,
        # passed through in requests_post_http/requests_post_gguf. As a
        # universally-compatible fallback we also append the '/no_think'
        # convention to the system message when thinking is disabled.
        if not thinking:
            system = f"{system}\n/no_think"

        # Extract relevant text for video memory
        relevant_memory = self.relevant_memory_retrieval(query, return_type='list')

        # Extract relevant text using retrieval method
        relevant_context = self.relevant_context_retrieval(query, context, return_type='list')

        # Append the relevant chunks of texts
        if isinstance(relevant_context, str): relevant_context = [relevant_context]
        total_context = (relevant_memory or []) + (relevant_context or [])

        # Context Strategu on the context
        processed_context = self.compute_context_strategy(query, total_context, instructions, system)

        # Set the prompt
        logger.debug(processed_context)

        # Make the prompt
        prompt = self.set_prompt(query, instructions, processed_context, response_format=response_format)
        logger.info(f'Running model: {self.model} ')

        # Run model
        if os.path.isfile(self.endpoint):
            # Run LLM from gguf model
            response = self.requests_post_gguf(prompt, system, temperature=temperature, top_p=top_p, task=self.task, stream=stream, return_type=return_type, thinking=thinking)
        else:
            # Run LLM with http model
            response = self.requests_post_http(prompt, system, temperature=temperature, top_p=top_p, task=self.task, stream=stream, return_type=return_type, thinking=thinking)

        # Return
        return response

    def requests_post_gguf(self, prompt, system, temperature=0.8, top_p=0.95, headers=None, task='max', stream=False, return_type='string', thinking=True):
        # Note that it is better to use messages_prompt instead of a dict (messages_dict) because most GGUF-based models don't have a tokenizer/parser that can interpret the JSON-style message structure.
        # Prepare data for request.
        if headers is None: headers = {"Content-Type": "application/json"}
        # Prepare messages
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        # Convert messages to string prompt
        prompt = convert_messages_to_model(messages, model=self.model)
        # Compute tokens
        used_tokens, max_tokens = compute_tokens(prompt, n_ctx=self.n_ctx, task=task)

        # Send post request to local GGUF model
        response = self.llm(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            max_tokens=max_tokens,
            stop=["<end_of_turn>", "<|im_end|>"]  # common stop tokens for chat formats
        )

        # Take only the output
        if 'string' in return_type:
            response = response.get('choices', [{}])[0].get('text', "No response")
        if return_type == 'string':
            # Remove thinking
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Return
        return response

    def requests_post_http(
        self,
        prompt,
        system,
        temperature=0.8,
        top_p=1,
        headers=None,
        task='max',
        stream=False,
        return_type='string',
        timeout=480,
        thinking=True):

        # Prepare headers
        if headers is None:
            headers = {"Content-Type": "application/json"}
    
        # Prepare messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    
        # Convert messages to string prompt for local token estimation
        prompt_string = convert_messages_to_model(messages, model=self.model)
    
        # Estimate prompt tokens and max generation budget
        estimated_prompt_tokens, max_tokens = compute_tokens(prompt_string, n_ctx=self.n_ctx, task=task)
    
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_tokens": max_tokens,
            # Hint for backends (e.g. vLLM / LM Studio serving Qwen3-style
            # models) that support toggling "thinking" via the chat template.
            # Backends that don't recognize this field simply ignore it.
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
    
        # Send POST request
        response = self.requests_post(headers, data, stream=stream, return_type=return_type)
    
        # Return response
        return response

    # def requests_post_http(self, prompt, system, temperature=0.8, top_p=1, headers=None, task='max', stream=False, return_type='string'):
    #     # Prepare data for request.
    #     if headers is None: headers = {"Content-Type": "application/json"}
    #     # Prepare messages
    #     messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    #     # Convert messages to string prompt
    #     prompt = convert_messages_to_model(messages, model=self.model)

    #     # Compute tokens
    #     if max_tokens is None:
    #         used_tokens, max_tokens = compute_tokens(prompt, n_ctx=self.n_ctx, task=task)
    #     # logger.info(f'Generating response with {self.model}')

    #     data = {
    #         "model": self.model,
    #         "messages": messages,
    #         "temperature": temperature,
    #         "top_p": top_p,
    #         "stream": stream,
    #         "max_tokens": max_tokens,
    #         }

    #     # Send POST request
    #     response = self.requests_post(headers, data, stream=stream, return_type=return_type)

    #     # Return
    #     return response

    def requests_post(self, headers, data, stream=False, return_type='string'):
        """Create the request to the LLM."""
        # Get response
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=self.timeout, stream=stream)

        # Handle the response
        if response.status_code == 200:
            try:
                # Create dictionary in case json
                response_text = response.json().get('choices', [{}])[0].get('message', {}).get('content', "No response")

                if return_type == 'dict':
                    response_text = utils.is_valid_json(response_text)
                    return response_text
                elif return_type == 'string_with_thinking':
                    return response_text
                elif return_type == 'string':
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                    return response_text
                else:
                    return response.json()
            except Exception as e:
                logger.debug(f"Failed to parse JSON response: {e}")
                return response
        else:
            logger.error(f"{response.status_code} - {response}")
            return f"Error: {response.status_code} - {response}"

    def memory_init(self, store_path: str = None, config: dict = None, embedding: str = None, backend: str = None, overwrite: bool = False):
        """Prepare a memory store for writing.

        Creates a new store or re-opens an existing one so that chunks can be
        added via :meth:`memory_add`.  Call :meth:`memory_save` afterwards to
        persist the buffered chunks (sqlite backend writes immediately; memvid
        requires an explicit save to rebuild the video).

        If the same resolved path is already initialised this call is a no-op.

        Parameters
        ----------
        store_path : str, optional
            Path to the store file.  Relative paths are resolved under the
            LLMlight temp directory.  Defaults to the path set at construction.
            The extension is normalised: ``.db`` -> sqlite, ``.mp4`` -> memvid.
        config : dict, optional
            Backend-specific configuration dict passed through to the backend.
        embedding : str, optional
            Override ``self.embedding['memory']`` for this store (e.g. ``'bert'``).
        backend : str, optional
            ``'sqlite'`` (default) or ``'memvid'``.  When ``None`` the factory
            infers the backend from the file extension.
        """
        resolved = self._resolve_file_path(store_path) or self.store_path
        if os.path.isfile(resolved) and not overwrite:
            logger.warning('sqlite database already exists and is loaded.')
        elif os.path.isfile(resolved) and overwrite:
            logger.warning('sqlite database is overwritten with a new empty database.')
        
        if overwrite and os.path.isfile(resolved):
            # Pass the already-open backend (if any) so close_and_remove can
            # flush WAL and release the file lock without opening a second
            # connection to the same file.
            existing_backend = getattr(self, 'memory', None)
            memory.close_and_remove(resolved, backend_instance=existing_backend)
            if hasattr(self, 'memory'):
                del self.memory

        # Skip if the same store is already initialised
        if hasattr(self, 'memory') and self.memory.store_path == resolved:
            logger.info(f'Memory already initialised: {resolved}')
            return

        self.memory = memory.create_memory_backend(resolved, config=config, backend=backend)
        self.store_path = self.memory.store_path

        if embedding is not None:
            self.embedding['memory'] = embedding
            logger.info(f'Memory embedding updated: {self.embedding}')

    def memory_close(self):
        """Close the memory backend and release the file lock.

        Calls ``close()`` on the active backend (flushing any WAL journal and
        releasing the SQLite file lock), then removes ``self.memory`` so the
        instance is in a clean state.  Safe to call even when no store is open.

        This is the recommended way to free the lock before deleting or
        overwriting the store file — especially on Windows where an open
        connection prevents deletion.

        Examples
        --------
        >>> client.memory_close()          # release lock
        >>> os.remove('knowledge.db')      # now safe to delete

        >>> # Overwrite the database entirely
        >>> client.memory_close()
        >>> client.memory_init('knowledge.db', overwrite=True)
        >>> client.memory_add(text=['fresh data'])
        >>> client.memory_save()
        """
        if not hasattr(self, 'memory'):
            logger.info('memory_close(): no memory store is open.')
            return

        path = getattr(self.memory, 'store_path', None)
        try:
            self.memory.close()
            logger.info(f'Memory store closed: {path}')
        except Exception as exc:
            logger.warning('memory_close(): close() raised %s — continuing.', exc)
        finally:
            del self.memory

    def memory_load(self, store_path: str = None, config: dict = None, backend: str = None):
        """Load an existing memory store from disk so it is ready for querying.

        When ``store_path`` differs from the currently loaded store the old
        backend is replaced and the new store is opened.  This means calling
        ``memory_load('other.db')`` after the store was previously initialised
        correctly switches to the new store rather than silently keeping the old one.

        Parameters
        ----------
        store_path : str, optional
            Path to the store file.  Defaults to the path set at construction.
            Relative paths are resolved under the LLMlight temp directory.
        config : dict, optional
            Backend-specific configuration.  Reuses the existing config when
            ``None`` and a backend is already loaded.
        backend : str, optional
            Force a specific backend.  Usually not needed -- the factory infers
            the backend from the file extension (``.db`` -> sqlite, ``.mp4`` -> memvid).
        """
        resolved = self._resolve_file_path(store_path) or self.store_path

        # Reuse existing config if none supplied
        if config is None and hasattr(self, 'memory') and hasattr(self.memory, 'config'):
            config = self.memory.config

        # Create a new backend when none exists OR when a different store is requested.
        current_path = getattr(getattr(self, 'memory', None), 'store_path', None)
        if not hasattr(self, 'memory') or (resolved and current_path != resolved):
            if hasattr(self, 'memory') and resolved and current_path != resolved:
                logger.info(
                    "memory_load: switching store from '%s' to '%s'.",
                    current_path, resolved,
                )
            self.memory = memory.create_memory_backend(resolved, config=config, backend=backend)
            self.store_path = self.memory.store_path

        self.memory.load()

    def memory_save(self,
                    store_path: str = None,
                    codec: str = 'mp4v',
                    auto_build_docker: bool = False,
                    allow_fallback: bool = True,
                    overwrite: bool = True,
                    show_progress: bool = True):
        """Persist buffered chunks to the store and reload the retriever.

        Parameters
        ----------
        store_path : str, optional
            Override the destination path.  Defaults to the current store path.
        codec : str, optional
            Video codec for the memvid backend (ignored by sqlite backend).
        auto_build_docker : bool
            Passed through to the memvid backend only.
        allow_fallback : bool
            Allow the memvid backend to fall back to mp4v on codec failure.
        overwrite : bool
            Overwrite an existing store.  Default True.
        show_progress : bool
            Show a progress bar during encoding.
        """
        if not hasattr(self, 'memory'):
            raise RuntimeError('No memory store initialised. Call memory_init() first.')

        if store_path is not None:
            self.store_path = self._resolve_file_path(store_path)

        self.memory.save(
            self.store_path,
            codec=codec,
            auto_build_docker=auto_build_docker,
            allow_fallback=allow_fallback,
            overwrite=overwrite,
            show_progress=show_progress,
        )
        self.memory.load()
        logger.info(f'Memory is saved to local database: {self.store_path}')

    def memory_remove(self,
                      ids: list = None,
                      query: str = None,
                      top_k: int = 1) -> list:
        """Remove chunks from the memory store by id or search query.

        Examples
        --------
        # Find what is stored first
        >>> results = client.memory.search('BMC')
        >>> # [(31, 0.23, {'text': 'BMC test', 'id': 31})]

        # Remove by id
        >>> client.memory_remove(ids=31)

        # Or remove by query (removes the single best match by default)
        >>> client.memory_remove(query='BMC test')

        # Remove the top-3 matches for a query
        >>> client.memory_remove(query='BMC', top_k=3)

        Parameters
        ----------
        ids : int or list of int, optional
            Row id(s) to delete, as returned in the first element of each
            search result tuple.
        query : str, optional
            Search term -- the top-*top_k* matching chunks are removed.
            Ignored when *ids* is provided.
        top_k : int
            How many top query matches to remove (default 1).

        Returns
        -------
        list of int
            The ids that were actually deleted.

        Notes
        -----
        For the **sqlite** backend the change is written to disk immediately.
        For the **memvid** backend the removal is staged in memory -- call
        :meth:`memory_save` to rebuild and persist the video without the
        removed chunks.
        """
        if not hasattr(self, 'memory'):
            raise RuntimeError("No memory store initialised. Call memory_init() first.")
        removed = self.memory.remove(ids=ids, query=query, top_k=top_k)
        if removed:
            logger.info("Removed %d chunk(s): ids=%s", len(removed), removed)
        return removed

    def memory_reindex(self, batch_size: int = 128, save_index: bool = True):
        """Rebuild the retrieval index for the current backend.

        Computes embeddings for all stored chunks and rebuilds the ANN index.
        Requires the ``sentence-transformers`` and ``hnswlib`` packages.

        Parameters
        ----------
        batch_size : int
            Number of texts to encode per batch.
        save_index : bool
            Persist the rebuilt index to disk after completion.

        Returns
        -------
        bool
            True on success.
        """
        if not hasattr(self, 'memory'):
            logger.info('No memory store loaded -- initialising from current store_path.')
            self.memory_init(self.store_path)

        if hasattr(self.memory, 'reindex') and callable(self.memory.reindex):
            logger.info('Rebuilding retrieval index...')
            try:
                result = self.memory.reindex(batch_size=batch_size, save_index=save_index)
                logger.info('Index rebuild complete.')
                return result
            except Exception as exc:
                logger.error(f'Index rebuild failed: {exc}')
                raise
        else:
            raise NotImplementedError(
                'The current memory backend does not support reindex(). '
                "Switch to backend='sqlite' or implement reindex() in your backend."
            )

    def memory_add(self,
                   text: Union[str, List[str]] = None,
                   files: Union[str, List[str]] = None,
                   dirpath: str = None,
                   filetypes: List[str] = None,
                   chunk_size: int = 512,
                   chunk_overlap: int = 100,
                   overwrite: bool = True):
        """Add text chunks or files to the memory store buffer.

        Parameters
        ----------
        text : str or list of str, optional
            Raw text strings to add directly.
        files : str or list of str, optional
            File paths or HTTP URLs to ingest.
        dirpath : str, optional
            Directory to scan recursively for supported file types.
        filetypes : list of str, optional
            File extensions to include when scanning *dirpath*.
            Defaults to a standard set (pdf, txt, epub, md, doc, docx, ...).
        chunk_size : int
            Characters (or words) per chunk.
        chunk_overlap : int
            Overlap between consecutive chunks.
        overwrite : bool
            When False, skip adding if a store file already exists on disk.
        """
        if not hasattr(self, 'memory'):
            raise RuntimeError('No memory store initialised. Call memory_init() first.')

        if filetypes is None:
            filetypes = ['.pdf', '.txt', '.epub', '.md', '.doc', '.docx',
                         '.rtf', '.html', '.htm']

        # Normalise text: read_pdf() can return a dict or a plain string.
        # Convert dict to a flat list of its non-empty string values so the
        # backend never receives a raw dict (which would be iterated as keys).
        if isinstance(text, dict):
            text = [str(v) for v in text.values() if v and str(v).strip()]
        elif isinstance(text, str):
            text = [text] if text.strip() else None

        self.memory.add(
            text=text,
            input_files=files,
            dirpath=dirpath,
            filetypes=filetypes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overwrite=overwrite,
            tempdir=self.tempdir,
        )


    def memory_chunks(self, n: int = None) -> list:
        """Return up to *n* stored chunks from the memory store.

        Parameters
        ----------
        n : int
            Maximum number of chunks to return.

        Returns
        -------
        list of str
        """
        if not hasattr(self, 'memory'):
            logger.warning('No memory store loaded. Call memory_init() or memory_load() first.')
            return []

        # Warn if there are unsaved chunks in the buffer (memvid backend)
        if (hasattr(self.memory, 'encoder')
                and hasattr(self.memory.encoder, 'chunks')
                and len(self.memory.encoder.chunks) > 0):
            logger.warning(
                f'{len(self.memory.encoder.chunks)} buffered chunk(s) not yet saved. '
                'Call memory_save() to persist them.'
            )


        chunks = self.memory.get_all_chunks()
        if n is None:
            return chunks

        logger.info(f'Returning {min(n, len(chunks))} of {len(chunks)} stored chunks.')
        return chunks[:n]

    def compute_probability(self, query, scores, embedding, n=5000):
        """Fit a null distribution over retrieval scores and return significance flags.

        Uses the *distfit* package.  Returns None when the test cannot run
        reliably (no memory, too few scores, degenerate distribution, or any
        distfit internal error -- including the IndexError caused by an empty
        histogram when all scores are identical).
        """
        if not hasattr(self, 'memory'):
            logger.debug('No memory store loaded -- skipping significance test.')
            return None

        scores = np.asarray(scores, dtype=float)
        if len(scores) < 2:
            logger.debug('Too few scores for significance test (need >= 2).')
            return None

        logger.info(f'Building null distribution with {n} samples for similarity-score significance testing.')

        if self.embedding['memory'] == 'memvid':
            scored = self.memory.search_with_scores(query, top_k=n)
            random_scores = np.array([s for s, _ in scored], dtype=float)
            bound = 'left'
        else:
            random_chunks = self.memory.get_random_chunks(n=n)
            if not random_chunks:
                logger.debug('No random chunks available -- skipping significance test.')
                return None
            query_vector, chunk_vectors = self._embed(query, random_chunks, embedding)
            random_scores = cosine_similarity(query_vector, chunk_vectors)[0].astype(float)
            random_scores = random_scores[random_scores != 0]
            bound = 'right'

        # Guard: need enough unique values for distfit to build a histogram.
        # Too few unique values causes an empty histogram and an IndexError
        # inside distfit when it tries widths[-1] on a zero-length array.
        if len(random_scores) < 10 or len(np.unique(random_scores)) < 5:
            logger.warning(
                'Null distribution has too few unique values (%d) -- '
                'skipping significance test.',
                len(np.unique(random_scores)) if len(random_scores) >= 1 else 0,
            )
            return None

        try:
            from distfit import distfit
        except Exception as exc:
            raise ImportError(
                "The 'distfit' package is required for significance testing. "
                "Install it with: pip install distfit"
            ) from exc

        try:
            model = distfit(method='parametric', alpha=self.alpha, bound=bound, verbose='warning')
            model.fit_transform(random_scores)

            # Guard: distfit stores histogram data used by plot(); if it is
            # missing or empty the plot call raises an IndexError.
            if (not hasattr(model, 'histdata')
                    or model.histdata is None
                    or len(model.histdata[0]) == 0):
                logger.warning('distfit histogram is empty -- skipping significance test.')
                return None

            results = model.predict(scores, alpha=self.alpha, todf=False, multtest='fdr_bh')

            plot_result = model.plot(
                title=f'Retrieval: {self.retrieval_method}, Embedding: {embedding}'
            )
            # model.plot() returns (fig, ax) but guard against unexpected return values
            if isinstance(plot_result, (tuple, list)) and len(plot_result) == 2:
                fig, ax = plot_result
            else:
                fig, ax = None, None

            self.distfit     = model
            self.distfit.fig = fig
            self.distfit.ax  = ax
            return results

        except (IndexError, ValueError) as exc:
            logger.warning('Significance test failed (%s) -- returning all chunks.', exc)
            return None

    def summarize(self,
             query="Extract key insights while maintaining coherence of the previous summaries.",
             instructions="Extract key insights from the **new text chunk** while maintaining coherence with **Previous summaries",
             system="You are a professional summarizer with over two decades of experience. Your strength is that you know how to deal with partial and incomplete texts but you do not make up new stuff. Keep the focus on the original input.",
             response_format="**Make a comprehensive, structured document covering all key insights**",
             context=None,
             return_type='string',
             ):
        """
        Summarize large documents iteratively while maintaining coherence across text chunks.
        
        This function splits the input text into smaller chunks and processes each part in sequence.
        For every chunk, it generates a partial summary while incorporating the context of the
        previous summaries. After all chunks have been processed, the function combines the partial
        results into a final, coherent, and structured summary.  

        Parameters
        ----------
        query : str, optional
            The guiding task or question for summarization (default extracts key insights).  
        instructions : str, optional
            Additional instructions for the summarizer, tailored to each chunk.  
        system : str
            System message that sets the role and behavior of the summarizer.  
        response_format : str, optional
            Defines the format of the final output (default is a structured document).  
        context : str or dict, optional
            Input text or structured content to be summarized. If None, uses `self.context`.  
        return_type : str, optional
            Format of the returned result (default "string").  
        
        Returns
        -------
        str
        A comprehensive, coherent summary that integrates insights across all chunks.  

        """
        if system is None:
            logger.error('system can not be None. <return>')
            return
        if (context is None) and (not hasattr(self, 'text') or self.context is None):
            logger.error('No input text found. Use context or <model.read_pdf("here comes your file path to the pdf")> first. <return>')
            return

        if context is None:
            if isinstance(self.context, dict):
                context = self.context['body'] + '\n---\n' + self.context['references']
            else:
                context = self.context

        # Create chunks based on words
        chunks = utils.chunk_text(context, method=self.chunks['method'], chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])

        logger.info(f'Processing the document using {len(chunks)} for the given task..')

        # Build a structured prompt that includes all previous summaries
        response_list = []
        for i, chunk in enumerate(chunks):
            logger.info(f'Working on text chunk {i}/{len(chunks)}')

            # Keep last N summaries for context (this needs to be within the context-window otherwise it will return an error.)
            previous_results = "\n---\n".join(response_list[-self.top_chunks:])

            prompt = (
            "### Context:\n"
            + (f"Previous results:\n{previous_results}\n" if len(response_list) > 0 else "")

            + "\n---\nNew text chunk (Part of a larger document, maintain context):\n"
            + f"{chunk}\n\n"

            "### Instructions:\n"
            + f"{instructions}**.\n\n"

            f"### Question:\n"
            f"{query}\n\n"

            "### Improved Results:\n"
            )

            # Get the summary for the current chunk
            # chunk_result = self.query_llm(prompt, system=system)
            chunk_result = self.requests_post_http(prompt, system, temperature=self.temperature, top_p=self.top_p, task='max', stream=False, return_type='string')

            response_list.append(f"Results {i+1}:\n" + chunk_result)

        # Final summarization pass over all collected summaries
        results_total = "\n---\n".join(response_list[-self.top_chunks:])
        final_prompt = f"""
        ### Context:
        {results_total}

        ### Task:
        Your task is to connect all the parts into a **coherent, well-structured document**. Make sure it becomes is a very good summary.

        ### Instructions:
        - Maintain as much as possible the key insights but ensure logical flow.
        - Connect insights smoothly while keeping essential details intact.
        - Only use bulletpoints when really needed.
        - {response_format}

        Begin your response below:
        """
        logger.info('Combining all information to create a single coherent output..')
        # Create the final summary.
        # final_result = self.query_llm(final_prompt, system=system, return_type=return_type)
        final_result = self.requests_post_http(final_prompt, system, temperature=self.temperature, top_p=self.top_p, task='max', stream=False, return_type=return_type)
        # Return
        return final_result
        # return {'summary': final_result, 'summary_per_chunk': results_total}

    def global_reasoning(self, query, context, instructions, system, return_per_chunk=False, rewrite_query=False, stream=False):
        """Global Reasoning.
            1. Rewrite the input user question into something like: "Based on the extracted summaries, does the document explain the societal relevance of the research? Justify your answer."
            2. Break the document into manageable chunks with overlapping parts to make sure we do not miss out.
            3. Create a global reasoning question based on the input user question.
            4. Take the summarized outputs and aggregate them.

            prompt = "Is the proposal well thought out?"
            instructions = "Your task is to rewrite questions for global reasoning. As an example, if there is a question like: 'Does this document section explain the societal relevance of the research?', the desired output would be: 'Does this document section explain the societal relevance of the research? If so, summarize it. If not, return 'No societal relevance found.''"
            response = model.llm.prompt(query=prompt, instructions=instructions, task='Task')

        """

        if rewrite_query:
            # 1. Rewrite user question in global reasoning question.
            logger.info('Rewriting user question for global reasoning..')
            instructions = """In the context are chunks of text from a document.
            Rewrite the user question in such a way that relevant information can be captured by a Large language model for summarization for the chunks of text in the context.
            Only return the new question with no other information.
            """
            # Initialize model for question refinement and summarization
            qmodel = LLMlight(model=self.model, temperature=0.7, endpoint=self.endpoint)
            # Create new query
            new_query = qmodel.prompt(query=query, instructions=instructions)
        else:
            new_query = query

        # Create chunks with overlapping parts to make sure we do not miss out
        if isinstance(context, str):
            chunks = utils.chunk_text(context, method=self.chunks['method'], chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
        else:
            chunks = context

        logger.info(f'Global-reasoning on {len(chunks)} chunks of text.')

        # Now summaries for the chunks
        summaries = []
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunk", unit="chunk")):
            # logger.info(f'Working on text chunk {i+1}/{len(chunks)}')

            prompt = f"""
            ### Context (Chunk {i+1} of {len(chunks)} from a larger document):
                {chunk}

            ### Instructions:
                You are an expert summarizer. For the given chunk of text:
                - Extract all **key points, decisions, facts, and actions**.
                - Ensure your analysis captures important ideas, implications, or patterns.
                - Preserve the **logical flow** and **chronological order**.
                - **Avoid repetition** or superficial statements.
                - Focus on **explicit and implicit information** that could be relevant in the full document.
                - Keep the summary **clear, precise**, and suitable for combining with other chunk summaries later.

            ### User Task:
                Summarize this chunk comprehensively and professionally.
                {query}

            """

            # Summarize
            response = self.requests_post_http(prompt, system, temperature=self.temperature, top_p=self.top_p, task='summarization', stream=stream)
            # Append
            summaries.append(response)
            # Show
            logger.debug(response)

        # Filter out "N/A" summaries
        summaries = [s for s in summaries if s.strip() != "N/A" and not any(err in s.strip()[:30] for err in ("400", "404"))]
        # Final summarization pass over all collected summaries
        summaries_final = "\n\n---\n\n".join([f"### Summary {i+1}:\n{s}" for i, s in enumerate(summaries)])
        # Return
        if return_per_chunk:
            return summaries_final

        # Create final prompt
        prompt_final = f"""### Context:
            Below are the individual summaries generated from multiple sequential chunks of a larger document. They are presented in order:
            {summaries_final}

            ---

            ### Instructions:
                {instructions}

            ### User Task:
            You are an expert editor. Your goal is to synthesize the above summaries into **one complete, well-structured, and logically coherent document**. Ensure:
            - Smooth transitions between sections.
            - Elimination of redundancies and overlaps.
            - Consistent tone, clarity, and structure.
            - That all essential information from the summaries is preserved.
            - The final result aligns with the given instructions.

            Produce the final, polished document below:
            """

        system_summaries = (
            "You are a helpful and detail-oriented assistant. "
            "Your task is to compile and structure summaries into a single coherent and well-formatted document. "
            "Follow all instructions precisely. "
            "Preserve important details, maintain logical flow, and respect any formatting requirements, "
            "such as using headings or bullet points when relevant. "
            "Output the final results in the same language as the instructions."
        )

        final_response = self.requests_post_http(prompt_final, system_summaries, temperature=self.temperature, top_p=self.top_p, task='summarization', stream=False, return_type='string')

        # Return
        return final_response

    def chunk_wise(self, query, context, instructions, system, top_chunks=0, return_per_chunk=False, stream=False):
        """Chunk-wise.
            1. Break the document into chunks with overlapping parts to make sure we do not miss out.
            2. Include the last two results in the prompt as context.
            3. Analyze each chunk seperately following the instructions and system messages and jointly with the last 2 results.

        """
        # Create chunks with overlapping parts to make sure we do not miss out
        if isinstance(context, str):
            chunks = utils.chunk_text(context, method=self.chunks['method'], chunk_size=self.chunks['size'], overlap=self.chunks['overlap'], return_type='list')
        else:
            chunks = context

        logger.info(f'Chunk wise analysis on {len(chunks)} chunks of text.')

        # Build a structured prompt that includes all previous summaries
        response_list = []
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunk", unit="chunk")):
            logger.info(f'Working on text chunk {i+1}/{len(chunks)}')

            if top_chunks > 0:
                previous_results = '\n\n---\n\n'.join(response_list[-top_chunks:])
                prev_section = previous_results if response_list else "No results because this is the initial chunk."
                prompt = f"""### Context:
                Previous Results:\n{prev_section}

                ---
                New Text Chunk (Part of a larger document, maintain continuity and coherence):
                {chunk}

                ### Instructions:
                - Apply the instructions to the new chunk **in the context** of the previous results.
                - Preserve logical structure and clarity.
                - Maintain coherence and avoid repetition with prior content.
                - Focus on extracting structured and relevant information.

                {instructions}

                ### User Question:
                {query}

                ### Final Improved Results:
                """
            else:
                prompt = f"""
                ### Context (Chunk {i+1} of {len(chunks)} -- part of a larger document):
                {chunk}

                ---

                ### Instructions:
                Carefully analyze the above chunk in isolation while considering that it is part of a broader document. Apply the following instructions to this specific chunk:
                {instructions}
                - Avoid repetition and irrelevant details.
                - Be clear and concise so this output can later be integrated with others.

                ---

                ### User Question:
                {query}

                ---

                ### Output:
                Provide your detailed, coherent analysis of this chunk below.
                """
            # Get the summary for the current chunk
            chunk_result = self.requests_post_http(prompt, system, temperature=self.temperature, top_p=self.top_p, task='summarization', stream=stream, return_type='string')
            response_list.append(chunk_result)

        # Filter out "N/A" summaries
        response_list = [s for s in response_list if s.strip() != "N/A" and not any(err in s.strip()[:30] for err in ("400", "404"))]
        # Combine all results
        response_total = "\n\n---\n\n".join([f"### Chunk {i+1}:\n{s}" for i, s in enumerate(response_list)])
        # Return all chunk information
        if return_per_chunk:
            return response_total

        if top_chunks > 0:
            prompt_final = f"""### Context (results based on {len(chunks)} chunk of text):
                {response_total}

                ---

                ### Task:
                    The context that is given to you contains the output of {len(chunks)} seperate text chunks.
                    Your task is to connect all the parts and make one output that is **coherent** and well-structured.

                ### Instructions:
                    - Maintain as much as possible the key insights but ensure logical flow.
                    - Connect insights smoothly while keeping essential details intact.
                    - If repetitions are detected across the parts, combine it.
                    {instructions}

                Begin your response below:
                """

            system_chunk_analysis = """
            You are a meticulous and structured AI assistant that performs detailed analyses of long documents, broken into smaller chunks.
            Your task is to analyze each chunk individually and extract relevant insights, observations, or structured responses based on specific user instructions.

            - Always follow the given instructions precisely.
            - If formatting is implied (e.g., headers, lists, bullet points), apply it clearly.
            - Do not add summaries or conclusions beyond the current chunk.
            - Avoid introducing outside knowledge or assumptions beyond what is present in the text.
            - Your analysis should be standalone, yet written clearly enough to be compiled later with other parts.
            """

            logger.info('Combining all information to create a single coherent output.')
            # Create the final summary.
            final_response = self.requests_post_http(prompt_final, system_chunk_analysis, temperature=self.temperature, top_p=self.top_p, task='summarization', stream=False, return_type='string')
        else:
            prompt_final = f"""### Context:
                {response_total}

                ### Task:
                    Given to you is a text that is compiled after analyzing multiple seperate chunks of text.
                    Your task is to restructure the text so that it complies with the instructions.

                ### Instructions:
                    - Maintain as much as possible the key insights but ensure logical flow.
                    - Connect insights smoothly while keeping essential details intact.
                    - If repetitions are detected across the parts, combine it.
                    - If there are vagues expressions, rewrite it to improve the quality.
                    {instructions}

                Begin your response below:
                """

            system = "You are a helpfull assistant specialized in combining multiple results that belong together. You are permitted to make assumptions if it improves the results."
            # Create the final summary.
            final_response = self.requests_post_http(prompt_final, system, temperature=self.temperature, top_p=self.top_p, task='summarization', stream=False, return_type='string')

        # Return
        return final_response
        # return {'response': final_response, 'response_per_chunk': response_total}

    # ------------------------------------------------------------------
    # Core embedding primitive
    # ------------------------------------------------------------------

    def _embed(self, query: str, chunks: list, embedding: str) -> tuple:
        """Embed *query* and *chunks* with the requested *embedding* method.

        Chunks are encoded in a single batched call (batch_size=32) when using
        sentence-transformers models, avoiding the per-chunk overhead of
        individual ``encode()`` calls.

        Parameters
        ----------
        query : str
        chunks : list of str
        embedding : str
            One of ``'tfidf'``, ``'bow'``, ``'bert'``, ``'bge-small'``.
            ``None`` is not accepted here -- callers must guard against it before
            calling ``_embed``.

        Returns
        -------
        (query_vector, chunk_vectors) : tuple
            Sparse scipy matrices for tfidf/bow; dense numpy arrays for bert/bge-small.
        """
        _ST_MODELS = {'bert': 'all-MiniLM-L6-v2', 'bge-small': 'BAAI/bge-small-en'}

        if embedding in ('tfidf', 'bow'):
            Vectorizer = TfidfVectorizer if embedding == 'tfidf' else CountVectorizer
            vec = Vectorizer()
            chunk_vectors = vec.fit_transform(chunks)
            query_vector  = vec.transform([query])
            return query_vector, chunk_vectors

        if embedding in _ST_MODELS:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise ImportError(
                    f"sentence-transformers is required for embedding='{embedding}'. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            model = SentenceTransformer(_ST_MODELS[embedding])
            chunk_vectors = model.encode(chunks, batch_size=32, convert_to_numpy=True)
            query_vector  = model.encode([query], convert_to_numpy=True).reshape(1, -1)
            return query_vector, chunk_vectors

        raise ValueError(
            f"Unsupported embedding method: '{embedding}'. "
            f"Valid options: {get_embeddings()}"
        )

    # ------------------------------------------------------------------
    # Core retrieval primitive  -- single shared implementation
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, chunks: list, embedding: str, top_k: int) -> list:
        """Rank *chunks* by relevance to *query* and return the top-*top_k* ones.

        Parameters
        ----------
        query     : str
        chunks    : list of str
        embedding : str
        top_k     : int  (capped at len(chunks))

        Returns
        -------
        list of (score: float, text: str) tuples, highest score first.
        """
        if not chunks: return []
        top_k = min(top_k, len(chunks))
        query_vector, chunk_vectors = self._embed(query, chunks, embedding)
        scores = cosine_similarity(query_vector, chunk_vectors)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(float(scores[i]), chunks[i]) for i in top_indices]

    # ------------------------------------------------------------------
    # Statistical significance filter
    # ------------------------------------------------------------------

    def _filter_by_significance(self, query: str, scored: list) -> list:
        """Remove chunks whose score is not statistically significant.

        Parameters
        ----------
        query  : str
        scored : list of (score, text) tuples

        Returns the original list unchanged when alpha is None, when there
        are too few results, or when the significance test fails.
        """
        if self.alpha is None or len(scored) < 2:
            return scored

        scores = np.array([s for s, _ in scored])
        logger.info("Testing retrieval score significance (alpha=%.3f).", self.alpha)
        out = self.compute_probability(query, scores, embedding=self.embedding['memory'], n=1000)

        if out is None:
            return scored

        mask = out.get('y_bool', np.ones(len(scored), dtype=bool))
        filtered = [pair for pair, keep in zip(scored, mask) if keep]
        logger.info(f"{len(filtered)} / {len(scored)} chunks retained after significance filtering with alpha={self.alpha}.")
        # Return
        return filtered

    # ------------------------------------------------------------------
    # Public retrieval entry points
    # ------------------------------------------------------------------

    def relevant_memory_retrieval(self, query: str, return_type: str = 'dict'):
        """Return the top-k most relevant chunks from the persistent memory store.

        Returns ``None`` when no store is loaded, the store file does not exist,
        or ``embedding['memory']`` is ``None`` (embedding disabled).

        The backing file is detected by checking both ``self.store_path`` and
        ``self.memory.store_path``, so extension normalisation (e.g. a user-supplied
        ``.mp4`` path resolved to a ``.db`` file by the sqlite backend) does not
        cause a silent miss.

        Parameters
        ----------
        query : str
            The search query.
        return_type : str, default ``'list'``
            ``'list'``   -- list of str chunks, highest relevance first.
            ``'string'`` -- chunks joined with ``### Chunk N:`` headers.
            ``'dict'`` -- score and string chunks

        Returns
        -------
        list of str, str, or None
        """
        # A store is "ready" when the backend is loaded and a backing file exists.
        # For the sqlite backend the file is always .db regardless of what the user
        # passed as file_path, so check both self.store_path and the backend's own
        # store_path (which may differ after extension normalisation).
        backend_path = getattr(getattr(self, 'memory', None), 'store_path', None)
        store_exists = any(
            p and os.path.isfile(p)
            for p in (self.store_path, backend_path)
        )
        if not (hasattr(self, 'memory') and store_exists):
            return None

        if self.embedding['memory'] is None:
            logger.warning("Memory retrieval skipped: embedding disabled.")
            return None

        logger.info("Retrieving [%d] chunks from memory store (embedding='%s').",
                    self.top_chunks, self.embedding['memory'])

        if self.embedding['memory'] == 'memvid':
            # Backend provides its own similarity scores
            scored = self.memory.search_with_scores(query, top_k=self.top_chunks)
        elif self.embedding['memory'] in get_embeddings():
            all_chunks = self.memory.get_all_chunks()
            scored = self._retrieve(query, all_chunks, self.embedding['memory'], self.top_chunks)
        else:
            logger.warning("Unknown memory embedding '%s', skipping retrieval.", self.embedding['memory'])
            return None

        # Filter chunks on significance
        scored  = self._filter_by_significance(query, scored)
        chunks  = [text for _, text in scored]

        if return_type == 'string':
            return "\n\n---\n\n".join(f"### Chunk {i+1}:\n{c}" for i, c in enumerate(chunks))
        elif return_type == 'list':
            return chunks
        else:
            return scored

    def relevant_context_retrieval(self, query: str, context: str, return_type: str = 'list'):
        """Return the most relevant portion of *context* for *query*.

        Behaviour depends on ``self.retrieval_method`` and ``self.embedding['context']``:

        - ``retrieval_method='naive_rag'`` and embedding not ``None``: context is split
          into chunks and the top-k most similar chunks are returned.
        - ``retrieval_method='RSE'`` with ``embedding`` in ``('bert', 'bge-small')``:
          contiguous relevant segments are extracted via RSE.
        - Any other combination (including ``embedding=None`` or ``retrieval_method=None``):
          the full *context* string is returned unchanged.  A warning is logged when
          the estimated token count exceeds ``n_ctx``.

        Parameters
        ----------
        query : str
        context : str
            Raw text to search through.
        return_type : str, default ``'list'``
            ``'list'``   -- list of str chunks.
            ``'string'`` -- chunks joined with ``### Chunk N:`` headers.

        Returns
        -------
        list of str, str, or the original *context* unchanged.
        """
        if not context:
            return context

        if self.retrieval_method == 'naive_rag' and self.embedding['context'] is not None and self.embedding['context'] in get_embeddings():
            logger.info(f"naive_rag: retrieving {self.top_chunks} chunks from context (embedding='{self.embedding['context']}').")
            # Create chunks
            chunks = utils.chunk_text(context, method=self.chunks['method'], chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
            scored = self._retrieve(query, chunks, self.embedding['context'], self.top_chunks)
            chunks_out = [text for _, text in scored]

            if return_type == 'string':
                return "\n\n---\n\n".join(f"### Chunk {i+1}:\n{c}" for i, c in enumerate(chunks_out))
            return chunks_out

        if self.retrieval_method == 'RSE' and self.embedding['context'] is not None and self.embedding['context'] in ('bert', 'bge-small'):
            logger.info("RSE retrieval applied.")
            return RAG.RSE(
                context, query,
                label=None,
                chunk_size=self.chunks['size'],
                irrelevant_chunk_penalty=0,
                embedding=self.embedding['context'],
                device='cpu',
                batch_size=32,
            )

        logger.info("No retrieval method applied -- using full context.")
        # Warn when the raw context is likely to exceed the context window.
        # Rough heuristic: 1 token ≈ 4 characters for Latin-script text.
        estimated_tokens = len(context) // 4
        if estimated_tokens > self.n_ctx:
            logger.warning(
                "Full context is ~%d tokens but n_ctx=%d. "
                "The model will likely truncate the input. "
                "Consider setting retrieval_method='naive_rag' or reducing the context.",
                estimated_tokens, self.n_ctx,
            )
        return context

    def compute_context_strategy(self, query, context, instructions, system):
        """Apply the configured context strategy before retrieval."""
        if context is None:
            return context
        if self.context_strategy == 'global-reasoning':
            return self.global_reasoning(query, context, instructions, system,
                                         rewrite_query=False, return_per_chunk=True)
        if self.context_strategy == 'chunk-wise':
            return self.chunk_wise(query, context, instructions, system,
                                   top_chunks=0, return_per_chunk=True)
        logger.info("No context strategy applied.")
        return context

    # ------------------------------------------------------------------
    # Backwards-compat wrappers
    # ------------------------------------------------------------------

    def search(self, query: str, chunks: list,
               return_type: str = 'score',
               top_chunks: int = None,
               embedding: str = None) -> list:
        """Rank *chunks* by relevance to *query*.

        Thin wrapper around :meth:`_retrieve` kept for backwards compatibility.

        return_type options
        -------------------
        'score'       : list of (score, text) tuples  [default]
        'list'        : list of text strings
        'string_flat' : space-joined string
        other         : newline-separated string
        """
        emb    = embedding or 'tfidf'
        k      = top_chunks or len(chunks)
        scored = self._retrieve(query, chunks, emb, k)

        if return_type == 'score':        return scored
        if return_type == 'list':         return [t for _, t in scored]
        if return_type == 'string_flat':  return " ".join(t for _, t in scored)
        return "\n---------\n".join(t for _, t in scored)

    def fit_transform(self, query, chunks, embedding=None):
        """Embed query and chunks.  Use :meth:`_embed` in new code."""
        if isinstance(embedding, dict):
            emb = embedding.get('context', 'tfidf')
        elif isinstance(embedding, str):
            emb = embedding
        else:
            emb = self.embedding.get('context', 'tfidf')
        return self._embed(query, chunks, emb)

    def _filter_proba(self, query, scores, relevant_context):
        """Backwards-compat wrapper -- use _filter_by_significance in new code."""
        scored   = list(zip(scores, relevant_context))
        filtered = self._filter_by_significance(query, scored)
        return [t for _, t in filtered]

    def set_prompt(self, query: str, instructions: str, context: (str, list), response_format: str = None):
        # Default and update when context and instructions are available.
        if isinstance(context, list):
            context = "\n\n---\n\n".join([f"### Chunk {i+1}:\n{s}" for i, s in enumerate(context)])
        if context=='':
            logger.info('No context is provided into the prompt.')

        prompt = (
            ("Context:\n" + context + "\n\n" if context else "")
            + ("Instructions:\n" + instructions + "\n\n" if instructions not in ("", None) else "")
            + ("Response format:\n" + response_format + "\n\n" if response_format not in ("", None) else "")
            + "User question:\n"
            + query
        )

        # Return
        return prompt

    def read_pdf(self, file_path, title_pages=[1, 2], body_pages=[], reference_pages=[-1], return_type='str'):
        """
        Reads a PDF file and extracts its text content as a string.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
            dict: dictionary

        """
        context = ''

        if 'http' in file_path[0:5]:
            logger.info('Downloading file from url..')
            url = file_path
            filename = wget.filename_from_url(url)
            file_path = os.path.join(self.tempdir, filename)
            wget.download(url, file_path)   # downloads to file_path; return value is None

        if os.path.isfile(file_path):
            # Read pdf
            context = utils.read_pdf(file_path, title_pages=title_pages, body_pages=body_pages, reference_pages=reference_pages, return_type=return_type)
        else:
            logger.error(f'file_path does not exist: {file_path}')

        # Return
        return context
    

    def get_model_info(self, model=None):
        logger.info(f'Collecting model info..')
        if model is None and hasattr(self, 'model'): model = self.model
        if model is None:
            return {}

        # Get model url
        model_url = self.get_model_endpoint()
        response = requests.get(model_url, timeout=10)

        # Check status
        if response.status_code == 200:
            try:
                modelnames = response.json()["models"]
                for modelname in modelnames:
                    if modelname["key"]==model:
                        logger.info(f'Model info collected: {model}')
                        return modelname
            except (KeyError, ValueError) as e:
                logger.error("Error parsing model data:", e)

        # If nothing is found, return None
        return {}

    def get_model_endpoint(self):
        base_url = '/'.join(self.endpoint.split('/')[:3])
        model_url = f"{base_url}/api/v1/models"
        # base_url = '/'.join(self.endpoint.split('/')[:3]) + '/'
        # model_url = base_url.rstrip('/') + '/v1/models'
        return model_url

    
    def get_available_models(self, validate=False):
        """Retrieve available models from the configured API endpoint.

        Optionally validates each model by sending a test prompt and filtering out
        models that return a 404 error or similar failure response.

        Parameters
        ----------
        validate : bool, optional
            If True, each model is tested with a prompt to ensure it can respond correctly.
            Models that fail validation (e.g., return a 404 error) are excluded from the result.

        Returns
        -------
        list of str
            A list of model identifiers (e.g., `"llama3"`, `"gpt-4"`) that are available and valid.

        Examples
        --------
        >>> # Import library
        >>> from LLMlight import LLMlight
        <<< # Initialize
        >>> client = LLMlight(endpoint='http://localhost:1234/v1/chat/completions')
        >>> # Get models
        >>> models = client.get_available_models(validate=False)
        >>> # Print
        >>> print(models)
        >>> ['llama3', 'mistral-7b']

        Notes
        -----
        - Requires an accessible endpoint and valid API response.
        - Relies on the `LLMlight` class for validation (must be importable).
        """
        logger.info(f'Collecting models at API endpoint: {self.endpoint}')
        model_url = self.get_model_endpoint()
        models = None

        try:
            response = requests.get(model_url, timeout=10)
            if response.status_code == 200:
                try:
                    get_models = response.json()["models"]
                    model_dict = {model["key"]: model for model in get_models}
                    models = list(model_dict.keys())
                except (KeyError, ValueError) as e:
                    logger.error("Error parsing model data:", e)
            else:
                logger.warning("Request failed with status code:", response.status_code)
                logger.warning("Response:", response.text)

        except requests.exceptions.RequestException as e:
            logger.error("Request error:", e)
            logger.error(f'No connection could be made with the endpoint: {model_url}')
            return None

        # Check each model whether it returns a response
        if validate and models:
            logger.info("Validating the working of each available model. Be patient.")
            keys = copy.deepcopy(list(model_dict.keys()))

            for key in keys:
                from LLMlight import LLMlight
                # logger.info(f'Checking: {key}')
                llm = LLMlight(model=key)
                response = llm.prompt('What is the capital of France?', instructions="You are only allowed to return one word.", return_type='string')
                response = response[0:30].replace('\n', ' ').replace('\r', ' ').lower()
                if 'error: 404' in response:
                    logger.error(f"{llm.model}: {response}")
                    model_dict.pop(key)
                else:
                    logger.debug(f"{llm.model}: {response}")
        
        if not models:
            logger.error(f'No models could be detected at endpoint. <return>')

        return models

    def check_logger(self):
        """Check the verbosity."""
        logger.debug('DEBUG')
        logger.info('INFO')
        logger.warning('WARNING')
        logger.critical('CRITICAL')

#%%
# ---------------------------------------------------------------------------
# Parameter constants & helpers
# ---------------------------------------------------------------------------

_CONTEXT_EMBEDDINGS = ('tfidf', 'bow', 'bert', 'bge-small')
_MEMORY_EMBEDDINGS  = ('tfidf', 'bow', 'bert', 'bge-small', 'memvid')
_VALID_RETRIEVAL    = (None, 'naive_rag', 'RSE')
_VALID_STRATEGIES   = (None, 'chunk-wise', 'global-reasoning')
_CHUNK_METHODS      = ('chars', 'words')
_CHUNKS_DEFAULTS    = {'method': 'chars', 'size': 1000, 'overlap': 200}


def get_embeddings():
    """Return all recognised embedding method names."""
    return list(_MEMORY_EMBEDDINGS)


def _resolve_embedding(embedding) -> dict:
    """Normalise *embedding* to ``{'memory': str|None, 'context': str|None}``.

    Parameters
    ----------
    embedding : None, str, or dict
        ``None``        -- Both paths disabled: ``{'memory': None, 'context': None}``.
        ``'automatic'`` -- Use defaults: ``{'memory': 'memvid', 'context': 'bert'}``.
        A valid string  -- Applied to both paths.  ``'memvid'`` on the context path
                          is silently corrected to ``'bert'``.
        A dict          -- Keys ``'memory'`` and/or ``'context'``; missing keys inherit
                          the ``'automatic'`` defaults.

    Returns
    -------
    dict with keys ``'memory'`` and ``'context'``, each a valid embedding name or ``None``.

    Raises
    ------
    ValueError
        Unknown embedding name or unknown dict key.
    TypeError
        *embedding* is not ``None``, a string, or a dict.
    """
    defaults = {'memory': 'memvid', 'context': 'bert'}

    if embedding is None:
        # Explicitly disabled -- no embedding for either path.
        return {'memory': None, 'context': None}

    if embedding == 'automatic':
        return dict(defaults)

    if isinstance(embedding, str):
        if embedding not in _MEMORY_EMBEDDINGS:
            raise ValueError(
                f"Unknown embedding '{embedding}'. Valid: {_MEMORY_EMBEDDINGS}."
            )
        resolved = {'memory': embedding, 'context': embedding}
    elif isinstance(embedding, dict):
        unknown = set(embedding) - {'memory', 'context'}
        if unknown:
            raise ValueError(
                f"Unknown key(s) in embedding dict: {unknown}. "
                "Expected keys: 'memory', 'context'."
            )
        resolved = dict(embedding)
    else:
        raise TypeError(
            f"embedding must be a str or dict, got {type(embedding).__name__}."
        )

    if resolved.get('context') == 'memvid':
        logger.warning(
            "embedding['context']='memvid' is not valid for context retrieval; "
            "falling back to 'bert'."
        )
        resolved['context'] = 'bert'

    mem_emb = resolved.get('memory', defaults['memory'])
    ctx_emb = resolved.get('context', defaults['context'])

    if mem_emb is not None and mem_emb not in _MEMORY_EMBEDDINGS:
        raise ValueError(
            f"embedding['memory']='{mem_emb}' not recognised. Valid: {_MEMORY_EMBEDDINGS}."
        )
    if ctx_emb is not None and ctx_emb not in _CONTEXT_EMBEDDINGS:
        raise ValueError(
            f"embedding['context']='{ctx_emb}' not recognised. Valid: {_CONTEXT_EMBEDDINGS}."
        )

    return {**defaults, **resolved}


def _resolve_chunks(chunks) -> dict:
    """Normalise *chunks* to {'method': str, 'size': int, 'overlap': int}.

    Handles legacy key aliases: 'type' -> 'method', 'chunk_size' -> 'size'.
    """
    if chunks is None:
        return dict(_CHUNKS_DEFAULTS)

    if not isinstance(chunks, dict):
        raise TypeError(f"chunks must be a dict or None, got {type(chunks).__name__}.")

    result = dict(chunks)
    if 'type' in result and 'method' not in result:
        result['method'] = result.pop('type')
    if 'chunk_size' in result and 'size' not in result:
        result['size'] = result.pop('chunk_size')

    merged = {**_CHUNKS_DEFAULTS, **result}

    if merged['method'] not in _CHUNK_METHODS:
        raise ValueError(
            f"chunks['method']='{merged['method']}' is not valid. Valid: {_CHUNK_METHODS}."
        )
    if not isinstance(merged['size'], int) or merged['size'] < 1:
        raise ValueError(
            f"chunks['size'] must be a positive integer, got {merged['size']!r}."
        )
    if not isinstance(merged['overlap'], int) or merged['overlap'] < 0:
        raise ValueError(
            f"chunks['overlap'] must be a non-negative integer, got {merged['overlap']!r}."
        )
    if merged['overlap'] >= merged['size']:
        raise ValueError(
            f"chunks['overlap'] ({merged['overlap']}) must be less than "
            f"chunks['size'] ({merged['size']})."
        )
    return merged


def _validate_params(model, retrieval_method, embedding, context_strategy, alpha, top_chunks, temperature, top_p, chunks) -> dict:
    """Validate all constructor parameters and return a normalised dict.

    Raises ValueError or TypeError with a clear message on any bad value.
    """
    if retrieval_method not in _VALID_RETRIEVAL:
        raise ValueError(
            f"retrieval_method='{retrieval_method}' is not valid. "
            f"Valid values: {_VALID_RETRIEVAL}."
        )
    if context_strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"context_strategy='{context_strategy}' is not valid. "
            f"Valid values: {_VALID_STRATEGIES}."
        )
    if alpha is not None:
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be a float or None, got {type(alpha).__name__}.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not isinstance(top_chunks, int) or top_chunks < 1:
        raise ValueError(f"top_chunks must be a positive integer, got {top_chunks!r}.")
    if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
        raise ValueError(f"temperature must be a float in [0, 2], got {temperature!r}.")
    if not isinstance(top_p, (int, float)) or not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be a float in (0, 1], got {top_p!r}.")

    return {
        'model':            model,
        'retrieval_method': retrieval_method,
        'embedding':        _resolve_embedding(embedding),
        'context_strategy': context_strategy,
        'alpha':            alpha,
        'top_chunks':       top_chunks,
        'temperature':      float(temperature),
        'top_p':            float(top_p),
        'chunks':           _resolve_chunks(chunks),
    }


# Legacy alias kept for any external callers
def _set_embedding(embedding):
    """Backwards-compat alias for _resolve_embedding()."""
    return _resolve_embedding(embedding)

#%%
def convert_messages_to_model(messages, model='llama', add_assistant_start=True):
    """
    Builds a prompt in the appropriate format for different model families.

    Supported families (matched by substring in the model id, case-insensitive):
      - gemma / grok : Gemma format  -- ``<start_of_turn>{role}\\n...<end_of_turn>\\n``
                       The assistant opener is ``<start_of_turn>model\\n``.
      - all others   : ChatML format -- ``<|im_start|>{role}\\n...<|im_end|>\\n``
                       The assistant opener is ``<|im_start|>assistant\\n``.
                       This covers llama, mistral, hermes, phi, qwen, deepseek,
                       and any unrecognised model id.

    Parameters
    ----------
    messages : list of dict
        Each dict must have keys ``'role'`` and ``'content'``.
    model : str, default ``'llama'``
        Model identifier string used to select the prompt format.
    add_assistant_start : bool, default ``True``
        Append the assistant turn opener so the model continues from there.

    Returns
    -------
    str
        The formatted prompt string.

    Examples
    --------
    >>> messages = [
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user',   'content': 'What is the capital of France?'},
    ... ]
    >>> # ChatML (llama, mistral, ...)
    >>> print(convert_messages_to_model(messages, model='mistralai/mistral-small'))
    >>> # Gemma format
    >>> print(convert_messages_to_model(messages, model='google/gemma-3-12b'))
    """
    prompt = ""
    model_lower = (model or '').lower()

    # Gemma-format models use <start_of_turn> / <end_of_turn>
    _GEMMA_FAMILIES = ('gemma', 'grok')
    use_gemma = any(f in model_lower for f in _GEMMA_FAMILIES)

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if use_gemma:
            prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        else:
            # ChatML -- covers llama, mistral, hermes, phi, qwen, deepseek, and unknown models
            prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

    if add_assistant_start:
        if use_gemma:
            prompt += "<start_of_turn>model\n"
        else:
            prompt += "<|im_start|>assistant\n"

    return prompt



def load_local_gguf_model(model_path: str, n_ctx: int=4096, n_threads: int=8, n_gpu_layers: int=0, verbose: bool=True) -> Llama:
    """
    Loads a local GGUF model using llama-cpp-python.

    Args:
        model_path (str): Path to the .gguf model file.
        n_ctx (int): Maximum context length. Default is 4096.
        n_threads (int): Number of CPU threads to use. Default is 8.
        n_gpu_layers (int): Number of layers to offload to GPU (if available). Default is 20.
        verbose (bool): Whether to print status info.

    Returns:
        Llama: The loaded Llama model object.

    Example:
        >>> model_path = r'C://Users//beeld//.lmstudio//models//NousResearch//Hermes-3-Llama-3.2-3B-GGUF//Hermes-3-Llama-3.2-3B.Q4_K_M.gguf'
        >>> llm = load_local_gguf_model(model_path, verbose=True)
        >>> prompt = "<start_of_turn>user\\nWhat is 2 + 2?\\n<end_of_turn>\\n<start_of_turn>model\\n"
        >>> response = llm(prompt=prompt, max_tokens=20, stop=["<end_of_turn>"])
        >>> print(response["choices"][0]["text"].strip())
        '4'

    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Context length: {n_ctx}, Threads: {n_threads}, GPU layers: {n_gpu_layers}")

    if Llama is None:
        raise ImportError(
            "llama-cpp-python is not installed. Install it with `pip install llama-cpp-python` "
            "or install the package extra: `pip install -e '.[llamacpp]'` to enable local llama models."
        )

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose
    )

    logger.info("Model loaded successfully!")
    # Return
    return llm

# def compute_tokens(string, n_ctx=4096, task='max'):
#     """Estimate the token count of *string* and compute the generation budget.

#     Uses the GPT-2 tokenizer as a fast, dependency-light approximation.  Token
#     counts for non-GPT-2 model families (Llama, Mistral, Gemma, ...) may differ
#     by up to ~30 %, so treat the result as an estimate rather than an exact value.

#     Unlike a naive approach the prompt is encoded **without** truncation so the
#     real length is always visible.  A warning is emitted when the prompt exceeds
#     ``n_ctx`` so the caller can react rather than silently losing context.

#     Parameters
#     ----------
#     string : str
#         The full prompt string to measure.
#     n_ctx : int, default ``4096``
#         Model context window in tokens.
#     task : str, default ``'max'``
#         Generation task passed to :func:`compute_max_tokens` to determine the
#         fraction of remaining tokens allocated for the response.

#     Returns
#     -------
#     (used_tokens, max_tokens) : tuple of int
#         ``used_tokens`` -- estimated tokens consumed by the prompt.
#         ``max_tokens``  -- tokens available for generation, capped by ``n_ctx``.
#     """
#     try:
#         from transformers import AutoTokenizer
#     except Exception as e:
#         raise ImportError("transformers is required for token counting. Install via 'pip install transformers'") from e

#     import warnings
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     # Encode WITHOUT truncation so we get the real token count.
#     # The GPT-2 tokenizer warns when the sequence exceeds its own 1024-token
#     # training limit, but here we are only *counting* tokens — not running
#     # GPT-2 — so the warning is a false alarm and we suppress it.
#     with warnings.catch_warnings():
#         warnings.filterwarnings(
#             "ignore",
#             message="Token indices sequence length is longer than the specified maximum",
#         )
#         tokens = tokenizer.encode(string)
#     used_tokens = len(tokens)
#     if used_tokens > n_ctx:
#         logger.warning(
#             "Prompt length (%d tokens) exceeds the model context window (%d tokens). "
#             "The model will truncate the input and important context may be lost.\n"
#             "\n"
#             "  How to fix:\n"
#             "  * Reduce chunk size:   LLMlight(..., chunks={'size': 500})\n"
#             "    Smaller chunks = fewer tokens per prompt.\n"
#             "  * Reduce chunk overlap: LLMlight(..., chunks={'overlap': 50})\n"
#             "  * Reduce top_chunks:   LLMlight(..., top_chunks=3)\n"
#             "    Fewer chunks combined into a single prompt.\n"
#             "  * Use summarize():     client.summarize(context=text)\n"
#             "    Splits the document automatically, chunk by chunk.\n"
#             "  * Increase n_ctx:      LLMlight(..., n_ctx=8192)\n"
#             "    Only works if your model actually supports a larger window.",
#             used_tokens, n_ctx,
#         )
#     # Determine how many tokens are available for the model to generate
#     max_tokens = compute_max_tokens(used_tokens, n_ctx=n_ctx, task=task)
#     # Show message
#     logger.info(f"Used_tokens={used_tokens}, max_tokens={max_tokens}, context_limit={n_ctx}")
#     # Return
#     return used_tokens, max_tokens

def compute_tokens(text: str, n_ctx: int = 16384, chars_per_token=3, task: str = "max"):
    """
    Estimate token usage using a model-agnostic approximation.

    We assume:
        1 token ≈ 3 characters

    This avoids model-specific tokenizers and provides a fast,
    reproducible estimate across different LLM providers.

    Parameters
    ----------
    text : str
        Prompt text.
    n_ctx : int, default=16384
        Model context window.
    chars_per_token : float, default=305
        3: for coding
        4: for english
        3.5 average usage
    task : str, default='max'
        Generation task.

    Returns
    -------
    used_tokens : int
        Estimated prompt tokens.
    max_tokens : int
        Recommended generation budget.
    """
    # Estimate the used tokens
    used_tokens = max(1, int(len(text) / chars_per_token))

    if used_tokens >= n_ctx:
        logger.warning(f"Prompt length ({used_tokens:,} estimated tokens) exceeds context window ({n_ctx:,}). Input will likely be truncated.")

    # Compute max tokens
    max_tokens = compute_max_tokens(used_tokens=used_tokens, n_ctx=n_ctx, task=task)
    logger.info(f"Estimated_tokens={used_tokens:,}, max_tokens={max_tokens:,}, context_limit={n_ctx:,}")
    # Return
    return used_tokens, max_tokens


def compute_max_tokens(used_tokens: int, n_ctx: int = 4096, task: str = "max"):
    """
    Compute a safe generation budget.

    The function reserves part of the context window for the prompt
    and allocates the remaining space according to the task.
    """

    available = max(n_ctx - used_tokens, 1)

    task_configs = {
        "summarization": {"ratio": 0.50, "minimum": 128},
        "chat":          {"ratio": 0.60, "minimum": 128},
        "code":          {"ratio": 0.75, "minimum": 256},
        "longform":      {"ratio": 0.90, "minimum": 512},
        "analysis":      {"ratio": 0.80, "minimum": 512},
        "max":           {"ratio": 1.00, "minimum": 1},
    }

    cfg = task_configs.get(task.lower(), task_configs["chat"])
    if task.lower() == "max":
        return available

    target = int(n_ctx * cfg["ratio"])

    return min(available, max(cfg["minimum"], target))


# def compute_max_tokens(used_tokens, n_ctx=4096, task="max"):
#     """
#     Compute the maximum number of tokens that can be generated for a given task,
#     taking into account the number of tokens already used and the model's context window.

#     Parameters
#     ----------
#     used_tokens : int
#         Number of tokens already consumed in the current context.
#     n_ctx : int, optional
#         Total context window size of the model (default is 4096 tokens).
#     task : str, optional
#         Type of generation task. Determines the proportion of the remaining tokens to use.
#         Options are:
#         - "summarization": Use up to 50% of the context window, minimum 128 tokens.
#         - "chat": Use up to 60% of the context window, minimum 128 tokens.
#         - "code": Use up to 75% of the context window, minimum 128 tokens.
#         - "longform": Use up to 90% of the context window, minimum 256 tokens.
#         - "max": Use all remaining tokens.
#         Any unrecognized task defaults to a safe fallback using 50% of the context window.

#     Returns
#     -------
#     max_tokens : int
#         Maximum number of tokens that can be generated for the specified task,
#         ensuring at least a minimum number of tokens as defined per task type.
#     """

#     available_tokens = max(n_ctx - used_tokens, 1)  # Ensure at least 1

#     task = task.lower()
#     if task == "summarization":
#         max_tokens = max(min(available_tokens, int(n_ctx * 0.5)), 128)
#     elif task == "chat":
#         max_tokens = max(min(available_tokens, int(n_ctx * 0.6)), 128)
#     elif task == "code":
#         max_tokens = max(min(available_tokens, int(n_ctx * 0.75)), 128)
#     elif task == "longform":
#         max_tokens = max(min(available_tokens, int(n_ctx * 0.9)), 256)
#     elif task == "max":
#         max_tokens = available_tokens
#     else:
#         # Default to safe fallback
#         max_tokens = max(min(available_tokens, int(n_ctx * 0.5)), 128)

#     return max_tokens


def set_system_message(system):
    if system is None:
        system = """You are a helpful AI assistant with access to a knowledge base.

        When answering questions:
        1. Use the provided context from the knowledge base when relevant
        2. When multiple sections are in the context; ### chunk 1:, ### chunk 2: or ### summary 1:, ### summary 2: etc, then the higher ranked chunks contain more relevant information.
        3. Be clear about what information comes from the knowledge base vs. your general knowledge
        4. If the context doesn't contain enough information, say so clearly
        5. Provide helpful, accurate, and concise responses

    The context will be provided with each query based on semantic similarity to the user's question."""

    return system


def get_logger():
    return logger.getEffectiveLevel()


def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)

# %% Retrieve files files.
class wget:
    """Retrieve file from url."""

    def filename_from_url(url, ext=True):
        """Return filename."""
        urlname = os.path.basename(url)
        if not ext: _, ext = os.path.splitext(urlname)
        return urlname

    def download(url, writepath):
        """Download.

        Parameters
        ----------
        url : str.
            Internet source.
        writepath : str.
            Directory to write the file.

        Returns
        -------
        None.

        """
        r = requests.get(url, stream=True)
        with open(writepath, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)