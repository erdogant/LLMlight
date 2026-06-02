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
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    _logger.debug(f"Relative imports failed: {e}. Falling back to top-level imports.")
    import memory
    import RAG
    import utils

logger = logging.getLogger(__name__)

# %%
class LLMlight:
    """Large Language Model Light.

    Run your LLM models local and with minimum dependencies.
    1. Go to LM-studio.
    2. Go to left panel and select developers mode.
    3. On top select your model of interest.
    4. Then go to settings in the top bar.
    5. Enable "server on local network" if you need.
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
    - Temperature control: Adjusts response randomness (0.7 default)
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
        'unsloth/gemma-4-26b-a4b-it'
        'qwen/qwen3-coder-30b'
        'openai/gpt-oss-20b'
    system : str
        String of the system message.
        "I am a helpfull assistant"
    retrieval_method : str (default: 'naive_rag')
        None:                   No processing. The entire context is used for the query.
        'naive_rag':            Context is processed using Navie RAG approach. Ideal for chats and when you need to answer specfic questions: Chunk of text are created. Use cosine similarity to for ranking. The top scoring chunks will be combined (n chunks) and used as input with the prompt.
        'RSE':                  Context is processed using Navie RSE approach. Identify and extract entire segments of relevant text.
    embedding : str, dict (default: 'automatic')
        Specify the embedding. When using both video-memory and context, it can be specified with a dictionary: {'memory': 'memvid', 'naive_rag': 'bert'}
        None:                   No embedding is performed.
        'automatic':            {'memory': 'memvid', 'naive_rag': 'bert'}
        'memvid':               This embedding can only be applied when using video-memory in the retrieval method.
        'tfidf':                Best use when it is a structured documents and the words in the queries are matching.
        'bow':                  Bag of words approach. Best use when you expect words in the document and queries to be matching.
        'bert':                 Best use when document is more free text and the queries may not match exactly the words or sentences in the document.
        'bge-small':
    context_strategy : str (default: None)
         None:                  No pre-processing. The original context is used in the pipeline of retrieval_method, embedding and the response.
        'chunk-wise':           The input context will be analyze chunkwise based on the query, instructions and system. The total set of answered-chunks is then returned. The normal pipeline proceeds for the query, instructions, system etc.
        'global-reasoning':     The input context will be summarized per chunk globally. The total set of summarized context is then returned. The normal pipeline proceeds for the query, instructions, system etc.
    temperature : float, optional
        Sampling temperature.
        0.7: (default)
        0: Deterministic
        1: Stochastic (max)
    top_p : float, optional
        Top-p (nucleus) sampling parameter (default is 1.0, no filtering).
    chunks: dict : {'method': 'chars', 'size': 1000, 'overlap': 250, 'top_chunks': 5}
        type : str
            'chars' or 'words': Chunks are created using chars or words.
            'size': Chunk length in chars or words.
                The accuracy increases with smaller chunk sizes. But it also reduces the input context for the LLM.
                Estimates: 1000 words or ~10.000 chars costs ~3000 tokens.
                With a context window (n_ctx) of 4096 your can set size=1000 words with n chunks=5 and leave some space for instructions, system and the query.
            'overlap': overlap between chunks
            'top_chunks': Retrieval of the top N chunks when performing RAG analysis.
    n_ctx : int, default: 4096
        The context window length is determined by the max tokens. A larger number of tokens will ask more cpu/gpu resources. Estimates: 1000 words or ~10.000 chars costs ~3000 tokens.
    file_path : str
        'knowledge_base.mp4'    Local or absolute path to your (video) memory file.
    endpoint : str:  Endpoint of the LLM API
        "http://localhost:1234/v1/chat/completions"

    Examples
    --------
    >>> # Examples
    >>> from LLMlight import LLMlight
    >>> client =  LLMlight()
    >>> client.prompt('hello, who are you?')
    >>> system_message = "You are a helpful assistant."
    >>> response = client.prompt('What is the capital of France?', system=system_message, top_p=0.9)
    >>> print(response)

    """
    def __init__(self,
                 model: str = None,
                 retrieval_method: (None, str) = 'naive_rag',
                 embedding: (str, dict) = {'memory': 'memvid', 'context': 'bert'},
                 context_strategy: str = None,
                 alpha: float = 0.05,
                 top_chunks: int = 5,
                 temperature: (int, float) = 0.7,
                 top_p: (int, float) = 1.0,
                 chunks: dict = {'method': 'chars', 'size': 1000, 'overlap': 200},
                 n_ctx: int = 4096,
                 file_path: str = None,
                 endpoint: str = "http://localhost:1234/v1/chat/completions",
                 verbose: (str, int) = 'info',
                 ):

        # Set the logger
        set_logger(verbose)

        # Store data in self
        self.model = model
        self.context_strategy = context_strategy
        self.retrieval_method = retrieval_method
        self.alpha = alpha
        self.top_chunks = top_chunks
        self.temperature = temperature
        self.top_p = top_p
        self.endpoint = endpoint
        self.n_ctx = n_ctx
        self.context = None
        self.embedding = _set_embedding(embedding)
        self.tempdir = os.path.join(tempfile.gettempdir(), 'temp_LLMlight')
        # Single, authoritative path resolution — used everywhere
        self.store_path = self._resolve_file_path(file_path)

        # Make checks
        if model is None:
            models = self.get_available_models(validate=False)
            if models is not None:
                logger.info(f'Available models: {models}')
                logger.info(f'Set model before proceeding: Example: client = LLMlight(model="{models[0]}", endpoint="{endpoint}").')
                self.models = models
            return

        # Create tempdir
        os.makedirs(self.tempdir, exist_ok=True)

        # Set chunk parameters
        if chunks is None:
            chunks = {}
        self.chunks = {**{'method': 'chars', 'size': 1000, 'overlap': 250}, **chunks}

        # Load memory from disk when a store path was provided
        if self.store_path:
            self.memory_load(self.store_path)

        # Load a local GGUF model when the endpoint points to a file
        if os.path.isfile(self.endpoint):
            self.llm = load_local_gguf_model(self.endpoint, n_ctx=self.n_ctx)

        logger.info(f'Model: {self.model}')
        logger.info(f'Context Strategy: {self.context_strategy or "disabled"}')
        logger.info(f'Retrieval method: {self.retrieval_method or "disabled"}')
        logger.info(f'Embedding: {self.embedding or "disabled"}')
        logger.info('LLMlight is initialized!')

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
               verbose=None,
               ):
        """Run the model with the provided parameters.

        The final prompt is created based on the query, instructions, and the context

        Parameters
        ----------
        query : str
            The question or query.
            "What is the capital for France?"
        context : str
            Large text string that will be chunked, and embedded. The answer for the query is based on the chunks.
        instructions : str
            Set your instructions.
            "Answer the question strictly based on the provided context."
        system : str, optional
            Optional system message to set context for the AI (default is None).
            "You are helpfull assistant."
        temperature : float, optional
            Sampling temperature (default is 0.7).
        top_p : float, optional
            Top-p (nucleus) sampling parameter (default is 1.0, no filtering).
        stream : bool, optional
            Whether to enable streaming (default is False).
        return_type: bool, optional
            Return dictionary in case the output is a json
            'max': Output the full json
            'dict': Convert json into dictionary.
            'string': Return only the string answer (remove thinking strings using tags: <think> </think>).
            'string_with_thinking' Return the full response which includes the thinking proces (if available).

        Examples
        --------
        >>> # Examples
        >>> from LLMlight import LLMlight
        >>> client =  LLMlight()
        >>> client.prompt('hello, who are you?')
        >>> system_message = "You are a helpful assistant."
        >>> response = client.prompt('What is the capital of France?', system=system_message, top_p=0.9)
        >>> print(response)

        Returns
        -------
        str
            The model's response or an error message if the request fails.
        """
        if verbose is not None: set_logger(verbose)
        logger.info(f'Creating response with {self.model}..')

        if context is None: context = self.context
        if temperature is None: temperature = self.temperature
        if top_p is None: top_p = self.top_p
        self.task = 'max'

        # Set system message
        system = set_system_message(system)

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
            response = self.requests_post_gguf(prompt, system, temperature=temperature, top_p=top_p, task=self.task, stream=stream, return_type=return_type)
        else:
            # Run LLM with http model
            response = self.requests_post_http(prompt, system, temperature=temperature, top_p=top_p, task=self.task, stream=stream, return_type=return_type)

        # Return
        return response

    # def _get_context(self, context):
    #     # First get the context
    #     if context is None:
    #         context = self.context
    #     # if isinstance(context, dict):
    #     #     context = '\n\n'.join(context.values())
    #     return context

    def requests_post_gguf(self, prompt, system, temperature=0.8, top_p=1, headers=None, task='max', stream=False, return_type='string'):
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

    def requests_post_http(self, prompt, system, temperature=0.8, top_p=1, headers=None, task='max', stream=False, return_type='string', max_tokens=None):
        # Prepare data for request.
        if headers is None: headers = {"Content-Type": "application/json"}
        # Prepare messages
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        # Convert messages to string prompt
        prompt = convert_messages_to_model(messages, model=self.model)

        # Compute tokens
        if max_tokens is None:
            used_tokens, max_tokens = compute_tokens(prompt, n_ctx=self.n_ctx, task=task)
        # logger.info(f'Generating response with {self.model}')

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_tokens": max_tokens,
            }

        # Send POST request
        response = self.requests_post(headers, data, stream=stream, return_type=return_type)

        # Return
        return response

    def requests_post(self, headers, data, stream=False, return_type='string'):
        """Create the request to the LLM."""
        # Get response
        response = requests.post(self.endpoint, headers=headers, json=data, stream=stream)

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

    def memory_init(self, store_path: str = None, config: dict = None,
                    embedding: str = None, backend: str = None):
        """Prepare a memory store for writing.

        Call this when you want to create a *new* store or re-open an existing
        one to add more chunks.  After adding chunks, call :meth:`memory_save`
        to persist them.

        Parameters
        ----------
        store_path : str, optional
            Path to the store file.  Relative paths are resolved against the
            LLMlight temp directory.  Defaults to the path set at construction.
        config : dict, optional
            Backend-specific configuration.
        embedding : str, optional
            Override the memory embedding method for this store.
        backend : str, optional
            ``'sqlite'`` (default) or ``'memvid'``.
        """
        resolved = self._resolve_file_path(store_path) or self.store_path

        # Skip if the same store is already initialised
        if hasattr(self, 'memory') and self.memory.store_path == resolved:
            logger.info(f'Memory already initialised: {resolved}')
            return

        self.memory = memory.create_memory_backend(resolved, config=config, backend=backend)
        self.store_path = self.memory.store_path

        if embedding is not None:
            self.embedding['memory'] = embedding
            logger.info(f'Memory embedding updated: {self.embedding}')

    def memory_load(self, store_path: str = None, config: dict = None, backend: str = None):
        """Load an existing memory store from disk so it is ready for querying.

        Parameters
        ----------
        store_path : str, optional
            Path to the store file.  Defaults to the path set at construction.
        config : dict, optional
            Backend-specific configuration (reuses existing config if omitted).
        backend : str, optional
            Force a specific backend.  Usually not needed — the factory infers
            the backend from the file extension.
        """
        resolved = self._resolve_file_path(store_path) or self.store_path

        # Reuse existing config if none supplied
        if config is None and hasattr(self, 'memory') and hasattr(self.memory, 'config'):
            config = self.memory.config

        if not hasattr(self, 'memory'):
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
            logger.info('No memory store loaded — initialising from current store_path.')
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

    def memory_chunks(self, n: int = 10) -> list:
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
        logger.info(f'Returning {min(n, len(chunks))} of {len(chunks)} stored chunks.')
        return chunks[:n]

    def compute_probability(self, query, scores, embedding, n=5000):
        """Fit a null distribution over retrieval scores and return significance flags.

        Uses the *distfit* package.  Returns None when no memory store is loaded
        or *distfit* is not installed.
        """
        if not hasattr(self, 'memory'):
            logger.debug('No memory store loaded — skipping null distribution.')
            return None

        scores = np.asarray(scores)
        if len(scores) < 2:
            return None

        logger.info('Building null distribution for retrieval score significance testing.')

        # The memvid backend can provide its own similarity scores for the
        # full corpus, which gives a better null distribution.
        if self.embedding['memory'] == 'memvid':
            scored = self.memory.search_with_scores(query, top_k=n)
            random_scores = np.array([s for s, _ in scored])
            bound = 'left'
        else:
            random_chunks = self.memory.get_random_chunks(n=n)
            if not random_chunks:
                logger.debug('No random chunks available — skipping null distribution.')
                return None
            query_vector, chunk_vectors = self.fit_transform(query, random_chunks, embedding=embedding)
            random_scores = cosine_similarity(query_vector, chunk_vectors)[0]
            random_scores = random_scores[random_scores != 0]
            bound = 'right'

        try:
            from distfit import distfit
        except Exception as exc:
            raise ImportError(
                "The 'distfit' package is required for compute_probability. "
                "Install it with: pip install distfit"
            ) from exc

        model = distfit(method='parametric', alpha=self.alpha, bound=bound, verbose='warning')
        model.fit_transform(random_scores)
        results = model.predict(scores, alpha=self.alpha, todf=False, multtest='fdr_bh')

        fig, ax = model.plot(
            title=f'Retrieval: {self.retrieval_method}, Embedding: {embedding}'
        )
        self.distfit = model
        self.distfit.fig = fig
        self.distfit.ax = ax
        return results

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
            "Follow all instructions precisely."
            "Preserve important details, maintain logical flow, and respect any formatting requirements, such as using headings or bullet points when relevant.",
            "Output the final results in the same language as the instructions.",
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
                prompt = f"""### Context:
                Previous Results:\n{previous_results}" if response_list else "Previous Results: No results because this is the initial chunk."

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
                ### Context (Chunk {i+1} of {len(chunks)} — part of a larger document):
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
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, query: str, chunks: list, embedding: str) -> tuple:
        """Embed *query* and *chunks* using the requested *embedding* method.

        Parameters
        ----------
        query : str
        chunks : list of str
        embedding : str  — one of 'tfidf', 'bow', 'bert', 'bge-small'

        Returns
        -------
        (query_vector, chunk_vectors)
        """
        _SENTENCE_MODELS = {
            'bert':      'all-MiniLM-L6-v2',
            'bge-small': 'BAAI/bge-small-en',
        }

        if embedding in ('tfidf', 'bow'):
            Vectorizer = TfidfVectorizer if embedding == 'tfidf' else CountVectorizer
            vec = Vectorizer()
            chunk_vectors = vec.fit_transform(chunks)
            query_vector  = vec.transform([query])
            return query_vector, chunk_vectors

        if embedding in _SENTENCE_MODELS:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise ImportError(
                    f"sentence-transformers is required for embedding='{embedding}'. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            model = SentenceTransformer(_SENTENCE_MODELS[embedding])
            chunk_vectors = np.vstack([model.encode(c) for c in chunks])
            query_vector  = model.encode([query]).reshape(1, -1)
            return query_vector, chunk_vectors

        raise ValueError(
            f"Unsupported embedding method: '{embedding}'. "
            f"Valid options: {get_embeddings()}"
        )

    # ------------------------------------------------------------------
    # Core retrieval primitive  (single shared implementation)
    # ------------------------------------------------------------------

    def _retrieve(self,
                  query: str,
                  chunks: list,
                  embedding: str,
                  top_k: int) -> list:
        """Rank *chunks* by relevance to *query* and return the top-*top_k* ones.

        Returns a list of (score, text) tuples, highest score first.

        Parameters
        ----------
        query     : str
        chunks    : list of str
        embedding : str — embedding method name
        top_k     : int — number of results to return (capped at len(chunks))
        """
        if not chunks:
            return []

        top_k = min(top_k, len(chunks))
        query_vector, chunk_vectors = self._embed(query, chunks, embedding)
        scores = cosine_similarity(query_vector, chunk_vectors)[0]

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(float(scores[i]), chunks[i]) for i in top_indices]

    # ------------------------------------------------------------------
    # Statistical significance filter
    # ------------------------------------------------------------------

    def _filter_by_significance(self, query: str, scored: list) -> list:
        """Remove chunks whose retrieval score is not statistically significant.

        Parameters
        ----------
        query  : str
        scored : list of (score, text) tuples

        Returns
        -------
        list of (score, text) — filtered to significant results only,
        or the original list unchanged when alpha is None / filtering fails.
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
        logger.info("%d / %d chunks retained after significance filtering.", len(filtered), len(scored))
        return filtered

    # ------------------------------------------------------------------
    # Public retrieval entry points
    # ------------------------------------------------------------------

    def relevant_memory_retrieval(self, query: str, return_type: str = 'list'):
        """Return the top-k chunks from the persistent memory store.

        Parameters
        ----------
        query       : str
        return_type : 'list' (default) or 'string'

        Returns
        -------
        list of str, a joined str, or None when no store is loaded.
        """
        if not (hasattr(self, 'memory') and self.store_path and os.path.isfile(self.store_path)):
            return None

        logger.info(
            "Retrieving [%d] chunks from memory store (embedding='%s').",
            self.top_chunks, self.embedding['memory'],
        )

        if self.embedding['memory'] == 'memvid':
            # Backend provides its own similarity scores
            scored = self.memory.search_with_scores(query, top_k=self.top_chunks)
        elif self.embedding['memory'] in get_embeddings():
            all_chunks = self.memory.get_all_chunks()
            scored = self._retrieve(query, all_chunks, self.embedding['memory'], self.top_chunks)
        else:
            logger.warning("Unknown memory embedding '%s', skipping retrieval.", self.embedding['memory'])
            return None

        scored = self._filter_by_significance(query, scored)
        chunks = [text for _, text in scored]

        if return_type == 'string':
            return "\n\n---\n\n".join(f"### Chunk {i+1}:\n{c}" for i, c in enumerate(chunks))
        return chunks

    def relevant_context_retrieval(self, query: str, context: str, return_type: str = 'list'):
        """Return the most relevant portion of *context* for *query*.

        Parameters
        ----------
        query       : str
        context     : str — raw text to search through
        return_type : 'list' (default) or 'string'

        Returns
        -------
        list of str, str, or None when *context* is empty.
        """
        if not context:
            return context

        if self.retrieval_method == 'naive_rag' and self.embedding['context'] in get_embeddings():
            logger.info(
                "naive_rag: retrieving [%d] chunks from context (embedding='%s').",
                self.top_chunks, self.embedding['context'],
            )
            chunks = utils.chunk_text(
                context,
                method=self.chunks['method'],
                chunk_size=self.chunks['size'],
                overlap=self.chunks['overlap'],
            )
            scored = self._retrieve(query, chunks, self.embedding['context'], self.top_chunks)
            # Note: significance filtering uses memory embedding; skip for pure-context retrieval
            chunks_out = [text for _, text in scored]

            if return_type == 'string':
                return "\n\n---\n\n".join(f"### Chunk {i+1}:\n{c}" for i, c in enumerate(chunks_out))
            return chunks_out

        if self.retrieval_method == 'RSE' and self.embedding['context'] in ('bert', 'bge-small'):
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

        logger.info("No retrieval method applied — using full context.")
        return context

    # Public alias kept for any external callers that use the old names
    def search(self, query: str, chunks: list,
               return_type: str = 'score',
               top_chunks: int = None,
               embedding: str = None) -> list:
        """Rank *chunks* by relevance to *query*.

        Thin wrapper around :meth:`_retrieve` kept for backwards compatibility.

        Returns
        -------
        list — format depends on *return_type*:
            'score'       : list of (score, text) tuples  [default]
            'list'        : list of text strings
            'string_flat' : space-joined string
            str (other)   : newline-separated string
        """
        emb = embedding or 'tfidf'
        k   = top_chunks or len(chunks)
        scored = self._retrieve(query, chunks, emb, k)

        if return_type == 'score':
            return scored
        if return_type == 'list':
            return [t for _, t in scored]
        if return_type == 'string_flat':
            return " ".join(t for _, t in scored)
        return "\n---------\n".join(t for _, t in scored)

    # fit_transform kept for any code that calls it directly
    def fit_transform(self, query, chunks, embedding=None):
        """Embed query and chunks.  Use :meth:`_embed` in new code."""
        emb = embedding if isinstance(embedding, str) else (
            self.embedding.get('context', 'tfidf') if not isinstance(embedding, dict) else embedding.get('context', 'tfidf')
        )
        return self._embed(query, chunks, emb)

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

    # Legacy alias
    def _filter_proba(self, query, scores, relevant_context):
        """Backwards-compat wrapper — use _filter_by_significance in new code."""
        scored = list(zip(scores, relevant_context))
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
            context = wget.download(url, file_path)

        if os.path.isfile(file_path):
            # Read pdf
            context = utils.read_pdf(file_path, title_pages=title_pages, body_pages=body_pages, reference_pages=reference_pages, return_type=return_type)
            # if return_type=='dict':
            #     counts = utils.count_words(self.context['body'])
            #     self.context['body'] = self.context['body']
        else:
            logger.error(f'file_path does not exist: {file_path}')

        # Return
        return context

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
        base_url = '/'.join(self.endpoint.split('/')[:3]) + '/'
        logger.info(f'Collecting models at API endpoint: {self.endpoint}')
        models = None

        try:
            model_url = base_url.rstrip('/') + '/v1/models'
            response = requests.get(model_url, timeout=10)
            if response.status_code == 200:
                try:
                    get_models = response.json()["data"]
                    model_dict = {model["id"]: model for model in get_models}
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
                # logger.info(f'Checking: {key}')
                from LLMlight import LLMlight
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
def get_embeddings():
    return ['tfidf', 'bow', 'bert', 'bge-small', 'memvid']

def _set_embedding(embedding):
    if embedding is None: embedding = {}
    if isinstance(embedding, str):
        if embedding == 'automatic':
            embedding = {'memory': 'memvid', 'context': 'bert'}
        elif embedding in get_embeddings():
            embedding = {'memory': embedding, 'context': embedding}
        else:
            embedding = {'memory': 'memvid', 'context': 'bert'}

    if embedding.get('context') == 'memvid':
        embedding['context'] = 'tfidf'

    embedding = {**{'memory': 'memvid', 'context': 'tfidf'}, **embedding}

    # Return
    return embedding

#%%
def convert_messages_to_model(messages, model='llama', add_assistant_start=True):
    """
    Builds a prompt in the appropriate format for different models (LLaMA, Grok, Mistral).

    Args:
        messages (list of dict): Each dict must have 'role' ('system', 'user', 'assistant') and 'content'.
        model (str): The type of model to generate the prompt for ('llama', 'grok', or 'mistral').
        add_assistant_start (bool): Whether to add the assistant start (default True).
        add_bos_token (bool): Helps models know it's a fresh conversation. Useful for llama/mistral/hermes-style models

    Returns:
        str: The final prompt string in the correct format for the given model.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is the capital of France?"}
        ... ]
        >>> prompt = convert_messages_to_model(messages, model='llama')
         >>> print(prompt)

    """
    prompt = ""

    # if add_bos_token and ('llama' in model or 'mistral' in model):
    #     prompt += "<|begin_of_text|>\n"

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if 'llama' in model or 'mistral' in model:
            prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
        elif 'grok' in model:
            prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        else:
            # Default to ChatML format if model not recognized
            prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

    if add_assistant_start:
        if 'llama' in model or 'mistral' in model:
            prompt += "<|im_start|>assistant\n"
        elif 'grok' in model:
            prompt += "<start_of_turn>assistant\n"

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

def compute_tokens(string, n_ctx=4096, task='max'):
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise ImportError("transformers is required for token counting. Install via 'pip install transformers'") from e

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Tokenize the input string
    tokens = tokenizer.encode(string, truncation=True, max_length=n_ctx)
    # Get the number of tokens
    used_tokens = len(tokens)
    # Determine how many tokens are available for the model to generate
    max_tokens = compute_max_tokens(used_tokens, n_ctx=n_ctx, task=task)
    # Show message
    logger.debug(f"Used_tokens={used_tokens}, max_tokens={max_tokens}, context_limit={n_ctx}")
    # Return
    return used_tokens, max_tokens


def compute_max_tokens(used_tokens, n_ctx=4096, task="max"):
    """
    Compute the maximum number of tokens that can be generated for a given task,
    taking into account the number of tokens already used and the model's context window.

    Parameters
    ----------
    used_tokens : int
        Number of tokens already consumed in the current context.
    n_ctx : int, optional
        Total context window size of the model (default is 4096 tokens).
    task : str, optional
        Type of generation task. Determines the proportion of the remaining tokens to use.
        Options are:
        - "summarization": Use up to 50% of the context window, minimum 128 tokens.
        - "chat": Use up to 60% of the context window, minimum 128 tokens.
        - "code": Use up to 75% of the context window, minimum 128 tokens.
        - "longform": Use up to 90% of the context window, minimum 256 tokens.
        - "max": Use all remaining tokens.
        Any unrecognized task defaults to a safe fallback using 50% of the context window.

    Returns
    -------
    max_tokens : int
        Maximum number of tokens that can be generated for the specified task,
        ensuring at least a minimum number of tokens as defined per task type.
    """

    available_tokens = max(n_ctx - used_tokens, 1)  # Ensure at least 1

    task = task.lower()
    if task == "summarization":
        max_tokens = max(min(available_tokens, int(n_ctx * 0.5)), 128)
    elif task == "chat":
        max_tokens = max(min(available_tokens, int(n_ctx * 0.6)), 128)
    elif task == "code":
        max_tokens = max(min(available_tokens, int(n_ctx * 0.75)), 128)
    elif task == "longform":
        max_tokens = max(min(available_tokens, int(n_ctx * 0.9)), 256)
    elif task == "max":
        max_tokens = available_tokens
    else:
        # Default to safe fallback
        max_tokens = max(min(available_tokens, int(n_ctx * 0.5)), 128)

    return max_tokens


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


# %%
def convert_verbose_to_new(verbose):
    """Convert old verbosity to the new."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if not isinstance(verbose, str) and verbose<10:
        status_map = {
            'None': 'silent',
            0: 'silent',
            6: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'debug'}
        if verbose>=2: print('[LLMlight] WARNING use the standardized verbose status. The status [1-6] will be deprecated in future versions.')
        return status_map.get(verbose, 0)
    else:
        return verbose

def get_logger():
    return logger.getEffectiveLevel()


def set_logger(verbose: [str, int] = 'info'):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical', 'error']: Messages from critical level and higher.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Convert verbose to new
    verbose = convert_verbose_to_new(verbose)
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {'silent': 60,
                  'off': 60,
                  'no': 60,
                  'debug': 10,
                  'info': 20,
                  'warning': 30,
                  'error': 50,
                  'critical': 50}
        verbose = levels[verbose]

    # Configure root logger if no handlers exist
    # if not logger.handlers:
    #     handler = logging.StreamHandler()
    #     fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
    #     formatter = logging.Formatter(fmt=fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)

    # Set the level
    logger.setLevel(verbose)


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