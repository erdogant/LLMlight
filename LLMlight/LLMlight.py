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
from llama_cpp import Llama
from transformers import AutoTokenizer
import copy
import re
from tqdm import tqdm

from typing import List, Union

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from memvid import MemvidEncoder, MemvidRetriever
# from memvid.config import get_default_config as memvid_get_default_config

# from .RAG import RAG_with_RSE
# from . import utils
# DEBUG
import utils
from RAG import RAG_with_RSE

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
    -----------------
    LLMlight processes text through several key stages to generate intelligent responses:

    1. Preprocessing Stage
    ---------------------
    The input context can be processed in different ways:
    - No preprocessing: Uses the raw context directly
    - Chunk-wise processing: Breaks down the context into manageable chunks, processes each chunk independently, and combines results
    - Global reasoning: Creates a global summary of the context before processing

    2. Retrieval Method Stage
    ------------------------
    Three main approaches for retrieving relevant information:
    - Naive RAG: Splits text into chunks and uses similarity scoring to find the most relevant sections
    - RSE (Relevant Segment Extraction): Identifies and extracts complete relevant text segments
    - No retrieval: Uses the entire context directly

    3. Embedding Stage
    -----------------
    Multiple embedding options for text representation:
    - TF-IDF: Best for structured documents with matching query terms
    - Bag of Words: Simple word frequency approach
    - BERT: Advanced contextual embeddings for free-form text
    - BGE-small: Efficient embedding model for general use

    4. Prompting Stage
    -----------------
    The system constructs prompts by combining:
    - System message: Defines the AI's role and behavior
    - Context: Processed and retrieved relevant information
    - User query: The specific question or request
    - Instructions: Additional guidance for response generation

    5. Response Generation
    ---------------------
    The model generates responses using:
    - Temperature control: Adjusts response randomness (0.7 default)
    - Top-p sampling: Controls response diversity
    - Context window management: Handles token limits efficiently

    The system can be configured through various parameters to optimize for different use cases, from simple Q&A to complex document analysis.

    Processing Flow
    --------------
    The system follows a sequential processing flow where each stage builds upon the previous one. First, the input context undergoes preprocessing, where it can be either used as-is or transformed into chunks for more manageable processing. These chunks are then passed through the retrieval method stage, which determines how relevant information is extracted and organized.
    During the embedding stage, the text is converted into numerical representations that capture its semantic meaning. This is crucial for the system to understand and process the content effectively. The embedding method chosen can significantly impact the system's ability to match queries with relevant content.
    The prompting stage brings together all the processed information, combining it with the user's query and any specific instructions. This creates a comprehensive prompt that guides the model in generating an appropriate response. The final response generation stage uses this prompt to create a coherent and relevant output, with parameters like temperature and top-p sampling helping to control the response's characteristics.
    Throughout this process, the system maintains flexibility through various configuration options, allowing it to adapt to different types of queries and contexts. This modular approach enables the system to handle everything from simple questions to complex document analysis tasks efficiently.


    Parameters
    ----------
    model : str
        'hermes-3-llama-3.2-3b'
        'mistral-7b-grok'
        'openhermes-2.5-mistral-7b'
        'gemma-2-9b-it'
    system : str
        String of the system message.
        "I am a helpfull assistant"
    retrieval_method : str (default: 'naive_RAG')
         None:              No processing. The entire context is used for the query.
        'naive_RAG':        Ideal for chats and when you need to answer specfic questions: Chunk of text are created. Use cosine similarity to for ranking. The top scoring chunks will be combined (n chunks) and used as input with the prompt.
        'RSE':              Identify and extract entire segments of relevant text.
    embedding : str (default: 'bert')
        None
        'tfidf': Best use when it is a structured documents and the words in the queries are matching.
        'bow': Bag of words approach. Best use when you expect words in the document and queries to be matching.
        'bert': Best use when document is more free text and the queries may not match exactly the words or sentences in the document.
        'bge-small':
    preprocessing : str (default: None)
         None:              No pre-processing. The original context is used in the pipeline of retrieval_method, embedding and the response.
        'chunk-wise':       The input context will be analyze chunkwise based on the query, instructions and system. The total set of answered-chunks is then returned. The normal pipeline proceeds for the query, instructions, system etc.
        'global-reasoning': The input context will be summarized per chunk globally. The total set of summarized context is then returned. The normal pipeline proceeds for the query, instructions, system etc.
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
    endpoint : str
        Endpoint of the LLM API
        "http://localhost:1234/v1/chat/completions"
        './models/Hermes-3-Llama-3.2-3B.Q4_K_M.gguf'
        r'C:/Users/username/.lmstudio/models/lmstudio-community/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf'
    n_ctx : int, default: 4096
        The context window length is determined by the max tokens. A larger number of tokens will ask more cpu/gpu resources. Estimates: 1000 words or ~10.000 chars costs ~3000 tokens.

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
                 model: str = "hermes-3-llama-3.2-3b",
                 retrieval_method: str = 'naive_RAG',
                 embedding: str = 'bert',
                 preprocessing: str = None,
                 temperature: (int, float) = 0.7,
                 top_p: (int, float) = 1.0,
                 chunks: dict = {'method': 'chars', 'size': 1024, 'overlap': 200, 'top_chunks': 5},
                 endpoint: str = "http://localhost:1234/v1/chat/completions",
                 n_ctx: int = 4096,
                 path_to_memory: str = None,
                 verbose: (str, int) = 'info',
                 ):

        # Set the logger
        set_logger(verbose)

        # Store data in self
        self.model = model
        self.preprocessing = preprocessing
        self.retrieval_method = retrieval_method
        self.embedding = embedding
        self.temperature = temperature
        self.top_p = top_p
        self.endpoint = endpoint
        self.n_ctx = n_ctx
        self.context = None

        # Set chunk parameters
        if chunks is None: chunks = {}
        self.chunks = {**{'method': 'chars', 'size': 1000, 'overlap': 250, 'top_chunks': 5}, **chunks}

        # Memvid default parameters
        self.memory_video_file = None
        self.memory_index_file = None
        self.memory_config = None
        # Update parameters when memory video memory exists.
        if path_to_memory is not None:
            self.memory_init(path_to_memory)

        # Set embedding model parameters.
        if embedding == 'bert':
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif embedding == 'bge-small':
            self.embedding_model = SentenceTransformer('BAAI/bge-small-en')
        else:
            self.embedding_model = None

        # Load local LLM gguf model
        if os.path.isfile(self.endpoint):
            self.llm = load_local_gguf_model(self.endpoint, n_ctx=self.n_ctx)
        logger.info('LLMlight is initialized.')

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
            'full': Output the full json
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
        logger.info(f'{self.model} is loaded..')

        if temperature is None: temperature = self.temperature
        if top_p is None: top_p = self.top_p
        self.task = 'full'

        # Set system message
        system = set_system_message(system)
        # Get context
        context = self._get_context(context)
        # Extract relevant text for video memory
        context = self.get_video_memory(query, context)
        # Preprocessing on the context
        processed_context = self.compute_preprocessing(query, context, instructions, system)
        # Extract relevant text using retrieval method
        relevant_context = self.relevant_text_retrieval(query, processed_context, instructions, system)
        # Set the prompt
        logger.debug(relevant_context)
        prompt = self.set_prompt(query, instructions, relevant_context, response_format=response_format)
        logger.info('Running model now..')

        # Run model
        if os.path.isfile(self.endpoint):
            # Run LLM from gguf model
            response = self.requests_post_gguf(prompt, system, temperature=temperature, top_p=top_p, task=self.task, stream=stream, return_type=return_type)
        else:
            # Run LLM with http model
            response = self.requests_post_http(prompt, system, temperature=temperature, top_p=top_p, task=self.task, stream=stream, return_type=return_type)
        # Return
        return response

    def _get_context(self, context):
        # First get the context
        if context is None:
            context = self.context
        if isinstance(context, dict):
            context = '\n\n'.join(context.values())
        return context

    def requests_post_gguf(self, prompt, system, temperature=0.8, top_p=1, headers=None, task='full', stream=False, return_type='string'):
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

    def requests_post_http(self, prompt, system, temperature=0.8, top_p=1, headers=None, task='full', stream=False, return_type='string', max_tokens=None):
        # Prepare data for request.
        if headers is None: headers = {"Content-Type": "application/json"}
        # Prepare messages
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        # Convert messages to string prompt
        prompt = convert_messages_to_model(messages, model=self.model)

        # Create full prompt
        prompt = messages[0]['content'] + messages[1]['content']
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
            except:
                return response
        else:
            logger.error(f"{response.status_code} - {response}")
            return f"Error: {response.status_code} - {response}"

    def memory_init(self, path_to_memory: str = "llmlight_memory.mp4", config: dict = None):
        """Build QR code video and index from chunks with unified codec handling.

        Parameters
        ----------
        path_to_memory : str
            Path to output video memory file.
        config : dict
            Dictionary containing configuration parameters.

        """
        if os.path.isfile(path_to_memory):
            logger.info(f'Initializing existing video memory: {path_to_memory}')
        else:
            logger.info(f'Initializing new video memory: {path_to_memory}')

        # Set memory path
        self._set_memory_path(path_to_memory)

        # Initialize new encoder
        self.encoder = MemvidEncoder(config=config)
        # Update config from memvid Encoder
        self.memory_config = self.encoder.config

    def memory_add(self,
                   text: Union[str, List[str]] = None,
                   input_files: Union[str, List[str]] = None,
                   dirpath: str = None,
                   filetypes: List[str] = ['.pdf', '.txt', '.epub', '.md', '.doc', '.docx', '.rtf', '.html', '.htm'],
                   overwrite=False):
        """Add chunks to memory.

        Parameters
        ----------
        input_files : (str, list)
            Path to file(s).

        """
        # Make checks
        if not hasattr(self, 'encoder'):
            logger.error('Memory is not yet initialized. Use client.memory_init() first')
            raise AssertionError('Memory is not yet initialized. Use client.memory_init() first')
        if os.path.isfile(self.memory_video_file) and not overwrite:
            logger.warning(f'Video memory already exists appending is not possible: {self.memory_video_file}')
            return

        # Make lists
        if isinstance(text, str): text = [text]
        if isinstance(input_files, str): input_files = [input_files]

        if dirpath is not None and os.path.isdir(dirpath):
            if input_files is None: input_files = []
            for root, _, files in os.walk(dirpath):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in filetypes):
                        input_files.append(os.path.join(root, file))

        # Add text chunk to video-memory
        if text is not None:
            logger.info(f'Adding {len(text)} text chunks to memory.')
            self.encoder.add_chunks(text)

        # Run over all files
        if input_files is not None:
            logger.info(f'Adding {len(input_files)} into memory.')

            for filepath in input_files:
                if not os.path.isfile(filepath):
                    logger.warning(f"File not found: {filepath}")
                else:
                    # full filename with extension
                    filename = os.path.basename(filepath)
                    # split name and extension
                    name, ext = os.path.splitext(filename.lower())
                    # Add to encoder
                    if (ext == '.pdf') and (ext in filetypes):
                        self.encoder.add_pdf(filepath, chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
                    elif ext == '.epub' and (ext in filetypes):
                        self.encoder.add_epub(filepath, chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
                    elif ext == '.txt' and (ext in filetypes):
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            self.encoder.add_text(text, chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
                    elif ext in ['.html', '.htm']:
                        # Process HTML with BeautifulSoup
                        try:
                            from bs4 import BeautifulSoup
                        except ImportError:
                            print(f"Warning: BeautifulSoup not available for HTML processing. Skipping {filepath}")
                            continue
 
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            soup = BeautifulSoup(f.read(), 'html.parser')
                            for script in soup(["script", "style"]):
                                script.decompose()
                            text = soup.get_text()
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            clean_text = ' '.join(chunk for chunk in chunks if chunk)
                            if clean_text.strip():
                                self.encoder.add_text(clean_text, chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])

                    elif ext in filetypes:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            self.encoder.add_text(text, chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
                    else:
                        continue
                    # Show message
                    logger.info(f'Added to memory: {filename}')

    def memory_save(self,
                    filepath: str = None,
                    codec: str = 'mp4v',
                    auto_build_docker: bool = True,
                    allow_fallback: bool = True,
                    overwrite: bool = False,
                    show_progress: bool = True,
                    ):
        """Build QR code video and index from chunks with unified codec handling.

        Parameters
        ----------
        filepath : str (default is the initialization memory-path)
            Path to output video memory file.
        codec : str, optional
            Video codec ('mp4v', 'h265', 'h264', etc.)
            'mp4v': Default
        auto_build_docker : bool (default: True)
            Whether to auto-build Docker if needed.
        allow_fallback : bool
            Whether to fall back to MP4V if advanced codec fails.
        show_progress : bool (default: True)

        """
        # Make checks
        if not hasattr(self, 'encoder') or len(self.encoder.chunks) == 0:
            logger.error('No chunks to encode. Use client.add_chunks() first')
            raise AssertionError('No chunks to encode. Use client.add_chunks() first')

        if filepath is not None:
            self._set_memory_path(filepath)

        # Check
        if os.path.isfile(self.memory_video_file) and not overwrite:
            logger.warning(f'File already exists and not allowed to overwrite: {self.memory_video_file}')
            return

        # Remove files when overwrite
        if overwrite:
            if os.path.isfile(self.memory_video_file):
                logger.info(f'Video memory file is overwriten: {self.memory_video_file}')
                os.remove(self.memory_video_file)
            # Also remove the index file
            if os.path.isfile(self.memory_index_file):
                os.remove(self.memory_index_file)

        logger.info(f'Building video-memory for {len(self.encoder.chunks)} chunks: {self.memory_video_file}')
        # Build the by passing all parameters to the building proces
        self.encoder.build_video(output_file=self.memory_video_file,
                                 index_file=self.memory_index_file,
                                 codec=codec,
                                 show_progress=show_progress,
                                 auto_build_docker=auto_build_docker,
                                 allow_fallback=allow_fallback,
                                 )

        # Clear all chunks of text from list because it will use the video memory file.
        self.encoder.chunks = []
        logger.info('✅ Video Memory saved to disk.')

    def _set_memory_path(self, path_to_memory):
        """Set Memory paths."""
        # Get the absolute filepath
        filepath = os.path.abspath(path_to_memory)
        # Get directory path (folder)
        directory = os.path.dirname(filepath)
        # full filename with extension
        filename = os.path.basename(filepath)
        # split name and extension
        name, extension = os.path.splitext(filename)
        # Store file names
        self.memory_video_file = filepath
        self.memory_index_file = os.path.join(directory, name) + '.json'

    def task(self,
             query="Extract key insights while maintaining coherence of the previous summaries.",
             instructions="",
             system="You are a helpfull assistant.",
             response_format="**comprehensive, structured document covering all key insights**",
             task='question',
             context=None,
             return_type='string',
             ):
        """
        Analyze the large text in an iterative, coherent manner.
        - Each chunk is processed while keeping track of previous summaries.
        - After all chunks are processed, a final coherent text is made.
        - The query can for example be to summarize the text or to extract key insights.

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

        logger.info(f'Processing the document for the given task..')

        # Create chunks based on words
        chunks = utils.chunk_text(context, method=self.chunks['method'], chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])

        # Build a structured prompt that includes all previous summaries
        response_list = []
        for i, chunk in enumerate(chunks):
            logger.info(f'Working on text chunk {i}/{len(chunks)}')

            # Keep last N summaries for context (this needs to be within the context-window otherwise it will return an error.)
            previous_results = "\n---\n".join(response_list[-self.chunks['top_chunks']:])

            prompt = (
            "### Context:\n"
            + (f"Previous results:\n{previous_results}\n" if len(response_list) > 0 else "")

            + "\n---\nNew text chunk (Part of a larger document, maintain context):\n"
            + f"{chunk}\n\n"

            "### Instructions:\n"
            + "- Extract key insights from the **new text chunk** while maintaining coherence with **Previous summaries**.\n"
            + f"{instructions}\n\n"

            f"### {task}:\n"
            f"{query}\n\n"

            "### Improved Results:\n"
            )

            # Get the summary for the current chunk
            # chunk_result = self.query_llm(prompt, system=system)
            chunk_result= self.requests_post_http(prompt, system, temperature=self.temperature, top_p=self.top_p, task='full', stream=False, return_type='string')

            response_list.append(f"Results {i+1}:\n" + chunk_result)

        # Final summarization pass over all collected summaries
        results_total = "\n---\n".join(response_list[-self.chunks['top_chunks']:])
        final_prompt = f"""
        ### Context:
        {results_total}

        ### Task:
        Connect the result parts in context into a **coherent, well-structured document**.

        ### Instructions:
        - Maintain as much as possible the key insights but ensure logical flow.
        - Connect insights smoothly while keeping essential details intact.

        {response_format}

        f"### {task}:\n"
        {query}\n\n

        Begin your response below:
        """
        logger.info('Combining all information to create a single coherent output..')
        # Create the final summary.
        # final_result = self.query_llm(final_prompt, system=system, return_type=return_type)
        final_result = self.requests_post_http(final_prompt, system, temperature=self.temperature, top_p=self.top_p, task='full', stream=False, return_type=return_type)
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
        chunks = utils.chunk_text(context, method=self.chunks['method'], chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
        logger.info(f'Total number of chunks created: {len(chunks)}')

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
            response = self.requests_post_http(prompt, system, temperature=self.temperature, top_p=self.top_p, task='summarization', stream=stream, return_type='string')
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
        chunks = utils.chunk_text(context, method=self.chunks['method'], chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
        logger.info(f'Total number of chunks created: {len(chunks)}')

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

    def compute_distances(self, query: str, chunks: list, return_type: str = 'string', top_chunks: int = None):
        """Splits large text into chunks and finds the most relevant ones."""
        # Embedding
        query_vector, chunk_vectors = self.fit_transform(query, chunks)
        # Compute similarity
        D = cosine_similarity(query_vector, chunk_vectors)[0]
        # Get top scoring chunks
        if top_chunks is None:
            top_chunks = len(D)
            logger.info('Number of top chunks selected is set to: {top_chunks}')

        # Top indices
        top_indices = np.argsort(D)[-top_chunks:][::-1]
        # Join relevant chunks and send as prompt
        relevant_chunks = [chunks[i] for i in top_indices]
        relevant_scores = [D[i] for i in top_indices]

        # Set the return type
        if return_type == 'score':
            return list(zip(relevant_scores, relevant_chunks))
        elif return_type == 'list':
            return relevant_chunks
        elif return_type == 'string_flat':
            return " ".join(relevant_chunks)
        else:
            return "\n---------\n".join(relevant_chunks)

    def fit_transform(self, query, chunks):
        """Converts context chunks and query into vector space representations based on the selected embedding method."""
        if self.embedding == 'tfidf':
            vectorizer = TfidfVectorizer()
            chunk_vectors = vectorizer.fit_transform(chunks)
            # dense_matrix = chunk_vectors.toarray()  # Converts to a NumPy array
            query_vector = vectorizer.transform([query])
        elif self.embedding == 'bow':
            vectorizer = CountVectorizer()
            chunk_vectors = vectorizer.fit_transform(chunks)
            query_vector = vectorizer.transform([query])
        # elif self.embedding_model is not None:
        elif self.embedding == 'bert' or self.embedding == 'bge-small':
            chunk_vectors = np.vstack([self.embedding_model.encode(chunk) for chunk in chunks])
            query_vector = self.embedding_model.encode([query])
            query_vector = query_vector.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported embedding method: {self.embedding}")
        # Return
        return query_vector, chunk_vectors

    def compute_preprocessing(self, query, context, instructions, system):
        # Create advanced prompt using relevant chunks of text, the input query and instructions
        if context is not None:
            if self.preprocessing=='global-reasoning':
                # Global Reasoning
                relevant_context = self.global_reasoning(query, context, instructions, system, rewrite_query=False, return_per_chunk=True)
            elif self.preprocessing=='chunk-wise':
                # Analyze per chunk
                relevant_context = self.chunk_wise(query, context, instructions, system, top_chunks=0, return_per_chunk=True)
            else:
                logger.info(f'No preprocessing method is applied.')
                relevant_context = context
        else:
            # Default
            relevant_context = context

        # Return
        return relevant_context

    def get_video_memory(self, query, context):
        # Get context from video Memory
        if hasattr(self, 'encoder') and len(self.encoder.chunks) > 0:
            logger.warning('Documents are stored in the encoder but not saved into video memory! Use save first: client.memory_save() to include the information.')

        if (self.memory_video_file is not None) and os.path.isfile(self.memory_video_file) and os.path.isfile(self.memory_index_file):
            logger.info(f"Text retrieval from video memory for [{self.chunks['top_chunks']}] chunks.")
            # Initialize retriever
            retriever = MemvidRetriever(video_file=self.memory_video_file, index_file=self.memory_index_file, config=self.memory_config)

            # self.embedding = 'bow'
            # self.embedding = 'tfidf'
            # Get relevant chunks
            # chunks = retriever.index_manager.metadata
            # chunks = list(map(lambda x: x.get('text'), retriever.index_manager.metadata))
            # Compute distances and get top k chunks
            # memory_chunks = self.compute_distances(query, chunks, top_chunks=self.chunks['top_chunks'], return_type='list')

            memory_chunks = retriever.search(query, top_k=self.chunks['top_chunks'])
            # memory_chunks = retriever.index_manager.search(query, top_k=self.chunks['top_chunks'])
            # Append context
            if context is not None and context != '': memory_chunks.append(context)
            # Join the chunks in context
            relevant_context = "\n\n---\n\n".join([f"### Chunk {i+1}:\n{s}" for i, s in enumerate(memory_chunks)])
            # Return
            return relevant_context
        else:
            return context

    def relevant_text_retrieval(self, query, context, instructions, system):
        if context is not None:
            # Create advanced prompt using relevant chunks of text, the input query and instructions
            if self.retrieval_method == 'naive_RAG' and np.isin(self.embedding, ['tfidf', 'bow', 'bert', 'bge-small']):
                # Find the best matching parts using simple retrieval method approach.
                logger.info(f'[{self.retrieval_method}] method is applied with [{self.embedding}] embedding for text retrieval.')
                # Create chunks
                chunks = utils.chunk_text(context, method=self.chunks['method'], chunk_size=self.chunks['size'], overlap=self.chunks['overlap'])
                # Compute distances and get top k chunks
                relevant_context = self.compute_distances(query, chunks, top_chunks=self.chunks['top_chunks'], return_type='string')
            elif self.retrieval_method == 'RSE' and np.isin(self.embedding, ['bert', 'bge-small']):
                logger.info(f'RAG approach [{self.retrieval_method}] is applied.')
                relevant_context = RAG_with_RSE(context, query, label=None, chunk_size=self.chunks['size'], irrelevant_chunk_penalty=0, embedding=self.embedding, device='cpu', batch_size=32)
            else:
                logger.info(f'No retrieval method is applied.')
                relevant_context = context
        else:
            relevant_context = context

        # Return
        return relevant_context

    def set_prompt(self, query, instructions, context, response_format=''):
        # Default and update when context and instructions are available.
        prompt = (
            ("Context:\n" + context + "\n\n" if context else "")
            + ("Instructions:\n" + instructions + "\n\n" if instructions not in ("", None) else "")
            + ("Response format:\n" + response_format + "\n\n" if response_format not in ("", None) else "")
            + "User question:\n"
            + query
        )

        # Return
        return prompt

    def read_pdf(self, filepath, title_pages=[1, 2], body_pages=[], reference_pages=[-1], return_type='dict'):
        """
        Reads a PDF file and extracts its text content as a string.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.

        """
        if os.path.isfile(filepath):
            # Read pdf
            context = utils.read_pdf(filepath, title_pages=title_pages, body_pages=body_pages, reference_pages=reference_pages, return_type=return_type)
            # Return
            return context
            # if return_type=='dict':
            #     counts = utils.count_words(self.context['body'])
            #     self.context['body'] = self.context['body']
        else:
            logger.error(f'Filepath does not exist: {filepath}')

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
        >>> client = LLMlight(endpoint='http://localhost:1234/v1/chat/completions', verbose='info')
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
        logger.info('Collecting models in the API endpoint..')

        try:
            model_url = base_url.rstrip('/') + '/v1/models'
            logger.info(f'Looking up models: {model_url}')
            response = requests.get(model_url, timeout=10)
            if response.status_code == 200:
                try:
                    models = response.json()["data"]
                    model_dict = {model["id"]: model for model in models}
                except (KeyError, ValueError) as e:
                    logger.error("Error parsing model data:", e)
            else:
                logger.warning("Request failed with status code:", response.status_code)
                logger.warning("Response:", response.text)

        except requests.exceptions.RequestException as e:
            logger.error("Request error:", e)

        # Check each model whether it returns a response
        if validate:
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

        return list(model_dict.keys())

    def check_logger(self):
        """Check the verbosity."""
        logger.debug('DEBUG')
        logger.info('INFO')
        logger.warning('WARNING')
        logger.critical('CRITICAL')

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

def compute_tokens(string, n_ctx=4096, task='full'):
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


def compute_max_tokens(used_tokens, n_ctx=4096, task="full"):
    """
    Compute the number of tokens that can be generated based on the task type.

    Parameters:
    - used_tokens (int): Tokens already used in the input prompt.
    - n_ctx (int): Total context window size of the model.
    - task (str): The use case for generation. Options:
        'summarization', 'chat', 'code', 'longform', 'full'

    Returns:
    - max_tokens (int): Tokens allowed for generation.
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
    elif task == "full":
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
        2. Be clear about what information comes from the knowledge base vs. your general knowledge
        3. If the context doesn't contain enough information, say so clearly
        4. Provide helpful, accurate, and concise responses

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

