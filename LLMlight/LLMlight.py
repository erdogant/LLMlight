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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .RAG import RAG_with_RSE
from . import utils
# DEBUG
# import utils
# from RAG import RAG_with_RSE

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

    Parameters
    ----------
    modelname : str
        'hermes-3-llama-3.2-3b'
        'mistral-7b-grok'
        'openhermes-2.5-mistral-7b'
        'gemma-2-9b-it'
    system : str
        String of the system message.
        "I am a helpfull assistant"
    method : str
        'naive_RAG': Simple RAG. Chunk text in fixed parts. Use cosine similarity to for ranking. The top scoring chunks will be combined (n chunks) and used as input with the prompt.
        'global_reasoning': Apply a a two steps proces, first the user question is re-formulated into a more generic question so that text chunks can be summarized more accurately. The total combined summarized context is then used as the new context for the user question following the rest of the pipeline.
        'RSE': Identify and extract entire segments of relevant text.
    embedding : str
        None
        'tfidf': Best use when it is a structured documents and the words in the queries are matching.
        'bow': Bag of words approach. Best use when you expect words in the document and queries to be matching.
        'bert': Best use when document is more free text and the queries may not match exactly the words or sentences in the document.
        'bge-small':
    temperature : float, optional
        Sampling temperature (default is 0.7).
    top_p : float, optional
        Top-p (nucleus) sampling parameter (default is 1.0, no filtering).
    task (str): The use case for generation. (default: 'full')
        'summarization'
        'chat'
        'code'
        'longform'
        'full'
    chunks: dict : {'type': 'words', 'size': 250, 'n': 5}
        type : str
            'words': Chunks are created using words.
            'chars': Chunks are created using chars.
        size : str
            The chunk size is measured in words. When lower chunk sizes are taken, the cosine similarity will increase in accuracy but the smaller chunk size reduces the input context for the LLM.
            Note that 1000 words (~10.000 chars) costs ~3000 tokens. Thus with a context window (n_ctx) of 4096 your can set chunk size=100 words with n chunks=10.
            256: Create chunks every 256 words.
            512: Create chunks every 512 words.
        n : int, optional
            Top scoring chunks to be used in the context of the prompt (all with length chunk size).
    endpoint : str
        Endpoint of the LLM API
        "http://localhost:1234/v1/chat/completions"
        './models/Hermes-3-Llama-3.2-3B.Q4_K_M.gguf'
        r'C:/Users/username/.lmstudio/models/lmstudio-community/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf'
    n_ctx : int, default: 4096
        The context window length is determined by the max tokens. A larger number of tokens will ask more cpu/gpu resources.
        Note that 1000 words (~10.000 chars) costs ~3000 tokens. Thus with a context window (n_ctx) of 4096 your can set chunk size=100 words with n=10.

    Examples
    --------
    >>> model = LLMlight()
    >>> model.prompt('hello, who are you?')
    >>> system_message = "You are a helpful assistant."
    >>> response = model.prompt('What is the capital of France?', system=system_message, top_p=0.9)

    """
    def __init__(self,
                 modelname="hermes-3-llama-3.2-3b",
                 method='naive_RAG',
                 embedding='bert',
                 temperature=0.7,
                 top_p=1.0,
                 task='full',
                 chunks={'type': 'words', 'size': 250, 'n': 5},
                 endpoint="http://localhost:1234/v1/chat/completions",
                 n_ctx=4096,
                 verbose='info',
                 ):

        # Set the logger
        set_logger(verbose)
        # Store data in self
        self.modelname = modelname
        self.endpoint = endpoint
        self.temperature = temperature
        self.top_p = top_p
        self.task = task
        self.method = method
        self.embedding = embedding
        if chunks is None: chunks = {'type': 'words', 'size': None, 'n': None}
        self.chunks = {**{'type': 'words', 'size': 250, 'n': 5}, **chunks}
        self.n_ctx = n_ctx
        self.context = None

        # Set the correct name for the model.
        if embedding == 'bert':
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif embedding == 'bge-small':
            self.embedding_model = SentenceTransformer('BAAI/bge-small-en')
        else:
            self.embedding_model = None
        # Load local model
        if os.path.isfile(self.endpoint):
            self.llm = load_local_gguf_model(self.endpoint, n_ctx=self.n_ctx)

    def check_logger(self):
        """Check the verbosity."""
        logger.debug('DEBUG')
        logger.info('INFO')
        logger.warning('WARNING')
        logger.critical('CRITICAL')

    def get_available_models(self, validate=False):
        # Set your local API base URL
        # base_url = "http://localhost:1234/v1"
        base_url = self.endpoint.split("/chat")[0]
        logger.info('Collecting models in the API endpoint..')

        # Query available models
        response = requests.get(f"{base_url}/models")
        model_dict = {}

        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=10)
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
            logger.info("Validating models:")
            keys = copy.deepcopy(list(model_dict.keys()))

            for key in keys:
                from LLMlight import LLMlight
                llm = LLMlight(modelname=key)
                response = llm.prompt('What is the capital of France?', system="You are a helpful assistant.", return_type='string')
                response = response[0:30].replace('\n', ' ').replace('\r', ' ').lower()
                if 'error: 404' in response:
                    logger.error(f"{llm.modelname}: {response}")
                    model_dict.pop(key)
                else:
                    logger.info(f"{llm.modelname}: {response}")

        return list(model_dict.keys())

    def prompt(self,
            query,
            instructions=None,
            system=None,
            context=None,
            response_format=None,
            temperature=None,
            top_p=None,
            stream=False,
            return_type='string',
            ):
        """
        Run the model with the provided parameters.
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
            'string': Return only the string output

        Returns
        -------
        str
            The model's response or an error message if the request fails.
        """
        logger.debug(f'{self.modelname} is loaded..')
        headers = {"Content-Type": "application/json"}

        if temperature is None: temperature = self.temperature
        if top_p is None: top_p = self.top_p
        if context is None: context = self.context
        # if embedding is not None: self.embedding = embedding
        if isinstance(context, dict): context = '\n\n'.join(context.values())
        if self.method=='global_reasoning' and context is None:
            logger.error('Context must be provided when using method="global_reasoning"')
            return None

        # Set system message
        system = set_system_message(system)
        # Extract relevant text using retrieval method
        relevant_text = self.relevant_text_retrieval(query, context)
        # Set the prompt
        prompt = self.set_prompt(query, instructions, response_format, relevant_text)
        # Prepare messages
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        # Run model
        if os.path.isfile(self.endpoint):
            # Run LLM from gguf model
            response = self.requests_post_gguf(messages, temperature, top_p, headers, task=self.task, stream=stream, return_type=return_type)
        else:
            # Run LLM with http model
            response = self.requests_post_http(messages, temperature, top_p, headers, task=self.task, stream=stream, return_type=return_type)

        # Return
        return response

    def requests_post_gguf(self, messages, temperature, top_p, headers, task='full', stream=False, return_type='string'):
        # Note that it is better to use messages_prompt instead of a dict (messages_dict) because most GGUF-based models don't have a tokenizer/parser that can interpret the JSON-style message structure.
        # Convert messages to string prompt
        prompt = convert_prompt(messages, modelname=self.modelname)
        # Prepare data for request.
        used_tokens = compute_used_tokens(prompt, n_ctx=self.n_ctx)
        # Determine how many tokens are available for the model to generate
        max_tokens = compute_max_tokens(used_tokens, self.n_ctx, task=task)

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
        if return_type == 'string':
            response = response.get('choices', [{}])[0].get('text', "No response")

        # Return
        return response

    def requests_post_http(self, messages, temperature, top_p, headers, task='full', stream=False, return_type='string'):
        # Prepare data for request.
        used_tokens = compute_used_tokens(messages[0]['content'] + messages[1]['content'], n_ctx=self.n_ctx)
        # Determine how many tokens are available for the model to generate
        max_tokens = compute_max_tokens(used_tokens, n_ctx=self.n_ctx, task=task)

        data = {
            "model": self.modelname,
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
                elif return_type == 'string':
                    return response_text
                else:
                    return response.json()
            except:
                return response_text
        else:
            logger.error(f"Error: {response.status_code} - {response}")
            return f"Error: {response.status_code} - {response}"

    def task(self,
             query="Extract key insights while maintaining coherence of the previous summaries.",
             instructions="",
             system="You are a helpfull assistant.",
             response_format="**comprehensive, structured document covering all key insights**",
             task='question',
             context=None,
             chunks={'type': 'words', 'size': None, 'n': None},
             return_type='string',
             ):
        """
        Analyze the large text in an iterative, coherent manner.
        - Each chunk is processed while keeping track of previous summaries.
        - After all chunks are processed, a final coherent text is made.
        - The query can for example be to summarize the text or to extract key insights.

        """
        chunks = {**{'type': 'words', 'size': None, 'n': None}, **chunks}
        if chunks['size'] is None: chunks['size'] = self.chunks['size']
        if chunks['type'] is None: chunks['type'] = self.chunks['type']
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
        chunks = utils.chunk_text(context, chunks['size'], method=chunks['type'])

        # Build a structured prompt that includes all previous summaries
        results_list = []
        for i, chunk in enumerate(chunks):
            logger.info(f'Working on text chunk {i}/{len(chunks)}')

            # Keep last N summaries for context (this needs to be within the context-window otherwise it will return an error.)
            previous_results = "\n---\n".join(results_list[-self.chunks['n']:])

            prompt = (
            "### Context:\n"
            + (f"Previous results:\n{previous_results}\n" if len(results_list) > 0 else "")

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
            chunk_result = self.query_llm(prompt, system=system)
            results_list.append(f"Results {i+1}:\n" + chunk_result)

        # Final summarization pass over all collected summaries
        results_total = "\n---\n".join(results_list[-self.chunks['n']:])
        final_prompt = f"""
        ### Context:
        {results_total}

        ### Task:
        Connect the result parts in context into a **coherent, well-structured document**.

        ### Instructions:
        - Maintain as much as possible the key insights but ensure logical flow.
        - Connect insights smoothly while keeping essential details intact.
        - Do **not** use any external knowledge or assumptions.

        {response_format}

        f"### {task}:\n"
        {query}\n\n

        Begin your response below:
        """
        logger.info('Combining all information to create a single coherent output..')
        # Create the final summary.
        final_result = self.query_llm(final_prompt, system=system, return_type=return_type)
        # Return
        return final_result
        # return {'summary': final_result, 'summary_per_chunk': results_total}


    def query_llm(self, prompt, system=None, return_type='string'):
        """Calls the LLM and returns the response."""
        headers = {"Content-Type": "application/json"}
        if system is None: system = "You are a helpful assistant."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        data = {"model": self.modelname, "messages": messages, "temperature": self.temperature, "top_p": self.top_p}

        # Send POST request
        response = self.requests_post(headers, data, stream=False, return_type=return_type)

        # Return
        return response

    def global_reasoning(self, query, context):
        """Global Reasoning.
            1. Rewrite the input user question into something like: "Based on the extracted summaries, does the document explain the societal relevance of the research? Justify your answer."
            2. Break the document into manageable chunks with overlapping parts to make sure we do not miss out.
            3. Create a global reasoning question based on the input user question.
            4. Take the summarized outputs and aggregate them.

            prompt = "Is the proposal well thought out?"
            instructions = "Your task is to rewrite questions for global reasoning. As an example, if there is a question like: 'Does this document section explain the societal relevance of the research?', the desired output would be: 'Does this document section explain the societal relevance of the research? If so, summarize it. If not, return 'No societal relevance found.''"
            response = model.llm.prompt(query=prompt, instructions=instructions, task='Task')

        """

        # Initialize model for question refinement and summarization
        qmodel = LLMlight(modelname=self.modelname, temperature=0.7, endpoint=self.endpoint)

        # if rewrite_query:
        #     # 1. Rewrite user question in global reasoning question.
        #     logger.info('Rewriting user question for global reasoning..')
        #     instructions = """In the context are chunks of text from a document.
        #     Rewrite the user question in such a way that relevant information can be captured by a Large language model for summarization for the chunks of text in the context.
        #     Only return the new question with no other information.
        #     """
        #     # Create new query
        #     new_query = qmodel.prompt(query=query, instructions=instructions)
        # else:
        #     new_query = query

        # Create chunks with overlapping parts to make sure we do not miss out
        chunks = utils.chunk_text(context, chunk_size=6000, method='chars', overlap=1000)

        # Now summaries for the chunks
        instructions = """- Base your answer **strictly** on the provided text.
        - Do **not** use any external knowledge or assumptions.
        - Determine the language of the context and answer in the **exact** same language.
        - If the answer is not available, Return **ONLY**: 'N/A'
        """

        summaries = []
        for chunk in chunks:
            prompt = f"""Context:
                {chunk}

                User question: Make a an extensive and complete summary.
                """

            response = qmodel.prompt(query=prompt, instructions=instructions)
            summaries.append(response)

        # Final summarization pass over all collected summaries
        # Filter out "N/A" summaries
        summaries = [s for s in summaries if s.strip() != "N/A"]
        summaries_final = "\n\n---\n\n".join(summaries)
        # Return
        return summaries_final


    def parse_large_document(self, query, context, top_chunks=3, chunk_size=512, return_type='string'):
        """Splits large text into chunks and finds the most relevant ones."""
        # Create chunks
        chunks = utils.chunk_text(context, chunk_size=chunk_size, method='words')
        # Embedding
        query_vector, chunk_vectors = self.fit_transform(query, chunks)
        # Compute similarity
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        # Get top scoring chunks
        if top_chunks is None: top_chunks = len(similarities)
        top_indices = np.argsort(similarities)[-top_chunks:][::-1]

        # Join relevant chunks and send as prompt
        relevant_chunks = [chunks[i] for i in top_indices]
        relevant_scores = [similarities[i] for i in top_indices]

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
            raise ValueError("Unsupported embedding method. Choose a supported embedding method.")
        return query_vector, chunk_vectors


    def relevant_text_retrieval(self, query, context):
        # Create advanced prompt using relevant chunks of text, the input query and instructions
        if context is not None:
            if self.method=='global_reasoning':
                # Global Reasoning
                relevant_chunks = self.global_reasoning(query, context)
            elif self.method == 'naive_RAG' and np.isin(self.embedding, ['tfidf', 'bow', 'bert', 'bge-small']):
                # Find the best matching parts using simple retrieval method approach.
                logger.info(f'[{self.method}] approach is applied with [{self.embedding}] embedding.')
                relevant_chunks = self.parse_large_document(query, context, chunk_size=self.chunks['size'], top_chunks=self.chunks['n'], return_type='string')
            elif self.method == 'RSE' and np.isin(self.embedding, ['bert', 'bge-small']):
                logger.info(f'RAG approach [{self.method}] is applied.')
                relevant_chunks = RAG_with_RSE(context, query, label=None, chunk_size=self.chunks['size'], irrelevant_chunk_penalty=0, embedding=self.embedding, device='cpu', batch_size=32)
            else:
                logger.info(f'No chunking: The entire text is used as context.')
        else:
            # Default
            relevant_chunks = context

        # Return
        return relevant_chunks

    def set_prompt(self, query, instructions, response_format, context):
        # Default and update when context and instructions are available.
        prompt = (
            ("Context:\n" + context + "\n\n" if context else "")
            + ("Instructions:\n" + instructions + "\n\n" if instructions else "")
            + ("Response format:\n" + response_format + "\n\n" if response_format else "")
            + f"User question:\n"
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
            self.context = utils.read_pdf(filepath, title_pages=title_pages, body_pages=body_pages, reference_pages=reference_pages, return_type=return_type)
            if self.context is None:
                logger.error('No input text gathered. <return>')
                return
            if return_type=='dict':
                counts = utils.count_words(self.context['body'])
                self.context['body'] = self.context['body'] + f"\n---\nThe exact word count in this document is: {counts}"
        else:
            logger.warning(f'{filepath} does not exist.')
            self.context = None


def convert_prompt(messages, modelname='llama', add_assistant_start=True):
    """
    Builds a prompt in the appropriate format for different models (LLaMA, Grok, Mistral).

    Args:
        messages (list of dict): Each dict must have 'role' ('system', 'user', 'assistant') and 'content'.
        modelname (str): The type of model to generate the prompt for ('llama', 'grok', or 'mistral').
        add_assistant_start (bool): Whether to add the assistant start (default True).
        add_bos_token (bool): Helps models know it's a fresh conversation. Useful for llama/mistral/hermes-style models

    Returns:
        str: The final prompt string in the correct format for the given model.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is the capital of France?"}
        ... ]
        >>> prompt = convert_prompt(messages, modelname='llama')
         >>> print(prompt)

    """
    prompt = ""

    # if add_bos_token and ('llama' in modelname or 'mistral' in modelname):
    #     prompt += "<|begin_of_text|>\n"

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if 'llama' in modelname or 'mistral' in modelname:
            prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
        elif 'grok' in modelname:
            prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        else:
            # Default to ChatML format if model not recognized
            prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

    if add_assistant_start:
        if 'llama' in modelname or 'mistral' in modelname:
            prompt += "<|im_start|>assistant\n"
        elif 'grok' in modelname:
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

def compute_used_tokens(string, n_ctx=4096):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Tokenize the input string
    tokens = tokenizer.encode(string, truncation=True, max_length=n_ctx)
    # Get the number of tokens
    return len(tokens)


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
    return "You are a helpful assistant." if system is None else system



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


def check_logger(verbose: [str, int] = 'info'):
    """Check the logger."""
    set_logger(verbose)
    logger.debug('DEBUG')
    logger.info('INFO')
    logger.warning('WARNING')
    logger.critical('CRITICAL')
