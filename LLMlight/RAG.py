import pymupdf
import os
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
import logging

from . import utils
# import utils

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """
    A lightweight vector store implementation using NumPy.
    """
    def __init__(self, dimension=1536):
        """
        Initialize the vector store.
        
        Args:
            dimension (int): Dimension of embeddings
        """
        self.dimension = dimension
        self.vectors = []
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents, vectors=None, metadata=None):
        """
        Add documents to the vector store.
        
        Args:
            documents (List[str]): List of document chunks
            vectors (List[List[float]], optional): List of embedding vectors
            metadata (List[Dict], optional): List of metadata dictionaries
        """
        if vectors is None:
            vectors = [None] * len(documents)
        
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        
        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(vec)
            self.metadata.append(meta)
    
    def search(self, query_vector, top_k=5):
        """
        Search for most similar documents.
        
        Args:
            query_vector (List[float]): Query embedding vector
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: List of results with documents, scores, and metadata
        """
        if not self.vectors or not self.documents:
            return []
        
        # Convert query vector to numpy array
        query_array = np.array(query_vector)
        
        # Compute similarity
        # similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        # # Get top scoring chunks
        # if top_chunks is None: top_chunks = len(similarities)
        # top_indices = np.argsort(similarities)[-top_chunks:][::-1]
        # # Join relevant chunks and send as prompt
        # relevant_chunks = [chunks[i] for i in top_indices]
        # relevant_scores = [similarities[i] for i in top_indices]

        # Calculate similarities
        similarities = []
        for i, vector in enumerate(self.vectors):
            if vector is not None:
                # Compute cosine similarity
                similarity = np.dot(query_array, vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(vector)
                )
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                "score": float(score),
                "metadata": self.metadata[i]
            })
        
        return results


def RAG_with_RSE(context, query, label=None, chunk_size=800, irrelevant_chunk_penalty=0.2, embedding_method='bert', device='cpu', batch_size=16):
    """
    Complete RAG pipeline with Relevant Segment Extraction.

    Args:
        text (str):
            Text (context)
            Path to the PDF document
        query (str): User query
        chunk_size (int): Size of chunks
        irrelevant_chunk_penalty (float): Penalty for irrelevant chunks
        embedding_method:
            "bge":          The large retrieval-optimized embedding model for search or ranking
            "bge-small":    The small retrieval-optimized embedding model for search or ranking
            "bert":         General-purpose sentence embedding model

    Returns:
        Dict: Result with query, segments, and response
    """
    print("\n=== STARTING RAG WITH RELEVANT SEGMENT EXTRACTION ===")
    print(f"Query: {query}")

    if embedding_method == 'bge':
        embedding_method = "BAAI/bge-en-icl"
    elif embedding_method == 'bge-small':
        embedding_method = "BAAI/bge-small-en"
    elif embedding_method == 'bert':
        embedding_method = "all-MiniLM-L6-v2"
    else:
        raise Exception(f'The embedding method [{embedding_method}] is not valid.')

    # Process the document to extract text, chunk it, and create embeddings
    chunks, vector_store, doc_info = process_document(context, label=label, chunk_size=chunk_size, embedding_method=embedding_method, device=device, batch_size=batch_size)

    # Calculate relevance scores and chunk values based on the query
    print("\nCalculating relevance scores and chunk values...")
    chunk_values = calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty)

    # Find the best segments of text based on chunk values
    best_segments, scores = find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2)

    # Reconstruct text segments from the best chunks
    print("\nReconstructing text segments from chunks...")
    segments = reconstruct_segments(chunks, best_segments)

    # Format the segments into a context string for the language model
    context = format_segments_for_context(segments)

    return context

    # # Generate a response from the language model using the context
    # response = generate_response(query, context)
    # # Compile the result into a dictionary
    # result = {
    #     "query": query,
    #     "segments": segments,
    #     "response": response
    # }
    # print("\n=== FINAL RESPONSE ===")
    # print(response)
    # return result


def create_embeddings(texts, model_path="BAAI/bge-small-en", batch_size=32, device="cpu"):
    """
    Generate embeddings using a local model.

    Args:
        texts (List[str]): List of texts to embed
        model_path (str): Path or name of the local embedding model
        batch_size: int : 16 when you run out of memory. Default: 32
        embedding_method:
            "BAAI/bge-en-icl":      The large version of retrieval-optimized embedding model for search or ranking.
            "BAAI/bge-small-en":    The smaller version of retrieval-optimized embedding model for search or ranking.
            "all-MiniLM-L6-v2":     general-purpose sentence embedding model
        device: str: "cuda" or 'cpu'

    Returns:
        List[List[float]]: List of embedding vectors
    """
    if not texts:
        return []

    # Load the model once and reuse it
    model = SentenceTransformer(model_path, device=device)

    # Compute embeddings
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_list=True)
    
    # Return the list of all embeddings
    return embeddings

# def create_embeddings(texts, model="BAAI/bge-en-icl"):
#     """
#     Generate embeddings for texts.
    
#     Args:
#         texts (List[str]): List of texts to embed
#         model (str): Embedding model to use
        
#     Returns:
#         List[List[float]]: List of embedding vectors
#     """
#     if not texts:
#         return []  # Return an empty list if no texts are provided
        
#     # Process in batches if the list is long
#     batch_size = 100  # Adjust based on your API limits
#     all_embeddings = []  # Initialize a list to store all embeddings
    
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]  # Get the current batch of texts
        
#         # Create embeddings for the current batch using the specified model
#         response = client.embeddings.create(
#             input=batch,
#             model=model
#         )
        
#         # Extract embeddings from the response
#         batch_embeddings = [item.embedding for item in response.data]
#         all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list
        
#     return all_embeddings  # Return the list of all embeddings

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and prints the first `num_chars` characters.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = pymupdf.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string

    return all_text  # Return the extracted text

def process_document(context, label=None, chunk_size=800, embedding_method="all-MiniLM-L6-v2", batch_size=16, device='cpu'):
    """
    Process a document for use with RSE.
    
    Args:
        context (str): 
            Text (context)
            The Path to the PDF document
        label (str):
            Add label to identify where the specific text originates from.
        chunk_size (int): Size of each chunk in characters
        embedding_method:
            "BAAI/bge-en-icl":      The large version of retrieval-optimized embedding model for search or ranking.
            "BAAI/bge-small-en":    The smaller version of retrieval-optimized embedding model for search or ranking.
            "all-MiniLM-L6-v2":     bert: general-purpose sentence embedding model

    Returns:
        Tuple[List[str], SimpleVectorStore, Dict]: Chunks, vector store, and document info
    """
    print("Extracting context from document...")
    # Extract text from the PDF file
    if os.path.isfile(context):
        label = context
        context = extract_text_from_pdf(context)
        # text = utils.read_pdf(pdf_path, return_type='string')

    print("Chunking context into non-overlapping segments...")
    # Chunk the extracted text into non-overlapping segments
    chunks = utils.chunk_text(context, chunk_size=chunk_size, overlap=0, method='chars')
    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings for chunks...")
    # Generate embeddings for the context chunks
    chunk_embeddings = create_embeddings(chunks, model_path=embedding_method, batch_size=batch_size, device=device)

    # Create an instance of the SimpleVectorStore
    vector_store = SimpleVectorStore()

    # Add documents with metadata (including chunk index for later reconstruction)
    metadata = [{"chunk_index": i, "source": label} for i in range(len(chunks))]
    vector_store.add_documents(chunks, chunk_embeddings, metadata)

    # Track original document structure for segment reconstruction
    doc_info = {
        "chunks": chunks,
        "source": label,
    }

    return chunks, vector_store, doc_info


def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    """
    Calculate chunk values by combining relevance and position.
    
    Args:
        query (str): Query text
        chunks (List[str]): List of document chunks
        vector_store (SimpleVectorStore): Vector store containing the chunks
        irrelevant_chunk_penalty (float): Penalty for irrelevant chunks
        
    Returns:
        List[float]: List of chunk values
    """
    # Create query embedding
    query_embedding = create_embeddings([query])[0]
    
    # Get all chunks with similarity scores
    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)
    
    # Create a mapping of chunk_index to relevance score
    relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}
    
    # Calculate chunk values (relevance score minus penalty)
    chunk_values = []
    for i in range(num_chunks):
        # Get relevance score or default to 0 if not in results
        score = relevance_scores.get(i, 0.0)
        # Apply penalty to convert to a value where irrelevant chunks have negative value
        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)
    
    return chunk_values


def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    Find the best segments using a variant of the maximum sum subarray algorithm.
    
    Args:
        chunk_values (List[float]): Values for each chunk
        max_segment_length (int): Maximum length of a single segment
        total_max_length (int): Maximum total length across all segments
        min_segment_value (float): Minimum value for a segment to be considered
        
    Returns:
        List[Tuple[int, int]]: List of (start, end) indices for best segments
    """
    print("Finding optimal continuous text segments...")
    
    best_segments = []
    segment_scores = []
    total_included_chunks = 0
    
    # Keep finding segments until we hit our limits
    while total_included_chunks < total_max_length:
        best_score = min_segment_value  # Minimum threshold for a segment
        best_segment = None
        
        # Try each possible starting position
        for start in range(len(chunk_values)):
            # Skip if this start position is already in a selected segment
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue
                
            # Try each possible segment length
            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length
                
                # Skip if end position is already in a selected segment
                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue
                
                # Calculate segment value as sum of chunk values
                segment_value = sum(chunk_values[start:end])
                
                # Update best segment if this one is better
                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)
        
        # If we found a good segment, add it
        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            print(f"Found segment {best_segment} with score {best_score:.4f}")
        else:
            # No more good segments to find
            break
    
    # Sort segments by their starting position for readability
    best_segments = sorted(best_segments, key=lambda x: x[0])
    
    return best_segments, segment_scores


def reconstruct_segments(chunks, best_segments):
    """
    Reconstruct text segments based on chunk indices.
    
    Args:
        chunks (List[str]): List of all document chunks
        best_segments (List[Tuple[int, int]]): List of (start, end) indices for segments
        
    Returns:
        List[str]: List of reconstructed text segments
    """
    reconstructed_segments = []  # Initialize an empty list to store the reconstructed segments
    
    for start, end in best_segments:
        # Join the chunks in this segment to form the complete segment text
        segment_text = " ".join(chunks[start:end])
        # Append the segment text and its range to the reconstructed_segments list
        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })
    
    return reconstructed_segments  # Return the list of reconstructed text segments


def format_segments_for_context(segments):
    """
    Format segments into a context string for the LLM.
    
    Args:
        segments (List[Dict]): List of segment dictionaries
        
    Returns:
        str: Formatted context text
    """
    context = []  # Initialize an empty list to store the formatted context
    
    for i, segment in enumerate(segments):
        # Create a header for each segment with its index and chunk range
        segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
        context.append(segment_header)  # Add the segment header to the context list
        context.append(segment['text'])  # Add the segment text to the context list
        context.append("-" * 80)  # Add a separator line for readability
    
    # Join all elements in the context list with double newlines and return the result
    return "\n\n".join(context)
