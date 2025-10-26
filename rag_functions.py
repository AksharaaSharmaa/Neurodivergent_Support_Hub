import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
from typing import List, Dict, Tuple
import re
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from youtube_transcript_api import YouTubeTranscriptApi

# Initialize embedding model (do this once at startup)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, any]]:
    """
    Split text into overlapping chunks for better context retrieval.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of dictionaries with chunk text and metadata
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'start_pos': len(''.join([c['text'] for c in chunks])) if chunks else 0
                })
                chunk_id += 1
            
            # Start new chunk with overlap
            words = current_chunk.split()
            overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence + " "
    
    # Add the last chunk
    if current_chunk:
        chunks.append({
            'id': chunk_id,
            'text': current_chunk.strip(),
            'start_pos': len(''.join([c['text'] for c in chunks])) if chunks else 0
        })
    
    return chunks

def create_embeddings(chunks: List[Dict[str, any]]) -> np.ndarray:
    """
    Create embeddings for text chunks using SentenceTransformer.
    
    Args:
        chunks: List of chunk dictionaries
    
    Returns:
        Numpy array of embeddings
    """
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    
    # Use IndexFlatL2 for exact search (good for smaller datasets)
    # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
    index = faiss.IndexFlatL2(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings.astype('float32'))
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    return index

def cosine_similarity_search(query: str, index: faiss.Index, chunks: List[Dict[str, any]], 
                            top_k: int = 5) -> List[Tuple[Dict[str, any], float]]:
    """
    Search for most similar chunks using cosine similarity.
    
    Args:
        query: Query text
        index: FAISS index
        chunks: List of chunk dictionaries
        top_k: Number of top results to return
    
    Returns:
        List of tuples (chunk_dict, similarity_score)
    """
    # Create query embedding
    query_embedding = embedding_model.encode([query], show_progress_bar=False)
    faiss.normalize_L2(query_embedding)
    
    # Search index
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # Convert L2 distances to cosine similarity scores
    # After normalization, L2 distance = 2(1 - cosine_similarity)
    # So: cosine_similarity = 1 - (L2_distance / 2)
    similarity_scores = 1 - (distances[0] / 2)
    
    # Prepare results
    results = []
    for idx, score in zip(indices[0], similarity_scores):
        if idx < len(chunks):  # Ensure valid index
            results.append((chunks[idx], float(score)))
    
    return results

def retrieve_context(query: str, index: faiss.Index, chunks: List[Dict[str, any]], 
                     top_k: int = 3, similarity_threshold: float = 0.3) -> str:
    """
    Retrieve relevant context for a query using RAG.
    
    Args:
        query: Query text
        index: FAISS index
        chunks: List of chunk dictionaries
        top_k: Number of top results to retrieve
        similarity_threshold: Minimum similarity score to include
    
    Returns:
        Concatenated context string
    """
    results = cosine_similarity_search(query, index, chunks, top_k)
    
    # Filter by similarity threshold
    relevant_chunks = [chunk for chunk, score in results if score >= similarity_threshold]
    
    # Concatenate relevant chunks
    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    return context

def save_rag_components(index: faiss.Index, chunks: List[Dict[str, any]], 
                       filepath: str = "rag_components.pkl"):
    """
    Save FAISS index and chunks to disk.
    
    Args:
        index: FAISS index
        chunks: List of chunk dictionaries
        filepath: Path to save file
    """
    # Save FAISS index
    faiss.write_index(index, f"{filepath}.faiss")
    
    # Save chunks
    with open(f"{filepath}.chunks", 'wb') as f:
        pickle.dump(chunks, f)

def load_rag_components(filepath: str = "rag_components.pkl") -> Tuple[faiss.Index, List[Dict[str, any]]]:
    """
    Load FAISS index and chunks from disk.
    
    Args:
        filepath: Path to saved file
    
    Returns:
        Tuple of (index, chunks)
    """
    # Load FAISS index
    index = faiss.read_index(f"{filepath}.faiss")
    
    # Load chunks
    with open(f"{filepath}.chunks", 'rb') as f:
        chunks = pickle.load(f)
    
    return index, chunks

def hybrid_search(query: str, index: faiss.Index, chunks: List[Dict[str, any]], 
                 top_k: int = 5, semantic_weight: float = 0.7) -> List[Tuple[Dict[str, any], float]]:
    """
    Perform hybrid search combining semantic (FAISS) and keyword matching.
    
    Args:
        query: Query text
        index: FAISS index
        chunks: List of chunk dictionaries
        top_k: Number of results to return
        semantic_weight: Weight for semantic search (1-weight for keyword)
    
    Returns:
        List of tuples (chunk_dict, combined_score)
    """
    # Semantic search
    semantic_results = cosine_similarity_search(query, index, chunks, top_k * 2)
    
    # Keyword matching (simple BM25-like scoring)
    query_terms = set(query.lower().split())
    keyword_scores = {}
    
    for chunk in chunks:
        chunk_terms = set(chunk['text'].lower().split())
        # Calculate Jaccard similarity
        intersection = query_terms.intersection(chunk_terms)
        union = query_terms.union(chunk_terms)
        keyword_score = len(intersection) / len(union) if union else 0
        keyword_scores[chunk['id']] = keyword_score
    
    # Combine scores
    combined_results = {}
    for chunk, semantic_score in semantic_results:
        chunk_id = chunk['id']
        keyword_score = keyword_scores.get(chunk_id, 0)
        combined_score = (semantic_weight * semantic_score + 
                         (1 - semantic_weight) * keyword_score)
        combined_results[chunk_id] = (chunk, combined_score)
    
    # Sort by combined score
    sorted_results = sorted(combined_results.values(), 
                          key=lambda x: x[1], 
                          reverse=True)
    
    return sorted_results[:top_k]

def rerank_results(query: str, results: List[Tuple[Dict[str, any], float]], 
                  rerank_top_k: int = 3) -> List[Tuple[Dict[str, any], float]]:
    """
    Rerank top results using cross-encoder for better relevance.
    Note: Requires cross-encoder model installation
    
    Args:
        query: Query text
        results: Initial search results
        rerank_top_k: Number of results to rerank
    
    Returns:
        Reranked results
    """
    from sentence_transformers import CrossEncoder
    
    # Initialize cross-encoder (do this once in practice)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Prepare pairs for reranking
    pairs = [(query, chunk['text']) for chunk, _ in results[:rerank_top_k]]
    
    # Get reranking scores
    rerank_scores = cross_encoder.predict(pairs)
    
    # Combine with original results
    reranked = [(results[i][0], float(score)) for i, score in enumerate(rerank_scores)]
    reranked.extend(results[rerank_top_k:])
    
    return sorted(reranked, key=lambda x: x[1], reverse=True)

def expand_query(query: str, transcript: str, top_terms: int = 5) -> str:
    """
    Expand query with related terms from transcript for better retrieval.
    
    Args:
        query: Original query
        transcript: Full transcript text
        top_terms: Number of related terms to add
    
    Returns:
        Expanded query string
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    try:
        # Fit on transcript
        tfidf_matrix = vectorizer.fit_transform([transcript, query])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get query terms
        query_vector = tfidf_matrix[1].toarray()[0]
        
        # Get top terms
        top_indices = query_vector.argsort()[-top_terms:][::-1]
        expansion_terms = [feature_names[i] for i in top_indices if query_vector[i] > 0]
        
        # Expand query
        expanded = query + " " + " ".join(expansion_terms)
        return expanded
    except:
        return query

def get_chunk_context(chunk: Dict[str, any], chunks: List[Dict[str, any]], 
                     context_window: int = 1) -> str:
    """
    Get surrounding context for a chunk.
    
    Args:
        chunk: Target chunk dictionary
        chunks: All chunks
        context_window: Number of chunks before/after to include
    
    Returns:
        Expanded context string
    """
    chunk_id = chunk['id']
    start_id = max(0, chunk_id - context_window)
    end_id = min(len(chunks), chunk_id + context_window + 1)
    
    context_chunks = chunks[start_id:end_id]
    return "\n\n".join([c['text'] for c in context_chunks])