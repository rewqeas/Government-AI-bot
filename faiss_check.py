import faiss  
import numpy as np
from gemini_client import genai,model
from document_embedder import pdf_loader, chunk_file
import os
import pickle
import re

merged_embeddings = np.load("merged_embeddings.npy")

print(merged_embeddings.shape)

with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)


def preprocess_query(query: str) -> str:
    """
    LOCAL translation of Nepali-English mixed text to plain English.
    NO API CALLS to avoid rate limits!
    """
    
    # Dictionary for common Nepali/Roman Nepali to English translations
    nepali_to_english = {
        # Common words
        'malai': 'I need',
        'mero': 'my',
        'timro': 'your', 
        'hamro': 'our',
        'tapai': 'you',
        'ma': 'I',
        
        # Verbs
        'garnu': 'to do',
        'garne': 'doing',
        'gareko': 'done',
        'garera': 'by doing',
        'parne': 'need to',
        'parcha': 'needed',
        'pareko': 'needed',
        'parera': 'by getting',
        'huncha': 'will be',
        'hune': 'to be',
        'bhayo': 'happened',
        'thiyo': 'was',
        
        # Question words
        'kasari': 'how',
        'kaha': 'where', 
        'kahile': 'when',
        'kati': 'how much',
        'kun': 'which',
        'ke': 'what',
        'ko': 'of',
        'ka': 'of',
        
        # Common phrases
        'cha': 'is',
        'chha': 'is',
        'chaina': 'is not',
        'chaincha': 'is needed',
        'chahincha': 'is needed',
        'bataideu': 'please tell',
        'bhannu': 'to say',
        'dekhau': 'show',
        'sikau': 'teach',
        
        # Government terms
        'nagarikta': 'citizenship',
        'nagrita': 'citizenship',
        'pramanpatra': 'certificate',
        'janma': 'birth',
        'sahayog': 'help',
        'karyalaya': 'office',
        'sarkar': 'government',
        'shulka': 'fee',
        'dastavej': 'documents',
        'kagajat': 'documents',
        'form': 'form',
        'apply': 'apply',
        'renew': 'renew',
        
        # Common suffixes and particles
        'lai': '',  # Remove particle
        'ma': 'in',
        'bata': 'from',
        'dekhi': 'from',
        'samma': 'until',
    }
    
    # Clean and normalize the query
    query = query.lower().strip()
    
    # Remove Devanagari script if present (keep only Roman text)
    query = re.sub(r'[\u0900-\u097F]', '', query)
    
    # Split into words
    words = query.split()
    translated_words = []
    
    for word in words:
        # Clean word (remove punctuation)
        clean_word = re.sub(r'[^\w]', '', word)
        
        # Try exact match first
        if clean_word in nepali_to_english:
            translated_words.append(nepali_to_english[clean_word])
        # Try partial matches for compound words
        else:
            translated = clean_word
            for nepali, english in nepali_to_english.items():
                if nepali in clean_word and len(nepali) > 2:  # Avoid short false matches
                    translated = translated.replace(nepali, english)
            translated_words.append(translated)
    
    # Join and clean up the result
    result = ' '.join(translated_words)
    
    # Clean up extra spaces and common patterns
    result = re.sub(r'\s+', ' ', result)  # Multiple spaces to single
    result = result.replace('I need I need', 'I need')  # Remove duplicates
    result = result.replace('is needed is needed', 'is needed')
    result = result.strip()
    
    # If translation didn't change much, return original (likely already English)
    if len(result) == 0 or result == query:
        return query
    
    return result


def embed_query(query: str):
    # If already a vector, just return it
    if isinstance(query, np.ndarray):
        return query.astype("float32")
    
    # LOCAL preprocessing - no API call!
    processed_query = preprocess_query(query)
    
    if not processed_query.strip():
        return np.zeros(768, dtype="float32")  # fallback

    try:
        # Only ONE API call for embedding
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=processed_query,
            task_type="retrieval_query"
        )
        vec = np.array(res["embedding"], dtype="float32")
        return vec
    except Exception as e:
        print(f"Error generating embeddings for query: {e}")
        return np.zeros(768, dtype="float32")

    

def faiss_index(embeddings):
    faiss.normalize_L2(embeddings)

    #build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_similar_chunk(query, chunks, index, top_k=5):
    query_embed = embed_query(query).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embed)
    distances, indices = index.search(query_embed, top_k)
    print("chunks length:", len(chunks))
    print("indices:", indices)

    valid_indices = [i for i in indices[0] if 0 <= i < len(chunks)]
    if len(valid_indices) < len(indices[0]):
        print("Warning: Some FAISS indices out of range:", indices[0])

    top_chunks = [chunks[i] for i in valid_indices]

    return top_chunks