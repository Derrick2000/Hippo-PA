import openai
import os
import pickle
from tqdm import tqdm

# Function to get embeddings using OpenAI's embedding model
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Read the book from a text file and split into chunks
# def load_and_split_book(file_path, chunk_size=22000):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def load_and_split_book(file_path, n=5):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]
    
    # Combine every N paragraphs into one chunk
    combined_chunks = ["\n\n".join(paragraphs[i:i+n]) for i in range(0, len(paragraphs), n)]
    
    return combined_chunks

def load_or_compute_embeddings(embedding_path, book_chunks):
    # Check if embeddings file exists6
    if os.path.exists(embedding_path):
        with open(embedding_path, "rb") as f:
            print("Loading embeddings from file...")
            return pickle.load(f)
    
    # Compute embeddings if file doesn't exist
    chunk_embeddings = [get_embedding(chunk) for chunk in tqdm(book_chunks, desc="Generating Embeddings")]
    
    # Save embeddings to file
    with open(embedding_path, "wb") as f:
        pickle.dump(chunk_embeddings, f)

    return chunk_embeddings 