#%%

import json
import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import numpy as np

def vectorize_query_retrieve(user_query, embedding_model, faiss_index, cursor):
    """
    Vectorize query and retrieve relevant chunks from FAISS
    """
    # 1. Vectorize query
    query_vector = embedding_model.encode(user_query)
    query_vector = query_vector.reshape(1, -1).astype('float32')
    
    # 2. Search FAISS
    k = 3
    distances, indices = faiss_index.search(query_vector, k)
    
    # 3. Display results
    print(f"\nQuery: '{user_query}'")
    print(f"Query vector (first 10 dims): {query_vector[0][:10]}\n")
    
    chunks = []
    for idx, distance in zip(indices[0], distances[0]):
        sqlite_id = int(idx) + 1
        cursor.execute("SELECT text FROM chunks WHERE id = ?", (sqlite_id,))
        result = cursor.fetchone()
        if result:
            chunks.append(result[0])
            print(f"Distance: {distance:.4f} | {result[0][:100]}...")  # Show first 100 chars
    
    return chunks

def embed_add(chunk_dict, embedding_model, faiss_index, cursor):
    """
    Converts the text to vector, adds to FAISS, and stores in SQLite with metadata
    """
    # 1. Vectorize (only the text gets embedded)
    vector = embedding_model.encode(chunk_dict['text'])
    vector = vector.reshape(1, -1).astype('float32')
    
    # 2. Add to FAISS
    faiss_index.add(vector)
    
    # 3. Add to SQLite with metadata
    cursor.execute(
        "INSERT INTO chunks (text, document, section) VALUES (?, ?, ?)",
        (chunk_dict['text'], chunk_dict['document'], chunk_dict['section'])
    )

# ============================================================================
# SLIDING WINDOW CHUNKING
# ============================================================================

def create_chunks_from_json(file_path, chunk_size=500, overlap=100):
    """
    Takes a single JSON file and creates chunks using sliding window.
    
    Args:
        file_path: Path to a single JSON file
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Returns:
        list: List of chunk dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get disease name from metadata
    disease_name = data['metadata']['disease_name']
    category = data['metadata']['category']
    
    # Combine ALL sections into one document
    full_text = ""
    for section_data in data['sections']:
        section_name = section_data['section']
        
        # Skip navigation sections
        if section_name.lower() in ['clinical trials', 'doctors&departments']:
            continue
        
        section_text = ' '.join(section_data['content'])
        if len(section_text) > 30:
            full_text += f"\n\n[{section_name}]\n{section_text}"
    
    # Apply sliding window
    chunks = []
    start = 0
    chunk_num = 0
    
    while start < len(full_text):
        end = start + chunk_size
        chunk_text = full_text[start:end]
        
        # Don't create tiny chunks at the end
        if len(chunk_text) < 50 and chunks:
            break
        
        # Add context header
        contextualized_text = f"Disease: {disease_name}\nCategory: {category}\nChunk: {chunk_num}\n\n{chunk_text}"
        
        chunk = {
            'text': contextualized_text,
            'document': disease_name,
            'section': f"chunk_{chunk_num}"
        }
        chunks.append(chunk)
        
        # Slide window
        start += chunk_size - overlap
        chunk_num += 1
    
    return chunks

def create_all_chunks(base_folder='common_diseases', chunk_size=500, overlap=100):
    """
    Loop through all subfolders and JSON files to create chunks using sliding window.
    
    Args:
        base_folder: Path to folder containing disease subfolders
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Returns:
        list: All chunks from all files
    """
    all_chunks = []
    base_path = Path(base_folder)
    
    # Check if we're looking at subfolders or direct JSON files
    json_files = []
    
    # First check for JSON files directly in the folder
    direct_jsons = list(base_path.glob('*.json'))
    if direct_jsons:
        json_files.extend(direct_jsons)
        print(f"Found {len(direct_jsons)} JSON files in {base_folder}")
    
    # Then check subfolders
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            subfolder_jsons = list(subfolder.glob('*.json'))
            if subfolder_jsons:
                json_files.extend(subfolder_jsons)
                print(f"Found {len(subfolder_jsons)} JSON files in {subfolder.name}")
    
    if not json_files:
        print(f"No JSON files found in {base_folder} or its subfolders")
        return []
    
    print(f"\nTotal JSON files found: {len(json_files)}")
    print(f"Chunking params: chunk_size={chunk_size}, overlap={overlap}")
    
    # Process each file
    for file_path in json_files:
        print(f"Processing: {file_path.name}")
        try:
            chunks = create_chunks_from_json(file_path, chunk_size, overlap)
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    print(f"Average chunks per file: {len(all_chunks)/len(json_files):.1f}")
    return all_chunks

def save_chunks(chunks, filename='chunks.pkl'):
    """Save chunks to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {filename}")

def load_chunks(filename='chunks.pkl'):
    """Load chunks from pickle file"""
    with open(filename, 'rb') as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from {filename}")
    return chunks

def process_and_store_chunks(chunk_size=500, overlap=100):
    """
    Main function to process and store all chunks using sliding window
    """
    # Create chunks from all JSON files
    all_chunks = create_all_chunks('common_diseases', chunk_size, overlap)
    
    if not all_chunks:
        print("No chunks created. Check your file paths and JSON structure.")
        return
    
    save_chunks(all_chunks)
    
    # Initialize ONLY what we need for embedding and storage
    dimension = 384  # for all-MiniLM-L6-v2
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = faiss.IndexFlatL2(dimension)
    
    # Setup SQLite
    conn = sqlite3.connect('medical_chunks.db')
    cursor = conn.cursor()
    
    # Drop existing table and recreate
    cursor.execute('DROP TABLE IF EXISTS chunks')
    cursor.execute('''
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            document TEXT NOT NULL,
            section TEXT NOT NULL)
    ''')
    
    print("\nAdding chunks to database...")
    for i, chunk in enumerate(all_chunks):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(all_chunks)} chunks...")
        embed_add(chunk, embedding_model, faiss_index, cursor)
    
    # Save everything
    conn.commit()
    faiss.write_index(faiss_index, 'medical_rag.index')
    conn.close()
    
    print(f"\n✓ Successfully stored {len(all_chunks)} chunks!")
    print(f"✓ FAISS index saved to: medical_rag.index")
    print(f"✓ SQLite database saved to: medical_chunks.db")
    print(f"✓ Ready for RAG queries!")

# Run if this script is executed directly
if __name__ == "__main__":
    process_and_store_chunks(chunk_size=500, overlap=100)
# %%
