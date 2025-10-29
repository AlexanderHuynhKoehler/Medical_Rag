def vectorize_query_retrieve(user_query, embedding_model, faiss_index, cursor):
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
            print(f"Distance: {distance:.4f} | {result[0]}")
    
    return chunks


def embed_add(chunk_dict, embedding_model, faiss_index, cursor):
    """
    Converts the text to vector, adds to FAISS, and stores in SQLite with metadata
    
    Args:
        chunk_dict: Dictionary with 'text', 'document', and 'section' keys
        embedding_model: SentenceTransformer model
        faiss_index: FAISS index
        cursor: SQLite cursor
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
