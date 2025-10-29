from sentence_transformers import SentenceTransformer
import faiss 
import sqlite3
import numpy as np
from rag_functions import embed_add, vectorize_query_retrieve
from Rag_model import RAG
from chunking import create_all_chunks, load_chunks
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    
  # Setup RAG model
  dimension = 768
  embedding_model = 'moka-ai/m3e-base'
  llm_model = "Qwen/Qwen2-1.5B-Instruct"

  rag = RAG(dimension=dimension, embedding_model=embedding_model, model_name=llm_model)

  print("Loading and chunking documents...")
  all_chunks = load_chunks('chunks.pkl')

  print(f"\nAdding {len(all_chunks)} chunks to database...")
  for chunk in all_chunks:
      rag.add_chunk(chunk)

  rag.commit()

  # === VERIFICATION TESTS ===
  print("\n" + "="*60)
  print("VERIFICATION TESTS")
  print("="*60)

  # Test 1: Check alignment
  print(f"\n✓ Test 1: Database Alignment")
  print(f"  FAISS total vectors: {rag.faiss_index.ntotal}")
  rag.cursor.execute("SELECT COUNT(*) FROM chunks")
  sqlite_count = rag.cursor.fetchone()[0]
  print(f"  SQLite total rows: {sqlite_count}")
  print(f"  Aligned: {'YES ✓' if rag.faiss_index.ntotal == sqlite_count else 'NO ✗'}")

  # Test 2: Check a random chunk was stored correctly
  print(f"\n✓ Test 2: Sample Chunk Retrieval")
  rag.cursor.execute("SELECT text, document, section FROM chunks WHERE id = 1")
  result = rag.cursor.fetchone()
  print(f"  Document: {result[1]}")
  print(f"  Section: {result[2]}")
  print(f"  Text preview: {result[0][:100]}...")

  # Test 3: Test semantic search
  print(f"\n✓ Test 3: Semantic Search Test")
  test_query = "什么是焦虑症？"  # What is anxiety disorder?
  print(f"  Query: {test_query}")
  retrieved_chunks = rag.query_chunks(test_query)
  print(f"  Retrieved {len(retrieved_chunks)} chunks")

  # Test 4: Check documents are diverse
  print(f"\n✓ Test 4: Document Diversity Check")
  rag.cursor.execute("SELECT DISTINCT document FROM chunks")
  unique_docs = rag.cursor.fetchall()
  print(f"  Unique documents: {len(unique_docs)}")
  for doc in unique_docs[:5]:  # Show first 5
      print(f"    - {doc[0]}")

  print("\n" + "="*60)
  print("All tests complete!")
  print("="*60)

  rag.llm_generate("什么是焦虑症？")


  rag.close()
main()
