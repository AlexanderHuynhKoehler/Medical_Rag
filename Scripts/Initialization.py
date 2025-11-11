from sentence_transformers import SentenceTransformer
import faiss 
import sqlite3
import numpy as np
from rag_functions import embed_add, vectorize_query_retrieve
from Rag_model import RAG
from chunking import create_all_chunks, load_chunks
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from chunking import create_all_chunks, save_chunks


def process_and_store_chunks(path = 'Json_files'):
  all_chunks = create_all_chunks(path)
  save_chunks(all_chunks)
  print(f"Created {len(all_chunks)} chunks")

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
  rag.save_databases()


process_and_store_chunks()
