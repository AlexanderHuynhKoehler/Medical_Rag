from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import numpy as np
from rag_functions import embed_add, vectorize_query_retrieve
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

class RAG:
    def __init__(self, dimension, embedding_model='all-MiniLM-L6-v2', model_name="Qwen/Qwen2-1.5B-Instruct"):

        self.system_prompt = "You are a medical assistant. Give answers to the questions using your knowledge in combination with retrieved information"

        # LLM setup
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Embedding setup
        self.dimension = dimension
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index = faiss.IndexFlatL2(dimension)

        # Database setup
        self.conn = sqlite3.connect('medical_chunks.db')
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                document TEXT NOT NULL,
                section TEXT NOT NULL)
            ''')

        self.context = []

    def add_chunk(self, text):
        embed_add(text, self.embedding_model, self.faiss_index, self.cursor)

    def query_chunks(self, query):
        # Step 1: Rewrite query for better retrieval
        user_query = self.rewrite_query(query)
        
        # Step 2: Use rewritten query for retrieval
        self.context = vectorize_query_retrieve(
            user_query,
            self.embedding_model,
            self.faiss_index,
            self.cursor)
        return self.context

    def rewrite_query(self, user_query):
        """
        Transform user query into a search-optimized query for better FAISS retrieval.
        """
        messages = [
            {"role": "system", "content": "You are a query rewriter. Transform the user's question into a search query optimized for retrieving relevant medical documents. Output ONLY the rewritten query, nothing else. Keep it concise - focus on key medical terms and concepts."},
            {"role": "user", "content": user_query}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,  # Short output - just the rewritten query
            temperature=0.3,    # Low temperature for consistent rewrites
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the rewritten query
        if "assistant\n" in response:
            response = response.split("assistant\n")[-1].strip()
        elif "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"Original query: {user_query}")
        print(f"Rewritten query: {response}")
        
        return response


    def llm_generate(self, query):
        """
        Generate response for English query with English context
        Args:
            query: Question in English
        """
        # Query RAG system with English query to get English chunks
        context = self.query_chunks(query)
        
        # Build context string from retrieved chunks
        context_str = "\n".join(context)
        
        # Set self.context for the evaluator
        self.context = context
        
        # Generate response in English
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context_str}\n\nQuestion: {query}"}
        ]
        
        # Generate response
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up to get just the assistant's response
        if "assistant\n" in response:
            response = response.split("assistant\n")[-1].strip()
        
        print(f"Response: {response}\n")
        return response

    def commit(self):
        self.conn.commit()
        
    def close(self):
        self.conn.close()
        
    def save_databases(self, faiss_path='medical_rag.index', 
                    sqlite_path='medical_chunks.db'):
        """Save both FAISS index and SQLite database to files"""
        faiss.write_index(self.faiss_index, faiss_path)
        print(f"✓ Saved FAISS index to {faiss_path}")
        
        self.conn.commit()
        print(f"✓ Saved SQLite database to {sqlite_path}")
        print(f"✓ Total chunks saved: {self.faiss_index.ntotal}")

    @classmethod
    def load_from_saved(cls, faiss_path='medical_rag.index', 
                       sqlite_path='medical_chunks.db',
                       embedding_model='all-MiniLM-L6-v2',
                       model_name="Qwen/Qwen2-1.5B-Instruct"):
        """Load a pre-built RAG system from saved files"""
        import os
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
        
        # Load FAISS index
        faiss_index = faiss.read_index(faiss_path)
        dimension = faiss_index.d
        print(f"✓ Loaded FAISS index: {faiss_index.ntotal} vectors")
        
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set up basic attributes
        instance.dimension = dimension
        instance.embedding_model = SentenceTransformer(embedding_model)
        instance.faiss_index = faiss_index
        
        # Load SQLite
        instance.conn = sqlite3.connect(sqlite_path)
        instance.cursor = instance.conn.cursor()
        
        instance.cursor.execute("SELECT COUNT(*) FROM chunks")
        sqlite_count = instance.cursor.fetchone()[0]
        print(f"✓ Loaded SQLite database: {sqlite_count} chunks")
        
        if faiss_index.ntotal != sqlite_count:
            print("⚠ WARNING: FAISS and SQLite counts don't match!")
        
        # Load LLM
        instance.system_prompt = "You are a medical assistant. Give answers to the questions using your knowledge in combination with retrieved information"
        instance.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
        instance.context = []
        
        print(f"✓ RAG system ready!")
        return instance