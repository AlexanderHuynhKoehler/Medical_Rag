from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import numpy as np
from rag_functions import embed_add, vectorize_query_retrieve
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
import torch
import json

class RAG:
    def __init__(self, dimension, embedding_model='all-MiniLM-L6-v2', model_name="Qwen/Qwen2-1.5B-Instruct", enable_translation=True):

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
        
        # Translation setup
        self.enable_translation = enable_translation
        if enable_translation:
            print("Loading translation models...")
            
            # English → Chinese
            self.en_zh_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            self.en_zh_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            print("✓ Loaded EN→ZH translator")
            
            # Chinese → English
            self.zh_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
            self.zh_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
            print("✓ Loaded ZH→EN translator")

    def translate_en_to_zh(self, text):
        """Translate English to Chinese"""
        if not self.enable_translation:
            return text
        inputs = self.en_zh_tokenizer(text, return_tensors="pt", padding=True)
        translated = self.en_zh_model.generate(**inputs)
        result = self.en_zh_tokenizer.decode(translated[0], skip_special_tokens=True)
        return result
    
    def translate_zh_to_en(self, text):
        """Translate Chinese to English"""
        if not self.enable_translation:
            return text
        inputs = self.zh_en_tokenizer(text, return_tensors="pt", padding=True)
        translated = self.zh_en_model.generate(**inputs)
        result = self.zh_en_tokenizer.decode(translated[0], skip_special_tokens=True)
        return result

    def add_chunk(self, text):
        embed_add(text, self.embedding_model, self.faiss_index, self.cursor)

    def query_chunks(self, user_query):
        self.context = vectorize_query_retrieve(
            user_query,
            self.embedding_model,
            self.faiss_index,
            self.cursor)
        return self.context

    def llm_generate(self, query, source_language='en'):
        """
        Generate response with optional translation
        
        Args:
            query: Question in English or Chinese
            source_language: 'en' or 'zh' - language of the input query
        """
        # Step 1: Translate query to Chinese if needed
        if source_language == 'en' and self.enable_translation:
            chinese_query = self.translate_en_to_zh(query)
            print(f"Original Query (EN): {query}")
            print(f"Translated Query (ZH): {chinese_query}")
        else:
            chinese_query = query
        
        # Step 2: Query RAG system (always in Chinese)
        context = self.query_chunks(chinese_query)
        context_str = "\n".join(context)
        
        # Step 3: Generate response in Chinese
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context_str}\n\nQuestion: {chinese_query}"}
        ]
        
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
        
        chinese_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up to get just the assistant's response
        if "assistant\n" in chinese_response:
            chinese_response = chinese_response.split("assistant\n")[-1].strip()
        
        print(f"\nResponse (Chinese):\n{chinese_response}\n")
        
        # Step 4: Translate response to English if needed
        if source_language == 'en' and self.enable_translation:
            english_response = self.translate_zh_to_en(chinese_response)
            print(f"Response (English):\n{english_response}\n")
            return english_response
            
        else:
            return chinese_response

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
                       model_name="Qwen/Qwen2-1.5B-Instruct",
                       enable_translation=True):
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
        
        # Load translation models if enabled
        instance.enable_translation = enable_translation
        if enable_translation:
            print("Loading translation models...")
            
            instance.en_zh_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            instance.en_zh_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            print("✓ Loaded EN→ZH translator")
            
            instance.zh_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
            instance.zh_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
            print("✓ Loaded ZH→EN translator")
        
        print(f"✓ RAG system ready!")
        return instance