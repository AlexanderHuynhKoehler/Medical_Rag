from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import numpy as np
from rag_functions import embed_add, vectorize_query_retrieve
from transformers import AutoTokenizer, AutoModelForCausalLM  # ← This line is missing!
import torch

class RAG:
    def __init__(self, dimension, embedding_model = 'all-MiniLM-L6-v2', model_name = "Qwen/Qwen2-1.5B-Instruct"):

        self.system_prompt = "You are a medical assistant. Give answers to the questions using your knowledge in combinatuion with retrieved information"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dimension = dimension
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index = faiss.IndexFlatL2(dimension)

        self.conn = sqlite3.connect('medical_chunks.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('DROP TABLE IF EXISTS chunks')
        
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                document TEXT NOT NULL,
                section TEXT NOT NULL)
            ''')

        self.context = []

    def add_chunk(self,text):
        embed_add(text, self.embedding_model,self.faiss_index, self.cursor)

    def query_chunks(self, user_query):
        self.context = vectorize_query_retrieve(
            user_query,
            self.embedding_model,
            self.faiss_index,
            self.cursor)
        return self.context


    def llm_generate(self, query):
        context = self.query_chunks(query)
        context_str = "\n".join(context) #converts list to string
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context_str}\n\nQuestion: {query}"}  # Fixed: context→context_str, user_query→query
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature = 0.7,
            top_p = 0.9,
            do_sample = True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

    def commit(self):
        self.conn.commit()
    def close(self):
        self.conn.close()

