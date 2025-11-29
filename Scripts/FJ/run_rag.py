import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ================== Configuration ==================
DATA_DIR = "/Users/volvol/NEU/courses/DS5500 Capstone/P2/disease_data"  # JSON folder - Updated for local path
VECTOR_STORE_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # English-specific embedding model
LLM_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"  # Can be replaced with other models

# Confidence thresholds
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.5
ANSWER_CONFIDENCE_THRESHOLD = 0.6

# Disease categories for informative responses
DISEASE_CATEGORIES = [
    "Cardiovascular (e.g., heart disease, hypertension, stroke)",
    "Neurological (e.g., Alzheimer's, Parkinson's, epilepsy)",
    "Gastrointestinal (e.g., Crohn's disease, GERD, cirrhosis)",
    "Respiratory (e.g., asthma, COPD, pneumonia)",
    "Endocrine/Metabolic (e.g., diabetes, thyroid disorders)",
    "Musculoskeletal (e.g., arthritis, osteoporosis, fibromyalgia)",
    "Kidney/Renal (e.g., kidney disease, UTI, kidney stones)"
]

# ================== Data Loading ==================
def load_all_json(data_dir):
    all_docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    section = item.get("section", "Unknown Section")
                    for content in item.get("content", []):
                        all_docs.append(f"File: {filename}\nSection: {section}\nContent: {content}")
    print(f"Loaded {len(all_docs)} knowledge entries.")
    return all_docs

# ================== Text Splitting ==================
def split_texts(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.create_documents(docs)

# ================== Build Vector Store ==================
def build_vectorstore(docs, embedding_model, persist_path=None):
    print("Computing text embeddings...")
    texts = split_texts(docs)
    vectorstore = FAISS.from_documents(texts, embedding_model)
    if persist_path:
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        vectorstore.save_local(persist_path)
        print(f"Vector database saved to: {persist_path}")
    return vectorstore

def load_vectorstore(persist_path, embedding_model):
    return FAISS.load_local(persist_path, embedding_model, allow_dangerous_deserialization=True)

# ================== Confidence Scoring Class ==================
class RAGWithConfidence:
    def __init__(self, vectorstore, llm_pipe, embedding_model_name=EMBEDDING_MODEL_NAME,
                 retrieval_threshold=RETRIEVAL_CONFIDENCE_THRESHOLD,
                 answer_threshold=ANSWER_CONFIDENCE_THRESHOLD):
        self.vectorstore = vectorstore
        self.llm_pipe = llm_pipe
        self.retrieval_threshold = retrieval_threshold
        self.answer_threshold = answer_threshold
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def _calculate_retrieval_confidence(self, query: str, docs_with_scores: List[Tuple]) -> float:
        """
        Calculate confidence score based on retrieval similarity scores.
        Returns average similarity score normalized to [0, 1].
        """
        if not docs_with_scores:
            return 0.0

        # FAISS returns distances, we need to convert to similarities
        # For L2 distance, similarity = 1 / (1 + distance)
        scores = []
        for doc, score in docs_with_scores:
            # Normalize L2 distance to similarity
            similarity = 1 / (1 + score)
            scores.append(similarity)

        return np.mean(scores)

    def _calculate_answer_confidence(self, answer: str, context: str, query: str) -> float:
        """
        Calculate answer confidence using semantic similarity between:
        1. Answer and retrieved context (groundedness)
        2. Answer and query (relevance)
        """
        # Encode texts
        answer_emb = self.embedding_model.encode(answer, convert_to_tensor=True)
        context_emb = self.embedding_model.encode(context, convert_to_tensor=True)
        query_emb = self.embedding_model.encode(query, convert_to_tensor=True)

        # Calculate similarities
        groundedness = util.cos_sim(answer_emb, context_emb).item()
        relevance = util.cos_sim(answer_emb, query_emb).item()

        # Combined confidence score (weighted average)
        confidence = 0.6 * groundedness + 0.4 * relevance

        return max(0.0, min(1.0, confidence))  # Clip to [0, 1]

    def _generate_low_confidence_response(self) -> str:
        """Generate informative response when confidence is too low."""
        categories_str = "\n".join([f"- {cat}" for cat in DISEASE_CATEGORIES])
        return (
            "I was not able to retrieve an answer with high confidence. "
            "The question may be outside my current knowledge scope.\n\n"
            "I currently have information about the following disease categories:\n"
            f"{categories_str}\n\n"
            "Please rephrase your question or ask about topics within these categories."
        )

    def query(self, question: str, k: int = 3) -> Dict:
        """
        Query the RAG system with confidence scoring.

        Returns:
            Dict with keys: answer, confidence_score, retrieval_confidence,
                          answer_confidence, should_answer, retrieved_chunks
        """
        # Retrieve documents with similarity scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)

        # Calculate retrieval confidence
        retrieval_confidence = self._calculate_retrieval_confidence(question, docs_with_scores)

        # Extract documents and create context
        docs = [doc for doc, score in docs_with_scores]
        context = "\n".join([d.page_content for d in docs])

        # Check if retrieval confidence is too low
        if retrieval_confidence < self.retrieval_threshold:
            return {
                "answer": self._generate_low_confidence_response(),
                "confidence_score": retrieval_confidence,
                "retrieval_confidence": retrieval_confidence,
                "answer_confidence": 0.0,
                "should_answer": False,
                "retrieved_chunks": [d.page_content for d in docs],
                "retrieval_scores": [score for doc, score in docs_with_scores]
            }

        # Generate answer
        answer = generate_answer(self.llm_pipe, context, question)

        # Calculate answer confidence
        answer_confidence = self._calculate_answer_confidence(answer, context, question)

        # Combined confidence score
        overall_confidence = 0.5 * retrieval_confidence + 0.5 * answer_confidence

        # Check if overall confidence is too low
        if overall_confidence < self.answer_threshold:
            return {
                "answer": self._generate_low_confidence_response(),
                "confidence_score": overall_confidence,
                "retrieval_confidence": retrieval_confidence,
                "answer_confidence": answer_confidence,
                "should_answer": False,
                "retrieved_chunks": [d.page_content for d in docs],
                "retrieval_scores": [score for doc, score in docs_with_scores]
            }

        return {
            "answer": answer,
            "confidence_score": overall_confidence,
            "retrieval_confidence": retrieval_confidence,
            "answer_confidence": answer_confidence,
            "should_answer": True,
            "retrieved_chunks": [d.page_content for d in docs],
            "retrieval_scores": [score for doc, score in docs_with_scores]
        }

# ================== Local LLM Answer Generation ==================
def generate_answer(llm_pipe, context, query):
    prompt = f"Based on the following information, answer the question:\n\n{context}\n\nQuestion: {query}\n\nPlease provide a concise answer in English."
    result = llm_pipe(prompt, max_new_tokens=512)
    if isinstance(result, list):
        return result[0]['generated_text']
    return result

# ================== Main Pipeline ==================
def main(use_confidence=True):
    # ---- Load embedding model ----
    print("Loading HuggingFace Embeddings model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    # ---- Load or build knowledge base ----
    if os.path.exists(VECTOR_STORE_PATH):
        print("Detected existing knowledge base, loading...")
        vectorstore = load_vectorstore(VECTOR_STORE_PATH, embedding_model)
    else:
        print("Building knowledge base...")
        docs = load_all_json(DATA_DIR)
        vectorstore = build_vectorstore(docs, embedding_model, VECTOR_STORE_PATH)

    # ---- Load local language model ----
    print("Loading local language model...")
    llm_pipe = pipeline("text-generation", model=LLM_MODEL_NAME, device_map="auto")

    if use_confidence:
        # ---- Use RAG with confidence scoring ----
        print("\n RAG System with Confidence Scoring ready!\n")
        rag_system = RAGWithConfidence(vectorstore, llm_pipe)

        # ---- Test queries (including out-of-scope questions) ----
        test_queries = [
            "How to treat pneumonia",
            "What are the symptoms of diabetes",
            "How to prevent high blood pressure",
            "What is quantum computing?",  # Out of scope
            "How to bake a chocolate cake?"  # Out of scope
        ]

        print("=" * 80)
        print("Running test queries with confidence scoring...")
        print("=" * 80)

        for query in test_queries:
            print(f"\n Question: {query}")
            result = rag_system.query(query)

            print(f"Confidence Score: {result['confidence_score']:.3f}")
            print(f"   - Retrieval Confidence: {result['retrieval_confidence']:.3f}")
            print(f"   - Answer Confidence: {result['answer_confidence']:.3f}")
            print(f"Should Answer: {result['should_answer']}")
            print(f"\n Answer:\n{result['answer']}\n")
            print("-" * 80)

        print("\n Testing complete!")

    else:
        # ---- Original pipeline without confidence scoring ----
        print("\n Knowledge base ready! (Original pipeline - no confidence scoring)\n")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        test_queries = [
            "How to treat pneumonia",
            "What are the symptoms of diabetes",
            "How to prevent high blood pressure"
        ]

        print("=" * 60)
        print("Running test queries...")
        print("=" * 60)

        for query in test_queries:
            print(f"\n Question: {query}")
            docs = retriever.invoke(query)
            context = "\n".join([d.page_content for d in docs])
            print(f" Found {len(docs)} relevant document chunks")
            answer = generate_answer(llm_pipe, context, query)
            print(f" Answer: {answer}\n")
            print("-" * 60)

        print("\n Testing complete!")

if __name__ == "__main__":
    import sys
    # Use confidence scoring by default, pass --no-confidence to disable
    use_conf = "--no-confidence" not in sys.argv
    main(use_confidence=use_conf)
