# %%
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import numpy as np
from rag_functions import embed_add, vectorize_query_retrieve
from Rag_model import RAG
from chunking import create_all_chunks, load_chunks
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



rag = RAG.load_from_saved(
    faiss_path='medical_rag.index',
    sqlite_path='medical_chunks.db',
    embedding_model='moka-ai/m3e-base',
    model_name="Qwen/Qwen2-1.5B-Instruct"
)


# Add this to your notebook

def test_retrieval_quality():
    """Test if the system retrieves relevant documents for different queries"""

    test_cases = [
        {
            "query": "什么是焦虑症?",  # What is anxiety?
            "expected_docs": ["Anxiety Disorder"],
            "description": "Direct match query"
        },
        {
            "query": "心脏病的症状",  # Heart disease symptoms
            "expected_docs": ["Hypertension", "Heart"],  # Should retrieve related
            "description": "Symptom-based query"
        },
        {
            "query": "如何治疗糖尿病?",  # How to treat diabetes?
            "expected_docs": ["Diabetes"],
            "description": "Treatment query"
        },
        {
            "query": "癌症的风险因素",  # Cancer risk factors
            "expected_docs": ["Breast Cancer", "Lung Cancer", "Colorectal Cancer"],
            "description": "Multi-document query"
        },
        {
            "query": "老年人常见疾病",  # Common diseases in elderly
            "expected_docs": ["Alzheimer", "Parkinson", "Osteoporosis"],
            "description": "Demographic query"
        }
    ]

    print("="*60)
    print("RETRIEVAL QUALITY TEST")
    print("="*60)

    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Query: {test['query']}")

        chunks = rag.query_chunks(test['query'])

        # Extract document names from chunks
        retrieved_docs = []
        for chunk in chunks:
            # Parse document name from chunk text
            if "文档:" in chunk:
                doc = chunk.split("文档:")[1].split("\n")[0].strip()
                retrieved_docs.append(doc)

        print(f"Retrieved documents: {retrieved_docs}")
        print(f"Expected documents: {test['expected_docs']}")

        # Check if at least one expected doc was retrieved
        has_match = any(exp in retrieved_docs for exp in test['expected_docs'])

        results.append({
            'query': test['query'],
            'retrieved': retrieved_docs,
            'expected': test['expected_docs'],
            'success': has_match
        })

        print(f"Result: {'✓ PASS' if has_match else '✗ FAIL'}")

    # Summary
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    print(f"\n{'='*60}")
    print(f"Overall Success Rate: {success_rate:.1f}%")
    print(f"{'='*60}")

    return results

# Run the test
retrieval_results = test_retrieval_quality()


rag.llm_generate("What are the causes of cancer?")
rag.close()