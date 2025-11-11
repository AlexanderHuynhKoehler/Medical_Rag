# %%
"""
RAGAS Evaluation Pipeline for Medical RAG System
Uses Gemini for LLM-as-judge evaluation
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from Rag_model import RAG
import json
import os

import google.generativeai as genai

genai.configure(api_key='AIzaSyBVYyyzHMXxRB2tlqhXnEqPfWflFq2AZHk')


model = genai.GenerativeModel('gemini-2.5-flash',
system_instruction='You are my assistant')


# %%
# STEP 1: CREATE TEST CASES

def create_test_cases_from_docs(json_dir='Json_files'):
    """
    Auto-generate test cases from your structured medical documents
    """
    test_cases = []
    
    manual_tests = [
        {
            'question': "What are the symptoms of anxiety disorder?",
            'ground_truth': "Excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep problems"
        },
        {
            'question': "What causes diabetes?",
            'ground_truth': "Type 1: autoimmune destruction of insulin-producing beta cells. Type 2: insulin resistance combined with relative insulin deficiency"
        },
        {
            'question': "How is hypertension treated?",
            'ground_truth': "Lifestyle modifications including diet changes, exercise, weight loss, and medications such as ACE inhibitors, diuretics, and beta blockers"
        },
        {
            'question': "What are the risk factors for stroke?",
            'ground_truth': "High blood pressure, smoking, diabetes, high cholesterol, obesity, physical inactivity, excessive alcohol use, family history"
        },
        {
            'question': "What are the complications of COPD?",
            'ground_truth': "Respiratory infections, heart problems, lung cancer, high blood pressure in lung arteries, depression and anxiety"
        }
    ]
    
    return manual_tests


# STEP 2: RUN YOUR RAG SYSTEM

def run_rag_evaluation(rag, test_cases):
    """
    Run your RAG system on all test cases and collect results
    """
    data = {
        'question': [],
        'contexts': [],
        'answer': [],
        'ground_truth': []
    }
    
    print(f"\nRunning RAG on {len(test_cases)} test cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Processing: {test['question'][:50]}...")
        
        # Run your RAG system
        answer = rag.llm_generate(test['question'], source_language='en')
        contexts = rag.context  # Retrieved chunks
        
        # Store results
        data['question'].append(test['question'])
        data['contexts'].append(contexts)
        data['answer'].append(answer if answer else "No answer generated")
        data['ground_truth'].append(test['ground_truth'])
    
    return data


# STEP 3: EVALUATE WITH RAGAS

def evaluate_rag_system(data):
    """
    Run RAGAS evaluation metrics
    """
    print("\n" + "="*60)
    print("RUNNING RAGAS EVALUATION")
    print("="*60)
    
    # Convert to RAGAS dataset format
    dataset = Dataset.from_dict(data)
    
    # Run evaluation (this calls Gemini)
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm
    )
    
    return results


# STEP 4: DISPLAY RESULTS

def display_results(results):
    """
    Pretty print evaluation results
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Scores:")
    print(f"  Context Precision:  {results['context_precision']:.3f}")
    print(f"  Context Recall:     {results['context_recall']:.3f}")
    print(f"  Faithfulness:       {results['faithfulness']:.3f}")
    print(f"  Answer Relevancy:   {results['answer_relevancy']:.3f}")
    
    print(f"\n💡 Interpretation:")
    
    if results['context_precision'] < 0.7:
        print("  ⚠️  Context Precision low - retrieval finding irrelevant docs")
    else:
        print("  ✅ Context Precision good - retrieval is accurate")
    
    if results['context_recall'] < 0.7:
        print("  ⚠️  Context Recall low - missing relevant information")
    else:
        print("  ✅ Context Recall good - retrieving comprehensive info")
    
    if results['faithfulness'] < 0.8:
        print("  🚨 Faithfulness low - MODEL IS HALLUCINATING!")
    else:
        print("  ✅ Faithfulness good - answers grounded in context")
    
    if results['answer_relevancy'] < 0.7:
        print("  ⚠️  Answer Relevancy low - not addressing questions well")
    else:
        print("  ✅ Answer Relevancy good - answers are on-topic")
    
    return results


# STEP 5: SAVE RESULTS

def save_results(results, filename='evaluation_results.json'):
    """
    Save results for tracking over time
    """
    import datetime
    
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'metrics': {
            'context_precision': float(results['context_precision']),
            'context_recall': float(results['context_recall']),
            'faithfulness': float(results['faithfulness']),
            'answer_relevancy': float(results['answer_relevancy'])
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")


================================

if __name__ == "__main__":
    # Load your RAG system
    print("Loading RAG system...")
    rag = RAG.load_from_saved(
        faiss_path='medical_rag.index',
        sqlite_path='medical_chunks.db',
        embedding_model='moka-ai/m3e-base',
        model_name="Qwen/Qwen2-1.5B-Instruct",
        enable_translation=True
    )
    
    # Create test cases
    test_cases = create_test_cases_from_docs()
    
    # Run RAG on test cases
    data = run_rag_evaluation(rag, test_cases)
    
    # Evaluate with RAGAS
    results = evaluate_rag_system(data)
    
    # Display results
    display_results(results)
    
    # Save results
    save_results(results)
    
    # Cleanup
    rag.close()
    
    print("\n✅ Evaluation complete!")