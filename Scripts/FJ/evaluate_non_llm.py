import os
import json
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from run_rag import RAGWithConfidence, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, VECTOR_STORE_PATH

# Suppress RoBERTa pooler warning from BERT Score (harmless, pooler not used)
warnings.filterwarnings('ignore', message='Some weights of RobertaModel')

# Optional imports with fallback
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge-score not installed. ROUGE metrics will be skipped.")
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert-score not installed. BERT Score will be skipped.")
    BERT_SCORE_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    print("Warning: rank-bm25 not installed. BM25 metrics will be skipped.")
    BM25_AVAILABLE = False


class NonLLMEvaluator:
    """Non-LLM based evaluation metrics for RAG systems."""

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

    def calculate_retrieval_metrics(self, question: str, ground_truth: str,
                                   retrieved_chunks: List[str]) -> Dict[float, float]:
        """
        Calculate retrieval quality metrics.

        Returns:
            Dictionary with retrieval metrics
        """
        metrics = {}

        # Encode texts
        question_emb = self.embedding_model.encode(question, convert_to_tensor=True)
        ground_truth_emb = self.embedding_model.encode(ground_truth, convert_to_tensor=True)
        chunk_embs = self.embedding_model.encode(retrieved_chunks, convert_to_tensor=True)

        # 1. Question-to-Context Semantic Similarity
        q_to_ctx_sims = [util.cos_sim(question_emb, chunk_emb).item()
                         for chunk_emb in chunk_embs]
        metrics['avg_q_to_ctx_similarity'] = np.mean(q_to_ctx_sims)
        metrics['max_q_to_ctx_similarity'] = np.max(q_to_ctx_sims)

        # 2. Ground Truth-to-Context Semantic Similarity
        gt_to_ctx_sims = [util.cos_sim(ground_truth_emb, chunk_emb).item()
                          for chunk_emb in chunk_embs]
        metrics['avg_gt_to_ctx_similarity'] = np.mean(gt_to_ctx_sims)
        metrics['max_gt_to_ctx_similarity'] = np.max(gt_to_ctx_sims)

        # 3. Retrieval Diversity (average pairwise distance between chunks)
        if len(chunk_embs) > 1:
            pairwise_sims = []
            for i in range(len(chunk_embs)):
                for j in range(i + 1, len(chunk_embs)):
                    sim = util.cos_sim(chunk_embs[i], chunk_embs[j]).item()
                    pairwise_sims.append(sim)
            # Diversity = 1 - similarity (higher is more diverse)
            metrics['retrieval_diversity'] = 1 - np.mean(pairwise_sims)
        else:
            metrics['retrieval_diversity'] = 0.0

        # 4. BM25 Score (if available)
        if BM25_AVAILABLE:
            tokenized_chunks = [chunk.lower().split() for chunk in retrieved_chunks]
            bm25 = BM25Okapi(tokenized_chunks)
            tokenized_question = question.lower().split()
            bm25_scores = bm25.get_scores(tokenized_question)
            metrics['avg_bm25_score'] = np.mean(bm25_scores)
            metrics['max_bm25_score'] = np.max(bm25_scores)

        return metrics

    def calculate_answer_metrics(self, generated_answer: str,
                                ground_truth: str, question: str) -> Dict[str, float]:
        """
        Calculate answer quality metrics.

        Returns:
            Dictionary with answer quality metrics
        """
        metrics = {}

        # Encode texts
        gen_emb = self.embedding_model.encode(generated_answer, convert_to_tensor=True)
        gt_emb = self.embedding_model.encode(ground_truth, convert_to_tensor=True)
        q_emb = self.embedding_model.encode(question, convert_to_tensor=True)

        # 1. Semantic Similarity (cosine similarity)
        metrics['semantic_similarity'] = util.cos_sim(gen_emb, gt_emb).item()

        # 2. Answer-to-Question Relevance
        metrics['answer_question_relevance'] = util.cos_sim(gen_emb, q_emb).item()

        # 3. ROUGE Scores (if available)
        if ROUGE_AVAILABLE:
            rouge_scores = self.rouge_scorer.score(ground_truth, generated_answer)
            metrics['rouge1_f'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2_f'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL_f'] = rouge_scores['rougeL'].fmeasure

        # 4. BERT Score (if available)
        if BERT_SCORE_AVAILABLE:
            try:
                P, R, F1 = bert_score_fn(
                    [generated_answer],
                    [ground_truth],
                    lang='en',
                    verbose=False,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                metrics['bert_score_precision'] = P.item()
                metrics['bert_score_recall'] = R.item()
                metrics['bert_score_f1'] = F1.item()
            except Exception as e:
                print(f"Warning: BERT Score calculation failed: {e}")

        return metrics

    def evaluate_sample(self, question: str, ground_truth: str,
                       generated_answer: str, retrieved_chunks: List[str],
                       is_relevant: bool, should_answer: bool) -> Dict[str, float]:
        """
        Evaluate a single sample with all metrics.
        """
        metrics = {}

        # Retrieval metrics
        retrieval_metrics = self.calculate_retrieval_metrics(
            question, ground_truth, retrieved_chunks
        )
        metrics.update(retrieval_metrics)

        # Answer metrics
        answer_metrics = self.calculate_answer_metrics(
            generated_answer, ground_truth, question
        )
        metrics.update(answer_metrics)

        # Confidence detection metrics
        metrics['is_relevant'] = is_relevant
        metrics['should_answer'] = should_answer
        metrics['correct_abstention'] = (not is_relevant) and (not should_answer)
        metrics['incorrect_abstention'] = is_relevant and (not should_answer)
        metrics['hallucination'] = (not is_relevant) and should_answer

        return metrics


def load_evaluation_dataset(csv_path: str = 'evaluation_dataset.csv') -> pd.DataFrame:
    """Load the evaluation dataset."""
    print(f"Loading evaluation dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} evaluation samples")
    return df


def run_evaluation(rag_system: RAGWithConfidence, eval_df: pd.DataFrame,
                   evaluator: NonLLMEvaluator) -> List[Dict]:
    """
    Run RAG system and evaluate with non-LLM metrics.
    """
    print("\nRunning RAG system and evaluating...")
    results = []

    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
        question = row['question']
        ground_truth = row['answer']
        is_relevant = row['is_relevant']
        category = row['category']

        # Query RAG system
        rag_result = rag_system.query(question)

        # Evaluate
        metrics = evaluator.evaluate_sample(
            question=question,
            ground_truth=ground_truth,
            generated_answer=rag_result['answer'],
            retrieved_chunks=rag_result['retrieved_chunks'],
            is_relevant=is_relevant,
            should_answer=rag_result['should_answer']
        )

        # Add metadata
        metrics['question'] = question
        metrics['category'] = category
        metrics['confidence_score'] = rag_result['confidence_score']
        metrics['retrieval_confidence'] = rag_result['retrieval_confidence']
        metrics['answer_confidence'] = rag_result['answer_confidence']

        results.append(metrics)

    return results


def analyze_abstention_performance(results: List[Dict]) -> Dict:
    """
    Analyze the performance of the confidence-based abstention mechanism.
    """
    print("\nAnalyzing confidence-based abstention performance...")

    # Ground truth: is_relevant indicates if question is in scope
    # Prediction: should_answer indicates if system chose to answer
    y_true = [r['is_relevant'] for r in results]
    y_pred = [r['should_answer'] for r in results]

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # Calculate confusion matrix elements
    correct_answers = sum(r['is_relevant'] and r['should_answer'] for r in results)
    correct_abstentions = sum(r['correct_abstention'] for r in results)
    incorrect_abstentions = sum(r['incorrect_abstention'] for r in results)
    hallucinations = sum(r['hallucination'] for r in results)

    total = len(results)
    relevant = sum(y_true)
    irrelevant = total - relevant

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': (correct_answers + correct_abstentions) / total,
        'correct_answers': correct_answers,
        'correct_abstentions': correct_abstentions,
        'incorrect_abstentions': incorrect_abstentions,
        'false_negatives': incorrect_abstentions,
        'hallucinations': hallucinations,
        'false_positives': hallucinations,
        'total_samples': total,
        'relevant_samples': relevant,
        'irrelevant_samples': irrelevant
    }


def save_results(results: List[Dict], abstention_metrics: Dict,
                 output_dir: str = 'evaluation_results'):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save detailed results
    detailed_path = os.path.join(output_dir, 'non_llm_detailed_results.csv')
    results_df.to_csv(detailed_path, index=False)
    print(f"\n Saved detailed results to: {detailed_path}")

    # Calculate aggregate metrics
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in
                    ['is_relevant', 'should_answer', 'correct_abstention',
                     'incorrect_abstention', 'hallucination']]

    overall_metrics = results_df[numeric_cols].mean().to_dict()

    # Add abstention metrics
    overall_metrics.update(abstention_metrics)

    # By category metrics
    category_metrics = results_df.groupby('category')[numeric_cols].mean().to_dict()

    # Save summary
    summary = {
        'overall_metrics': overall_metrics,
        'by_category': category_metrics,
        'abstention_performance': abstention_metrics
    }

    summary_path = os.path.join(output_dir, 'non_llm_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f" Saved summary to: {summary_path}")

    # Save category statistics
    category_stats = results_df.groupby('category')[numeric_cols].mean()
    category_path = os.path.join(output_dir, 'non_llm_category_stats.csv')
    category_stats.to_csv(category_path)
    print(f" Saved category statistics to: {category_path}")


def main():
    # Load evaluation dataset
    eval_df = load_evaluation_dataset('evaluation_dataset.csv')

    # Initialize RAG system
    print("\n Loading RAG system...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    llm_pipe = pipeline("text-generation", model=LLM_MODEL_NAME, device_map="auto")
    rag_system = RAGWithConfidence(vectorstore, llm_pipe)

    # Initialize evaluator
    print(" Initializing non-LLM evaluator...")
    evaluator = NonLLMEvaluator()

    # Run evaluation
    results = run_evaluation(rag_system, eval_df, evaluator)

    # Analyze abstention performance
    abstention_metrics = analyze_abstention_performance(results)

    # Calculate aggregate metrics
    results_df = pd.DataFrame(results)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in
                    ['is_relevant', 'should_answer', 'correct_abstention',
                     'incorrect_abstention', 'hallucination']]

    # Print results
    print("\n" + "=" * 80)
    print("NON-LLM EVALUATION RESULTS")
    print("=" * 80)

    print("\n Retrieval Quality Metrics:")
    retrieval_metrics = ['avg_q_to_ctx_similarity', 'max_q_to_ctx_similarity',
                        'avg_gt_to_ctx_similarity', 'max_gt_to_ctx_similarity',
                        'retrieval_diversity']
    if BM25_AVAILABLE:
        retrieval_metrics.extend(['avg_bm25_score', 'max_bm25_score'])

    for metric in retrieval_metrics:
        if metric in results_df.columns:
            print(f"  {metric:30s}: {results_df[metric].mean():.3f}")

    print("\n Answer Quality Metrics:")
    answer_metrics = ['semantic_similarity', 'answer_question_relevance']
    if ROUGE_AVAILABLE:
        answer_metrics.extend(['rouge1_f', 'rouge2_f', 'rougeL_f'])
    if BERT_SCORE_AVAILABLE:
        answer_metrics.extend(['bert_score_precision', 'bert_score_recall', 'bert_score_f1'])

    for metric in answer_metrics:
        if metric in results_df.columns:
            print(f"  {metric:30s}: {results_df[metric].mean():.3f}")

    print("\n Confidence & Abstention Performance:")
    print(f"  Accuracy:                      {abstention_metrics['accuracy']:.3f}")
    print(f"  Precision:                     {abstention_metrics['precision']:.3f}")
    print(f"  Recall:                        {abstention_metrics['recall']:.3f}")
    print(f"  F1 Score:                      {abstention_metrics['f1_score']:.3f}")
    print(f"  Correct Answers:               {abstention_metrics['correct_answers']}")
    print(f"  Correct Abstentions:           {abstention_metrics['correct_abstentions']}")
    print(f"  Incorrect Abstentions (FN):    {abstention_metrics['incorrect_abstentions']}")
    print(f"  Hallucinations (FP):           {abstention_metrics['hallucinations']}")

    # Save results
    save_results(results, abstention_metrics)

    print("\n" + "=" * 80)
    print(" Non-LLM evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
