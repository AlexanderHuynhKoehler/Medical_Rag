# RAG System Evaluation Framework 

This repository contains a comprehensive framework for developing and evaluating a **Retrieval-Augmented Generation (RAG) system** focused on health and disease information. The system incorporates a novel **confidence scoring mechanism** to identify and abstain from answering out-of-scope or low-confidence queries, mitigating hallucination risk.

## Repository Structure

The core functionality is split into two main files:

| File | Description |
| :--- | :--- |
| `run_rag.py` | **RAG Pipeline:** Handles data loading, vector store creation, LLM integration, and the **`RAGWithConfidence`** class for answering queries with built-in confidence scoring and abstention logic. |
| `evaluate_non_llm.py` | **Evaluation Script:** Executes the RAG system against a benchmark dataset and calculates a wide array of non-LLM metrics (e.g., ROUGE, BERTScore, Semantic Similarity, BM25) for retrieval and answer quality, and analyzes the performance of the abstention mechanism. |
| `evaluation_dataset.csv` | The required dataset file containing `question`, `answer` (ground truth), `is_relevant`, and `category` columns for evaluation. |
| `vectorstore/` | Directory where the FAISS vector index is persisted. |
| `disease_data/` | Directory containing the source JSON files used to build the knowledge base. |

-----

## Key Features

### 1\. Confidence-Aware RAG Pipeline (`run_rag.py`)

  * **Custom LLM Integration:** Uses the HuggingFace `pipeline` for local LLM inference (default: `Qwen/Qwen2-1.5B-Instruct`).
  * **Vector Store:** Utilizes **FAISS** and **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) for efficient knowledge retrieval.
  * **`RAGWithConfidence` Class:**
      * **Retrieval Confidence:** Measures the quality of context retrieval based on normalized similarity scores.
      * **Answer Confidence:** Measures the **groundedness** (answer similarity to context) and **relevance** (answer similarity to query) using semantic similarity.
      * **Abstention Logic:** If either confidence score falls below a defined threshold (`RETRIEVAL_CONFIDENCE_THRESHOLD` or `ANSWER_CONFIDENCE_THRESHOLD`), the system returns a specific **low-confidence response** instead of generating a potentially erroneous answer.

### 2\. Comprehensive Non-LLM Evaluation (`evaluate_non_llm.py`)

This script provides an objective, large-scale assessment of the RAG system without relying on an external LLM for judgment (which can be costly and biased).

| Metric Category | Metrics Used | Purpose |
| :--- | :--- | :--- |
| **Answer Quality** | **ROUGE** (1, 2, L), **BERTScore** (P, R, F1), **Semantic Similarity** (Generated vs. Ground Truth) | Measures lexical and semantic overlap between the generated answer and the ground truth answer.  |
| **Retrieval Quality** | **Avg/Max Q-to-Context Similarity**, **Avg/Max GT-to-Context Similarity**, **Retrieval Diversity**, **BM25 Score** | Assesses how relevant and diverse the retrieved chunks are to both the input question and the true answer. |
| **Abstention Performance** | **Precision**, **Recall**, **F1 Score**, **Accuracy**, **Hallucination Rate** (False Positives), **Incorrect Abstention** (False Negatives) | Analyzes the effectiveness of the confidence mechanism in correctly choosing to answer in-scope questions and abstain from irrelevant or unsupported ones. |

-----

## Setup and Installation

### Prerequisites

You need Python 3.8+ and the following dependencies.

```bash
# Install core dependencies
pip install pandas numpy torch tqdm scikit-learn sentence-transformers transformers
# Install LangChain components
pip install langchain-community langchain-huggingface
# Install optional evaluation metrics
pip install rouge-score bert-score rank-bm25
```

### Configuration

Ensure the following paths and constants in `run_rag.py` are set correctly for your environment:

  * `DATA_DIR`: Path to your folder containing source JSON data.
  * `VECTOR_STORE_PATH`: Path where the FAISS index will be saved.
  * `EMBEDDING_MODEL_NAME`: The Sentence Transformer model for embeddings.
  * `LLM_MODEL_NAME`: The Hugging Face model for text generation.

-----

## Usage

### 1\. Build the Knowledge Base (Run RAG Pipeline)

If the vector store path does not exist, running `run_rag.py` will build it first. It will then run a set of test queries.

```bash
python run_rag.py
```

### 2\. Run Evaluation

Ensure you have your **`evaluation_dataset.csv`** prepared. This file must contain the columns: `question`, `answer` (ground truth), `is_relevant` (boolean), and `category`.

Run the evaluation script:

```bash
python evaluate_non_llm.py
```

### Output

The script will print summarized metrics to the console and save detailed results to the `evaluation_results/` directory:

  * `evaluation_results/non_llm_detailed_results.csv`: Contains all calculated metrics for every sample.
  * `evaluation_results/non_llm_summary.json`: Contains mean overall metrics and performance of the abstention mechanism.
  * `evaluation_results/non_llm_category_stats.csv`: Mean metrics broken down by question category.
