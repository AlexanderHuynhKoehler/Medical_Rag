# RAG Evaluation Framework Documentation

This document provides a comprehensive breakdown of the evaluation metrics used in our RAG (Retrieval-Augmented Generation) system evaluation framework. All metrics are computed using mathematical calculations—no LLM-as-judge methods are used.

## Table of Contents

1. [Overview](#overview)
2. [Retrieval Metrics](#retrieval-metrics)
3. [Answer Quality Metrics](#answer-quality-metrics)
4. [Faithfulness Metrics](#faithfulness-metrics)
5. [Abstention Metrics](#abstention-metrics)
6. [Metric Calculation Details](#metric-calculation-details)

---

## Overview

Our evaluation framework assesses RAG system performance across three core dimensions:

| Dimension | Question Answered |
|-----------|-------------------|
| **Retrieval Quality** | Are we finding the right chunks? |
| **Answer Quality** | Is the generated answer correct? |
| **Faithfulness** | Is the answer grounded in retrieved context? |

### Why No LLM-as-Judge?

Many RAG evaluation frameworks (e.g., RAGAS) use an LLM to score outputs subjectively. We avoid this approach because:

- **Non-deterministic**: Same input can yield different scores across runs
- **Circular dependency**: Using AI to evaluate AI without ground truth validation
- **Black box**: Difficult to understand why a score was assigned

Our metrics use embedding similarity, n-gram overlap, and statistical formulas—all deterministic and reproducible.

---

## Retrieval Metrics

These metrics evaluate how well the retrieval system finds relevant chunks.

### `avg_q_to_ctx_similarity`

**What it measures**: Average semantic similarity between the user query and retrieved chunks.

**Calculation**:
```
For each retrieved chunk:
    similarity = cosine_similarity(embed(query), embed(chunk))
avg_q_to_ctx_similarity = mean(all similarities)
```

**Interpretation**:
- High score (>0.7): Retrieved chunks are semantically related to the query
- Low score (<0.5): Retrieved chunks may be off-topic

**Range**: 0 to 1

---

### `max_q_to_ctx_similarity`

**What it measures**: The similarity of the single best-matching chunk to the query.

**Calculation**:
```
max_q_to_ctx_similarity = max(cosine_similarity(embed(query), embed(chunk)) for each chunk)
```

**Interpretation**:
- Shows the quality of your best retrieval
- If high but `avg` is low, you're retrieving one good chunk and several irrelevant ones

**Range**: 0 to 1

---

### `avg_gt_to_ctx_similarity`

**What it measures**: Average semantic similarity between the ground truth answer and retrieved chunks.

**Calculation**:
```
For each retrieved chunk:
    similarity = cosine_similarity(embed(ground_truth), embed(chunk))
avg_gt_to_ctx_similarity = mean(all similarities)
```

**Interpretation**:
- High score: Retrieved chunks contain information similar to the correct answer
- Low score: Even if chunks match the query, they don't contain answer-relevant content

**Range**: 0 to 1

**Note**: This is often more informative than query-to-context similarity because it measures whether chunks actually contain the answer, not just whether they're topically related.

---

### `max_gt_to_ctx_similarity` / `context_coverage`

**What it measures**: The similarity of the single best chunk to the ground truth answer.

**Calculation**:
```
context_coverage = max(cosine_similarity(embed(ground_truth), embed(chunk)) for each chunk)
```

**Interpretation**:
- Represents the ceiling of what your retriever can provide
- If this is low, no retrieved chunk contains the answer

**Range**: 0 to 1

---

### `retrieval_diversity`

**What it measures**: How different the retrieved chunks are from each other.

**Calculation**:
```
pairwise_similarities = [cosine_similarity(chunk_i, chunk_j) for all pairs i,j where i≠j]
retrieval_diversity = 1 - mean(pairwise_similarities)
```

**Interpretation**:
- High score (>0.5): Chunks cover different aspects/sections
- Low score (<0.3): Chunks are redundant (saying the same thing)

**Range**: 0 to 1

**Trade-off**: High diversity isn't always better—sometimes you want multiple chunks reinforcing the same answer.

---

### `context_relevance`

**What it measures**: Proportion of retrieved chunks that exceed a relevance threshold.

**Calculation**:
```
threshold = 0.5
relevant_chunks = count(chunks where similarity_to_ground_truth > threshold)
context_relevance = relevant_chunks / total_chunks
```

**Interpretation**:
- 1.0: All retrieved chunks are relevant
- 0.33: Only 1 of 3 chunks is relevant

**Range**: 0 to 1

---

### `ndcg@k`

**What it measures**: Normalized Discounted Cumulative Gain—whether the best chunks are ranked first.

**Calculation**:
```
DCG = Σ (relevance_i / log2(i + 1)) for i = 1 to k
IDCG = DCG with chunks sorted by relevance (ideal ordering)
NDCG = DCG / IDCG
```

Where `relevance_i` is the similarity of chunk at position i to the ground truth.

**Interpretation**:
- 1.0: Perfect ranking—most relevant chunks appear first
- <1.0: Better chunks are buried lower in the results

**Critical Note**: NDCG measures *relative ordering*, not *absolute quality*. You can have NDCG=1.0 with three terrible chunks if they're ordered correctly (e.g., similarities [0.3, 0.2, 0.1] in that order).

**Range**: 0 to 1

---

## Answer Quality Metrics

These metrics evaluate the quality of the generated answer compared to the ground truth.

### `semantic_similarity`

**What it measures**: Embedding-based similarity between generated answer and ground truth.

**Calculation**:
```
semantic_similarity = cosine_similarity(embed(generated_answer), embed(ground_truth))
```

**Interpretation**:
- High score (>0.7): Answer captures similar meaning to ground truth
- Low score (<0.5): Answer is semantically different from expected

**Range**: 0 to 1

**Limitation**: Two answers can be semantically similar but factually different. "The capital of France is Paris" and "The capital of France is Lyon" might have moderate similarity despite one being wrong.

---

### `answer_question_relevance`

**What it measures**: How well the answer addresses the original question.

**Calculation**:
```
answer_question_relevance = cosine_similarity(embed(answer), embed(question))
```

**Interpretation**:
- High score: Answer is on-topic
- Low score: Answer may be tangential or off-topic

**Range**: 0 to 1

---

### `rouge1_f`, `rouge2_f`, `rougeL_f`

**What they measure**: N-gram overlap between generated answer and ground truth.

**Calculation**:
```
ROUGE-1: Unigram (single word) overlap
ROUGE-2: Bigram (two consecutive words) overlap
ROUGE-L: Longest Common Subsequence

F-score = 2 * (precision * recall) / (precision + recall)

Where:
  precision = matched_ngrams / ngrams_in_generated
  recall = matched_ngrams / ngrams_in_ground_truth
```

**Interpretation**:
- ROUGE-1: General word overlap
- ROUGE-2: Phrase-level overlap (stricter)
- ROUGE-L: Structural similarity

**Range**: 0 to 1

**Limitation**: ROUGE is purely lexical. "The dog bit the man" and "The man bit the dog" have perfect ROUGE-1 but opposite meanings.

---

### `key_term_coverage`

**What it measures**: Whether important terms from the ground truth appear in the generated answer.

**Calculation**:
```
1. Extract key terms from ground truth using TF-IDF weighting
2. Check which key terms appear in the generated answer
3. key_term_coverage = matched_key_terms / total_key_terms
```

**Interpretation**:
- High score: Answer includes domain-specific terminology from the expected answer
- Low score: Answer may be missing critical terms

**Range**: 0 to 1

---

### `answer_length`

**What it measures**: Word count of the generated answer.

**Calculation**:
```
answer_length = len(answer.split())
```

**Interpretation**: Raw count, useful for detecting overly verbose or terse responses.

**Range**: 0 to ∞

---

### `length_ratio`

**What it measures**: Ratio of generated answer length to ground truth length.

**Calculation**:
```
length_ratio = len(generated_answer.split()) / len(ground_truth.split())
```

**Interpretation**:
- 1.0: Same length as expected
- <1.0: Shorter than expected
- >1.0: Longer than expected

**Range**: 0 to ∞

---

## Faithfulness Metrics

These metrics evaluate whether the generated answer is grounded in the retrieved context (not hallucinated from parametric knowledge).

### `faithfulness_score`

**What it measures**: Average similarity between each sentence in the answer and the retrieved context.

**Calculation**:
```
For each sentence in the generated answer:
    sentence_support = max(cosine_similarity(embed(sentence), embed(chunk)) for each chunk)
faithfulness_score = mean(all sentence_support scores)
```

**Interpretation**:
- High score (>0.6): Answer sentences are grounded in retrieved context
- Low score (<0.4): Answer likely relies on parametric knowledge (LLM's training data)

**Range**: 0 to 1

**Key Insight**: You can have high `semantic_similarity` (correct answer) but low `faithfulness_score` (not from context). This indicates the LLM is using its own knowledge rather than the retrieved documents.

---

### `min_sentence_support`

**What it measures**: The support score of the least-grounded sentence.

**Calculation**:
```
min_sentence_support = min(sentence_support for each sentence)
```

**Interpretation**:
- Shows the "weakest link" in faithfulness
- If low, at least one sentence is likely hallucinated

**Range**: 0 to 1

---

### `unsupported_sentences`

**What it measures**: Count of sentences with support below threshold.

**Calculation**:
```
threshold = 0.5
unsupported_sentences = count(sentences where support < threshold)
```

**Interpretation**:
- 0: All sentences are grounded
- High count: Many sentences lack context support

**Range**: 0 to (number of sentences in answer)

---

## Abstention Metrics

These metrics evaluate the system's ability to recognize when it should not answer (e.g., out-of-domain questions).

### Confusion Matrix Components

| Metric | Description |
|--------|-------------|
| `correct_answers_tp` | True Positives: Correctly answered relevant questions |
| `correct_abstentions_tn` | True Negatives: Correctly abstained on irrelevant questions |
| `incorrect_abstentions_fn` | False Negatives: Wrongly abstained on relevant questions |
| `hallucinations_fp` | False Positives: Answered irrelevant questions (hallucination risk) |

### `precision`

**Calculation**:
```
precision = TP / (TP + FP)
```

**Interpretation**: Of all questions answered, what proportion should have been answered?

---

### `recall`

**Calculation**:
```
recall = TP / (TP + FN)
```

**Interpretation**: Of all questions that should be answered, what proportion were answered?

---

### `f1_score`

**Calculation**:
```
f1_score = 2 * (precision * recall) / (precision + recall)
```

**Interpretation**: Harmonic mean of precision and recall.

---

### `accuracy`

**Calculation**:
```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation**: Overall correctness of answer/abstain decisions.

---

## Metric Calculation Details

### Embedding Model

All semantic similarity calculations use the `all-MiniLM-L6-v2` sentence transformer model:

- **Dimension**: 384
- **Max sequence length**: 256 tokens
- **Similarity function**: Cosine similarity

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text)
similarity = util.cos_sim(embedding1, embedding2)
```

### Cosine Similarity

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- Returns 1.0 for identical vectors
- Returns 0.0 for orthogonal vectors
- Returns -1.0 for opposite vectors (rare with text embeddings)

---

## Summary Table

| Metric | Family | Range | Higher is Better? |
|--------|--------|-------|-------------------|
| `avg_q_to_ctx_similarity` | Retrieval | 0-1 | ✓ |
| `max_q_to_ctx_similarity` | Retrieval | 0-1 | ✓ |
| `avg_gt_to_ctx_similarity` | Retrieval | 0-1 | ✓ |
| `context_coverage` | Retrieval | 0-1 | ✓ |
| `retrieval_diversity` | Retrieval | 0-1 | Depends |
| `context_relevance` | Retrieval | 0-1 | ✓ |
| `ndcg@k` | Retrieval | 0-1 | ✓ |
| `semantic_similarity` | Answer | 0-1 | ✓ |
| `answer_question_relevance` | Answer | 0-1 | ✓ |
| `rouge1_f` | Answer | 0-1 | ✓ |
| `rouge2_f` | Answer | 0-1 | ✓ |
| `rougeL_f` | Answer | 0-1 | ✓ |
| `key_term_coverage` | Answer | 0-1 | ✓ |
| `answer_length` | Answer | 0-∞ | Depends |
| `length_ratio` | Answer | 0-∞ | ~1.0 ideal |
| `faithfulness_score` | Faithfulness | 0-1 | ✓ |
| `min_sentence_support` | Faithfulness | 0-1 | ✓ |
| `unsupported_sentences` | Faithfulness | 0-∞ | ✗ (lower better) |

---

## Common Patterns and Interpretations

### Pattern: High Semantic Similarity + Low Faithfulness

**What it means**: The LLM generates correct answers using its parametric knowledge rather than the retrieved context.

**Implication**: The RAG system is functioning more as a vanilla LLM than a retrieval-augmented system.

### Pattern: High NDCG + Low Context Coverage

**What it means**: Chunks are ranked correctly, but even the best chunks don't contain the answer.

**Implication**: Corpus coverage issue—the answer may not exist in your document collection.

### Pattern: High Query-Context Similarity + Low GT-Context Similarity

**What it means**: Retrieved chunks are topically related to the question but don't contain the specific answer.

**Implication**: May need finer-grained chunking or more documents on the topic.

---

## References

- [Sentence-BERT](https://www.sbert.net/) - Embedding model
- [ROUGE Score](https://aclanthology.org/W04-1013/) - Lin, 2004
- [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) - Normalized Discounted Cumulative Gain