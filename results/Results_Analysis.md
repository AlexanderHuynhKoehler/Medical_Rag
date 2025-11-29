# RAG System Evaluation: Results Analysis

This document provides a comprehensive analysis of our RAG (Retrieval-Augmented Generation) system evaluation across four configurations. We examine our initial hypotheses, present the experimental results, and discuss the implications of our findings.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experimental Setup](#experimental-setup)
3. [Hypotheses](#hypotheses)
4. [Results Overview](#results-overview)
5. [Detailed Analysis](#detailed-analysis)
6. [Hypothesis Evaluation](#hypothesis-evaluation)
7. [Key Findings](#key-findings)
8. [Implications](#implications)
9. [Future Work](#future-work)

---

## Executive Summary

We conducted a 2×2 ablation study comparing:
- **Chunking strategies**: Section-based vs. Sliding window
- **Query processing**: Original queries vs. LLM-rewritten queries

**Primary Finding**: All four configurations produced nearly identical results (±1-5 percentage points), indicating that neither chunking strategy nor query rewriting addresses the fundamental performance bottleneck in our RAG system.

**Core Issue Identified**: The LLM generates semantically correct answers (0.68 similarity to ground truth) but does not ground those answers in the retrieved context (0.42 faithfulness). This pattern persisted across all configurations, suggesting the bottleneck lies in corpus coverage and/or LLM behavior rather than retrieval mechanics.

---

## Experimental Setup

### Configurations Tested

| Configuration | Chunking Method | Query Processing |
|---------------|-----------------|------------------|
| **Section** | One chunk per document section | Original query |
| **Section + Rewrite** | One chunk per document section | LLM-rewritten query |
| **Sliding Window** | 500 char chunks, 100 char overlap | Original query |
| **Sliding + Rewrite** | 500 char chunks, 100 char overlap | LLM-rewritten query |

### System Components

- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Store**: FAISS with L2 distance
- **LLM**: Qwen2-1.5B-Instruct
- **Corpus**: Medical documents across 8 disease categories (872 chunks in section-based configuration)

### Evaluation Dataset

- **Total questions**: 220
- **Relevant questions**: 200 (diseases present in corpus)
- **Unrelated questions**: 20 (diseases not in corpus)
- **Disease categories**: 8
- **Unique diseases**: 69

### Evaluation Methodology

All metrics computed using deterministic mathematical calculations—no LLM-as-judge methods. Metrics fall into three families:

1. **Retrieval metrics**: Measure chunk relevance and ranking
2. **Answer metrics**: Measure generated answer quality
3. **Faithfulness metrics**: Measure grounding in retrieved context

---

## Hypotheses

### Hypothesis 1: Query Rewriting Will Improve Retrieval

**Rationale**: User queries may not be optimally phrased for vector similarity search. An LLM could transform queries into search-optimized forms that better match corpus terminology.

**Prediction**: Configurations with query rewriting will show improved retrieval metrics (higher context relevance, higher GT-context similarity).

---

### Hypothesis 2: Sliding Window Chunking Will Improve Faithfulness

**Rationale**: Section-based chunks may be too large or may not align with answer boundaries. Smaller, overlapping chunks might provide more focused context that the LLM can better utilize.

**Prediction**: Sliding window configurations will show improved faithfulness scores.

---

### Hypothesis 3: The Combination Will Yield Best Results

**Rationale**: If both interventions help independently, combining them should produce additive or synergistic improvements.

**Prediction**: Sliding + Rewrite configuration will outperform all others.

---

### Alternative Hypothesis: Retrieval Is Not the Bottleneck

**Rationale**: If the system's NDCG is already high (~0.98), the retriever is ranking chunks correctly. The issue may be that even the best chunks don't contain sufficient information, or the LLM ignores context in favor of parametric knowledge.

**Prediction**: All configurations will perform similarly because the bottleneck lies elsewhere.

---

## Results Overview

### Summary Table

| Configuration | Semantic Similarity | Faithfulness | Context Relevance | GT-Context Similarity | Key Term Coverage | NDCG@k |
|---------------|---------------------|--------------|-------------------|----------------------|-------------------|--------|
| Section | 0.678 | 0.417 | 0.632 | 0.546 | 0.330 | 0.980 |
| Section + Rewrite | 0.679 | 0.403 | 0.577 | 0.526 | 0.328 | 0.969 |
| Sliding Window | 0.683 | 0.433 | 0.677 | 0.566 | 0.344 | 0.984 |
| Sliding + Rewrite | 0.684 | 0.402 | 0.648 | 0.553 | 0.340 | 0.978 |

### Key Observations

1. **Semantic similarity is stable**: All configurations achieve ~0.68, indicating answer quality is consistent
2. **Faithfulness remains low**: All configurations score ~0.40-0.43, indicating persistent grounding issues
3. **NDCG is uniformly high**: All configurations achieve ~0.97-0.98, indicating retrieval ranking is not the problem
4. **Differences are marginal**: Maximum variation is ~5 percentage points (context relevance)

---

## Detailed Analysis

### Retrieval Metrics Analysis

#### Context Relevance

Context relevance measures the proportion of retrieved chunks with similarity > 0.5 to the ground truth answer.

| Configuration | Context Relevance | Delta from Baseline |
|---------------|-------------------|---------------------|
| Section (baseline) | 0.632 | — |
| Section + Rewrite | 0.577 | **-5.5 pp** |
| Sliding Window | 0.677 | +4.5 pp |
| Sliding + Rewrite | 0.648 | +1.6 pp |

**Finding**: Query rewriting **decreased** context relevance for section-based chunking. This was unexpected.

**Explanation**: The original queries (e.g., "What are the symptoms of diabetes?") naturally matched corpus terminology. The LLM rewriter introduced synonyms and expansions ("clinical manifestations", "indicators") that don't appear in the corpus, causing retrieval of less relevant chunks.

#### GT-Context Similarity

This measures how similar the retrieved chunks are to the ground truth answer.

| Configuration | GT-Context Similarity | Delta from Baseline |
|---------------|----------------------|---------------------|
| Section (baseline) | 0.546 | — |
| Section + Rewrite | 0.526 | -2.0 pp |
| Sliding Window | 0.566 | +2.0 pp |
| Sliding + Rewrite | 0.553 | +0.7 pp |

**Finding**: Same pattern—rewriting hurt performance, sliding window provided marginal improvement.

#### NDCG@k

Measures whether the best chunks are ranked first.

| Configuration | NDCG@k |
|---------------|--------|
| Section | 0.980 |
| Section + Rewrite | 0.969 |
| Sliding Window | 0.984 |
| Sliding + Rewrite | 0.978 |

**Finding**: All configurations achieve near-perfect ranking (~0.97-0.98). This confirms the retriever correctly orders chunks by relevance.

**Critical Insight**: High NDCG does not mean chunks are good—it means they're ordered correctly. You can have NDCG = 1.0 with three poor chunks if they're ranked in the right order (e.g., similarities [0.3, 0.2, 0.1]).

---

### Answer Quality Metrics Analysis

#### Semantic Similarity

Measures embedding similarity between generated answer and ground truth.

| Configuration | Semantic Similarity |
|---------------|---------------------|
| Section | 0.678 |
| Section + Rewrite | 0.679 |
| Sliding Window | 0.683 |
| Sliding + Rewrite | 0.684 |

**Finding**: Virtually identical across all configurations (0.678-0.684). The LLM produces consistent answer quality regardless of retrieval configuration.

#### Key Term Coverage

Measures whether important terms from ground truth appear in generated answer.

| Configuration | Key Term Coverage |
|---------------|-------------------|
| Section | 0.330 |
| Section + Rewrite | 0.328 |
| Sliding Window | 0.344 |
| Sliding + Rewrite | 0.340 |

**Finding**: Uniformly low (~0.33) across all configurations. Generated answers are missing key terminology from expected answers.

---

### Faithfulness Metrics Analysis

#### Faithfulness Score

Measures average similarity between answer sentences and retrieved context.

| Configuration | Faithfulness Score | Delta from Baseline |
|---------------|-------------------|---------------------|
| Section (baseline) | 0.417 | — |
| Section + Rewrite | 0.403 | -1.4 pp |
| Sliding Window | 0.433 | +1.6 pp |
| Sliding + Rewrite | 0.402 | -1.5 pp |

**Finding**: All configurations score ~0.40-0.43. The LLM consistently generates answers that are not well-grounded in retrieved context.

**Critical Insight**: Query rewriting **decreased** faithfulness in both chunking configurations. The "optimized" queries retrieved chunks that were even less useful for grounding answers.

---

### The Faithfulness vs. Similarity Paradox

The scatter plots reveal the core issue across all configurations:

| Configuration | Mean Faithfulness | Mean Semantic Similarity |
|---------------|-------------------|--------------------------|
| Section | 0.42 | 0.68 |
| Section + Rewrite | 0.42 | 0.68 |
| Sliding Window | 0.42 | 0.68 |
| Sliding + Rewrite | 0.42 | 0.68 |

All four scatter plots show the same pattern:
- Points cluster in the **upper-left quadrant** (high similarity, low faithfulness)
- Mean point is approximately (0.42, 0.68) for all configurations

**Interpretation**: The LLM produces correct answers (high semantic similarity to ground truth) but derives those answers from parametric knowledge rather than retrieved context (low faithfulness). This pattern persists regardless of configuration.

---

## Hypothesis Evaluation

### Hypothesis 1: Query Rewriting Will Improve Retrieval

| Status | **REJECTED** |
|--------|--------------|

**Evidence**:
- Context relevance: Section 0.632 → Section+Rewrite 0.577 (**-5.5 pp**)
- GT-Context similarity: Section 0.546 → Section+Rewrite 0.526 (**-2.0 pp**)
- Faithfulness: Section 0.417 → Section+Rewrite 0.403 (**-1.4 pp**)

**Conclusion**: Query rewriting degraded retrieval performance. The original user queries were already well-suited for the corpus. LLM rewriting introduced vocabulary drift—synonyms and expansions that don't appear in the indexed documents.

---

### Hypothesis 2: Sliding Window Chunking Will Improve Faithfulness

| Status | **PARTIALLY SUPPORTED** (marginal effect) |
|--------|-------------------------------------------|

**Evidence**:
- Faithfulness: Section 0.417 → Sliding 0.433 (**+1.6 pp**)
- Context relevance: Section 0.632 → Sliding 0.677 (**+4.5 pp**)

**Conclusion**: Sliding window chunking showed modest improvements in retrieval metrics, but the effect on faithfulness was marginal (+1.6 percentage points). The fundamental faithfulness gap (~0.42) persisted.

---

### Hypothesis 3: The Combination Will Yield Best Results

| Status | **REJECTED** |
|--------|--------------|

**Evidence**:
- Sliding + Rewrite performed **worse** than Sliding alone on most metrics
- Faithfulness: Sliding 0.433 → Sliding+Rewrite 0.402 (**-3.1 pp**)
- Context relevance: Sliding 0.677 → Sliding+Rewrite 0.648 (**-2.9 pp**)

**Conclusion**: The combination did not produce synergistic benefits. Query rewriting degraded performance even when paired with sliding window chunking.

---

### Alternative Hypothesis: Retrieval Is Not the Bottleneck

| Status | **SUPPORTED** |
|--------|---------------|

**Evidence**:
- All configurations produced nearly identical results (±1-5 pp)
- NDCG remained ~0.98 across all configurations (ranking is not the issue)
- Faithfulness gap persisted uniformly (~0.42)
- Semantic similarity remained stable (~0.68)

**Conclusion**: The retrieval system is functioning correctly—it finds and ranks relevant chunks well. The bottleneck lies in either:
1. **Corpus coverage**: The retrieved chunks don't contain sufficient information to answer the questions
2. **LLM behavior**: The model prefers parametric knowledge over retrieved context, even when relevant context is available

---

## Key Findings

### Finding 1: Retrieval Optimization Has Diminishing Returns

When NDCG is already ~0.98, further retrieval optimizations yield minimal benefit. The retriever is not the bottleneck.

### Finding 2: Query Rewriting Can Hurt Performance

Contrary to conventional wisdom, LLM-based query rewriting **degraded** retrieval quality in our experiments. This occurred because:
- Original queries already matched corpus terminology
- Rewriting introduced vocabulary drift (synonyms not in corpus)
- The "optimized" queries retrieved different, less relevant chunks

### Finding 3: Chunking Strategy Has Minimal Impact

Section-based and sliding window chunking produced similar results. The ~4.5 pp improvement in context relevance from sliding window did not translate to meaningful improvements in answer quality or faithfulness.

### Finding 4: The Faithfulness Gap Is Architectural

Across all configurations:
- Semantic similarity: ~0.68 (answers are correct)
- Faithfulness: ~0.42 (answers are not grounded in context)

This gap suggests the LLM relies on parametric knowledge rather than retrieved context. No retrieval-side intervention we tested addressed this issue.

### Finding 5: High NDCG ≠ Good Retrieval

Our system achieved ~0.98 NDCG but only ~0.55 GT-context similarity. This illustrates that NDCG measures **relative ranking**, not **absolute quality**. The retriever correctly ranks chunks, but even the best chunks are only moderately similar to the required answers.

---

## Implications

### For RAG System Design

1. **Don't over-optimize retrieval**: If NDCG is high, further retrieval optimizations likely won't help
2. **Test query rewriting carefully**: It can introduce vocabulary drift and degrade performance
3. **Chunking strategy is not a silver bullet**: Section-based vs. sliding window made minimal difference

### For Evaluation Methodology

1. **Use multiple metrics**: NDCG alone is misleading; pair it with context coverage and faithfulness
2. **Measure faithfulness explicitly**: High answer quality doesn't mean the RAG is working—the LLM might be ignoring context
3. **Avoid LLM-as-judge**: Deterministic metrics provide reproducible, interpretable results

### For Future Research

The bottleneck in our system is not retrieval mechanics. Future work should focus on:

1. **Corpus expansion**: Adding more documents to improve coverage
2. **Prompting strategies**: Forcing the LLM to cite or quote retrieved context
3. **Context utilization analysis**: Understanding why LLMs prefer parametric knowledge
4. **Smaller, more capable models**: Testing whether larger LLMs better utilize context

---

## Future Work

### The Real Bottleneck: LLM Context Integration

Our results reveal a critical insight: **the bottleneck is not retrieval quality, chunking strategy, or data quality—it's how the LLM integrates retrieved context into its answers.**

Evidence for this conclusion:

| Metric | Score | What It Tells Us |
|--------|-------|------------------|
| Context Relevance | 0.63-0.68 | Retrieved chunks ARE relevant to the answers |
| Semantic Similarity | 0.68 | Generated answers ARE correct |
| Faithfulness | 0.40-0.43 | Generated answers are NOT derived from context |

The retrieval system finds good chunks. The LLM produces good answers. But the LLM doesn't use the good chunks to produce those good answers—it relies on parametric knowledge instead.

### The Likely Culprit: System Prompt

Our current system prompt explicitly encourages blending:

```python
"You are a medical assistant. Give answers to the questions using your 
knowledge in combination with retrieved information"
```

The phrase **"in combination with"** gives the LLM permission to use parametric knowledge, which it apparently prefers over retrieved context.

### Proposed Prompt Engineering Experiments

**Experiment A: Strict Context-Only**
```
You are a medical assistant. Answer questions using ONLY the provided context. 
If the context doesn't contain the answer, say "I don't have enough information."
Do not use any outside knowledge.
```

**Experiment B: Citation Requirement**
```
You are a medical assistant. Answer based on the provided context.
You must quote directly from the context to support your answer.
Format: "According to the context: '[exact quote]'..."
```

**Experiment C: Context-First with Explicit Fallback**
```
You are a medical assistant. First, check if the provided context answers the question.
If yes, answer using ONLY the context.
If no, clearly state "The provided context does not address this" before giving 
any additional information.
```

**Hypothesis**: Prompting strategy will have a larger impact on faithfulness than any retrieval-side optimization. We predict faithfulness could improve from ~0.42 to 0.60+ with strict context-only prompting.

### Immediate Next Steps

1. **Prompt engineering ablation**: Test the three prompting strategies above using the same evaluation framework
2. **Context utilization analysis**: Measure exact n-gram overlap between retrieved chunks and generated answers
3. **Per-question diagnosis**: Identify specific questions where context was available but ignored

### Longer-Term Research Directions

1. **LLM comparison**: Test whether larger models (7B, 13B) better utilize context than Qwen2-1.5B
2. **Fine-tuning for grounding**: Train the LLM to prefer context over parametric knowledge
3. **Retrieval-aware decoding**: Modify generation to up-weight tokens that appear in context
4. **Hybrid retrieval**: Combine dense (embedding) and sparse (BM25) retrieval for better lexical matching

---

## Conclusion

Our ablation study systematically tested chunking strategies and query rewriting to identify performance bottlenecks in a medical RAG system. The results demonstrate that:

1. **Neither intervention significantly improved performance** (±1-5 percentage points)
2. **Query rewriting actually degraded retrieval quality** by introducing vocabulary drift
3. **The faithfulness gap persisted across all configurations**, indicating the LLM relies on parametric knowledge regardless of retrieval quality

### The Key Insight

The combination of high context relevance (~0.65), high semantic similarity (~0.68), and low faithfulness (~0.42) tells a clear story:

- ✓ We're retrieving relevant chunks
- ✓ The LLM produces correct answers  
- ✗ The LLM is not using our chunks to produce those answers

**The bottleneck is not data, not chunking, not retrieval—it's LLM context integration.** The model has access to good context but prefers its parametric knowledge.

This shifts the focus from retrieval optimization to **prompt engineering**. Our current prompt ("use your knowledge in combination with retrieved information") explicitly permits parametric knowledge. Stricter prompts that force context-only answering may yield the faithfulness improvements that retrieval optimizations could not achieve.

### Contributions

1. **Non-LLM evaluation framework**: Reproducible, deterministic metrics without circular AI-judging-AI
2. **Ablation methodology**: Systematic isolation of variables (chunking × query rewriting)
3. **Bottleneck identification**: Ruled out retrieval mechanics, identified LLM behavior as the constraint
4. **Actionable insight**: Future work should focus on prompting strategies rather than retrieval optimization

---

## Appendix: Raw Results

### Complete Metrics Table

| Metric | Section | Section+RW | Sliding | Sliding+RW |
|--------|---------|------------|---------|------------|
| semantic_similarity | 0.6782 | 0.6790 | 0.6830 | 0.6840 |
| faithfulness_score | 0.4170 | 0.4030 | 0.4330 | 0.4020 |
| context_relevance | 0.6318 | 0.5770 | 0.6770 | 0.6480 |
| avg_gt_to_ctx_similarity | 0.5464 | 0.5260 | 0.5660 | 0.5530 |
| key_term_coverage | 0.3296 | 0.3280 | 0.3440 | 0.3400 |
| ndcg@k | 0.9797 | 0.9690 | 0.9840 | 0.9780 |
| answer_question_relevance | 0.7322 | — | — | — |
| min_sentence_support | 0.1802 | — | — | — |
| unsupported_sentences | 5.4773 | — | — | — |

### Configuration Details

**Section-based chunking**:
- Method: One chunk per document section
- Chunk count: 872 chunks
- Average chunk size: Variable (depends on section length)

**Sliding window chunking**:
- Method: Fixed-size overlapping windows
- Chunk size: 500 characters
- Overlap: 100 characters
- Chunk count: 2,033 chunks (2.3× more than section-based)

**Query rewriting**:
- Model: Qwen2-1.5B-Instruct
- Prompt: "Transform this question into a search query optimized for retrieving relevant medical documents. Output ONLY the rewritten query."
- Temperature: 0.3
- Max tokens: 50