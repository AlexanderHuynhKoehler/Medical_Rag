# %%

"""
RAG Configuration Comparison Visualization Script
Generates combined plots comparing all 4 configurations:
- Section-based chunking (no rewrite)
- Section-based chunking (with query rewrite)
- Sliding window chunking (no rewrite)
- Sliding window chunking (with query rewrite)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = '../results'

CONFIGS = {
    'section_no_rewrite': 'Section',
    'section_with_rewrite': 'Section + Rewrite',
    'sliding_no_rewrite': 'Sliding Window',
    'sliding_with_rewrite': 'Sliding + Rewrite'
}

OUTPUT_DIR = '../results/comparison_plots'

# Colors for each configuration
COLORS = {
    'section_no_rewrite': '#3498db',      # Blue
    'section_with_rewrite': '#2980b9',    # Dark Blue
    'sliding_no_rewrite': '#e74c3c',      # Red
    'sliding_with_rewrite': '#c0392b'     # Dark Red
}

LINE_STYLES = {
    'section_no_rewrite': '-',
    'section_with_rewrite': '--',
    'sliding_no_rewrite': '-',
    'sliding_with_rewrite': '--'
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_summaries() -> Dict[str, Dict]:
    """Load evaluation_summary.json from each configuration folder."""
    summaries = {}
    
    for config_key, config_name in CONFIGS.items():
        json_path = os.path.join(BASE_DIR, config_key, 'evaluation_summary.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                summaries[config_key] = json.load(f)
            print(f"✓ Loaded: {config_key}")
        else:
            print(f"✗ Missing: {json_path}")
    
    return summaries


def load_all_detailed_results() -> Dict[str, pd.DataFrame]:
    """Load detailed_results.csv from each configuration folder."""
    results = {}
    
    for config_key, config_name in CONFIGS.items():
        csv_path = os.path.join(BASE_DIR, config_key, 'detailed_results.csv')
        
        if os.path.exists(csv_path):
            results[config_key] = pd.read_csv(csv_path)
            print(f"✓ Loaded: {config_key} ({len(results[config_key])} rows)")
        else:
            print(f"✗ Missing: {csv_path}")
    
    return results


# ============================================================================
# COMBINED VISUALIZATIONS
# ============================================================================

def plot_radar_comparison(summaries: Dict[str, Dict], output_dir: str):
    """
    Radar chart with all configurations overlaid.
    """
    print("\n📊 Creating combined radar chart...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to display (all should be 0-1 scale)
    metrics = {
        'semantic_similarity': 'Answer\nSimilarity',
        'answer_question_relevance': 'Question\nRelevance',
        'faithfulness_score': 'Faithfulness',
        'context_relevance': 'Context\nRelevance',
        'context_coverage': 'Context\nCoverage',
        'key_term_coverage': 'Key Term\nCoverage'
    }
    
    labels = list(metrics.values())
    num_vars = len(labels)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for config_key, summary in summaries.items():
        overall = summary.get('overall_metrics', {})
        values = [overall.get(m, 0) for m in metrics.keys()]
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 
                color=COLORS[config_key], 
                linestyle=LINE_STYLES[config_key],
                linewidth=2.5, 
                label=CONFIGS[config_key])
        ax.fill(angles, values, color=COLORS[config_key], alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('RAG Configuration Comparison', fontsize=16, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'radar_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_grouped_bar_comparison(summaries: Dict[str, Dict], output_dir: str):
    """
    Grouped bar chart comparing key metrics across configurations.
    """
    print("\n📊 Creating grouped bar chart...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Key metrics to compare
    metrics = {
        'semantic_similarity': 'Semantic\nSimilarity',
        'faithfulness_score': 'Faithfulness',
        'context_relevance': 'Context\nRelevance',
        'avg_gt_to_ctx_similarity': 'GT-Context\nSimilarity',
        'key_term_coverage': 'Key Term\nCoverage',
        'ndcg@k': 'NDCG@k'
    }
    
    x = np.arange(len(metrics))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, (config_key, summary) in enumerate(summaries.items()):
        overall = summary.get('overall_metrics', {})
        values = [overall.get(m, 0) for m in metrics.keys()]
        
        bars = ax.bar(x + i * width, values, width, 
                      label=CONFIGS[config_key],
                      color=COLORS[config_key],
                      alpha=0.85)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Key Metrics Comparison Across Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics.values(), fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'grouped_bar_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_faithfulness_vs_similarity_grid(detailed_results: Dict[str, pd.DataFrame], output_dir: str):
    """
    2x2 grid of scatter plots showing faithfulness vs semantic similarity.
    """
    print("\n📊 Creating faithfulness vs similarity grid...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    config_keys = list(CONFIGS.keys())
    
    for idx, config_key in enumerate(config_keys):
        ax = axes[idx]
        
        if config_key in detailed_results:
            df = detailed_results[config_key]
            
            if 'category' in df.columns:
                categories = df['category'].unique()
                colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
                
                for cat, color in zip(categories, colors):
                    mask = df['category'] == cat
                    ax.scatter(df.loc[mask, 'faithfulness_score'],
                              df.loc[mask, 'semantic_similarity'],
                              label=cat, alpha=0.6, s=40, c=[color])
            else:
                ax.scatter(df['faithfulness_score'], df['semantic_similarity'],
                          alpha=0.6, s=40, c=COLORS[config_key])
            
            # Add quadrant lines
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
            
            # Calculate and show mean point
            mean_faith = df['faithfulness_score'].mean()
            mean_sim = df['semantic_similarity'].mean()
            ax.scatter([mean_faith], [mean_sim], c='red', s=200, marker='X', 
                      edgecolors='black', linewidths=2, zorder=5, label=f'Mean ({mean_faith:.2f}, {mean_sim:.2f})')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Faithfulness Score', fontsize=10)
        ax.set_ylabel('Semantic Similarity', fontsize=10)
        ax.set_title(CONFIGS[config_key], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Add single legend for categories (from first plot with data)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)
    
    plt.suptitle('Faithfulness vs Answer Quality by Configuration\n(Upper-left = Parametric knowledge, Lower-right = Grounded but wrong)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'faithfulness_vs_similarity_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_metric_improvement_heatmap(summaries: Dict[str, Dict], output_dir: str):
    """
    Heatmap showing metric values across configurations.
    """
    print("\n📊 Creating metrics heatmap...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = [
        'semantic_similarity',
        'faithfulness_score',
        'context_relevance',
        'avg_gt_to_ctx_similarity',
        'key_term_coverage',
        'ndcg@k'
    ]
    
    metric_labels = [
        'Semantic Similarity',
        'Faithfulness',
        'Context Relevance',
        'GT-Context Similarity',
        'Key Term Coverage',
        'NDCG@k'
    ]
    
    # Build data matrix
    data = []
    config_labels = []
    
    for config_key in CONFIGS.keys():
        if config_key in summaries:
            overall = summaries[config_key].get('overall_metrics', {})
            row = [overall.get(m, 0) for m in metrics]
            data.append(row)
            config_labels.append(CONFIGS[config_key])
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=metric_labels, yticklabels=config_labels,
                ax=ax, vmin=0, vmax=1, annot_kws={'size': 11})
    
    ax.set_title('Metrics Comparison Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Configuration', fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_delta_bar_chart(summaries: Dict[str, Dict], output_dir: str):
    """
    Bar chart showing improvement/regression from baseline (section_no_rewrite).
    """
    print("\n📊 Creating delta comparison chart...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_key = 'section_no_rewrite'
    
    if baseline_key not in summaries:
        print("  ⚠️ Baseline (section_no_rewrite) not found. Skipping delta chart.")
        return
    
    metrics = {
        'semantic_similarity': 'Semantic Sim',
        'faithfulness_score': 'Faithfulness',
        'context_relevance': 'Context Rel',
        'avg_gt_to_ctx_similarity': 'GT-Context Sim',
        'key_term_coverage': 'Key Term Cov'
    }
    
    baseline = summaries[baseline_key].get('overall_metrics', {})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    other_configs = [k for k in CONFIGS.keys() if k != baseline_key]
    
    for i, config_key in enumerate(other_configs):
        if config_key in summaries:
            overall = summaries[config_key].get('overall_metrics', {})
            deltas = [(overall.get(m, 0) - baseline.get(m, 0)) * 100 for m in metrics.keys()]  # as percentage points
            
            bars = ax.bar(x + i * width, deltas, width,
                          label=CONFIGS[config_key],
                          color=COLORS[config_key],
                          alpha=0.85)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Change from Baseline (percentage points)', fontsize=12)
    ax.set_title(f'Performance Change vs Baseline ({CONFIGS[baseline_key]})', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics.values(), fontsize=10)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'delta_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def generate_summary_table(summaries: Dict[str, Dict], output_dir: str):
    """
    Generate a summary table as CSV for easy inclusion in papers.
    """
    print("\n📋 Generating summary table...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = [
        'semantic_similarity',
        'faithfulness_score',
        'context_relevance',
        'avg_gt_to_ctx_similarity',
        'key_term_coverage',
        'ndcg@k'
    ]
    
    rows = []
    for config_key, config_name in CONFIGS.items():
        if config_key in summaries:
            overall = summaries[config_key].get('overall_metrics', {})
            row = {'Configuration': config_name}
            for m in metrics:
                row[m] = round(overall.get(m, 0), 4)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    save_path = os.path.join(output_dir, 'summary_table.csv')
    df.to_csv(save_path, index=False)
    print(f"  ✓ Saved to {save_path}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("RAG CONFIGURATION COMPARISON")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading evaluation summaries...")
    summaries = load_all_summaries()
    
    print("\n📂 Loading detailed results...")
    detailed_results = load_all_detailed_results()
    
    if not summaries:
        print("\n❌ No data found. Check your folder structure.")
        print(f"   Expected: {BASE_DIR}/<config_name>/evaluation_summary.json")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate all visualizations
    plot_radar_comparison(summaries, OUTPUT_DIR)
    plot_grouped_bar_comparison(summaries, OUTPUT_DIR)
    plot_metric_improvement_heatmap(summaries, OUTPUT_DIR)
    plot_delta_bar_chart(summaries, OUTPUT_DIR)
    
    if detailed_results:
        plot_faithfulness_vs_similarity_grid(detailed_results, OUTPUT_DIR)
    
    generate_summary_table(summaries, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - radar_comparison.png")
    print("  - grouped_bar_comparison.png")
    print("  - metrics_heatmap.png")
    print("  - delta_comparison.png")
    print("  - faithfulness_vs_similarity_grid.png")
    print("  - summary_table.csv")


if __name__ == "__main__":
    main()
# %%
