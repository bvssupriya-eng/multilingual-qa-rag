"""
Fairness and Bias Metrics for Multilingual RAG QA System.

Provides:
- Cross-language fairness analysis
- Performance disparity detection
- Bias metrics computation
- Visualization functions for fairness/bias plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class FairnessMetrics:
    """Compute fairness and bias metrics for multilingual QA system"""
    
    def __init__(self):
        self.language_names = {
            'en': 'English',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ar': 'Arabic'
        }
    
    def compute_language_fairness(self, language_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute fairness metrics across languages.
        
        Args:
            language_scores: Dict mapping language code to overall score
            
        Returns:
            Dict with fairness metrics
        """
        scores = list(language_scores.values())
        
        if len(scores) < 2:
            return {
                'mean_score': scores[0] if scores else 0.0,
                'std_dev': 0.0,
                'coefficient_variation': 0.0,
                'max_disparity': 0.0,
                'fairness_score': 1.0
            }
        
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        max_disparity = max_score - min_score
        
        # Coefficient of variation (lower is better)
        cv = std_dev / mean_score if mean_score > 0 else 0.0
        
        # Fairness score: 1.0 = perfect fairness, 0.0 = highly unfair
        # Penalize high disparity and high variation
        fairness_score = 1.0 - min(1.0, (max_disparity + cv) / 2)
        
        return {
            'mean_score': round(mean_score, 4),
            'std_dev': round(std_dev, 4),
            'coefficient_variation': round(cv, 4),
            'max_disparity': round(max_disparity, 4),
            'min_score': round(min_score, 4),
            'max_score': round(max_score, 4),
            'fairness_score': round(fairness_score, 4)
        }
    
    def compute_demographic_parity(self, language_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """
        Check if all language groups have similar positive outcome rates.
        
        Args:
            language_metrics: Dict mapping language to metrics dict
            
        Returns:
            Demographic parity metrics
        """
        threshold = 0.7  # Score above this is "positive outcome"
        
        positive_rates = {}
        for lang, metrics in language_metrics.items():
            overall_score = metrics.get('overall_score', 0)
            positive_rates[lang] = 1.0 if overall_score >= threshold else 0.0
        
        rates = list(positive_rates.values())
        if not rates:
            return {'demographic_parity': 1.0, 'parity_difference': 0.0}
        
        max_rate = max(rates)
        min_rate = min(rates)
        parity_diff = max_rate - min_rate
        
        # Perfect parity = 0 difference
        parity_score = 1.0 - parity_diff
        
        return {
            'demographic_parity': round(parity_score, 4),
            'parity_difference': round(parity_diff, 4),
            'positive_rates': positive_rates
        }
    
    def compute_equalized_odds(self, language_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """
        Measure if true positive rates are similar across languages.
        
        Args:
            language_metrics: Dict mapping language to metrics dict
            
        Returns:
            Equalized odds metrics
        """
        tpr_scores = {}
        for lang, metrics in language_metrics.items():
            # Use retrieval precision as proxy for TPR
            tpr_scores[lang] = metrics.get('precision_at_5', 0)
        
        scores = list(tpr_scores.values())
        if len(scores) < 2:
            return {'equalized_odds': 1.0, 'tpr_disparity': 0.0}
        
        max_tpr = max(scores)
        min_tpr = min(scores)
        disparity = max_tpr - min_tpr
        
        # Lower disparity = better equalized odds
        eq_odds_score = 1.0 - min(1.0, disparity)
        
        return {
            'equalized_odds': round(eq_odds_score, 4),
            'tpr_disparity': round(disparity, 4),
            'tpr_by_language': tpr_scores
        }
    
    def compute_bias_metrics(self, language_metrics: Dict[str, Dict]) -> Dict[str, any]:
        """
        Comprehensive bias analysis across languages.
        
        Args:
            language_metrics: Dict mapping language to full metrics dict
            
        Returns:
            Comprehensive bias metrics
        """
        # Extract scores by metric type
        retrieval_scores = {lang: m.get('retrieval_score', 0) for lang, m in language_metrics.items()}
        generation_scores = {lang: m.get('generation_score', 0) for lang, m in language_metrics.items()}
        overall_scores = {lang: m.get('overall_score', 0) for lang, m in language_metrics.items()}
        
        # Compute disparities
        retrieval_disparity = max(retrieval_scores.values()) - min(retrieval_scores.values()) if retrieval_scores else 0
        generation_disparity = max(generation_scores.values()) - min(generation_scores.values()) if generation_scores else 0
        overall_disparity = max(overall_scores.values()) - min(overall_scores.values()) if overall_scores else 0
        
        # Bias score: 0 = no bias, 1 = high bias
        bias_score = (retrieval_disparity + generation_disparity + overall_disparity) / 3
        
        return {
            'bias_score': round(bias_score, 4),
            'retrieval_disparity': round(retrieval_disparity, 4),
            'generation_disparity': round(generation_disparity, 4),
            'overall_disparity': round(overall_disparity, 4),
            'retrieval_scores': retrieval_scores,
            'generation_scores': generation_scores,
            'overall_scores': overall_scores
        }
    
    def detect_language_bias(self, language_metrics: Dict[str, Dict], reference_lang: str = 'en') -> Dict:
        """
        Detect bias relative to a reference language (typically English).
        
        Args:
            language_metrics: Dict mapping language to metrics
            reference_lang: Reference language code (default: 'en')
            
        Returns:
            Bias detection results
        """
        if reference_lang not in language_metrics:
            return {'error': f'Reference language {reference_lang} not found'}
        
        ref_score = language_metrics[reference_lang].get('overall_score', 0)
        
        bias_gaps = {}
        for lang, metrics in language_metrics.items():
            if lang == reference_lang:
                continue
            lang_score = metrics.get('overall_score', 0)
            gap = ref_score - lang_score
            bias_gaps[lang] = round(gap, 4)
        
        # Average bias gap
        avg_gap = np.mean(list(bias_gaps.values())) if bias_gaps else 0.0
        
        # Bias severity
        if avg_gap < 0.05:
            severity = 'low'
        elif avg_gap < 0.15:
            severity = 'moderate'
        else:
            severity = 'high'
        
        return {
            'reference_language': reference_lang,
            'reference_score': round(ref_score, 4),
            'bias_gaps': bias_gaps,
            'average_gap': round(avg_gap, 4),
            'bias_severity': severity
        }


class FairnessVisualizer:
    """Visualization functions for fairness and bias analysis"""
    
    def __init__(self):
        self.language_names = {
            'en': 'English',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ar': 'Arabic'
        }
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    def plot_language_performance_comparison(self, language_metrics: Dict[str, Dict], 
                                            save_path: str = None):
        """
        Bar chart comparing overall performance across languages.
        """
        languages = list(language_metrics.keys())
        scores = [language_metrics[lang].get('overall_score', 0) for lang in languages]
        lang_labels = [self.language_names.get(lang, lang) for lang in languages]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(lang_labels, scores, color=self.colors[:len(languages)], alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Overall Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Language', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Language Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fairness_heatmap(self, language_metrics: Dict[str, Dict], 
                             save_path: str = None):
        """
        Heatmap showing all metrics across languages.
        """
        languages = list(language_metrics.keys())
        lang_labels = [self.language_names.get(lang, lang) for lang in languages]
        
        # Select key metrics
        metric_keys = ['retrieval_score', 'generation_score', 'overall_score', 
                      'precision_at_5', 'completeness', 'relevance']
        metric_labels = ['Retrieval', 'Generation', 'Overall', 
                        'Precision@5', 'Completeness', 'Relevance']
        
        # Build matrix
        data = []
        for lang in languages:
            row = [language_metrics[lang].get(key, 0) for key in metric_keys]
            data.append(row)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_yticks(np.arange(len(lang_labels)))
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.set_yticklabels(lang_labels)
        
        # Add values in cells
        for i in range(len(lang_labels)):
            for j in range(len(metric_labels)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Fairness Heatmap: Metrics Across Languages', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Score')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_bias_disparity(self, bias_metrics: Dict, save_path: str = None):
        """
        Bar chart showing disparity across different metric types.
        """
        categories = ['Retrieval', 'Generation', 'Overall']
        disparities = [
            bias_metrics.get('retrieval_disparity', 0),
            bias_metrics.get('generation_disparity', 0),
            bias_metrics.get('overall_disparity', 0)
        ]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(categories, disparities, color=['#3498db', '#e74c3c', '#2ecc71'], 
                     alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Disparity Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metric Category', fontsize=12, fontweight='bold')
        ax.set_title('Performance Disparity Across Languages', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(disparities) * 1.2 if disparities else 1.0)
        ax.axhline(y=0.1, color='orange', linestyle='--', label='Acceptable Threshold (0.1)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_language_bias_gaps(self, bias_detection: Dict, save_path: str = None):
        """
        Bar chart showing bias gaps relative to reference language.
        """
        if 'bias_gaps' not in bias_detection:
            print("No bias gaps data available")
            return
        
        bias_gaps = bias_detection['bias_gaps']
        languages = list(bias_gaps.keys())
        gaps = list(bias_gaps.values())
        lang_labels = [self.language_names.get(lang, lang) for lang in languages]
        
        # Color based on gap magnitude
        colors = ['#e74c3c' if g > 0.1 else '#f39c12' if g > 0.05 else '#2ecc71' for g in gaps]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(lang_labels, gaps, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ref_lang = bias_detection.get('reference_language', 'en')
        ref_name = self.language_names.get(ref_lang, ref_lang)
        
        ax.set_ylabel('Performance Gap', fontsize=12, fontweight='bold')
        ax.set_xlabel('Language', fontsize=12, fontweight='bold')
        ax.set_title(f'Language Bias Gaps (Reference: {ref_name})', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Low Bias (0.05)')
        ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='High Bias (0.15)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fairness_radar(self, language_metrics: Dict[str, Dict], save_path: str = None):
        """
        Radar chart comparing multiple fairness dimensions across languages.
        """
        from math import pi
        
        languages = list(language_metrics.keys())
        categories = ['Retrieval', 'Generation', 'Precision', 'Completeness', 'Relevance']
        metric_keys = ['retrieval_score', 'generation_score', 'precision_at_5', 
                      'completeness', 'relevance']
        
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for idx, lang in enumerate(languages):
            values = [language_metrics[lang].get(key, 0) for key in metric_keys]
            values += values[:1]
            
            lang_label = self.language_names.get(lang, lang)
            ax.plot(angles, values, 'o-', linewidth=2, label=lang_label, 
                   color=self.colors[idx % len(self.colors)])
            ax.fill(angles, values, alpha=0.15, color=self.colors[idx % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.set_title('Fairness Radar: Multi-Dimensional Comparison', 
                    size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fairness_summary(self, fairness_metrics: Dict, save_path: str = None):
        """
        Summary visualization of key fairness indicators.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Fairness Score Gauge
        ax1 = axes[0, 0]
        fairness_score = fairness_metrics.get('fairness_score', 0)
        colors_gauge = ['#e74c3c', '#f39c12', '#2ecc71']
        bounds = [0, 0.6, 0.8, 1.0]
        
        ax1.barh(['Fairness'], [fairness_score], color='#2ecc71' if fairness_score > 0.8 else '#f39c12' if fairness_score > 0.6 else '#e74c3c', height=0.5)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Score', fontweight='bold')
        ax1.set_title(f'Overall Fairness Score: {fairness_score:.3f}', fontweight='bold')
        ax1.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5)
        ax1.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
        
        # 2. Disparity Metrics
        ax2 = axes[0, 1]
        metrics = ['Max Disparity', 'Std Dev', 'Coeff. Variation']
        values = [
            fairness_metrics.get('max_disparity', 0),
            fairness_metrics.get('std_dev', 0),
            fairness_metrics.get('coefficient_variation', 0)
        ]
        ax2.barh(metrics, values, color=['#3498db', '#e74c3c', '#f39c12'], alpha=0.8)
        ax2.set_xlabel('Value', fontweight='bold')
        ax2.set_title('Disparity Indicators', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Score Range
        ax3 = axes[1, 0]
        min_score = fairness_metrics.get('min_score', 0)
        max_score = fairness_metrics.get('max_score', 0)
        mean_score = fairness_metrics.get('mean_score', 0)
        
        ax3.plot([1, 2, 3], [min_score, mean_score, max_score], 'o-', linewidth=2, markersize=10, color='#3498db')
        ax3.fill_between([1, 2, 3], [min_score, mean_score, max_score], alpha=0.3, color='#3498db')
        ax3.set_xticks([1, 2, 3])
        ax3.set_xticklabels(['Min', 'Mean', 'Max'])
        ax3.set_ylabel('Score', fontweight='bold')
        ax3.set_title('Score Distribution', fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # 4. Fairness Interpretation
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        interpretation = []
        if fairness_score >= 0.8:
            interpretation.append("✓ HIGH FAIRNESS")
            interpretation.append("System performs consistently")
            interpretation.append("across all languages.")
        elif fairness_score >= 0.6:
            interpretation.append("⚠ MODERATE FAIRNESS")
            interpretation.append("Some performance gaps exist.")
            interpretation.append("Consider improvements.")
        else:
            interpretation.append("✗ LOW FAIRNESS")
            interpretation.append("Significant disparities detected.")
            interpretation.append("Action required.")
        
        interpretation.append("")
        interpretation.append(f"Max Disparity: {fairness_metrics.get('max_disparity', 0):.3f}")
        interpretation.append(f"Std Deviation: {fairness_metrics.get('std_dev', 0):.3f}")
        
        text = '\n'.join(interpretation)
        ax4.text(0.5, 0.5, text, ha='center', va='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax4.set_title('Fairness Assessment', fontweight='bold')
        
        plt.suptitle('Fairness Metrics Summary', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def format_fairness_report(fairness_metrics: Dict, bias_metrics: Dict, 
                          demographic_parity: Dict, equalized_odds: Dict) -> str:
    """
    Generate a formatted text report of fairness and bias analysis.
    """
    report = []
    report.append("\n" + "="*60)
    report.append("FAIRNESS & BIAS ANALYSIS REPORT")
    report.append("="*60)
    
    report.append("\n--- Language Fairness ---")
    report.append(f"  Overall Fairness Score:  {fairness_metrics.get('fairness_score', 0):.3f}")
    report.append(f"  Mean Performance:        {fairness_metrics.get('mean_score', 0):.3f}")
    report.append(f"  Std Deviation:           {fairness_metrics.get('std_dev', 0):.3f}")
    report.append(f"  Max Disparity:           {fairness_metrics.get('max_disparity', 0):.3f}")
    report.append(f"  Score Range:             [{fairness_metrics.get('min_score', 0):.3f}, {fairness_metrics.get('max_score', 0):.3f}]")
    
    report.append("\n--- Bias Metrics ---")
    report.append(f"  Overall Bias Score:      {bias_metrics.get('bias_score', 0):.3f}")
    report.append(f"  Retrieval Disparity:     {bias_metrics.get('retrieval_disparity', 0):.3f}")
    report.append(f"  Generation Disparity:    {bias_metrics.get('generation_disparity', 0):.3f}")
    report.append(f"  Overall Disparity:       {bias_metrics.get('overall_disparity', 0):.3f}")
    
    report.append("\n--- Demographic Parity ---")
    report.append(f"  Parity Score:            {demographic_parity.get('demographic_parity', 0):.3f}")
    report.append(f"  Parity Difference:       {demographic_parity.get('parity_difference', 0):.3f}")
    
    report.append("\n--- Equalized Odds ---")
    report.append(f"  Equalized Odds Score:    {equalized_odds.get('equalized_odds', 0):.3f}")
    report.append(f"  TPR Disparity:           {equalized_odds.get('tpr_disparity', 0):.3f}")
    
    report.append("\n--- Interpretation ---")
    fairness_score = fairness_metrics.get('fairness_score', 0)
    if fairness_score >= 0.8:
        report.append("  ✓ System demonstrates HIGH fairness across languages")
    elif fairness_score >= 0.6:
        report.append("  ⚠ System shows MODERATE fairness - improvements recommended")
    else:
        report.append("  ✗ System exhibits LOW fairness - significant action needed")
    
    report.append("="*60 + "\n")
    
    return "\n".join(report)
