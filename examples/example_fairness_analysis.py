"""
Example usage of Fairness and Bias Metrics module.

This script demonstrates how to:
1. Compute fairness metrics across languages
2. Detect bias in the system
3. Generate fairness visualizations
"""

import sys
from evaluation.fairness_metrics import FairnessMetrics, FairnessVisualizer, format_fairness_report

# Example: Simulated metrics from multiple language runs
# In practice, you would collect these from actual MLflow runs or system outputs

def example_fairness_analysis():
    """Example fairness analysis with simulated data"""
    
    # Simulated metrics from 4 language runs
    language_metrics = {
        'en': {
            'overall_score': 0.82,
            'retrieval_score': 0.85,
            'generation_score': 0.79,
            'precision_at_5': 0.88,
            'completeness': 0.85,
            'relevance': 0.76,
            'mrr': 0.90,
            'context_use': 0.72
        },
        'hi': {
            'overall_score': 0.75,
            'retrieval_score': 0.78,
            'generation_score': 0.72,
            'precision_at_5': 0.80,
            'completeness': 0.78,
            'relevance': 0.68,
            'mrr': 0.82,
            'context_use': 0.65
        },
        'bn': {
            'overall_score': 0.71,
            'retrieval_score': 0.74,
            'generation_score': 0.68,
            'precision_at_5': 0.76,
            'completeness': 0.72,
            'relevance': 0.65,
            'mrr': 0.78,
            'context_use': 0.62
        },
        'ar': {
            'overall_score': 0.77,
            'retrieval_score': 0.80,
            'generation_score': 0.74,
            'precision_at_5': 0.82,
            'completeness': 0.76,
            'relevance': 0.70,
            'mrr': 0.84,
            'context_use': 0.68
        }
    }
    
    print("\n" + "="*60)
    print("FAIRNESS & BIAS ANALYSIS - EXAMPLE")
    print("="*60)
    
    # Initialize
    fairness = FairnessMetrics()
    visualizer = FairnessVisualizer()
    
    # 1. Compute Language Fairness
    print("\n[1/6] Computing language fairness metrics...")
    overall_scores = {lang: metrics['overall_score'] for lang, metrics in language_metrics.items()}
    fairness_metrics = fairness.compute_language_fairness(overall_scores)
    print(f"  ✓ Fairness Score: {fairness_metrics['fairness_score']:.3f}")
    print(f"  ✓ Max Disparity: {fairness_metrics['max_disparity']:.3f}")
    
    # 2. Compute Bias Metrics
    print("\n[2/6] Computing bias metrics...")
    bias_metrics = fairness.compute_bias_metrics(language_metrics)
    print(f"  ✓ Bias Score: {bias_metrics['bias_score']:.3f}")
    print(f"  ✓ Overall Disparity: {bias_metrics['overall_disparity']:.3f}")
    
    # 3. Demographic Parity
    print("\n[3/6] Computing demographic parity...")
    demographic_parity = fairness.compute_demographic_parity(language_metrics)
    print(f"  ✓ Parity Score: {demographic_parity['demographic_parity']:.3f}")
    
    # 4. Equalized Odds
    print("\n[4/6] Computing equalized odds...")
    equalized_odds = fairness.compute_equalized_odds(language_metrics)
    print(f"  ✓ Equalized Odds: {equalized_odds['equalized_odds']:.3f}")
    
    # 5. Language Bias Detection
    print("\n[5/6] Detecting language bias (reference: English)...")
    bias_detection = fairness.detect_language_bias(language_metrics, reference_lang='en')
    print(f"  ✓ Average Gap: {bias_detection['average_gap']:.3f}")
    print(f"  ✓ Bias Severity: {bias_detection['bias_severity'].upper()}")
    
    # 6. Generate Report
    print("\n[6/6] Generating fairness report...")
    report = format_fairness_report(fairness_metrics, bias_metrics, 
                                   demographic_parity, equalized_odds)
    print(report)
    
    # Generate Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    print("\n[1/6] Language Performance Comparison...")
    visualizer.plot_language_performance_comparison(
        language_metrics, 
        save_path='outputs/fairness_language_comparison.png'
    )
    
    print("[2/6] Fairness Heatmap...")
    visualizer.plot_fairness_heatmap(
        language_metrics,
        save_path='outputs/fairness_heatmap.png'
    )
    
    print("[3/6] Bias Disparity Chart...")
    visualizer.plot_bias_disparity(
        bias_metrics,
        save_path='outputs/bias_disparity.png'
    )
    
    print("[4/6] Language Bias Gaps...")
    visualizer.plot_language_bias_gaps(
        bias_detection,
        save_path='outputs/language_bias_gaps.png'
    )
    
    print("[5/6] Fairness Radar Chart...")
    visualizer.plot_fairness_radar(
        language_metrics,
        save_path='outputs/fairness_radar.png'
    )
    
    print("[6/6] Fairness Summary Dashboard...")
    visualizer.plot_fairness_summary(
        fairness_metrics,
        save_path='outputs/fairness_summary.png'
    )
    
    print("\n✓ All visualizations saved to outputs/ directory")
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


def example_with_mlflow_data():
    """
    Example showing how to collect metrics from MLflow runs.
    
    This is a template - you would need to adapt it to your actual MLflow setup.
    """
    print("\n" + "="*60)
    print("COLLECTING METRICS FROM MLFLOW")
    print("="*60)
    
    try:
        import mlflow
        
        # Set experiment
        mlflow.set_experiment("multilingual_qa_system")
        
        # Get all runs
        runs = mlflow.search_runs()
        
        # Group by language
        language_metrics = {}
        
        for _, run in runs.iterrows():
            lang = run.get('params.detected_language')
            if lang and lang not in language_metrics:
                language_metrics[lang] = {
                    'overall_score': run.get('metrics.overall_score', 0),
                    'retrieval_score': run.get('metrics.retrieval_score', 0),
                    'generation_score': run.get('metrics.generation_score', 0),
                    'precision_at_5': run.get('metrics.precision_at_5', 0),
                    'completeness': run.get('metrics.completeness', 0),
                    'relevance': run.get('metrics.relevance', 0),
                }
        
        print(f"✓ Collected metrics for {len(language_metrics)} languages")
        
        # Now run fairness analysis
        fairness = FairnessMetrics()
        overall_scores = {lang: m['overall_score'] for lang, m in language_metrics.items()}
        fairness_metrics = fairness.compute_language_fairness(overall_scores)
        
        print(f"\nFairness Score: {fairness_metrics['fairness_score']:.3f}")
        
    except Exception as e:
        print(f"Note: MLflow integration example - {e}")
        print("Use example_fairness_analysis() for simulated data demo")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    
    # Run the example with simulated data
    example_fairness_analysis()
    
    # Uncomment to try MLflow integration (requires actual runs)
    # example_with_mlflow_data()
