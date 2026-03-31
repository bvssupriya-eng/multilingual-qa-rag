"""
Visualizer for LIME Explanations
Displays feature importance in readable format
"""

from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)


class LimeVisualizer:
    """Visualize LIME explanations in terminal and HTML"""
    
    @staticmethod
    def print_context_importance(explanation_result, top_n=5):
        """
        Print which context parts were most important
        
        Args:
            explanation_result: Result from RAGLimeExplainer.explain_context_importance()
            top_n: Number of top features to display
        """
        if not explanation_result.get("feature_weights"):
            print("\n❌ No explanation available")
            return
        
        print(f"\n{'='*60}")
        print(f"{Fore.CYAN}📊 CONTEXT IMPORTANCE ANALYSIS{Style.RESET_ALL}")
        print(f"{'='*60}\n")
        
        feature_weights = explanation_result["feature_weights"][:top_n]
        
        print(f"{Fore.YELLOW}Top {len(feature_weights)} Most Important Context Features:{Style.RESET_ALL}\n")
        
        for idx, (feature, weight) in enumerate(feature_weights, 1):
            # Color based on importance
            if weight > 0.3:
                color = Fore.GREEN
                importance = "HIGH"
            elif weight > 0.1:
                color = Fore.YELLOW
                importance = "MEDIUM"
            else:
                color = Fore.RED
                importance = "LOW"
            
            # Truncate long features
            feature_display = feature[:80] + "..." if len(feature) > 80 else feature
            
            print(f"{color}[{idx}] {importance} (weight: {weight:+.3f}){Style.RESET_ALL}")
            print(f"    \"{feature_display}\"")
            print()
    
    @staticmethod
    def print_query_importance(explanation_result):
        """
        Print which query words were most important for retrieval
        
        Args:
            explanation_result: Result from RAGLimeExplainer.explain_query_importance()
        """
        if not explanation_result.get("feature_weights"):
            print("\n❌ No explanation available")
            return
        
        print(f"\n{'='*60}")
        print(f"{Fore.CYAN}🔍 QUERY WORD IMPORTANCE ANALYSIS{Style.RESET_ALL}")
        print(f"{'='*60}\n")
        
        feature_weights = explanation_result["feature_weights"]
        
        print(f"{Fore.YELLOW}Word Importance for Retrieval:{Style.RESET_ALL}\n")
        
        for idx, (word, weight) in enumerate(feature_weights, 1):
            # Color based on importance
            if weight > 0.2:
                color = Fore.GREEN
                bar = "█" * int(weight * 20)
            elif weight > 0:
                color = Fore.YELLOW
                bar = "▓" * int(weight * 20)
            else:
                color = Fore.RED
                bar = "░" * int(abs(weight) * 20)
            
            print(f"{color}[{idx}] '{word}' {bar} {weight:+.3f}{Style.RESET_ALL}")
    
    @staticmethod
    def generate_html_report(context_explanation, query_explanation, output_file="lime_report.html"):
        """
        Generate HTML report with LIME explanations
        
        Args:
            context_explanation: Context importance results
            query_explanation: Query importance results
            output_file: Output HTML file path
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LIME Explainability Report - RAG System</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .feature {{
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .weight-positive {{
            color: #28a745;
            font-weight: bold;
        }}
        .weight-negative {{
            color: #dc3545;
            font-weight: bold;
        }}
        .importance-high {{
            border-left-color: #28a745;
        }}
        .importance-medium {{
            border-left-color: #ffc107;
        }}
        .importance-low {{
            border-left-color: #dc3545;
        }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 LIME Explainability Report</h1>
        <p>RAG System - Multilingual Question Answering</p>
    </div>
    
    <div class="section">
        <h2>📊 Context Importance Analysis</h2>
        <p>Shows which parts of the retrieved context influenced the generated answer:</p>
"""
        
        # Add context features
        if context_explanation and context_explanation.get("feature_weights"):
            for feature, weight in context_explanation["feature_weights"][:5]:
                importance_class = "high" if weight > 0.3 else "medium" if weight > 0.1 else "low"
                weight_class = "positive" if weight > 0 else "negative"
                html_content += f"""
        <div class="feature importance-{importance_class}">
            <span class="weight-{weight_class}">Weight: {weight:+.3f}</span><br>
            <em>"{feature[:150]}..."</em>
        </div>
"""
        else:
            html_content += "<p>No context explanation available</p>"
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>🔍 Query Word Importance Analysis</h2>
        <p>Shows which words in the query were most important for retrieval:</p>
"""
        
        # Add query features
        if query_explanation and query_explanation.get("feature_weights"):
            for word, weight in query_explanation["feature_weights"]:
                weight_class = "positive" if weight > 0 else "negative"
                html_content += f"""
        <div class="feature">
            <strong>'{word}'</strong>: <span class="weight-{weight_class}">{weight:+.3f}</span>
        </div>
"""
        else:
            html_content += "<p>No query explanation available</p>"
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ HTML report saved to: {output_file}")
