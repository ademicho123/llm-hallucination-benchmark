import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filepath="results_multi_model_improved.csv"):
    """Load and validate results data."""
    df = pd.read_csv(filepath)
    print(f"📊 Loaded {len(df)} results from {df['model_short'].nunique()} models")
    return df


def calculate_metrics(df):
    """Calculate all key metrics for each model."""
    metrics = []
    
    for model in df['model_short'].unique():
        model_data = df[df['model_short'] == model]
        
        # Filter out errors for accuracy calculation
        valid_data = model_data[~model_data['outcome'].isin(['api_error', 'parse_error'])]
        fake_data = valid_data[valid_data['label'] == 'fake']
        real_data = valid_data[valid_data['label'] == 'real']
        
        total_valid = len(valid_data)
        total_fake = len(fake_data)
        total_real = len(real_data)
        
        # Core metrics
        correct = len(valid_data[valid_data['outcome'].isin(['correct_acceptance', 'correct_rejection'])])
        accuracy = (correct / total_valid * 100) if total_valid > 0 else 0
        
        # Hallucination metrics
        hallucinations = len(fake_data[fake_data['outcome'] == 'accepted_false_claim'])
        hallucination_rate = (hallucinations / total_fake * 100) if total_fake > 0 else 0
        
        confident_hallucinations = len(fake_data[
            (fake_data['outcome'] == 'accepted_false_claim') & 
            (fake_data['confidence'] == 'high')
        ])
        confident_hallucination_rate = (confident_hallucinations / total_fake * 100) if total_fake > 0 else 0
        
        # False negative metrics
        false_negatives = len(real_data[real_data['outcome'] == 'false_negative'])
        false_negative_rate = (false_negatives / total_real * 100) if total_real > 0 else 0
        
        # Confidence distribution
        high_conf = len(valid_data[valid_data['confidence'] == 'high'])
        medium_conf = len(valid_data[valid_data['confidence'] == 'medium'])
        low_conf = len(valid_data[valid_data['confidence'] == 'low'])
        
        # Error rates
        parse_errors = len(model_data[model_data['outcome'] == 'parse_error'])
        api_errors = len(model_data[model_data['outcome'] == 'api_error'])
        
        metrics.append({
            'Model': model,
            'Accuracy': round(accuracy, 1),
            'Hallucination_Rate': round(hallucination_rate, 1),
            'Confident_Hallucination_Rate': round(confident_hallucination_rate, 1),
            'False_Negative_Rate': round(false_negative_rate, 1),
            'Total_Hallucinations': hallucinations,
            'Confident_Hallucinations': confident_hallucinations,
            'False_Negatives': false_negatives,
            'High_Confidence_Pct': round((high_conf / total_valid * 100) if total_valid > 0 else 0, 1),
            'Parse_Errors': parse_errors,
            'API_Errors': api_errors,
            'Valid_Responses': total_valid,
            'Total_Queries': len(model_data)
        })
    
    return pd.DataFrame(metrics).sort_values('Accuracy', ascending=False)


def create_accuracy_chart(metrics_df, output_dir="charts"):
    """Create accuracy comparison bar chart."""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics_df)))
    
    bars = plt.bar(range(len(metrics_df)), metrics_df['Accuracy'], color=colors)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison\n(Higher is Better)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(metrics_df)), metrics_df['Model'], rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Random baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/accuracy_comparison.png")
    plt.close()


def create_hallucination_chart(metrics_df, output_dir="charts"):
    """Create hallucination rate comparison."""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, metrics_df['Hallucination_Rate'], width, 
                    label='All Hallucinations', alpha=0.8, color='#ff6b6b')
    bars2 = plt.bar(x + width/2, metrics_df['Confident_Hallucination_Rate'], width,
                    label='Confident Hallucinations', alpha=0.8, color='#c92a2a')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Hallucination Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Hallucination Rates by Model\n(Lower is Better)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, metrics_df['Model'], rotation=45, ha='right')
    plt.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hallucination_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/hallucination_comparison.png")
    plt.close()


def create_confidence_analysis(df, output_dir="charts"):
    """Create confidence distribution analysis."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Confidence distribution by outcome
    valid_data = df[~df['outcome'].isin(['api_error', 'parse_error'])]
    
    outcome_labels = {
        'correct_acceptance': 'Correct (Real→True)',
        'correct_rejection': 'Correct (Fake→False)',
        'accepted_false_claim': 'Hallucination (Fake→True)',
        'false_negative': 'False Negative (Real→False)'
    }
    
    confidence_dist = valid_data.groupby(['outcome', 'confidence']).size().unstack(fill_value=0)
    confidence_dist = confidence_dist.reindex(['high', 'medium', 'low'], axis=1, fill_value=0)
    
    # Normalize to percentages
    confidence_dist_pct = confidence_dist.div(confidence_dist.sum(axis=1), axis=0) * 100
    
    # Rename outcomes for better labels
    confidence_dist_pct.index = [outcome_labels.get(x, x) for x in confidence_dist_pct.index]
    
    confidence_dist_pct.plot(kind='bar', stacked=True, ax=axes[0], 
                             color=['#51cf66', '#ffd43b', '#ff6b6b'])
    axes[0].set_title('Confidence Distribution by Outcome Type', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Outcome Type', fontweight='bold')
    axes[0].set_ylabel('Percentage', fontweight='bold')
    axes[0].legend(title='Confidence', labels=['High', 'Medium', 'Low'])
    axes[0].tick_params(axis='x', rotation=45)
    
    # Chart 2: High confidence accuracy by model
    high_conf_metrics = []
    for model in df['model_short'].unique():
        model_data = df[df['model_short'] == model]
        high_conf_data = model_data[model_data['confidence'] == 'high']
        
        correct_high = len(high_conf_data[high_conf_data['outcome'].isin(['correct_acceptance', 'correct_rejection'])])
        total_high = len(high_conf_data[~high_conf_data['outcome'].isin(['api_error', 'parse_error'])])
        
        if total_high > 0:
            accuracy = (correct_high / total_high * 100)
            high_conf_metrics.append({'Model': model, 'Accuracy': accuracy, 'Count': total_high})
    
    hc_df = pd.DataFrame(high_conf_metrics).sort_values('Accuracy', ascending=False)
    
    bars = axes[1].bar(range(len(hc_df)), hc_df['Accuracy'], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(hc_df))))
    axes[1].set_title('Accuracy of High-Confidence Responses', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Model', fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1].set_xticks(range(len(hc_df)))
    axes[1].set_xticklabels(hc_df['Model'], rotation=45, ha='right')
    axes[1].set_ylim(0, 100)
    
    # Add labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confidence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/confidence_analysis.png")
    plt.close()


def create_error_breakdown(df, output_dir="charts"):
    """Create error type breakdown chart."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Error types by model
    error_data = []
    for model in df['model_short'].unique():
        model_data = df[df['model_short'] == model]
        error_data.append({
            'Model': model,
            'Parse Errors': len(model_data[model_data['outcome'] == 'parse_error']),
            'API Errors': len(model_data[model_data['outcome'] == 'api_error']),
            'Success': len(model_data[~model_data['outcome'].isin(['parse_error', 'api_error'])])
        })
    
    error_df = pd.DataFrame(error_data).sort_values('Success', ascending=False)
    
    x = np.arange(len(error_df))
    width = 0.25
    
    axes[0].bar(x - width, error_df['Success'], width, label='Success', color='#51cf66')
    axes[0].bar(x, error_df['Parse Errors'], width, label='Parse Errors', color='#ffd43b')
    axes[0].bar(x + width, error_df['API Errors'], width, label='API Errors', color='#ff6b6b')
    
    axes[0].set_title('Response Quality by Model', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Model', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(error_df['Model'], rotation=45, ha='right')
    axes[0].legend()
    
    # Chart 2: Overall outcome distribution
    valid_data = df[~df['outcome'].isin(['api_error', 'parse_error'])]
    outcome_counts = valid_data['outcome'].value_counts()
    
    colors = {'correct_acceptance': '#51cf66', 'correct_rejection': '#51cf66',
              'accepted_false_claim': '#ff6b6b', 'false_negative': '#ffd43b'}
    
    outcome_labels = {
        'correct_acceptance': 'Correct\n(Real→True)',
        'correct_rejection': 'Correct\n(Fake→False)',
        'accepted_false_claim': 'Hallucination\n(Fake→True)',
        'false_negative': 'False Negative\n(Real→False)'
    }
    
    pie_colors = [colors.get(outcome, '#adb5bd') for outcome in outcome_counts.index]
    pie_labels = [outcome_labels.get(outcome, outcome) for outcome in outcome_counts.index]
    
    wedges, texts, autotexts = axes[1].pie(outcome_counts.values, labels=pie_labels, autopct='%1.1f%%',
                                            colors=pie_colors, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    axes[1].set_title('Overall Outcome Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_and_outcome_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/error_and_outcome_breakdown.png")
    plt.close()


def create_comprehensive_heatmap(df, output_dir="charts"):
    """Create a comprehensive performance heatmap."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculate metrics for heatmap
    heatmap_data = []
    for model in df['model_short'].unique():
        model_data = df[df['model_short'] == model]
        valid_data = model_data[~model_data['outcome'].isin(['api_error', 'parse_error'])]
        
        fake_data = valid_data[valid_data['label'] == 'fake']
        real_data = valid_data[valid_data['label'] == 'real']
        
        total_valid = len(valid_data)
        
        metrics = {
            'Model': model,
            'Accuracy': (len(valid_data[valid_data['outcome'].isin(['correct_acceptance', 'correct_rejection'])]) / total_valid * 100) if total_valid > 0 else 0,
            'Hallucination': (len(fake_data[fake_data['outcome'] == 'accepted_false_claim']) / len(fake_data) * 100) if len(fake_data) > 0 else 0,
            'False Reject': (len(real_data[real_data['outcome'] == 'false_negative']) / len(real_data) * 100) if len(real_data) > 0 else 0,
            'High Conf %': (len(valid_data[valid_data['confidence'] == 'high']) / total_valid * 100) if total_valid > 0 else 0,
            'Parse Errors': (len(model_data[model_data['outcome'] == 'parse_error']) / len(model_data) * 100),
        }
        heatmap_data.append(metrics)
    
    heat_df = pd.DataFrame(heatmap_data).set_index('Model').sort_values('Accuracy', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Invert hallucination and error metrics for color scheme (lower is better)
    display_df = heat_df.copy()
    display_df['Hallucination'] = 100 - display_df['Hallucination']
    display_df['False Reject'] = 100 - display_df['False Reject']
    display_df['Parse Errors'] = 100 - display_df['Parse Errors']
    
    sns.heatmap(display_df, annot=True, fmt='.1f', cmap='RdYlGn', center=70,
                cbar_kws={'label': 'Score (higher is better)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Model Performance Heatmap\n(All metrics normalized: higher=better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Update column labels
    ax.set_xticklabels(['Accuracy', 'Anti-Hallucinate', 'Anti-False Reject', 
                        'High Conf %', 'Anti-Parse Error'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/performance_heatmap.png")
    plt.close()


def generate_article_table(metrics_df, output_dir="charts"):
    """Generate markdown table for article."""
    Path(output_dir).mkdir(exist_ok=True)
    
    table_df = metrics_df[['Model', 'Accuracy', 'Hallucination_Rate', 
                           'Confident_Hallucination_Rate', 'False_Negative_Rate',
                           'Total_Hallucinations', 'False_Negatives']].copy()
    
    table_df.columns = ['Model', 'Accuracy (%)', 'Hallucination Rate (%)', 
                        'Confident Hallucination (%)', 'False Negative Rate (%)',
                        'Total Hallucinations', 'Total False Negatives']
    
    # Save as markdown
    with open(f'{output_dir}/results_table.md', 'w') as f:
        f.write("# Model Performance Comparison\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n\n## Key Metrics Explanation\n\n")
        f.write("- **Accuracy**: Percentage of correctly classified statements\n")
        f.write("- **Hallucination Rate**: Percentage of fake facts accepted as true\n")
        f.write("- **Confident Hallucination**: Hallucinations with high confidence\n")
        f.write("- **False Negative Rate**: Percentage of real facts rejected\n")
    
    # Save as CSV
    table_df.to_csv(f'{output_dir}/results_table.csv', index=False)
    
    print(f"✅ Saved: {output_dir}/results_table.md")
    print(f"✅ Saved: {output_dir}/results_table.csv")
    
    return table_df


def find_interesting_cases(df, output_file="charts/interesting_cases.txt"):
    """Find and document interesting edge cases for the article."""
    Path("charts").mkdir(exist_ok=True)
    
    cases = []
    
    # Find confident hallucinations
    confident_hallucinations = df[
        (df['outcome'] == 'accepted_false_claim') & 
        (df['confidence'] == 'high')
    ].sort_values('model_short')
    
    if not confident_hallucinations.empty:
        cases.append("\n🚨 CONFIDENT HALLUCINATIONS (High-confidence wrong answers):\n")
        cases.append("=" * 80 + "\n")
        for _, row in confident_hallucinations.head(10).iterrows():
            cases.append(f"\nModel: {row['model_short']}\n")
            cases.append(f"Statement: {row['statement']}\n")
            cases.append(f"Verdict: {row['verdict']} (Confidence: {row['confidence']})\n")
            cases.append(f"Raw: {row['raw_response'][:200]}\n")
            cases.append("-" * 80 + "\n")
    
    # Find statements that tripped up most models
    hallucination_counts = df[df['outcome'] == 'accepted_false_claim'].groupby('statement').size().sort_values(ascending=False)
    
    if not hallucination_counts.empty:
        cases.append("\n🎯 MOST DECEPTIVE FAKE FACTS (fooled multiple models):\n")
        cases.append("=" * 80 + "\n")
        for statement, count in hallucination_counts.head(5).items():
            cases.append(f"\n'{statement}'\n")
            cases.append(f"  Fooled {count} model(s)\n")
            cases.append("-" * 80 + "\n")
    
    # Find universally accepted real facts
    real_statements = df[df['label'] == 'real'].groupby('statement')['outcome'].apply(
        lambda x: (x == 'correct_acceptance').sum()
    ).sort_values(ascending=False)
    
    if not real_statements.empty:
        cases.append("\n✅ UNIVERSALLY ACCEPTED REAL FACTS:\n")
        cases.append("=" * 80 + "\n")
        for statement, count in real_statements.head(5).items():
            cases.append(f"\n'{statement}'\n")
            cases.append(f"  Accepted by {count} model(s)\n")
            cases.append("-" * 80 + "\n")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.writelines(cases)
    
    print(f"✅ Saved: {output_file}")
    
    # Also print to console
    print("\n" + "".join(cases))


def main():
    print("🎨 Generating visualizations and metrics for article...\n")
    
    # Load data
    df = load_data()
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics_df = calculate_metrics(df)
    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))
    
    # Generate table for article
    print("\n📝 Generating article table...")
    table_df = generate_article_table(metrics_df)
    
    # Create all visualizations
    print("\n🎨 Creating visualizations...")
    create_accuracy_chart(metrics_df)
    create_hallucination_chart(metrics_df)
    create_confidence_analysis(df)
    create_error_breakdown(df)
    create_comprehensive_heatmap(df)
    
    # Find interesting cases
    print("\n🔍 Finding interesting cases...")
    find_interesting_cases(df)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files in 'charts/' directory:")
    print("  • accuracy_comparison.png")
    print("  • hallucination_comparison.png")
    print("  • confidence_analysis.png")
    print("  • error_and_outcome_breakdown.png")
    print("  • performance_heatmap.png")
    print("  • results_table.md")
    print("  • results_table.csv")
    print("  • interesting_cases.txt")
    print("\n💡 Use these in your article!")


if __name__ == "__main__":
    main()