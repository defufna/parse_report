#!/usr/bin/env python
import re
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from io import StringIO
from glob import glob
import os
import sys

def parse_report(report_text):
    """Parse the performance report text into a structured format."""
    # Extract batch sizes and their corresponding metrics
    pattern = r"Affected object per transaction (\d+)\.([^A].*?)(?=Affected object per transaction \d+\.|$)"
    matches = re.findall(pattern, report_text, re.DOTALL)

    data = []

    for batch_size, section in matches:
        batch_size = int(batch_size)

        # Extract each operation and its value
        lines = [line.strip() for line in section.strip().split('\n') if line.strip()]
        for line in lines:
            # Extract operation name and value
            match = re.match(r"(.*?) \[\d+\]: ([\d.]+) T/s", line)
            if match:
                operation, value = match.groups()
                data.append({
                    'batch_size': batch_size,
                    'operation': operation,
                    'transactions_per_second': float(value)
                })

    return pd.DataFrame(data)

def compare_databases(db_reports):
    """Compare performance metrics across different databases."""
    all_data = []

    for db_name, report in db_reports.items():
        df = parse_report(report)
        df['database'] = db_name
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def generate_charts(comparison_df):
    """Generate comparative charts for database performance."""
    # Set a clean, modern style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)

    # Chart 1: Compare overall performance by database and batch size
    plt.figure(figsize=(16, 10))
    chart1 = sns.barplot(
        data=comparison_df,
        x='batch_size',
        y='transactions_per_second',
        hue='database',
        errorbar=None
    )
    plt.title('Overall Database Performance by Batch Size', fontsize=16)
    plt.xlabel('Batch Size', fontsize=14)
    plt.ylabel('Transactions per Second', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Database', fontsize=12)
    plt.tight_layout()
    plt.savefig('overall_performance_comparison.png', dpi=300)
    plt.close()

    # Chart 2: Compare specific operations across databases
    # Focus on the most common operations
    key_operations = ['Insert vehicles', 'Update vehicles', 'Get vehicles', 'Delete vehicles']
    filtered_df = comparison_df[comparison_df['operation'].isin(key_operations)]

    plt.figure(figsize=(16, 10))
    chart2 = sns.catplot(
        data=filtered_df,
        x='operation',
        y='transactions_per_second',
        hue='database',
        col='batch_size',
        kind='bar',
        height=6,
        aspect=0.8,
        errorbar=None
    )
    chart2.fig.suptitle('Key Operations Performance by Database and Batch Size', fontsize=16, y=1.05)
    chart2.set_axis_labels('Operation', 'Transactions per Second', fontsize=14)
    chart2.set_titles('Batch Size: {col_name}', fontsize=14)
    chart2.tight_layout()
    plt.savefig('operations_comparison.png', dpi=300)
    plt.close()

    # Chart 3: Radar chart comparing databases across all operations for a specific batch size
    # Using batch size 1 as an example
    batch1_df = comparison_df[comparison_df['batch_size'] == 1]
    operation_categories = batch1_df['operation'].unique()
    n_operations = len(operation_categories)

    # Create a function for a radar chart
    def radar_chart(df, batch_size):
        plt.figure(figsize=(14, 10))

        # Calculate angles for the radar chart
        angles = np.linspace(0, 2*np.pi, n_operations, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Create subplot with polar projection
        ax = plt.subplot(111, polar=True)

        # Get unique database names
        databases = df['database'].unique()

        # Define a color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(databases)))

        for i, db in enumerate(databases):
            db_data = df[df['database'] == db]

            # Prepare data for the radar chart
            values = []
            for op in operation_categories:
                val = db_data[db_data['operation'] == op]['transactions_per_second'].values
                values.append(float(val[0]) if len(val) > 0 else 0)

            values += values[:1]  # Close the loop

            # Plot the radar chart for this database
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=db)
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # Set ticks and labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(operation_categories, fontsize=10)

        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        plt.title(f'Database Performance Comparison (Batch Size {batch_size})', fontsize=16, y=1.08)

        plt.tight_layout()
        plt.savefig(f'radar_comparison_batch{batch_size}.png', dpi=300)
        plt.close()

    # Create radar charts for each batch size
    for batch in comparison_df['batch_size'].unique():
        batch_df = comparison_df[comparison_df['batch_size'] == batch]
        radar_chart(batch_df, batch)

def load_file(fn: str) -> Tuple[str, str]:
    with open(fn, 'r') as f:
        first = f.readline()
        rest = f.read()
    return first, rest

def dataframe_to_report(df: pd.DataFrame) -> str:
    """
    Convert a performance report DataFrame back to text format.
    
    Args:
        df: pandas DataFrame containing performance data with columns:
            'batch_size', 'operation', 'transactions_per_second'
            
    Returns:
        str: Formatted report text
    """
    # Sort DataFrame by batch_size and operation for consistent output
    df_sorted = df.sort_values(['batch_size', 'operation'])
    
    # Group by batch_size
    report_parts = []
    for batch_size, group in df_sorted.groupby('batch_size'):
        # Start section with batch size header
        section = [f"Affected object per transaction {batch_size}."]
        
        # Add each operation line
        for _, row in group.iterrows():
            # Format: "operation [0]: value T/s"
            line = f"{row['operation']} [0]: {row['transactions_per_second']:.1f} T/s"
            section.append(line)
        
        # Join section lines with newlines
        report_parts.append('\n'.join(section))
    
    # Join all sections with double newlines
    return '\n\n'.join(report_parts)

def average_reports(reports: list) -> pd.DataFrame:
    """
    Calculate the average values across multiple performance reports.

    Args:
        reports: List of pandas DataFrames containing performance data

    Returns:
        pd.DataFrame: A new DataFrame with averaged values
    """
    # Concatenate all reports
    combined = pd.concat(reports)

    # Group by relevant columns and calculate mean
    return combined.groupby(['batch_size', 'operation'])['transactions_per_second'].mean().reset_index()

def main():
    """Main function to process reports and generate visualizations."""
    if len(sys.argv) != 2:
        print("Usage: python parse_report.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        sys.exit(1)

    # Load .txt files from the specified directory
    db_reports = {name: report for name, report in (load_file(fn) for fn in glob(os.path.join(directory_path, "*.txt")))}

    # Parse and compare the data
    comparison_df = compare_databases(db_reports)

    # Generate charts
    generate_charts(comparison_df)

    print("Charts generated successfully!")

if __name__ == "__main__":
    main()
