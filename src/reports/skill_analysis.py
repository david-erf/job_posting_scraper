#!/usr/bin/env python3
"""
Skill Analysis Report

This script analyzes job descriptions to identify the most requested
skills for specific job titles or keywords. This helps identify which
skills are in highest demand for targeted job searches.
"""

import sys
import os
import pandas as pd
import argparse
import re
import logging
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

# Add the src directory to path if needed
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import from parent directory
from db import db

# Default skill categories to look for
DEFAULT_SKILLS = {
    'programming': [
        'python', 'java', 'javascript', 'c\\+\\+', 'sql', 'r', 'scala', 'golang', 'go', 
        'typescript', 'ruby', 'php', 'swift', 'kotlin', 'rust', 'perl', 'julia'
    ],
    'data_engineering': [
        'hadoop', 'spark', 'kafka', 'airflow', 'snowflake', 'databricks', 'etl', 
        'data warehouse', 'data lake', 'data pipeline', 'aws glue', 'dbt'
    ],
    'databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'dynamodb', 'oracle', 'sql server', 
        'cassandra', 'neo4j', 'elasticsearch', 'sqlite', 'firebase'
    ],
    'ml_libraries': [
        'numpy', 'pandas', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras', 
        'matplotlib', 'seaborn', 'huggingface', 'transformers', 'xgboost', 'lightgbm'
    ],
    'cloud': [
        'aws', 'azure', 'gcp', 'google cloud', 'ec2', 's3', 'lambda', 'cloudformation', 
        'terraform', 'docker', 'kubernetes', 'k8s', 'cloud formation'
    ],
    'collaboration': [
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'agile', 'scrum', 
        'kanban', 'ci/cd', 'continuous integration'
    ],
    'analytics': [
        'tableau', 'power bi', 'looker', 'metabase', 'google analytics', 'excel', 
        'spreadsheet', 'data studio', 'google data studio', 'dashboard', 'powerbi'
    ]
}

def get_jobs_with_descriptions(filters: Dict = None, min_relevance: float = 0, limit: int = 500) -> pd.DataFrame:
    """
    Retrieve jobs with their descriptions from the database.
    
    Args:
        filters (Dict): Dictionary of filter conditions
        min_relevance (float): Minimum title relevance score
        limit (int): Maximum number of jobs to return
        
    Returns:
        pd.DataFrame: DataFrame with job details including description text
    """
    # Build WHERE clause conditions
    where_clauses = [f"a.title_relevance >= {min_relevance}"]
    
    if filters:
        if title_filter := filters.get('title'):
            # Allow partial title matching
            where_clauses.append(f"LOWER(r.title) LIKE '%{title_filter.lower()}%'")
        
        if company_filter := filters.get('company'):
            where_clauses.append(f"LOWER(r.company) LIKE '%{company_filter.lower()}%'")
            
        if keyword_filter := filters.get('keyword'):
            where_clauses.append(f"LOWER(r.searched_keyword) LIKE '%{keyword_filter.lower()}%'")
    
    # Combine all WHERE clauses with AND
    where_clause = " AND ".join(where_clauses)
    
    query = f"""
    SELECT 
        r.job_id,
        r.title,
        r.company,
        r.location,
        r.searched_keyword,
        a.title_relevance,
        f.description_text
    FROM raw_jobs r
    JOIN job_analysis a ON r.job_id = a.job_id
    JOIN formatted_descriptions f ON r.job_id = f.job_id
    WHERE {where_clause}
    ORDER BY a.title_relevance DESC
    LIMIT {limit}
    """
    
    return db.query_to_df(query)

def analyze_skills(descriptions: List[str], skills_dict: Dict[str, List[str]]) -> Dict[str, Counter]:
    """
    Analyze descriptions to count occurrences of skills by category.
    
    Args:
        descriptions (List[str]): List of job description texts
        skills_dict (Dict[str, List[str]]): Dictionary mapping skill categories to skills
        
    Returns:
        Dict[str, Counter]: Dictionary with skill counts by category
    """
    results = {}
    
    # Combine all descriptions into a single string for analysis
    all_text = ' '.join(descriptions).lower()
    
    # Count skills by category
    for category, skills in skills_dict.items():
        skill_counts = Counter()
        
        for skill in skills:
            # Create regex pattern that matches whole words only
            pattern = r'\b' + skill + r'\b'
            count = len(re.findall(pattern, all_text))
            if count > 0:
                skill_counts[skill] = count
        
        results[category] = skill_counts
    
    return results

def display_results(skill_results: Dict[str, Counter], top_n: int = 10, show_plot: bool = False):
    """
    Display skill analysis results by category.
    
    Args:
        skill_results (Dict[str, Counter]): Results from analyze_skills
        top_n (int): Number of top skills to display per category
        show_plot (bool): Whether to display plots
    """
    print("\n===== SKILL ANALYSIS RESULTS =====\n")
    
    for category, counter in skill_results.items():
        if not counter:
            continue
            
        # Get most common skills
        most_common = counter.most_common(top_n)
        
        if most_common:
            # Calculate total mentions for this category
            total = sum(counter.values())
            
            # Print category header
            category_name = category.replace('_', ' ').title()
            print(f"\n{category_name} Skills (Total mentions: {total})")
            print("-" * 40)
            
            # Print skills table
            for skill, count in most_common:
                percentage = (count / total) * 100
                print(f"{skill.ljust(20)} {count} ({percentage:.1f}%)")
                
            # Create plot if requested
            if show_plot and len(most_common) > 1:
                skills, counts = zip(*most_common)
                plt.figure(figsize=(10, 6))
                plt.barh(skills, counts)
                plt.xlabel('Number of Mentions')
                plt.title(f'Top {category_name} Skills')
                plt.tight_layout()
                plt.show()

def save_to_csv(skill_results: Dict[str, Counter], filename: Optional[str] = None):
    """
    Save skill analysis results to CSV.
    
    Args:
        skill_results (Dict[str, Counter]): Results from analyze_skills
        filename (str, optional): Output filename
    """
    # Generate default filename if none provided
    if not filename:
        import datetime
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f'skill_analysis_{today}.csv'
    
    # Ensure the data directory exists
    data_dir = os.path.join(script_dir, '../data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    # Convert results to DataFrame
    rows = []
    for category, counter in skill_results.items():
        for skill, count in counter.items():
            rows.append({
                'category': category,
                'skill': skill,
                'mentions': count
            })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")
    else:
        print("\nNo results to save")

def main():
    parser = argparse.ArgumentParser(description='Analyze skills mentioned in job descriptions')
    
    # Filtering options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument('--title', type=str,
                        help='Filter jobs by title (e.g., "data scientist")')
    filter_group.add_argument('--company', type=str,
                        help='Filter jobs by company')
    filter_group.add_argument('--keyword', type=str,
                        help='Filter jobs by search keyword used')
    filter_group.add_argument('--min-relevance', type=float, default=0.0,
                        help='Minimum title relevance score (default: 0.0)')
    filter_group.add_argument('--limit', type=int, default=500,
                        help='Maximum number of jobs to analyze (default: 500)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--top', type=int, default=10,
                        help='Show top N skills per category (default: 10)')
    analysis_group.add_argument('--plot', action='store_true',
                        help='Display bar plots of skill frequencies')
    analysis_group.add_argument('--save', action='store_true',
                        help='Save results to CSV')
    analysis_group.add_argument('--output', type=str,
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Build filters dictionary from arguments
    filters = {}
    if args.title:
        filters['title'] = args.title
    if args.company:
        filters['company'] = args.company
    if args.keyword:
        filters['keyword'] = args.keyword
    
    # Fetch jobs with descriptions
    print(f"Fetching job descriptions (limit: {args.limit})...")
    df = get_jobs_with_descriptions(
        filters=filters,
        min_relevance=args.min_relevance,
        limit=args.limit
    )
    
    if df.empty:
        print("No matching jobs found.")
        return
        
    print(f"Retrieved {len(df)} job descriptions for analysis.")
    
    # Extract description texts for analysis
    descriptions = df['description_text'].dropna().tolist()
    if not descriptions:
        print("No job descriptions available for analysis.")
        return
    
    # Analyze skills
    print("Analyzing skills in job descriptions...")
    skill_results = analyze_skills(descriptions, DEFAULT_SKILLS)
    
    # Display results
    display_results(skill_results, top_n=args.top, show_plot=args.plot)
    
    # Save results if requested
    if args.save:
        save_to_csv(skill_results, args.output)

if __name__ == "__main__":
    main() 