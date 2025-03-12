#!/usr/bin/env python3
"""
Company Analysis Report

This script analyzes companies that are hiring to identify trends,
the most active employers, and patterns in job title variations
across different organizations.
"""

import sys
import os
import pandas as pd
import argparse
import re
import logging
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set

# Add the src directory to path if needed
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import from parent directory
from db import db

def get_company_data(filters: Dict = None, min_relevance: float = 0, limit: int = 1000) -> pd.DataFrame:
    """
    Retrieve company hiring data from jobs.
    
    Args:
        filters (Dict): Dictionary of filter conditions
        min_relevance (float): Minimum title relevance score
        limit (int): Maximum number of jobs to analyze
        
    Returns:
        pd.DataFrame: DataFrame with job details by company
    """
    # Build WHERE clause conditions
    where_clauses = [f"a.title_relevance >= {min_relevance}"]
    
    if filters:
        if title_filter := filters.get('title'):
            where_clauses.append(f"LOWER(r.title) LIKE '%{title_filter.lower()}%'")
        
        if location_filter := filters.get('location'):
            where_clauses.append(f"LOWER(r.location) LIKE '%{location_filter.lower()}%'")
            
        if keyword_filter := filters.get('keyword'):
            where_clauses.append(f"LOWER(r.searched_keyword) LIKE '%{keyword_filter.lower()}%'")
            
        # if date_filter := filters.get('days'):
        #     where_clauses.append(f"r.date_posted >= date('now', '-{date_filter} days')")
    
    # Combine all WHERE clauses with AND
    where_clause = " AND ".join(where_clauses)
    
    query = f"""
    SELECT 
        r.job_id,
        r.title,
        r.company,
        r.location,
        r.date_posted,
        r.searched_keyword,
        a.title_relevance,
        f.description_text
    FROM raw_jobs r
    JOIN job_analysis a ON r.job_id = a.job_id
    JOIN formatted_descriptions f ON r.job_id = f.job_id
    WHERE {where_clause}
    ORDER BY r.company, r.date_posted DESC
    LIMIT {limit}
    """
    
    return db.query_to_df(query)

def analyze_companies(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Analyze company hiring patterns.
    
    Args:
        df (pd.DataFrame): DataFrame with job data
        
    Returns:
        Tuple[pd.DataFrame, Dict, Dict]: Company stats, title variations by company, and hiring timeline
    """
    # Count jobs per company
    company_counts = Counter(df['company'])
    
    # Create company stats dataframe
    company_data = []
    
    # Analyze job titles by company
    title_variations = {}
    
    # Track hiring timeline
    hiring_timeline = defaultdict(lambda: defaultdict(int))
    
    # Process each company
    for company, jobs in df.groupby('company'):
        # Skip companies with no name
        if pd.isna(company) or not company.strip():
            continue
            
        # Get job count
        job_count = len(jobs)
        
        # Get average title relevance
        avg_relevance = jobs['title_relevance'].mean()
        
        # Count unique job titles
        unique_titles = set(jobs['title'].str.lower())
        
        # # Extract dates and locations
        # dates = pd.to_datetime(jobs['date_posted'])
        # earliest_date = dates.min() if not dates.empty else None
        # latest_date = dates.max() if not dates.empty else None
        
        # # Track monthly hiring
        # if not dates.empty:
        #     for date in dates:
        #         month_year = date.strftime('%Y-%m')
        #         hiring_timeline[company][month_year] += 1
        
        # Count locations and remote jobs
        locations = jobs['location'].str.lower()
        remote_count = locations.str.contains('remote|anywhere|work from home|wfh', 
                                             regex=True).sum()
        unique_locations = set([loc for loc in locations if not pd.isna(loc)])
        
        # Store company stats
        company_data.append({
            'company': company,
            'job_count': job_count,
            'unique_titles': len(unique_titles),
            'avg_relevance': avg_relevance,
            'unique_locations': len(unique_locations),
            'remote_jobs': remote_count,
            'remote_percentage': (remote_count / job_count * 100) if job_count > 0 else 0,
            # 'earliest_date': earliest_date,
            # 'latest_date': latest_date
        })
        
        # Store title variations
        title_variations[company] = list(unique_titles)
    
    # Convert to DataFrame and sort by job count
    company_df = pd.DataFrame(company_data)
    if not company_df.empty:
        company_df = company_df.sort_values('job_count', ascending=False)
    
    return company_df, title_variations, hiring_timeline

def standardize_company_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to standardize company names by removing common suffixes and fixing capitalization.
    
    Args:
        df (pd.DataFrame): DataFrame with job data
        
    Returns:
        pd.DataFrame: DataFrame with standardized company names
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Common company suffixes to remove or standardize
    suffixes = [
        r'\s+Inc\.?$', r'\s+LLC$', r'\s+Ltd\.?$', r'\s+Limited$', 
        r'\s+Corp\.?$', r'\s+Corporation$', r'\s+Co\.?$', r'\s+Company$',
        r'\s+Group$', r'\s+Holdings$', r'\s+Technologies$', r'\s+Technology$',
        r'\s+International$', r'\s+Incorporated$'
    ]
    
    # Apply standardization to company names
    if 'company' in result_df.columns:
        # Create a standardized company name column
        result_df['company_standardized'] = result_df['company'].copy()
        
        # Remove suffixes
        for suffix in suffixes:
            result_df['company_standardized'] = result_df['company_standardized'].str.replace(
                suffix, '', regex=True, case=False)
        
        # Fix capitalization and trim whitespace
        result_df['company_standardized'] = result_df['company_standardized'].str.strip()
    
    return result_df

def identify_duplicate_companies(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify potential duplicate company names using string similarity.
    
    Args:
        df (pd.DataFrame): DataFrame with company data
        
    Returns:
        Dict[str, List[str]]: Dictionary of potential duplicate groups
    """
    from difflib import SequenceMatcher
    
    # Extract unique company names
    if 'company_standardized' in df.columns:
        companies = df['company_standardized'].unique()
    else:
        companies = df['company'].unique()
    
    # Set similarity threshold
    threshold = 0.85
    
    # Dictionary to store potential duplicates
    duplicates = defaultdict(list)
    processed = set()
    
    # Compare each pair of company names
    for i, company1 in enumerate(companies):
        if pd.isna(company1) or company1 in processed:
            continue
            
        group = [company1]
        
        for company2 in companies[i+1:]:
            if pd.isna(company2):
                continue
                
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, company1.lower(), company2.lower()).ratio()
            
            if similarity >= threshold:
                group.append(company2)
                processed.add(company2)
        
        if len(group) > 1:
            duplicates[company1] = group
    
    return duplicates

def analyze_job_titles(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Analyze common job title patterns within companies.
    
    Args:
        df (pd.DataFrame): DataFrame with job data
        
    Returns:
        Dict[str, List[str]]: Dictionary of common title patterns by company
    """
    title_patterns = {}
    
    # Common job title components to look for
    title_components = [
        'analyst', 'senior', 'junior', 'lead', 'manager', 'director',
        'engineer', 'scientist', 'developer', 'consultant', 'architect',
        'intern', 'associate', 'principal', 'staff', 'head', 'vp',
        'specialist', 'administrator', 'executive', 'president'
    ]
    
    # Analyze titles by company
    for company, group in df.groupby('company'):
        if pd.isna(company) or not company.strip():
            continue
            
        title_counts = Counter()
        seniority_counts = Counter()
        
        for title in group['title']:
            if pd.isna(title):
                continue
                
            title_lower = title.lower()
            
            # Count occurrences of title components
            for component in title_components:
                if re.search(r'\b' + component + r'\b', title_lower):
                    title_counts[component] += 1
            
            # Check for seniority indicators
            if re.search(r'\b(senior|sr\.?|lead|principal|staff|head)\b', title_lower):
                seniority_counts['senior'] += 1
            elif re.search(r'\b(junior|jr\.?|associate|entry[\s-]level)\b', title_lower):
                seniority_counts['junior'] += 1
            elif re.search(r'\b(intern|internship|student)\b', title_lower):
                seniority_counts['intern'] += 1
            elif re.search(r'\b(manager|director|head|chief|vp|executive)\b', title_lower):
                seniority_counts['management'] += 1
            else:
                seniority_counts['mid-level'] += 1
        
        # Store patterns for this company
        title_patterns[company] = {
            'components': title_counts,
            'seniority': seniority_counts
        }
    
    return title_patterns

def display_company_results(company_df: pd.DataFrame, title_variations: Dict, 
                           hiring_timeline: Dict, top_n: int = 20, show_plot: bool = False):
    """
    Display company analysis results.
    
    Args:
        company_df (pd.DataFrame): DataFrame with company statistics
        title_variations (Dict): Dictionary of job title variations by company
        hiring_timeline (Dict): Dictionary of hiring activity by month
        top_n (int): Number of top companies to display
        show_plot (bool): Whether to display plots
    """
    print("\n===== COMPANY ANALYSIS RESULTS =====\n")
    
    if company_df.empty:
        print("No company data available for analysis.")
        return
    
    # Display top hiring companies
    print(f"\nTop {top_n} Hiring Companies:")
    print("-" * 50)
    
    top_companies = company_df.head(top_n)
    for idx, row in top_companies.iterrows():
        company = row['company']
        job_count = row['job_count']
        remote_pct = row['remote_percentage']
        avg_relevance = row['avg_relevance']
        
        print(f"{company.ljust(30)} {job_count} jobs | {remote_pct:.1f}% remote | Avg. relevance: {avg_relevance:.2f}")
    
    # Display companies with most diverse job titles
    title_diverse = company_df[company_df['job_count'] >= 3].sort_values('unique_titles', ascending=False).head(10)
    
    if not title_diverse.empty:
        print("\nCompanies with Most Diverse Job Titles:")
        print("-" * 50)
        
        for idx, row in title_diverse.iterrows():
            company = row['company']
            unique_count = row['unique_titles']
            job_count = row['job_count']
            
            print(f"{company.ljust(30)} {unique_count} unique titles | {job_count} total jobs")
            
            # Show example titles
            if company in title_variations and len(title_variations[company]) > 0:
                examples = title_variations[company][:3]
                print(f"  Example titles: {', '.join(examples)}")
    
    # Display companies with most remote jobs
    remote_focused = company_df[company_df['job_count'] >= 3].sort_values('remote_percentage', ascending=False).head(10)
    
    if not remote_focused.empty:
        print("\nMost Remote-Friendly Companies:")
        print("-" * 50)
        
        for idx, row in remote_focused.iterrows():
            company = row['company']
            remote_count = row['remote_jobs']
            remote_pct = row['remote_percentage']
            job_count = row['job_count']
            
            print(f"{company.ljust(30)} {remote_count}/{job_count} remote jobs ({remote_pct:.1f}%)")
    
    # Plot results if requested
    if show_plot:
        # Top companies bar chart
        plt.figure(figsize=(12, 8))
        companies = top_companies['company']
        counts = top_companies['job_count']
        plt.barh(companies, counts)
        plt.xlabel('Number of Job Postings')
        plt.title('Top Companies by Job Postings')
        plt.tight_layout()
        plt.show()
        
        # Remote percentage chart
        plt.figure(figsize=(12, 8))
        companies = remote_focused['company']
        remote_pcts = remote_focused['remote_percentage']
        plt.barh(companies, remote_pcts)
        plt.xlabel('Remote Job Percentage')
        plt.title('Most Remote-Friendly Companies')
        plt.tight_layout()
        plt.show()
        
        # Hiring timeline for top companies
        if hiring_timeline:
            plt.figure(figsize=(14, 8))
            
            # Get top 5 companies
            top5_companies = list(top_companies['company'].head(5))
            
            # Get all months across all companies
            all_months = set()
            for company in top5_companies:
                if company in hiring_timeline:
                    all_months.update(hiring_timeline[company].keys())
            
            # Sort months
            sorted_months = sorted(all_months)
            
            # Create data for each company
            for company in top5_companies:
                if company in hiring_timeline:
                    monthly_counts = [hiring_timeline[company].get(month, 0) for month in sorted_months]
                    plt.plot(sorted_months, monthly_counts, marker='o', linewidth=2, label=company)
            
            plt.xlabel('Month')
            plt.ylabel('Number of Job Postings')
            plt.title('Hiring Timeline for Top Companies')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

def save_to_csv(company_df: pd.DataFrame, title_variations: Dict, 
               hiring_timeline: Dict, filename: Optional[str] = None):
    """
    Save company analysis results to CSV.
    
    Args:
        company_df (pd.DataFrame): DataFrame with company statistics
        title_variations (Dict): Dictionary of job title variations by company
        hiring_timeline (Dict): Dictionary of hiring activity by month
        filename (str, optional): Output filename
    """
    # Generate default filename if none provided
    if not filename:
        import datetime
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f'company_analysis_{today}.csv'
    
    # Ensure the data directory exists
    data_dir = os.path.join(script_dir, '../data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    # Save company stats
    company_df.to_csv(filepath, index=False)
    
    # Create and save timeline data
    timeline_data = []
    
    for company, months in hiring_timeline.items():
        for month, count in months.items():
            timeline_data.append({
                'company': company,
                'month': month,
                'job_count': count
            })
    
    timeline_df = pd.DataFrame(timeline_data)
    timeline_filepath = os.path.join(data_dir, f'hiring_timeline_{today}.csv')
    timeline_df.to_csv(timeline_filepath, index=False)
    
    # Create and save title variations
    title_data = []
    
    for company, titles in title_variations.items():
        for title in titles:
            title_data.append({
                'company': company,
                'job_title': title
            })
    
    title_df = pd.DataFrame(title_data)
    title_filepath = os.path.join(data_dir, f'job_titles_{today}.csv')
    title_df.to_csv(title_filepath, index=False)
    
    print(f"\nCompany statistics saved to {filepath}")
    print(f"Hiring timeline saved to {timeline_filepath}")
    print(f"Job titles saved to {title_filepath}")

def main():
    parser = argparse.ArgumentParser(description='Analyze hiring companies and job trends')
    
    # Filtering options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument('--title', type=str,
                        help='Filter jobs by title (e.g., "data scientist")')
    filter_group.add_argument('--location', type=str,
                        help='Filter jobs by location')
    filter_group.add_argument('--keyword', type=str,
                        help='Filter jobs by search keyword used')
    filter_group.add_argument('--days', type=int, default=90,
                        help='Only include jobs posted within the last N days (default: 90)')
    filter_group.add_argument('--min-relevance', type=float, default=0.0,
                        help='Minimum title relevance score (default: 0.0)')
    filter_group.add_argument('--limit', type=int, default=1000,
                        help='Maximum number of jobs to analyze (default: 1000)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--top', type=int, default=20,
                        help='Show top N companies (default: 20)')
    analysis_group.add_argument('--detect-duplicates', action='store_true',
                        help='Attempt to detect duplicate company names')
    analysis_group.add_argument('--plot', action='store_true',
                        help='Display visualizations of company data')
    analysis_group.add_argument('--save', action='store_true',
                        help='Save results to CSV')
    analysis_group.add_argument('--output', type=str,
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Build filters dictionary from arguments
    filters = {}
    if args.title:
        filters['title'] = args.title
    if args.location:
        filters['location'] = args.location
    if args.keyword:
        filters['keyword'] = args.keyword
    if args.days:
        filters['days'] = args.days
    
    # Fetch company data
    print(f"Fetching company data (limit: {args.limit})...")
    df = get_company_data(
        filters=filters,
        min_relevance=args.min_relevance,
        limit=args.limit
    )
    
    if df.empty:
        print("No matching jobs found.")
        return
        
    print(f"Retrieved {len(df)} jobs from {df['company'].nunique()} companies for analysis.")
    
    # Standardize company names
    print("Standardizing company names...")
    df = standardize_company_names(df)
    
    # Check for duplicate company names if requested
    if args.detect_duplicates:
        print("Checking for potential duplicate company names...")
        duplicates = identify_duplicate_companies(df)
        
        if duplicates:
            print(f"\nPotential duplicate company names ({len(duplicates)} groups):")
            for primary, group in duplicates.items():
                print(f"  {primary}: {', '.join([c for c in group if c != primary])}")
    
    # Analyze companies
    print("Analyzing company hiring patterns...")
    company_df, title_variations, hiring_timeline = analyze_companies(df)
    
    # Analyze job titles
    print("Analyzing job title patterns...")
    title_patterns = analyze_job_titles(df)
    
    # Display results
    display_company_results(
        company_df,
        title_variations,
        hiring_timeline,
        top_n=args.top,
        show_plot=args.plot
    )
    
    # Save results if requested
    if args.save:
        save_to_csv(company_df, title_variations, hiring_timeline, args.output)

if __name__ == "__main__":
    main() 