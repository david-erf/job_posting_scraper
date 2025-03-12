#!/usr/bin/env python3
"""
Recent Jobs Report

This script queries and displays recent job postings based on various criteria
including posting date, application status, and relevance scores. It can filter
for early application opportunities or high-relevance positions.
"""

import sys
import os
import pandas as pd
import argparse
import logging
import datetime
from typing import Dict, List, Optional

# Add the src directory to path if needed
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import from parent directory
from db import db

# Set pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 40)  # Truncate long text fields

def get_recent_relevant_jobs(
    limit: int = 50, 
    days: int = 7,
    min_relevance: float = 0,
    early_only: bool = False,
    include_internships: bool = True,
    location_filter: Optional[str] = None,
    company_filter: Optional[str] = None,
    title_filter: Optional[str] = None,
    keyword_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Query recent job postings based on specified criteria.
    
    Args:
        limit (int): Maximum number of jobs to return
        days (int): Only include jobs posted within this many days
        min_relevance (float): Minimum relevance score threshold
        early_only (bool): Only include early application opportunities
        include_internships (bool): Include internship positions even if below relevance threshold
        location_filter (str, optional): Filter by location
        company_filter (str, optional): Filter by company name
        title_filter (str, optional): Filter by job title
        keyword_filter (str, optional): Filter by search keyword used
        
    Returns:
        pd.DataFrame: DataFrame with matching jobs
    """
    # Build query conditions
    where_clauses = []
    
    # Date filter
    if days > 0:
        where_clauses.append(f"r.date_posted >= date('now', '-{days} days') OR r.created_at >= datetime('now', '-{days} days')")
    
    # Early application filter
    if early_only:
        where_clauses.append("r.hiring_status = 'Be an early applicant'")
    
    # Build relevance condition with internship exception
    relevance_clause = f"COALESCE(a.title_relevance, 0) >= {min_relevance}"
    if include_internships:
        relevance_clause = f"({relevance_clause} OR LOWER(r.title) LIKE '%intern%')"
    where_clauses.append(relevance_clause)
    
    # Optional filters
    if location_filter:
        where_clauses.append(f"LOWER(r.location) LIKE '%{location_filter.lower()}%'")
    
    if company_filter:
        where_clauses.append(f"LOWER(r.company) LIKE '%{company_filter.lower()}%'")
    
    if title_filter:
        where_clauses.append(f"LOWER(r.title) LIKE '%{title_filter.lower()}%'")
        
    if keyword_filter:
        where_clauses.append(f"LOWER(r.searched_keyword) LIKE '%{keyword_filter.lower()}%'")
    
    # Combine all WHERE clauses with AND
    where_clause = " AND ".join(where_clauses)
    
    query = f"""
    SELECT 
        r.job_id,
        r.title,
        r.company,
        r.location,
        r.date_posted,
        r.hiring_status,
        r.job_url,
        r.searched_keyword,
        COALESCE(a.title_relevance, 0) as title_relevance,
        COALESCE(f.salary_range, 'Not specified') as salary_range,
        r.created_at,
        CASE
            WHEN r.date_posted LIKE '%hour%' OR r.date_posted LIKE '%minute%' THEN 1
            WHEN r.date_posted LIKE '%day%' AND CAST(SUBSTR(r.date_posted, 1, 1) AS INTEGER) <= 1 THEN 2
            WHEN r.date_posted LIKE '%day%' AND CAST(SUBSTR(r.date_posted, 1, 1) AS INTEGER) <= 3 THEN 3
            ELSE 4
        END as freshness
    FROM raw_jobs r
    LEFT JOIN job_analysis a ON r.job_id = a.job_id
    LEFT JOIN formatted_descriptions f ON r.job_id = f.job_id
    WHERE {where_clause}
    ORDER BY 
        freshness ASC,
        title_relevance DESC,
        r.created_at DESC
    LIMIT {limit}
    """
    
    try:
        return db.query_to_df(query)
    except Exception as e:
        print(f"Error querying database: {e}")
        return pd.DataFrame()

def display_results(df: pd.DataFrame, verbose: bool = False):
    """
    Format and display the job search results.
    
    Args:
        df (pd.DataFrame): DataFrame with job data
        verbose (bool): Whether to display additional details
    """
    if df.empty:
        print("No jobs found matching the criteria.")
        return
    
    # Format the output
    display_df = df.copy()
    
    # Convert timestamps to readable format
    if 'created_at' in display_df.columns:
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Round relevance scores
    if 'title_relevance' in display_df.columns:
        display_df['title_relevance'] = display_df['title_relevance'].round(2)
    
    # Add "remote" indicator
    if 'location' in display_df.columns:
        display_df['remote'] = display_df['location'].str.contains('remote', case=False).map({True: '✓', False: ''})
    
    # Select and reorder columns for display
    basic_cols = [
        'title', 'company', 'location', 'remote', 'date_posted', 
        'title_relevance'
    ]
    
    detailed_cols = basic_cols + [
        'hiring_status', 'salary_range', 'searched_keyword', 'created_at'
    ]
    
    display_cols = detailed_cols if verbose else basic_cols
    
    # Print job IDs separately for easy copying
    print("\nJob IDs for application tracking:")
    for job_id in df['job_id']:
        print(job_id)
    
    print("\nDetailed Job Information:")
    print(display_df[display_cols])
    
    print(f"\nTotal jobs found: {len(df)}")
    
    # Print URLs separately
    print("\nJob URLs:")
    for idx, (job_id, url) in enumerate(zip(df['job_id'], df['job_url']), 1):
        print(f"{idx}. {job_id}: {url}")

def save_to_csv(df: pd.DataFrame, filename: Optional[str] = None):
    """
    Save job results to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame with job data
        filename (str, optional): Output filename
    """
    if df.empty:
        print("No data to save.")
        return
        
    # Generate default filename if none provided
    if not filename:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f'recent_jobs_{today}.csv'
    
    # Ensure the data directory exists
    data_dir = os.path.join(script_dir, '../data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Query and display recent job postings')
    
    # Basic options
    parser.add_argument('--limit', type=int, default=50,
                    help='Maximum number of jobs to display (default: 50)')
    parser.add_argument('--days', type=int, default=7,
                    help='Only include jobs posted within this many days (default: 7)')
    parser.add_argument('--min-relevance', type=float, default=7.0,
                    help='Minimum relevance score threshold (default: 7.0)')
    parser.add_argument('--early-only', action='store_true',
                    help='Only include early application opportunities')
    parser.add_argument('--no-internships', action='store_true',
                    help='Exclude internship positions')
    
    # Filter options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument('--location', type=str,
                    help='Filter by location (e.g., "San Francisco" or "Remote")')
    filter_group.add_argument('--company', type=str,
                    help='Filter by company name')
    filter_group.add_argument('--title', type=str,
                    help='Filter by job title')
    filter_group.add_argument('--keyword', type=str,
                    help='Filter by search keyword used')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--verbose', action='store_true',
                    help='Display additional job details')
    output_group.add_argument('--save', action='store_true',
                    help='Save results to CSV')
    output_group.add_argument('--output', type=str,
                    help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Query jobs with specified filters
    df = get_recent_relevant_jobs(
        limit=args.limit,
        days=args.days,
        min_relevance=args.min_relevance,
        early_only=args.early_only,
        include_internships=not args.no_internships,
        location_filter=args.location,
        company_filter=args.company,
        title_filter=args.title,
        keyword_filter=args.keyword
    )
    
    # Display results
    display_results(df, verbose=args.verbose)
    
    # Save to CSV if requested
    if args.save:
        save_to_csv(df, args.output)

if __name__ == "__main__":
    main() 