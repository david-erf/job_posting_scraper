#!/usr/bin/env python3
"""
Query recent job postings that are either:
- Early applications or recently posted (hours ago)
AND
- High relevance (>7) or internship positions
"""

import pandas as pd
from db import db
import logging

# Set pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 40)  # Truncate long text fields

def get_recent_relevant_jobs():
    query = """
    SELECT 
        r.job_id,
        r.title,
        r.company,
        r.location,
        r.date_posted,
        r.hiring_status,
        r.job_url,
        COALESCE(a.title_relevance, 0) as title_relevance,
        COALESCE(f.salary_range, 'Not specified') as salary_range,
        r.created_at
    FROM raw_jobs r
    LEFT JOIN job_analysis a ON r.job_id = a.job_id
    LEFT JOIN formatted_descriptions f ON r.job_id = f.job_id
    WHERE (
        -- Early application opportunity or posted within hours
        r.hiring_status = 'Be an early applicant'
        OR r.date_posted LIKE '%hour%'
    )
    AND (
        -- High relevance or internship position
        COALESCE(a.title_relevance, 0) > 7
        OR LOWER(r.title) LIKE '%intern%'
    )
    ORDER BY 
        -- Prioritize most recent jobs
        r.created_at DESC,
        -- Then by relevance
        title_relevance DESC
    LIMIT 50
    """
    
    df = db.query_to_df(query)
    
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
    
    # Select and reorder columns for display
    display_cols = [
        'title', 'company', 'location', 'date_posted', 
        'hiring_status', 'title_relevance', 'salary_range',
        'created_at'
    ]
    
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

if __name__ == "__main__":
    get_recent_relevant_jobs() 