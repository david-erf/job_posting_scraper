#!/usr/bin/env python3
"""
Location Analysis Report

This script analyzes job locations to identify hiring trends by geography,
remote work opportunities, and location-based salary information.
"""

import sys
import os
import pandas as pd
import argparse
import re
import logging
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# Add the src directory to path if needed
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import from parent directory
from db import db

# Remote work keywords to search for
REMOTE_KEYWORDS = [
    'remote', 'work from home', 'wfh', 'virtual', 'telecommute',
    'telework', 'distributed team', 'anywhere'
]

# Hybrid work keywords
HYBRID_KEYWORDS = [
    'hybrid', 'flexible', 'partial remote', 'part-remote', 
    'remote optional', 'in-office', 'on-site'
]

def get_location_data(filters: Dict = None, min_relevance: float = 0, limit: int = 1000) -> pd.DataFrame:
    """
    Retrieve location data from jobs.
    
    Args:
        filters (Dict): Dictionary of filter conditions
        min_relevance (float): Minimum title relevance score
        limit (int): Maximum number of jobs to analyze
        
    Returns:
        pd.DataFrame: DataFrame with job details including location information
    """
    # Build WHERE clause conditions
    where_clauses = [f"a.title_relevance >= {min_relevance}"]
    
    if filters:
        if title_filter := filters.get('title'):
            where_clauses.append(f"LOWER(r.title) LIKE '%{title_filter.lower()}%'")
        
        if company_filter := filters.get('company'):
            where_clauses.append(f"LOWER(r.company) LIKE '%{company_filter.lower()}%'")
            
        if keyword_filter := filters.get('keyword'):
            where_clauses.append(f"LOWER(r.searched_keyword) LIKE '%{keyword_filter.lower()}%'")
            
        if date_filter := filters.get('days'):
            where_clauses.append(f"r.date_posted >= date('now', '-{date_filter} days')")
    
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
    ORDER BY r.date_posted DESC
    LIMIT {limit}
    """
    
    return db.query_to_df(query)

def parse_locations(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """
    Parse location data from job listings.
    
    Args:
        df (pd.DataFrame): DataFrame with job data
        
    Returns:
        Tuple[Dict, Dict, Dict]: Tuple containing city counts, state counts, and country counts
    """
    city_counts = Counter()
    state_counts = Counter()
    country_counts = Counter()
    
    # Regex patterns for US states
    us_state_pattern = r'(?:^|\W)(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming)(?:$|\W)'
    
    # Dictionary to standardize state names
    state_abbr_to_name = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    # Add reverse mapping (full name to abbreviation)
    name_to_abbr = {v.lower(): k for k, v in state_abbr_to_name.items()}
    
    for _, row in df.iterrows():
        location = str(row['location']).strip() if not pd.isna(row['location']) else ""
        
        if not location or location.lower() in ('remote', 'anywhere', 'work from home'):
            city_counts['Remote'] += 1
            state_counts['Remote'] += 1
            country_counts['Remote'] += 1
            continue
            
        # Extract location components
        parts = [p.strip() for p in re.split(r'[,;]', location) if p.strip()]
        
        if len(parts) >= 2:
            # Assume format is "City, State/Country" or "City, State, Country"
            city = parts[0]
            
            # Check if the second part is a US state
            second_part = parts[1].strip()
            state_match = re.search(us_state_pattern, second_part, re.IGNORECASE)
            
            if state_match:
                state_text = state_match.group(0).strip()
                # Standardize state name
                if state_text.upper() in state_abbr_to_name:
                    state = state_abbr_to_name[state_text.upper()]
                elif state_text.lower() in name_to_abbr:
                    state = state_abbr_to_name[name_to_abbr[state_text.lower()]]
                else:
                    state = state_text
                    
                country = "United States"
            else:
                state = "Unknown"
                country = second_part
                
                # If there's a third part, it's likely the country
                if len(parts) >= 3:
                    country = parts[2]
        else:
            # Only one part, could be a city, state, or country
            city = parts[0]
            
            # Check if it's a US state
            state_match = re.search(us_state_pattern, city, re.IGNORECASE)
            if state_match:
                city = "Unknown"
                state_text = state_match.group(0).strip()
                # Standardize state name
                if state_text.upper() in state_abbr_to_name:
                    state = state_abbr_to_name[state_text.upper()]
                elif state_text.lower() in name_to_abbr:
                    state = state_abbr_to_name[name_to_abbr[state_text.lower()]]
                else:
                    state = state_text
                country = "United States"
            else:
                # Assume it's a major city or country
                state = "Unknown"
                country = "Unknown"
                
                # Try to determine if it's a country
                major_countries = [
                    'United States', 'USA', 'U.S.', 'Canada', 'UK', 'United Kingdom',
                    'Australia', 'Germany', 'France', 'India', 'Japan', 'China'
                ]
                for c in major_countries:
                    if c.lower() in city.lower():
                        city = "Unknown"
                        country = c
                        break
        
        city_counts[city] += 1
        state_counts[state] += 1
        country_counts[country] += 1
    
    return city_counts, state_counts, country_counts

def analyze_remote_work(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Analyze remote work information from job listings.
    
    Args:
        df (pd.DataFrame): DataFrame with job data
        
    Returns:
        Tuple[Dict, pd.DataFrame]: Remote work stats and identified remote jobs
    """
    remote_stats = {
        'total': len(df),
        'remote': 0,
        'hybrid': 0,
        'onsite': 0,
        'unknown': 0
    }
    
    # Create new columns for work arrangement classification
    df['is_remote'] = False
    df['is_hybrid'] = False
    df['work_arrangement'] = 'Unknown'
    
    for idx, row in df.iterrows():
        location = str(row['location']).lower() if not pd.isna(row['location']) else ""
        description = str(row['description_text']).lower() if not pd.isna(row['description_text']) else ""
        
        # Check if explicitly remote in location field
        is_remote_location = any(kw in location for kw in ['remote', 'anywhere', 'work from home', 'wfh'])
        
        # Check for remote keywords in description
        is_remote_desc = any(re.search(r'\b' + re.escape(kw) + r'\b', description) for kw in REMOTE_KEYWORDS)
        
        # Check for hybrid keywords
        is_hybrid = any(re.search(r'\b' + re.escape(kw) + r'\b', description) for kw in HYBRID_KEYWORDS)
        
        # Update dataframe with classifications
        if is_remote_location or is_remote_desc:
            df.at[idx, 'is_remote'] = True
            df.at[idx, 'work_arrangement'] = 'Remote'
            remote_stats['remote'] += 1
        elif is_hybrid:
            df.at[idx, 'is_hybrid'] = True
            df.at[idx, 'work_arrangement'] = 'Hybrid'
            remote_stats['hybrid'] += 1
        elif 'onsite' in description or 'on-site' in description or 'in office' in description:
            df.at[idx, 'work_arrangement'] = 'On-site'
            remote_stats['onsite'] += 1
        else:
            remote_stats['unknown'] += 1
    
    # Calculate percentages
    remote_stats['remote_percent'] = round((remote_stats['remote'] / remote_stats['total']) * 100, 1)
    remote_stats['hybrid_percent'] = round((remote_stats['hybrid'] / remote_stats['total']) * 100, 1)
    remote_stats['onsite_percent'] = round((remote_stats['onsite'] / remote_stats['total']) * 100, 1)
    
    return remote_stats, df

def display_location_results(city_counts: Counter, state_counts: Counter, country_counts: Counter, 
                            remote_stats: Dict, top_n: int = 10, show_plot: bool = False):
    """
    Display location analysis results.
    
    Args:
        city_counts (Counter): City frequency counts
        state_counts (Counter): State frequency counts
        country_counts (Counter): Country frequency counts
        remote_stats (Dict): Remote work statistics
        top_n (int): Number of top locations to display
        show_plot (bool): Whether to display plots
    """
    print("\n===== LOCATION ANALYSIS RESULTS =====\n")
    
    # Display remote work statistics
    print("\nRemote Work Analysis:")
    print("-" * 40)
    print(f"Total jobs analyzed: {remote_stats['total']}")
    print(f"Remote jobs: {remote_stats['remote']} ({remote_stats['remote_percent']}%)")
    print(f"Hybrid jobs: {remote_stats['hybrid']} ({remote_stats['hybrid_percent']}%)")
    print(f"On-site jobs: {remote_stats['onsite']} ({remote_stats['onsite_percent']}%)")
    print(f"Unclassified: {remote_stats['unknown']} ({round((remote_stats['unknown'] / remote_stats['total']) * 100, 1)}%)")
    
    # Display top cities
    print("\nTop Cities:")
    print("-" * 40)
    for city, count in city_counts.most_common(top_n):
        print(f"{city.ljust(25)} {count} ({round((count / sum(city_counts.values())) * 100, 1)}%)")
    
    # Display top states
    print("\nTop States/Regions:")
    print("-" * 40)
    for state, count in state_counts.most_common(top_n):
        print(f"{state.ljust(25)} {count} ({round((count / sum(state_counts.values())) * 100, 1)}%)")
    
    # Display top countries
    print("\nTop Countries:")
    print("-" * 40)
    for country, count in country_counts.most_common(top_n):
        print(f"{country.ljust(25)} {count} ({round((count / sum(country_counts.values())) * 100, 1)}%)")
    
    # Create plots if requested
    if show_plot:
        # Remote work pie chart
        plt.figure(figsize=(10, 6))
        labels = ['Remote', 'Hybrid', 'On-site', 'Unknown']
        sizes = [remote_stats['remote'], remote_stats['hybrid'], 
                remote_stats['onsite'], remote_stats['unknown']]
        colors = ['#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Job Postings by Work Arrangement')
        plt.tight_layout()
        plt.show()
        
        # Top cities bar chart
        plt.figure(figsize=(12, 8))
        cities, counts = zip(*city_counts.most_common(top_n))
        plt.barh(cities, counts)
        plt.xlabel('Number of Job Postings')
        plt.title('Top Cities for Job Postings')
        plt.tight_layout()
        plt.show()
        
        # Top states bar chart
        plt.figure(figsize=(12, 8))
        states, counts = zip(*state_counts.most_common(top_n))
        plt.barh(states, counts)
        plt.xlabel('Number of Job Postings')
        plt.title('Top States/Regions for Job Postings')
        plt.tight_layout()
        plt.show()

def save_to_csv(df: pd.DataFrame, city_counts: Counter, state_counts: Counter, 
                country_counts: Counter, filename: Optional[str] = None):
    """
    Save location analysis results to CSV.
    
    Args:
        df (pd.DataFrame): DataFrame with job data and classifications
        city_counts (Counter): City frequency counts
        state_counts (Counter): State frequency counts 
        country_counts (Counter): Country frequency counts
        filename (str, optional): Output filename
    """
    # Generate default filename if none provided
    if not filename:
        import datetime
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f'location_analysis_{today}.csv'
    
    # Ensure the data directory exists
    data_dir = os.path.join(script_dir, '../data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    # Save the jobs dataframe with work arrangement classifications
    selected_cols = ['job_id', 'title', 'company', 'location', 'date_posted', 
                    'is_remote', 'is_hybrid', 'work_arrangement', 'title_relevance']
    df_subset = df[selected_cols].copy()
    df_subset.to_csv(filepath, index=False)
    
    # Save location counts to separate files
    location_data = []
    
    # Add city data
    for city, count in city_counts.items():
        location_data.append({
            'location_type': 'city',
            'location_name': city,
            'count': count,
            'percentage': round((count / sum(city_counts.values())) * 100, 1)
        })
    
    # Add state data
    for state, count in state_counts.items():
        location_data.append({
            'location_type': 'state',
            'location_name': state,
            'count': count,
            'percentage': round((count / sum(state_counts.values())) * 100, 1)
        })
    
    # Add country data
    for country, count in country_counts.items():
        location_data.append({
            'location_type': 'country',
            'location_name': country,
            'count': count,
            'percentage': round((count / sum(country_counts.values())) * 100, 1)
        })
    
    # Save location data to a separate file
    locations_filepath = os.path.join(data_dir, f'location_counts_{today}.csv')
    pd.DataFrame(location_data).to_csv(locations_filepath, index=False)
    
    print(f"\nJob data with work arrangements saved to {filepath}")
    print(f"Location counts saved to {locations_filepath}")

def main():
    parser = argparse.ArgumentParser(description='Analyze job locations and remote work trends')
    
    # Filtering options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument('--title', type=str,
                        help='Filter jobs by title (e.g., "data scientist")')
    filter_group.add_argument('--company', type=str,
                        help='Filter jobs by company')
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
    analysis_group.add_argument('--top', type=int, default=10,
                        help='Show top N locations per category (default: 10)')
    analysis_group.add_argument('--plot', action='store_true',
                        help='Display visualizations of location data')
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
    if args.days:
        filters['days'] = args.days
    
    # Fetch job data with location information
    print(f"Fetching job data (limit: {args.limit})...")
    df = get_location_data(
        filters=filters,
        min_relevance=args.min_relevance,
        limit=args.limit
    )
    
    if df.empty:
        print("No matching jobs found.")
        return
        
    print(f"Retrieved {len(df)} jobs for location analysis.")
    
    # Parse locations
    print("Analyzing location data...")
    city_counts, state_counts, country_counts = parse_locations(df)
    
    # Analyze remote work information
    print("Analyzing remote work trends...")
    remote_stats, df_classified = analyze_remote_work(df)
    
    # Display results
    display_location_results(
        city_counts, 
        state_counts, 
        country_counts,
        remote_stats,
        top_n=args.top, 
        show_plot=args.plot
    )
    
    # Save results if requested
    if args.save:
        save_to_csv(df_classified, city_counts, state_counts, country_counts, args.output)

if __name__ == "__main__":
    main() 