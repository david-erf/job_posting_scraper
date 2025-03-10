import os
import time
import pandas as pd
import pickle
from datetime import date, datetime
import concurrent.futures
from functools import partial
from dotenv import load_dotenv
from utils import (
    get_jobs_for_keyword,
    convert_to_days,
    split_location,
    mapping,
    abbreviation_mapping,
    rank_job_posting,
    keywords,
    parse_job_id
)
from db import db  # Import the singleton database manager instance

load_dotenv()  # Load environment variables

def score_job_title(title, resume):
    """Dedicated function for scoring job titles against resume (better for parallel processing)"""
    if not title or pd.isna(title):
        return 0
    return rank_job_posting(resume, title)

def main():
    today = date.today()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load resume and augment with keywords
    resume_path = os.getenv("RESUME_PATH")
    with open(resume_path, "r") as file:
        resume = file.read()
        resume += ", " + ", ".join(keywords)
    
    # Collect job postings
    print(f"Starting job search at {timestamp}")
    results = {}
    for idx, k in enumerate(keywords, start=1):
        print(f"Processing keyword: {k} ({idx}/{len(keywords)})")
        results[k] = get_jobs_for_keyword(k, pages=20)
    
    # Process results into a DataFrame
    dfs = []
    for k, job_list in results.items():
        if not job_list:
            continue
        df = pd.DataFrame(job_list)
        df['searched_keyword'] = k
        dfs.append(df)
    
    if not dfs:
        print("No jobs found. Exiting.")
        return
    
    # Combine and clean raw job data
    results_df = pd.concat(dfs, axis=0)
    results_df['company'].fillna('', inplace=True)
    
    # De-duplicate before any scoring or DB insertion (more efficient)
    print(f"Raw job count: {len(results_df)}")
    results_df.drop_duplicates(subset=['title', 'company', 'searched_keyword', 'job_url'], inplace=True)
    print(f"After deduplication: {len(results_df)}")
    
    # Insert raw jobs into database
    db.insert_raw_jobs(results_df)
    
    # Score job titles in parallel for better performance
    print("Scoring job titles for relevance...")
    
    # Use a ThreadPoolExecutor for parallel processing of the scoring
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() * 2)) as executor:
        score_func = partial(score_job_title, resume=resume)
        titles = results_df['title'].tolist()
        scores = list(executor.map(score_func, titles))
    
    # Add scores to DataFrame
    results_df['title_relevance'] = scores
    
    # Save analysis results to the database
    analysis_df = results_df[['job_id', 'title_relevance']]
    db.insert_job_analysis(analysis_df)
    
    # Filter based on relevance threshold and extract job descriptions
    threshold = 1
    matched_results_df = results_df[results_df['title_relevance'] >= threshold].copy()
    print('Count of relevant jobs:', matched_results_df.shape[0])
    
    # Only process job descriptions for relevant jobs
    jobs = {}
    raw_descriptions_data = []
    
    # First pass: Store raw HTML responses
    for job_id, url in zip(matched_results_df['job_id'], matched_results_df['job_url']):
        print(f"Fetching raw HTML for job {job_id}")
        job_data = parse_job_id(job_id, url, resume, extract_html=True)
        jobs[job_id] = job_data
        
        # Store raw HTML in db
        if 'raw_response' in job_data and job_data['raw_response']:
            raw_descriptions_data.append({
                'job_id': job_id,
                'html_content': job_data['raw_response'].text if hasattr(job_data['raw_response'], 'text') else '',
                'response_status': job_data['raw_response'].status_code if hasattr(job_data['raw_response'], 'status_code') else 0
            })
        time.sleep(2)  # Respect rate limits
    
    # Insert raw HTML into database
    if raw_descriptions_data:
        print(f"Storing {len(raw_descriptions_data)} raw HTML descriptions in database")
        db.insert_raw_descriptions(raw_descriptions_data)
    
    # Save job descriptions for future reference (legacy)
    with open(f'../data/job_descriptions_{timestamp}.pkl', "wb") as file:
        pickle.dump(jobs, file)
    
    # Format descriptions and insert into formatted_descriptions table
    formatted_descriptions = []
    for job_id, job_data in jobs.items():
        if 'job_description' not in job_data:
            continue
            
        formatted_descriptions.append({
            'job_id': job_id,
            'description_text': job_data.get('job_description', ''),
            'seniority_level': job_data.get('Seniority level'),
            'employment_type': job_data.get('Employment type'),
            'job_function': job_data.get('Job function'),
            'industries': job_data.get('Industries'),
            'num_applicants': job_data.get('num_applicants'),
            'salary_range': job_data.get('salary_range'),
            'min_salary': job_data.get('min_salary'),
            'max_salary': job_data.get('max_salary'),
            'salary_frequency': job_data.get('salary_frequency')
        })
    
    # Insert formatted descriptions into database
    if formatted_descriptions:
        print(f"Storing {len(formatted_descriptions)} formatted descriptions in database")
        db.insert_formatted_descriptions(formatted_descriptions)
    
    # Update job analysis with description relevance scores
    description_analysis = []
    for job_id, job_data in jobs.items():
        if 'job_description_relevance' in job_data:
            description_analysis.append({
                'job_id': job_id,
                'description_relevance': job_data['job_description_relevance']
            })
    
    if description_analysis:
        description_df = pd.DataFrame(description_analysis)
        # Merge with existing analysis records to preserve title_relevance
        analysis_records = db.query_to_df(
            "SELECT job_id, title_relevance FROM job_analysis WHERE job_id IN ({})".format(
                ','.join(['?'] * len(description_analysis))
            ), 
            [d['job_id'] for d in description_analysis]
        )
        
        if not analysis_records.empty:
            full_analysis = pd.merge(
                analysis_records,
                description_df,
                on='job_id',
                how='outer'
            )
            db.insert_job_analysis(full_analysis)
    
    # Get all job data with descriptions for reporting
    print("Generating final report with all job data...")
    relevant_jobs = db.get_jobs_with_descriptions(relevance_threshold=threshold)
    
    if not relevant_jobs.empty:
        # Save the processed data for reference
        relevant_jobs.to_csv(f'../data/relevant_jobs_{timestamp}.csv', index=False)
        print(f"Process complete. Found {len(relevant_jobs)} relevant jobs with descriptions.")
    else:
        print("No jobs with descriptions found that meet the relevance criteria.")

if __name__ == '__main__':
    main()