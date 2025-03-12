#!/usr/bin/env python3
"""
LinkedIn Job Search and Analysis Tool

This script searches for jobs on LinkedIn based on keywords,
scores their relevance, and downloads job descriptions.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import time
import re
import pickle
import argparse
import concurrent.futures
from functools import partial
from datetime import datetime, date
import hashlib  # Added for resume hash calculation

from dotenv import load_dotenv
from bs4 import BeautifulSoup
import argparse
import random

# Add the src directory to the path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import local modules
from utils import get_jobs_for_keyword, parse_job_id, rank_job_posting
from db import db
from config import config, load_config

def score_job_title(title, resume):
    """Dedicated function for scoring job titles against resume (better for parallel processing)"""
    if not title or pd.isna(title):
        return 0
    return rank_job_posting(resume, title)

def score_job_description(description, resume):
    """Score job description against resume"""
    if not description or pd.isna(description):
        return 0.0
    return float(rank_job_posting(resume, description))

def compute_resume_hash(content: str) -> str:
    """Compute a hash of the resume content for tracking changes."""
    return hashlib.sha256(content.encode()).hexdigest()

def get_jobs_needing_description_relevance():
    """
    Get all jobs with formatted descriptions that don't have description relevance scores.
    
    Returns:
        DataFrame with job_id, description_text, and title_relevance for jobs that need scoring
    """
    query = """
        SELECT f.job_id, f.description_text, a.title_relevance 
        FROM formatted_descriptions f
        JOIN job_analysis a ON f.job_id = a.job_id
        WHERE a.description_relevance IS NULL
        AND f.description_text IS NOT NULL
        AND LENGTH(f.description_text) > 0
    """
    return db.query_to_df(query)

def load_keywords_from_file(file_path):
    """Load keywords from a file (text or JSON format)
    
    Args:
        file_path: Path to the file containing keywords
        
    Returns:
        List of keywords or None if file couldn't be loaded
    """
    try:
        with open(file_path, 'r') as f:
            # Try JSON format first
            if file_path.lower().endswith('.json'):
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'keywords' in data:
                        return data['keywords']
                    else:
                        logging.error(f"Invalid JSON format in {file_path}. Expected a list or a dict with 'keywords' key")
                        return None
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON from {file_path}")
                    return None
            
            # Default to text format (one keyword per line)
            return [line.strip() for line in f 
                   if line.strip() and not line.strip().startswith('#')]
    except Exception as e:
        logging.error(f"Error loading keywords file {file_path}: {e}")
        return None

def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process job listings from LinkedIn')
    
    # Primary options - the ones you use most
    parser.add_argument('--process-only', action='store_true', 
                        help='Skip job search and start directly from processing existing jobs')
    parser.add_argument('--pages', type=int, default=20,
                        help='Number of pages to scrape per keyword (default: 20)')
    parser.add_argument('--keywords', type=str, nargs='+',
                        help='Override default keywords to search for (space-separated)')
    parser.add_argument('--resume', type=str,
                        help='Path to resume file (overrides RESUME_PATH from environment)')
    
    # Add config file option
    parser.add_argument('--config', type=str,
                        help='Path to configuration file (JSON)')
    parser.add_argument('--keywords-file', type=str,
                        help='Path to file containing search keywords (one per line or JSON format)')
    
    # Advanced options - grouped separately
    advanced_group = parser.add_argument_group('Advanced Options (typically used for database queries)')
    advanced_group.add_argument('--location', type=str,
                        help='Filter jobs by location (rarely used during scraping)')
    advanced_group.add_argument('--company', type=str, nargs='+',
                        help='Filter jobs by company names (space-separated)')
    advanced_group.add_argument('--max-age', type=int,
                        help='Maximum job age in days (default: 30)')
    advanced_group.add_argument('--date-posted', choices=['24h', 'week', 'month', 'any'],
                        default='any', help='Filter by date posted')
    
    # Processing related arguments
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument('--title-threshold', type=float,
                        help='Title relevance threshold for processing descriptions (default: 1.0)')
    process_group.add_argument('--no-descriptions', action='store_true',
                        help='Skip processing job descriptions, only score titles')
    process_group.add_argument('--batch-size', type=int,
                        help='Batch size for database operations (default: 500)')
    process_group.add_argument('--max-jobs', type=int,
                        help='Maximum number of jobs to process')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                        help='Minimize output (show only errors and important messages)')
    output_group.add_argument('--log-file', type=str,
                        help='Log output to the specified file')
    
    args = parser.parse_args(args)
    
    # Load configuration from file if specified
    if args.config:
        load_config(args.config)
    
    # Override config with command line arguments
    if args.pages:
        config.set('search.pages_per_keyword', args.pages)
    if args.location:
        config.set('search.location', args.location)
    if args.company:
        config.set('search.companies', args.company)
    if args.max_age:
        config.set('search.max_age_days', args.max_age)
    if args.date_posted and args.date_posted != 'any':
        config.set('search.date_posted', args.date_posted)
    if args.title_threshold:
        config.set('analysis.title_relevance_threshold', args.title_threshold)
    if args.no_descriptions:
        config.set('analysis.process_descriptions', False)
    if args.batch_size:
        config.set('analysis.batch_size', args.batch_size)
    if args.max_jobs:
        config.set('search.max_jobs', args.max_jobs)
    
    # Configure logging based on verbosity settings
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    
    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=args.log_file,
            filemode='a'
        )
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Check for keywords from file first (command-line option)
    if args.keywords_file and os.path.exists(args.keywords_file):
        keywords = load_keywords_from_file(args.keywords_file)
        if keywords:
            args.keywords = keywords
            logging.info(f"Loaded {len(keywords)} keywords from {args.keywords_file}")
    # If no keywords specified via command line, try environment variable
    elif not args.keywords:
        keywords_file_path = os.getenv("KEYWORDS_FILE_PATH")
        logging.info(f"KEYWORDS_FILE_PATH from env: {keywords_file_path}")
        
        if keywords_file_path and os.path.exists(keywords_file_path):
            logging.info(f"Keywords file exists at: {keywords_file_path}")
            
            # Read and print raw file contents for debugging
            try:
                with open(keywords_file_path, 'r') as f:
                    raw_content = f.read()
                    logging.info(f"Raw file content first 200 chars: {raw_content[:200]}")
                    # Count non-empty, non-comment lines
                    non_comment_lines = [line for line in raw_content.splitlines() 
                                         if line.strip() and not line.strip().startswith('#')]
                    logging.info(f"Non-comment lines count: {len(non_comment_lines)}")
            except Exception as e:
                logging.error(f"Error reading raw file: {e}")
            
            keywords = load_keywords_from_file(keywords_file_path)
            if keywords:
                logging.info(f"Parsed keywords list: {keywords}")
                args.keywords = keywords
                logging.info(f"Loaded {len(keywords)} keywords from environment variable path: {keywords_file_path}")
                logging.debug(f"Keywords from env file: {args.keywords}")
            else:
                logging.error(f"Failed to parse keywords from file: {keywords_file_path}")
        else:
            if keywords_file_path:
                logging.error(f"Keywords file does not exist: {keywords_file_path}")
            else:
                logging.info("KEYWORDS_FILE_PATH environment variable not set")
    
    # Validate that we have keywords if not in process-only mode
    if not args.process_only and not args.keywords:
        logging.warning("No keywords specified and no keywords loaded from file. Using default keywords.")
        # Default keywords if none provided 
        args.keywords = ["data scientist", "machine learning"]
        logging.info(f"Using default keywords: {args.keywords}")
    
    today = date.today()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # DEBUGGING: Print keywords before config.get
    logging.info(f"Keywords from args before config.get: {args.keywords}")
    
    # Only get keywords from config if args.keywords is None/empty
    if args.keywords:
        search_keywords = args.keywords
        logging.info(f"Using keywords from arguments/env file: {search_keywords}")
    else:
        search_keywords = config.get('search.keywords')
        logging.info(f"Using keywords from config: {search_keywords}")
    
    # Get other configuration settings
    pages_per_keyword = config.get('search.pages_per_keyword', args.pages)
    title_threshold = config.get('analysis.title_relevance_threshold', args.title_threshold if hasattr(args, 'title_threshold') else 1.0)
    process_descriptions = config.get('analysis.process_descriptions', not args.no_descriptions if hasattr(args, 'no_descriptions') else True)
    batch_size = config.get('analysis.batch_size', args.batch_size if hasattr(args, 'batch_size') else 500)
    max_jobs = config.get('search.max_jobs', args.max_jobs if hasattr(args, 'max_jobs') else None)
    
    # Build filter options
    search_filters = {}
    if location := config.get('search.location'):
        search_filters['location'] = location
    if companies := config.get('search.companies'):
        search_filters['companies'] = companies
    if date_posted := config.get('search.date_posted'):
        search_filters['date_posted'] = date_posted
    if max_age_days := config.get('search.max_age_days'):
        search_filters['max_age_days'] = max_age_days
    
    logging.info(f"Starting job search with keywords: {search_keywords}")
    if search_filters:
        logging.info(f"Applied filters: {search_filters}")
    
    # Load resume from path in config, command line argument, or environment variable
    resume_path = args.resume or config.get('resume.path', os.getenv('RESUME_PATH'))
    if not resume_path:
        logging.error("Resume path not specified in arguments, config or RESUME_PATH environment variable")
        sys.exit(1)
        
    try:
        with open(resume_path, "r") as file:
            resume = file.read()
            resume += ", " + ", ".join(search_keywords)
            logging.info(f"Loaded resume from {resume_path}")
    except Exception as e:
        logging.error(f"Error loading resume from {resume_path}: {e}")
        sys.exit(1)
    
    # Only run job search and initial processing if not in process-only mode
    if not args.process_only:
        # Collect job postings
        logging.info(f"Starting job search at {timestamp}")
        results = {}
        
        # Set up search parameters
        search_params = {
            'pages': pages_per_keyword
        }
        
        # Add any additional filters from search_filters
        if 'location' in search_filters:
            search_params['location'] = search_filters['location']
        if 'date_posted' in search_filters:
            search_params['date_posted'] = search_filters['date_posted']
        
        for idx, k in enumerate(search_keywords, start=1):
            logging.info(f"Processing keyword: {k} ({idx}/{len(search_keywords)}) - Pages: {pages_per_keyword}")
            results[k] = get_jobs_for_keyword(k, **search_params)
        
        # Process results into a DataFrame
        dfs = []
        for k, job_list in results.items():
            if not job_list:
                continue
            df = pd.DataFrame(job_list)
            df['searched_keyword'] = k
            dfs.append(df)
        
        if not dfs:
            logging.warning("No jobs found. Exiting.")
            return
        
        # Combine and clean raw job data
        results_df = pd.concat(dfs, axis=0)
        results_df['company'].fillna('', inplace=True)
        
        # Apply additional filters
        original_count = len(results_df)
        
        # Filter by company if specified
        if 'companies' in search_filters and search_filters['companies']:
            company_filter = [c.lower() for c in search_filters['companies']]
            results_df = results_df[results_df['company'].str.lower().isin(company_filter)]
            logging.info(f"Filtered by company: {original_count} → {len(results_df)} jobs")
            
        # Filter by max age if specified
        if 'max_age_days' in search_filters:
            # Implementation depends on date format in the data
            # This is a placeholder
            logging.info(f"Age filtering would apply here if implemented")
        
        if len(results_df) == 0:
            logging.warning("No jobs found after applying filters. Exiting.")
            return
            
        # De-duplicate before any scoring or DB insertion (more efficient)
        logging.info(f"Raw job count: {len(results_df)}")
        results_df.drop_duplicates(subset=['title', 'company', 'searched_keyword', 'job_url'], inplace=True)
        logging.info(f"After deduplication: {len(results_df)} jobs")
        
        # Limit number of jobs if specified
        if max_jobs and len(results_df) > max_jobs:
            logging.info(f"Limiting to {max_jobs} jobs (from {len(results_df)})")
            results_df = results_df.sample(max_jobs) if max_jobs < len(results_df) else results_df
        
        # Insert raw jobs into database
        db.insert_raw_jobs(results_df)
        
        # Check which jobs have already been analyzed
        logging.info("Checking for jobs that need analysis...")
        
        # Get list of all jobs in raw_jobs that aren't in job_analysis
        jobs_to_analyze_df = db.get_jobs_needing_analysis()
        
        if jobs_to_analyze_df.empty:
            logging.info("No new jobs to analyze.")
        else:
            logging.info(f"Found {len(jobs_to_analyze_df)} new jobs to analyze.")
            
            # Score job titles in parallel for better performance
            logging.info("Scoring job titles for relevance...")
            
            # Use a ThreadPoolExecutor for parallel processing of the scoring
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() * 2)) as executor:
                score_func = partial(score_job_title, resume=resume)
                titles = jobs_to_analyze_df['title'].tolist()
                scores = list(executor.map(score_func, titles))
            
            # Add scores to DataFrame
            jobs_to_analyze_df['title_relevance'] = scores
            
            # Save analysis results to the database
            analysis_df = jobs_to_analyze_df[['job_id', 'title_relevance']]
            db.insert_job_analysis(analysis_df)
            
            logging.info(f'Job title analysis complete! Analyzed {len(jobs_to_analyze_df)} new jobs.')
            
            # Check for jobs with descriptions that need relevance scoring
            logging.info("Checking for jobs with descriptions that need relevance scoring...")
            
            # Get all jobs with formatted descriptions that don't have description relevance
            jobs_needing_desc_scoring = get_jobs_needing_description_relevance()
            
            if not jobs_needing_desc_scoring.empty:
                logging.info(f"Found {len(jobs_needing_desc_scoring)} jobs with descriptions that need relevance scoring.")
                
                # Calculate resume hash to track version
                resume_version = os.path.basename(resume_path) if resume_path else "default_resume"
                resume_hash = compute_resume_hash(resume)
                current_time = datetime.now().isoformat()
                
                # Score descriptions in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
                    score_func = partial(score_job_description, resume=resume)
                    desc_scores = list(executor.map(score_func, jobs_needing_desc_scoring['description_text'].tolist()))
                
                # Prepare analysis data for update
                jobs_needing_desc_scoring['description_relevance'] = desc_scores
                jobs_needing_desc_scoring['analyzed_at'] = current_time
                jobs_needing_desc_scoring['resume_version'] = resume_version
                jobs_needing_desc_scoring['resume_hash'] = resume_hash
                
                # Create DataFrame for database update
                analysis_update_df = jobs_needing_desc_scoring[['job_id', 'title_relevance', 'description_relevance', 
                                                               'analyzed_at', 'resume_version', 'resume_hash']]
                
                # Update database in batches to reduce lock duration
                batch_size = 100
                for i in range(0, len(analysis_update_df), batch_size):
                    batch = analysis_update_df.iloc[i:i + batch_size]
                    db.insert_job_analysis(batch)
                    logging.info(f"Updated batch {i//batch_size + 1} of {(len(analysis_update_df) - 1)//batch_size + 1}")
                
                logging.info(f"Job description relevance scoring complete for {len(jobs_needing_desc_scoring)} jobs.")
            else:
                logging.info("No jobs with descriptions need relevance scoring.")
        
        logging.info('Job search and job analysis complete!')
    else:
        logging.info("Skipping job search and initial processing...")
    
    # Skip description processing if requested
    if not process_descriptions:
        logging.info("Skipping job description processing as requested.")
        return
    
    # Only process job descriptions for relevant jobs
    jobs = {}

    # Identify jobs that still need processing
    logging.info("Querying database for processing status...")
    
    # Use the helper method to get jobs that need processing
    processing_status = db.get_jobs_needing_descriptions(title_threshold)
    
    # Extract the sets for easier reference
    relevant_jobs = processing_status['relevant']
    need_both = processing_status['need_raw'] 
    need_formatted_only = processing_status['need_formatted']
    fully_processed = processing_status['fully_processed']

    logging.info(f"\nJob processing status (title relevance threshold: {title_threshold}):")
    logging.info(f"- Total relevant jobs: {len(relevant_jobs)}")
    logging.info(f"- Already fully processed: {len(fully_processed)}")
    logging.info(f"- Have raw data but need formatting: {len(need_formatted_only)}")
    logging.info(f"- Need complete processing: {len(need_both)}")
    
    # Get job URLs for jobs that need processing
    if need_both:
        logging.info(f"Getting URLs for {len(need_both)} jobs that need complete processing...")
        need_both_urls = db.get_job_urls(need_both, batch_size=batch_size)
    else:
        need_both_urls = {}

    if need_formatted_only:
        logging.info(f"Getting URLs for {len(need_formatted_only)} jobs that need formatting...")
        need_formatted_urls = db.get_job_urls(need_formatted_only, batch_size=batch_size)
    else:
        need_formatted_urls = {}

    # First pass: Process jobs that need raw descriptions
    if need_both:
        logging.info(f"\nProcessing {len(need_both)} jobs that need raw descriptions:")
        
        skipped = 0
        errors = 0
        success = 0
        
        try:
            # Process jobs using URLs from need_both_urls dictionary
            for idx, job_id in enumerate(need_both, 1):
                progress = (idx / len(need_both)) * 100
                job_url = need_both_urls.get(job_id)
                
                if not job_url:
                    errors += 1
                    logging.error(f"Error processing job {job_id}: No URL found")
                    continue
                
                # Print progress inline
                print(f"\rProgress: [{idx}/{len(need_both)}] {progress:.1f}% - Processing job {job_id}", end="", flush=True)
                
                # Periodically log progress to the log file without the carriage return
                if idx % 5 == 0 or idx == len(need_both):
                    logging.info(f"Progress: [{idx}/{len(need_both)}] {progress:.1f}% - Processing job {job_id}")
                
                try:
                    # Process the job
                    job_data = parse_job_id(job_id, job_url, resume, extract_html=True)
                    jobs[job_id] = job_data
                    
                    # Immediately store raw HTML in the database
                    if 'raw_response' in job_data and job_data['raw_response']:
                        raw_description = [{
                            'job_id': job_id,
                            'html_content': job_data['raw_response'].text if hasattr(job_data['raw_response'], 'text') else '',
                            'response_status': job_data['raw_response'].status_code if hasattr(job_data['raw_response'], 'status_code') else 0
                        }]
                        db.insert_raw_descriptions(raw_description)
                        
                        # If job has description, immediately store formatted description too
                        if 'job_description' in job_data:
                            formatted_description = [{
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
                            }]
                            db.insert_formatted_descriptions(formatted_description)
                    
                    success += 1
                except Exception as e:
                    errors += 1
                    logging.error(f"Error processing job {job_id}: {str(e)}")
                
                time.sleep(2)  # Respect rate limits
            
        except KeyboardInterrupt:
            logging.warning("\n\nInterrupted by user.")
        finally:
            logging.info(f"\nRaw job processing ended! Summary:")
            logging.info(f"Successfully processed: {success}")
            logging.info(f"Errors: {errors}")
    
    # Second pass: Process jobs that only need formatted descriptions
    if need_formatted_only:
        logging.info(f"\nProcessing {len(need_formatted_only)} jobs that only need formatted descriptions:")
        
        # Get the raw HTML for these jobs
        # Process in batches to avoid SQL IN clause limitations
        raw_html_df = pd.DataFrame()
        need_formatted_list = list(need_formatted_only)
        
        for i in range(0, len(need_formatted_list), batch_size):
            batch = need_formatted_list[i:i+batch_size]
            batch_ids_list = ", ".join([f"'{job_id}'" for job_id in batch])
            batch_df = db.query_to_df(f"""
                SELECT job_id, html_content FROM raw_descriptions
                WHERE job_id IN ({batch_ids_list})
            """)
            raw_html_df = pd.concat([raw_html_df, batch_df])
        
        formatted_count = 0
        format_errors = 0
        total_to_format = len(raw_html_df)
        
        # Process each job for formatted descriptions
        for idx, row in enumerate(raw_html_df.iterrows(), 1):
            _, row_data = row  # Unpack the tuple from iterrows()
            job_id = row_data['job_id']
            html_content = row_data['html_content']
            progress = (idx / total_to_format) * 100
            
            try:
                # Parse the HTML content
                soup = BeautifulSoup(html_content, "html.parser")
                
                # Extract job description
                job_data = {}
                job_description_div = soup.find('div', class_='show-more-less-html__markup')
                if job_description_div:
                    job_description = job_description_div.get_text(strip=True)
                    job_data['job_description'] = job_description
                    relevance = rank_job_posting(resume, job_description)
                    job_data['job_description_relevance'] = relevance
                
                # Extract additional header information
                for item in soup.find_all('li', class_='description__job-criteria-item'):
                    header_elem = item.find('h3', class_='description__job-criteria-subheader')
                    value_elem = item.find('span', class_='description__job-criteria-text')
                    if header_elem and value_elem:
                        header = header_elem.get_text(strip=True)
                        value = value_elem.get_text(strip=True)
                        job_data[header] = value
                
                # Store the formatted description
                if 'job_description' in job_data:
                    formatted_description = [{
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
                    }]
                    db.insert_formatted_descriptions(formatted_description)
                    formatted_count += 1
                    logging.info(f"\rFormatting progress: [{idx}/{total_to_format}] {progress:.1f}% - Processed job {job_id}", end="")
                    
                    # Calculate description relevance immediately after formatting
                    job_description = job_data.get('job_description', '')
                    if job_description:
                        relevance = score_job_description(job_description, resume)
                        resume_version = os.path.basename(resume_path) if resume_path else "default_resume"
                        resume_hash = compute_resume_hash(resume)
                        current_time = datetime.now().isoformat()
                        
                        # Create analysis data for update
                        analysis_data = {
                            'job_id': job_id,
                            'description_relevance': relevance,
                            'analyzed_at': current_time,
                            'resume_version': resume_version,
                            'resume_hash': resume_hash
                        }
                        
                        # Get existing title relevance if available
                        existing_analysis = db.query_to_df(f"SELECT title_relevance FROM job_analysis WHERE job_id = '{job_id}'")
                        if not existing_analysis.empty and pd.notna(existing_analysis.iloc[0]['title_relevance']):
                            analysis_data['title_relevance'] = existing_analysis.iloc[0]['title_relevance']
                        
                        # Update analysis in database
                        analysis_df = pd.DataFrame([analysis_data])
                        db.insert_job_analysis(analysis_df)
            except Exception as e:
                format_errors += 1
                logging.error(f"Error formatting job {job_id}: {str(e)}")
        
        logging.info(f"\nFormatting complete! Successfully formatted: {formatted_count}, Errors: {format_errors}")
    
    # Save job descriptions for future reference (legacy)
    if jobs:
        with open(f'../data/job_descriptions_{timestamp}.pkl', "wb") as file:
            pickle.dump(jobs, file)
    
    logging.info("\nJob processing complete!")

if __name__ == '__main__':
    main()