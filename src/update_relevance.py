#!/usr/bin/env python3
import os
import sys
import hashlib
import pandas as pd
import concurrent.futures
from functools import partial
from datetime import datetime
from typing import Optional, Dict, List
import logging
from dotenv import load_dotenv
from utils import rank_job_posting
from db import db
import sqlite3
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('relevance_update.log')
    ]
)

def compute_resume_hash(content: str) -> str:
    """Compute a hash of the resume content for tracking changes."""
    return hashlib.sha256(content.encode()).hexdigest()

def score_job_title(title: str, resume: str) -> float:
    """Score job title against resume"""
    if not title or pd.isna(title):
        return 0.0
    return float(rank_job_posting(resume, title))

def score_job_description(description: str, resume: str) -> float:
    """Score job description against resume"""
    if not description or pd.isna(description):
        return 0.0
    return float(rank_job_posting(resume, description))

def wait_for_database_availability(max_retries: int = 5, retry_delay: int = 10) -> bool:
    """
    Check if the database is available for writing to job_analysis table.
    Will retry several times before giving up.
    """
    for attempt in range(max_retries):
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Execute each statement separately
                    cursor.execute("BEGIN IMMEDIATE")
                    cursor.execute("SELECT 1 FROM job_analysis LIMIT 1")
                    cursor.execute("ROLLBACK")
                    return True
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        logging.warning(f"Database is locked, attempt {attempt + 1}/{max_retries}. Waiting {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logging.error(f"Database error: {e}")
                        return False
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    cursor.execute("ROLLBACK")
                    return False
        except Exception as e:
            logging.error(f"Connection error: {e}")
            time.sleep(retry_delay)
    
    logging.error("Could not acquire database lock after maximum retries")
    return False

def regenerate_relevance_scores(
    resume_path: str,
    min_title_relevance: float = 0.0,
    min_description_relevance: float = 0.0
) -> Optional[pd.DataFrame]:
    """
    Regenerate relevance scores for all jobs using a new resume.
    
    Args:
        resume_path: Path to the new resume file
        min_title_relevance: Minimum title relevance score to include in results
        min_description_relevance: Minimum description relevance score to include in results
        
    Returns:
        DataFrame with high-relevance jobs or None if no matches found
    """
    try:
        # Check database availability first
        if not wait_for_database_availability():
            logging.error("Database is not available for writing. Is another process using it heavily?")
            return None
            
        # Load and process the new resume
        with open(resume_path, "r") as file:
            resume_content = file.read()
        
        resume_version = os.path.basename(resume_path)
        resume_hash = compute_resume_hash(resume_content)
        
        logging.info(f"Loaded resume: {resume_version} (hash: {resume_hash[:8]})")
        
        # Get all job titles from raw_jobs table
        jobs_df = db.query_to_df("""
            SELECT r.job_id, r.title, r.company, r.job_url, r.location,
                   f.description_text
            FROM raw_jobs r
            LEFT JOIN formatted_descriptions f ON r.job_id = f.job_id
        """)
        
        if jobs_df.empty:
            logging.warning("No jobs found in database")
            return None
        
        logging.info(f"Processing {len(jobs_df)} jobs...")
        
        # Score titles in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() * 2)) as executor:
            score_func = partial(score_job_title, resume=resume_content)
            jobs_df['title_relevance'] = list(executor.map(score_func, jobs_df['title'].tolist()))
        
        # Initialize description_relevance column with NaN
        jobs_df['description_relevance'] = pd.NA
        
        # Score descriptions in parallel for jobs with description text
        desc_jobs = jobs_df[jobs_df['description_text'].notna()]
        if not desc_jobs.empty:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
                score_func = partial(score_job_description, resume=resume_content)
                desc_scores = list(executor.map(score_func, desc_jobs['description_text'].tolist()))
                
                # Update scores in main DataFrame
                jobs_df.loc[desc_jobs.index, 'description_relevance'] = desc_scores
        
        # Prepare analysis data
        analysis_df = jobs_df[['job_id', 'title_relevance', 'description_relevance']].copy()
        analysis_df['analyzed_at'] = datetime.now().isoformat()
        analysis_df['resume_version'] = resume_version
        analysis_df['resume_hash'] = resume_hash
        
        # Use smaller batch sizes for updates to reduce lock duration
        batch_size = 100
        for i in range(0, len(analysis_df), batch_size):
            batch = analysis_df.iloc[i:i + batch_size]
            db.insert_job_analysis(batch)
            logging.info(f"Updated batch {i//batch_size + 1} of {(len(analysis_df) - 1)//batch_size + 1}")
        
        # Filter for high-relevance jobs
        high_relevance = jobs_df[
            (jobs_df['title_relevance'] >= min_title_relevance) |
            (jobs_df['description_relevance'] >= min_description_relevance)
        ].copy()
        
        if not high_relevance.empty:
            # Sort by relevance scores
            high_relevance.sort_values(
                by=['description_relevance', 'title_relevance'],
                ascending=[False, False],
                inplace=True
            )
            
            logging.info(f"Found {len(high_relevance)} relevant jobs")
            
            # Save results
            output_file = f"relevant_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            high_relevance.to_csv(output_file, index=False)
            logging.info(f"Results saved to {output_file}")
            
            return high_relevance
        else:
            logging.info("No jobs found meeting relevance criteria")
            return None
            
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            logging.error("Database is locked by another process. Try again later.")
        else:
            logging.error(f"Database error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing resume: {str(e)}", exc_info=True)
        return None

def print_results(df: pd.DataFrame) -> None:
    """Print a formatted summary of the results."""
    print("\nHigh Relevance Jobs:")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Company: {row['company']}")
        print(f"Location: {row['location']}")
        print(f"Relevance Scores:")
        print(f"  - Title: {row['title_relevance']:.2f}")
        print(f"  - Description: {row['description_relevance']:.2f if pd.notna(row['description_relevance']) else 'N/A'}")
        print(f"URL: {row['job_url']}")
        print("-" * 80)

if __name__ == "__main__":
    load_dotenv()
    
    # Add warning about running alongside main.py
    print("Note: This script can run while main.py is running, but may be slower due to database contention.")
    print("Press Ctrl+C to cancel, or Enter to continue...")
    input()
    
    # Get resume path from environment or input
    resume_path = os.getenv("NEW_RESUME_PATH")
    if not resume_path:
        resume_path = input("Enter path to new resume file: ")
    
    if not os.path.exists(resume_path):
        logging.error(f"Resume file not found: {resume_path}")
        sys.exit(1)
    
    # Get minimum relevance scores (optional)
    try:
        min_title_rel = float(os.getenv("MIN_TITLE_RELEVANCE", "5.0"))
        min_desc_rel = float(os.getenv("MIN_DESCRIPTION_RELEVANCE", "7.0"))
    except ValueError:
        logging.warning("Invalid relevance thresholds in environment, using defaults")
        min_title_rel = 5.0
        min_desc_rel = 7.0
    
    results = regenerate_relevance_scores(
        resume_path,
        min_title_relevance=min_title_rel,
        min_description_relevance=min_desc_rel
    )
    
    if isinstance(results, pd.DataFrame):
        print_results(results)
    else:
        print("\nNo jobs found matching relevance criteria.") 