import os
import time
import pandas as pd
import pickle
from datetime import date, datetime
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
from db import insert_new_jobs  # Database functions are now in db.py

load_dotenv()  # Load environment variables

today = date.today()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Load resume and augment with keywords
resume_path = os.getenv("RESUME_PATH")
with open(resume_path, "r") as file:
    resume = file.read()
    resume += ", " + ", ".join(keywords)

# Collect job postings
results = {}
for idx, k in enumerate(keywords, start=1):
    print(f"Processing keyword: {k} ({idx}/{len(keywords)})")
    results[k] = get_jobs_for_keyword(k, pages=10)

# Process results into a DataFrame
dfs = []
for k, job_list in results.items():
    df = pd.DataFrame(job_list)
    df['searched_keyword'] = k
    dfs.append(df)
results_df = pd.concat(dfs, axis=0)
results_df.to_csv(f'../data/job_search_results_{timestamp}.csv')

# Clean and score job postings
results_df['company'].fillna('', inplace=True)
results_df.drop_duplicates(subset=['title', 'company', 'searched_keyword', 'job_url'], inplace=True)
results_df['title_relevance'] = results_df['title'].apply(lambda x: rank_job_posting(resume, x))
results_df.to_csv(f'../data/job_search_title_relevance_{timestamp}.csv')

# Insert new jobs into the database
insert_new_jobs(f'../data/job_search_title_relevance_{timestamp}.csv')

# Filter based on relevance threshold and extract job descriptions
threshold = 1
matched_results_df = results_df[results_df['title_relevance'] >= threshold].copy()
print('Count of relevant jobs:', matched_results_df.shape[0])

jobs = {}
for job_id, url in zip(matched_results_df['job_id'], matched_results_df['job_url']):
    jobs[job_id] = parse_job_id(job_id, url, resume)
    time.sleep(2)

with open(f'../data/job_descriptions_{timestamp}.pkl', "wb") as file:
    pickle.dump(jobs, file)

# Process job descriptions into a DataFrame and perform location cleaning
jobs_df = pd.DataFrame([
    {
        'id': id,
        'date_posted': job.get('date_posted'),
        'num_applicants': job.get('num_applicants'),
        'job_description': job.get('job_description'),
        'Seniority': job.get('Seniority level'),
        'Employment': job.get('Employment type'),
        'Job': job.get('Job function'),
        'Industries': job.get('Industries'),
        'title_content': job.get('title_content'),
        'location': job.get('location'),
        'salary_frequency': job.get('salary_frequency'),
        'min_salary': job.get('min_salary'),
        'max_salary': job.get('max_salary'),
        'salary_range': job.get('salary_range'),
        'job_description_relevance': job.get('job_description_relevance'),
    }
    for id, job in jobs.items()
])
jobs_df['location'] = jobs_df['location'].fillna('')
jobs_df['clean_location'] = jobs_df['location'].apply(lambda x: split_location(x, mapping))
jobs_df['city'] = jobs_df['clean_location'].apply(lambda x: x[0])
jobs_df['state'] = jobs_df['clean_location'].apply(lambda x: x[1])
jobs_df['time_since_posted'] = jobs_df['date_posted'].apply(lambda x: convert_to_days(x))
jobs_df.to_csv(f'../data/formatted_jobs_df_{timestamp}.csv')

if __name__ == '__main__':
    main()