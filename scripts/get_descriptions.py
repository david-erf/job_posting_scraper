import pandas as pd
import time
from datetime import date,datetime
today = date.today()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import requests
import pickle
from utils import get_jobs_for_keyword,convert_to_days,split_location,mapping,abbreviation_mapping,rank_job_posting,keywords,parse_job_id,job_exists,insert_new_jobs
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
import os
load_dotenv()  # Loads .env variables into environment


# get resume
resume_path = os.getenv("RESUME_PATH")  # Now accessible like an env var
with open(resume_path, "r") as file:
    resume = file.read()
    resume = resume + ", ".join(keywords)

# Get Job Cards for keywords 
results={}
ct=0
for k in keywords:
  ct+=1
  print(k,ct,len(keywords))
  results[k]=get_jobs_for_keyword(k, pages=10)

l=[]
for k in results.keys():
  df = pd.DataFrame(results[k])
  df['searched_keyword']= k
  l.append(df)
results_df = pd.concat(l,axis=0)
results_df.to_csv(f'../data/job_search_results_{timestamp}.csv')


# master_path = '../data/master_jobs.pkl'
# if not os.path.exists(master_path):
#   continue
# else:
#   with open('resume_path', "rb") as file:
#     master_jobs_dict = pickle.load(file)


# results_df=pd.read_csv(f'../data/job_search_results_{timestamp}.csv')
results_df['company'].fillna('', inplace=True)
results_df.drop_duplicates(subset=['title','company','searched_keyword','job_url'], inplace=True)
results_df['title_relevance'] = results_df['title'].apply(lambda x: rank_job_posting(resume, x))
results_df.to_csv(f'../data/job_search_title_relevance_{timestamp}.csv')

insert_new_jobs(f'../data/job_search_title_relevance_{timestamp}.csv')

  
# # to do: if not rerun
# results_df = pd.read_csv(f'../data/job_search_title_relevance_{timestamp}.csv')

# todo: decide if we want to increase relevance threshold
threshold = 1
bool_relevant = (results_df['title_relevance']>=threshold)
matched_results_df = results_df[bool_relevant].copy()
print('count of jobs',matched_results_df.shape)
# Loop through all job IDs and pull back html content
jobs = {}
ct=0
for job_id,url in list(zip(matched_results_df['job_id'],matched_results_df['job_url'])):
  jobs[job_id] = parse_job_id(job_id,url,resume)
  time.sleep(2)


# with open(f'../data/job_descriptions_raw_{timestamp}.pkl', "rb") as file:
#   jobs_raw = pickle.load(file)

# jobs=jobs_raw.copy()



# Save dictionary as a pickle file
with open(f'../data/job_descriptions_{timestamp}.pkl', "wb") as file:
    pickle.dump(jobs, file)


jobs_df = pd.DataFrame([
    {   'id':id,
        'date_posted': j.get('date_posted'),
        'num_applicants': j.get('num_applicants'),
        'job_description': j.get('job_description'),
        'Seniority': j.get('Seniority level'),
        'Employment': j.get('Employment type'),
        'Job': j.get('Job function'),
        'Industries': j.get('Industries'),
        'title_content': j.get('title_content'),
        'location': j.get('location'),
        'salary_frequency': j.get('salary_frequency'),
        'min_salary': j.get('min_salary'),
        'max_salary': j.get('max_salary'),
        'salary_range': j.get('salary_range'),
        'job_description_relevance':j.get('job_description_relevance'),
    }
    for id, j in jobs.items()
])

# Fix Location Field
jobs_df['location'] = jobs_df['location'].fillna('')
jobs_df['clean_location'] = jobs_df['location'].apply(lambda x: split_location(x, mapping))
jobs_df['city'] = jobs_df['clean_location'].apply(lambda x: x[0])
jobs_df['state'] = jobs_df['clean_location'].apply(lambda x: x[1])

# Process Date Posted
jobs_df['time_since_posted'] = jobs_df['date_posted'].apply(lambda x: convert_to_days(x))

jobs_df.to_csv(f'../data/formatted_jobs_df_{timestamp}.csv')

# quality checks
# jobs_df.groupby(['state']).agg({'id':'count'}).sort_values('id',ascending=False).head(100)
# jobs_df['time_since_posted'].plot.hist(bins=10)


# filter to values w/o commas
# jobs_df[~(jobs_df['location'].str.contains(','))]['location'].value_counts()




