import pandas as pd
import time
from datetime import date
today = date.today()
import requests
import pickle
from utils import get_jobs_for_keyword,convert_to_days,split_location,mapping,abbreviation_mapping,rank_job_posting,keywords

from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
import os
load_dotenv()  # Loads .env variables into environment


# get resume
resume_path = os.getenv("RESUME_PATH")  # Now accessible like an env var
with open(resume_path, "r") as file:
    resume = file.read()

# Get JOb Descriptions for keywords 
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
results_df.to_csv(f'../data/job_search_results_{today}.csv')


# master_path = '../data/master_jobs.pkl'
# if not os.path.exists(master_path):
#   continue
# else:
#   with open('resume_path', "rb") as file:
#     master_jobs_dict = pickle.load(file)


# results_df=pd.read_csv(f'../data/job_search_results_{today}.csv')
results_df['company'].fillna('', inplace=True)
results_df.drop_duplicates(subset=['title','company','searched_keyword','job_url'], inplace=True)
results_df['title_relevance'] = results_df['title'].apply(lambda x: rank_job_posting(resume, x))
results_df.to_csv(f'../data/job_search_title_relevance_{today}.csv')

  
# to do: if not rerun
# results_df = pd.read_csv(f'../data/job_search_title_relevance_{today}.csv')

# todo: decide if we want to increase relevance threshold
threshold = 1
bool_relevant = (results_df['title_relevance']>=threshold)
matched_results_df = results_df[bool_relevant].copy()
print('count of jobs',matched_results_df.shape)
# Loop through all job IDs and pull back html content
jobs_raw = {}
ct=0
for job_id,url in list(zip(matched_results_df['job_id'],matched_results_df['job_url'])):
  print(url)
  ct+=1 
  print(ct)
  time.sleep(2)
  try:
    response = requests.get(url, timeout=20)
  except:
    response = 'failed to pull'
  jobs_raw[job_id] = {
        'raw_response':response
        }

# with open(f'../data/job_descriptions_raw_{today}.pkl', "rb") as file:
#     jobs_raw = pickle.load(file)

jobs=jobs_raw.copy()

# Regular expression to find dollar amounts
dollar_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?'

# Parse Soup for each Job
ct=0
for k in list(jobs.keys()):
  ct+=1
  print(ct)
  criteria = {}
  try:
    soup = BeautifulSoup(jobs[k]['raw_response'].content, "html.parser")
    # Get date posted
    date_posted_tag = soup.find('span', class_='posted-time-ago__text')
    if date_posted_tag:
      jobs[k].update({'date_posted':date_posted_tag.get_text(strip=True)})

    # Get number of applicants
    num_applicants_tag = soup.find('figcaption', class_='num-applicants__caption')
    if num_applicants_tag:
      jobs[k].update({'num_applicants':num_applicants_tag.get_text(strip=True)})

    # Find the job description information
    job_description_div = soup.find('div', class_='show-more-less-html__markup')
    if job_description_div:
        job_description = job_description_div.text.strip()
        jobs[k].update({'job_description': job_description})
        
        # Get Relevance
        relevance= rank_job_posting(resume, job_description)
        jobs[k].update({'job_description_relevance':relevance})


    # Get Header information
    for item in soup.find_all('li', class_='description__job-criteria-item'):
        header = item.find('h3', class_='description__job-criteria-subheader').get_text(strip=True)
        value = item.find('span', class_='description__job-criteria-text').get_text(strip=True)
        criteria[header] = value
        jobs[k].update(criteria)
        # print( criteria)

    # Get Location Information
    og_title = soup.find('meta', property='og:title')
    if og_title:
        title_content = og_title['content']
        jobs[k].update({'title_content': title_content})
        # Extract the location from the title content
        if "in " in title_content:
            location = title_content.split("in ")[-1].split("|")[0].strip()
            jobs[k].update({'location': location})

    # Get Salary information
    salary_div = soup.find('div', class_='compensation__salary')
    if salary_div:
        salary = salary_div.text.strip()
        if "/yr" in salary:
          salary_frequency='annual'
        elif "/hr" in salary:
          salary_frequency='hourly'
        elif "/mo" in salary:
          salary_frequency='monthly'
        else:
          salary_frequency='unknown'
        jobs[k].update({'salary_frequency':salary_frequency})
        min_salary, max_salary = salary.replace("$", "").replace("/yr", "").replace('/hr',"").replace("/mo","").split(" - ")
        min_salary = float(min_salary.replace(",", ""))
        max_salary = float(max_salary.replace(",", ""))
        jobs[k].update({'min_salary':min_salary,
                        'max_salary':max_salary,
                        'salary_range':salary})
    # If salary isn't in the expected field, check for dollar values in the job description
    else:
        # if the field isn't explicit,
        dollar_amounts = re.findall(dollar_pattern, job_description)
        dollar_amounts = [float(s.replace("$", "").replace(",", "")) for s in dollar_amounts]
        # todo
        if len(dollar_amounts)>0:
          max_salary=max(dollar_amounts) # should be safe
          min_salary=min([s for s in dollar_amounts if s>max_salary*.1]) # but handle cases where annualized income and hourly are intermingled (prefer annualized)

          # handle likely data integrity issues
          # todo: replace with configurable variable
          if min_salary>500000:
            min_salary=None
          if max_salary>500000:
            max_salary=None
          jobs[k].update({'min_salary':min_salary,
                          'max_salary':max_salary})
  except:
    print(f'failed on {k}')


# Save dictionary as a pickle file
with open(f'../data/job_descriptions_{today}.pkl', "wb") as file:
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

jobs_df.to_csv(f'../data/formatted_jobs_df_{today}.csv')

# quality checks
# jobs_df.groupby(['state']).agg({'id':'count'}).sort_values('id',ascending=False).head(100)
# jobs_df['time_since_posted'].plot.hist(bins=10)


# filter to values w/o commas
# jobs_df[~(jobs_df['location'].str.contains(','))]['location'].value_counts()




