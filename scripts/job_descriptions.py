import pandas as pd
import time
from datetime import date
today = date.today()
import requests
import pickle
from agent import rank_job_posting,resume

results_df=pd.read_csv('../data/job_search_results_2025-02-05.csv')

results_df['company'].fillna('', inplace=True)

results_df['exact_match'] = [
    company.lower().__contains__(search.lower())
    for company, search in zip(results_df['company'], results_df['searched_company'])
]

matched_results_df = results_df[results_df['exact_match']].copy()

print(results_df.shape)
print(matched_results_df.shape)

print(matched_results_df[matched_results_df.duplicated(subset=['title','company','searched_company','job_url'], keep=False)].shape)
print(matched_results_df[matched_results_df.duplicated(subset=['job_url'], keep=False)].shape)

# check uniqueness
matched_results_df.drop_duplicates(subset=['title','company','searched_company','job_url'], inplace=True)
print(matched_results_df.shape)

matched_results_df['title_relevance'] = matched_results_df['title'].apply(lambda x: rank_job_posting(resume, x))

matched_results_df.to_csv(f'../data/job_search_title_relevance_{today}')

matched_results_df = matched_results_df[matched_results_df['title_relevance']>=4].copy()
print('count of jobs',matched_results_df.shape)

# Loop through all job IDs and pull back html content
jobs = {}
ct=0
for job_id,url in zip(matched_results_df['job_id'],matched_results_df['job_url']):
  print(url)
  ct+=1
  print(ct)
  time.sleep(2)
  response = requests.get(url, timeout=5)
  jobs[job_id] = {
      'raw_response':response
      }

# Regular expression to find dollar amounts
dollar_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?'

# Parse Soup for each Job
ct=0
for k in list(jobs.keys()):
  ct+=1
  print(ct)
  clear_output(wait=True)  # Clear the previous output
  criteria = {}
  soup = BeautifulSoup(jobs[k]['raw_response'].content
    , "html.parser")

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
      print( criteria)

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



# Save dictionary as a pickle file
with open(f'../data/job_descriptions_{today}.pkl', "wb") as file:
    pickle.dump(jobs, file)



