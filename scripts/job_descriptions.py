import pandas as pd
import time
from datetime import date
today = date.today()
import requests
import pickle
from agent import rank_job_posting,resume
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
import os

def convert_to_days(date_posted):
    """
    Convert a human-readable time string (e.g., '1 week ago', '2 months ago')
    into a float representing the number of days.
    """
    time_mapping = {
        "minute": 1 / 1440,  # 1 minute = 1/1440 days
        "hour": 1 / 24,      # 1 hour = 1/24 days
        "day": 1,            # 1 day = 1 day
        "week": 7,           # 1 week = 7 days
        "month": 30.4,       # 1 month = 30.4 days (average)
        "year": 365          # 1 year = 365 days
    }

    # Split the string into components
    if date_posted:
      parts = date_posted.split()

      if len(parts) < 2:
          return None  # Handle unexpected formats

      try:
          # Extract the number and time unit
          number = float(parts[0])
          unit = parts[1].rstrip('s')  # Remove plural (e.g., 'weeks' -> 'week')

          # Convert to days
          return number * time_mapping.get(unit, 1)
      except (ValueError, KeyError):
          return None  # Handle invalid cases gracefully

def split_location(location, mapping):
    """
    Splits a location string into city and state.

    Parameters:
        location (str): The location string (e.g., "Chicago, IL" or "Greater San Francisco Area").
        mapping (dict): A dictionary mapping special cases to (city, state) tuples.

    Returns:
        tuple: A tuple (city, state) with the split values.
    """
    if location in mapping:
        return mapping[location]

    # Split by comma for standard cases like "City, State"
    if "," in location:
        parts = location.split(",")
        city = parts[0].strip()
        state = parts[1].strip()

        # Handle cases like "New York, United States"
        if state.lower() == "united states":
            state = abbreviation_mapping.get(city, "Unknown")  # The first part is actually the state in this subcase
            city="Unknown"
        return city, state

    # Handle edge cases where location doesn't fit the standard pattern
    return "Unknown", "Unknown"

# Mappings to standardize location data for job postings
mapping = {
    'San Francisco Bay Area': ("San Francisco", "CA"),
    'Greater Chicago Area':("Chicago","IL"),
    'New York City Metropolitan Area': ("New York City", "NY"),
    'Greater Wilmington Area': ("Wilmington", "DE"),
    'Greater Sioux Falls Area': ("Sioux Falls", "SD"),
    'Raleigh-Durham-Chapel Hill Area': ("Raleigh", "NC"),
    'Buffalo-Niagara Falls Area': ("Buffalo", "NY"),
    'Greater Hartford': ("Hartford", "CT"),
    'Greater Boston': ("Boston", "MA"),
    'Greater Houston': ("Houston", "TX"),
    'Greater Reno Area': ("Reno", "NV"),
    'Greater Scranton Area': ("Scranton", "PA"),
    'Louisville Metropolitan Area': ("Louisville", "KY"),
    'Kansas City Metropolitan Area': ("Kansas City", "MO"),
    'Cincinnati Metropolitan Area': ("Cincinnati", "OH"),
    'Omaha Metropolitan Area': ("Omaha", "NE"),
    'Washington DC-Baltimore Area': ("Washington, DC", "DC"),
    'Atlanta Metropolitan Area': ("Atlanta", "GA"),
    'Greater Minneapolis-St. Paul Area': ("Minneapolis", "MN"),
    'Los Angeles Metropolitan Area': ("Los Angeles", "CA"),
    'Miami-Fort Lauderdale Area': ("Miami", "FL"),
    'Utica-Rome Area': ("Utica", "NY"),
    'Greater Cleveland': ("Cleveland", "OH"),
    'Las Vegas Metropolitan Area': ("Las Vegas", "NV"),
    'Albany, New York Metropolitan Area':("Albany","NY"),
    'Columbus, Ohio Metropolitan Area':("Columbus","OH"),
}

# Additional Mappings to standardize location data for job postings

abbreviation_mapping = {
    "Wisconsin": "WI",
    "Virginia": "VA",
    "New Jersey": "NJ",
    "Illinois": "IL",
    "Hawaii": "HI",
    "California": "CA",
    "Washington": "WA",
    "Texas": "TX",
    "Florida": "FL",
    "Arizona": "AZ",
    "Colorado": "CO",
    "Missouri": "MO",
    "New York": "NY",

}

load_dotenv()  # Loads .env variables into environment
resume_path = os.getenv("RESUME_PATH")  # Now accessible like an env var

with open(resume_path, "r") as file:
    resume = file.read()


results_df=pd.read_csv('../data/job_search_results_2025-02-06.csv')

results_df['company'].fillna('', inplace=True)

# print(results_df.shape)
# print(results_df.shape)
# print(results_df[results_df.duplicated(subset=['title','company','searched_company','job_url'], keep=False)].shape)
# print(results_df[results_df.duplicated(subset=['job_url'], keep=False)].shape)

# check uniqueness
results_df.drop_duplicates(subset=['title','company','searched_keyword','job_url'], inplace=True)
# print(results_df.shape)

results_df['title_relevance'] = results_df['title'].apply(lambda x: rank_job_posting(resume, x))
results_df.to_csv(f'../data/job_search_title_relevance_{today}.csv')

# to do: if not rerun
results_df = pd.read_csv(f'../data/job_search_title_relevance_{today}.csv')

matched_results_df = results_df[results_df['title_relevance']>=4].copy()
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
    response = requests.get(url, timeout=5)
  except:
    response = 'failed to pull'
  jobs_raw[job_id] = {
        'raw_response':response
        }



with open(f'../data/job_descriptions_raw_{today}.pkl', "rb") as file:
    jobs = pickle.load(file)

jobs=jobs_raw.copy()

# Regular expression to find dollar amounts
dollar_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?'

# Parse Soup for each Job
ct=0
for k in list(jobs.keys())[860:]:
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


# quality checks
# jobs_df.groupby(['state']).agg({'id':'count'}).sort_values('id',ascending=False).head(100)
# jobs_df['time_since_posted'].plot.hist(bins=10)


# filter to values w/o commas
# jobs_df[~(jobs_df['location'].str.contains(','))]['location'].value_counts()




