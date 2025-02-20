import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import re
import pandas as pd
from datetime import date
today = date.today()
import subprocess

keywords = ['data science','data analytics','fintech','AI intern',"ML Intern","Salesforce",'Salesforce AI']


def get_jobs_for_keyword(keyword, pages=10,pause=2):
    """
    Scrape job postings for a specific keyword.

    Args:
        keyword (str): The job title or keyword to search for.
        pages (int): The number of pages to scrape.

    Returns:
        list: A list of dictionaries containing job details.
    """
    jobs = []
    base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"

    for page in range(pages):
        print('.  ', page)
        # Removed location from the URL
        url = f"{base_url}?keywords={quote(keyword)}&start={25 * page}"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            print(f"Failed to fetch page {page + 1}: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        job_cards = soup.find_all('div', class_='base-search-card__info')

        for card in job_cards:
            title = card.find('h3').text.strip() if card.find('h3') else None
            company = card.find('a', class_='hidden-nested-link').text.strip() if card.find('a', class_='hidden-nested-link') else None
            # Extract date posted
            date_posted_elem = card.find_previous('time')  # Looks for the closest preceding <time> tag
            date_posted = date_posted_elem.text.strip() if date_posted_elem else None

            # Extract location
            location_elem = card.find('span', class_='job-search-card__location')
            location = location_elem.text.strip() if location_elem else None

            # Extract info about applicant count
            hiring_status_elem = card.find('span', class_='job-posting-benefits__text')
            hiring_status = hiring_status_elem.text.strip() if hiring_status_elem else None

            # Removed the location extraction
            if card.parent:
              job_id = card.parent.get('data-entity-urn').split(':')[-1]
              job_url = f"https://www.linkedin.com/jobs/view/{job_id}/"
            else:
              job_id = None
              job_url = None

            jobs.append({
            'title': title,
            'company': company,
            'date_posted': date_posted, 
            'location': location,       
            'job_id': job_id,
            'job_url': job_url,
            'hiring_status':hiring_status,
            })

        time.sleep(pause)
    return jobs

def call_ollama_run(model: str, prompt: str) -> str:
    """
    Calls the Ollama CLI using the `run` command with a given prompt and returns the model's output.
    
    Parameters:
        model (str): The name of the model to use (e.g., "mistral").
        prompt (str): The prompt to send to the model.
        
    Returns:
        str: The output from the model.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode("utf-8")
        raise RuntimeError(f"Error calling Ollama: {error_message}")

def rank_job_posting(resume: str, job_posting: str,lim:int = 2000) -> int:
    """
    Uses the Ollama CLI (with the Mistral model) to rank a job posting's relevance
    to the given resume on a scale from 1 to 10.
    
    Parameters:
        resume (str): The candidate's resume.
        job_posting (str): The job posting description or title.
        
    Returns:
        int: The relevance score.
    """
    resume=resume[0:lim]
    job_posting=job_posting[0:lim]
    

    prompt = f"""You are a numeric AI job relevance scorer. For each input, output only a single number (from 1 to 10) on a single line, with no additional text, explanation, or formatting. The number represents the relevance of the job posting to the candidate's resume (10 means highly relevant, 1 means not relevant).

    ### Resume:
    {resume}

    ### Job Posting:
    {job_posting}

    """
    prompt=prompt[0:4000]
    try:
        response = call_ollama_run("mistral", prompt)
        # Parse the response into an integer score
        score = float(response.strip())
    except Exception as e:
        print(f"Error processing job posting: {job_posting}\nError: {e}")
        score = 0

    print(job_posting[:50], score)
    return score

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

def parse_job_id(job_id,url,resume):

    ans = {}
    # Regular expression to find dollar amounts
    dollar_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?'

    try:
        response = requests.get(url, timeout=20)
    except:
        response = 'failed to pull'

    ans.update({'raw_response':response})
    soup = BeautifulSoup(response.content, "html.parser")
    # Get date posted
    date_posted_tag = soup.find('span', class_='posted-time-ago__text')
    if date_posted_tag:
      ans.update({'date_posted':date_posted_tag.get_text(strip=True)})

    # Get number of applicants
    num_applicants_tag = soup.find('figcaption', class_='num-applicants__caption')
    if num_applicants_tag:
      ans.update({'num_applicants':num_applicants_tag.get_text(strip=True)})

    # Find the job description information
    job_description_div = soup.find('div', class_='show-more-less-html__markup')
    if job_description_div:
        job_description = job_description_div.text.strip()
        ans.update({'job_description': job_description})
        
        # Get Relevance
        relevance= rank_job_posting(resume, job_description)
        ans.update({'job_description_relevance':relevance})


    # Get Header information
    for item in soup.find_all('li', class_='description__job-criteria-item'):
        header = item.find('h3', class_='description__job-criteria-subheader').get_text(strip=True)
        value = item.find('span', class_='description__job-criteria-text').get_text(strip=True)
        ans.update({header: value})

    # Get Location Information
    og_title = soup.find('meta', property='og:title')
    if og_title:
        title_content = og_title['content']
        ans.update({'title_content': title_content})
        # Extract the location from the title content
        if "in " in title_content:
            location = title_content.split("in ")[-1].split("|")[0].strip()
            ans.update({'location': location})

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
        ans.update({'salary_frequency':salary_frequency})
        min_salary, max_salary = salary.replace("$", "").replace("/yr", "").replace('/hr',"").replace("/mo","").split(" - ")
        min_salary = float(min_salary.replace(",", ""))
        max_salary = float(max_salary.replace(",", ""))
        ans.update({'min_salary':min_salary,
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
          ans.update({'min_salary':min_salary,
                          'max_salary':max_salary})
    # except:
    #     print(f'failed on {job_id}')
    return ans

def write_cover_letter(resume: str, job_posting: str,lim:int = 2000) -> int:
    """
    Uses the Ollama CLI (with the Mistral model) to write a cover letter
    
    Parameters:
        resume (str): The candidate's resume.
        job_posting (str): The job posting description or title.
        
    Returns:
        int: The relevance score.
    """
    resume=resume[0:lim]
    job_posting=job_posting[0:lim]
    
    prompt = f"""You are an AI professional cover letter writer. For each input, return a cover letter that adapts my Resume to the Job Posting. Answer the question: why does David want to work here?

    ### Resume:
    {resume}

    ### Job Posting:
    {job_posting}

    """
    prompt=prompt[0:4000]
    try:
        response = call_ollama_run("mistral", prompt)
    except Exception as e:
        print(f"Error processing job posting: {job_posting}\nError: {e}")

    return response
