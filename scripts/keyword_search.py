import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import pandas as pd
from datetime import date
today = date.today()

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
                'job_id':job_id,
                'job_url': job_url,
            })

        time.sleep(pause)


    return jobs

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

keywords = ['data science','SoFi','PayPal','Plaid','Salesforce','Affirm','Chime','Venmo','Upstart','Stripe','Wealthfront','Argyle','Square','Square','data science','data analyst','data manager','analyst','data engineer','fintech startup','credit analytics','product analytics','analytics']
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