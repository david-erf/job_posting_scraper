import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import re
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


keywords = ['data science']
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

results_df.to_csv(f'./data/job_search_results_{today}.csv')