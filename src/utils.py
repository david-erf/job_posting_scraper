"""
Utility functions for the Job Posting Scraper & AI Relevance Analyzer.

This module includes functions to:
- Scrape job postings from LinkedIn.
- Interface with the Ollama CLI to call an LLM (Mistral) for relevance scoring and cover letter generation.
- Process and clean scraped data.
- Compare job files and handle database insertion.
"""

import time
import requests
import re
import subprocess
import sqlite3
import logging
from datetime import date, datetime
from urllib.parse import quote
from typing import List, Dict, Tuple, Any, Set, Optional

import pandas as pd
from bs4 import BeautifulSoup

# Set logging level for requests' underlying libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Define keywords for job search
keywords: List[str] = [
    'data science', 'data analytics', 'fintech', 'AI intern',
    "ML Intern", "Salesforce", 'Salesforce AI', 'business intelligence'
]

def get_jobs_for_keyword(keyword: str, pages: int = 10, pause: int = 2) -> List[Dict[str, Any]]:
    """
    Scrape job postings for a given keyword from LinkedIn.

    Args:
        keyword (str): The job title or keyword to search for.
        pages (int): Number of pages to scrape.
        pause (int): Seconds to pause between page requests.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing details of a job posting.
    """
    jobs: List[Dict[str, Any]] = []
    base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    geoid = 102095887  # California

    for page in range(pages):
        logging.info(f"Fetching page {page + 1} for keyword '{keyword}'")
        url = f"{base_url}?geoId={geoid}&keywords={quote(keyword)}&start={25 * page}"
        try:
            response = requests.get(url, timeout=5)
        except requests.RequestException as e:
            logging.error(f"Request failed on page {page + 1}: {e}")
            continue

        if response.status_code != 200:
            logging.error(f"Failed to fetch page {page + 1}: HTTP {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        job_cards = soup.find_all('div', class_='base-search-card__info')

        for card in job_cards:
            title = card.find('h3').text.strip() if card.find('h3') else None
            company = card.find('a', class_='hidden-nested-link').text.strip() if card.find('a', class_='hidden-nested-link') else None
            # Extract date posted from the nearest <time> tag
            date_posted_elem = card.find_previous('time')
            date_posted = date_posted_elem.text.strip() if date_posted_elem else None

            # Extract location
            location_elem = card.find('span', class_='job-search-card__location')
            location = location_elem.text.strip() if location_elem else None

            # Extract hiring status (e.g., applicant count info)
            hiring_status_elem = card.find('span', class_='job-posting-benefits__text')
            hiring_status = hiring_status_elem.text.strip() if hiring_status_elem else None

            # Extract job_id and construct job URL
            if card.parent and card.parent.get('data-entity-urn'):
                job_id = card.parent.get('data-entity-urn').split(':')[-1]
                job_url = f"https://www.linkedin.com/jobs/view/{job_id}/"
            else:
                job_id, job_url = None, None

            jobs.append({
                'title': title,
                'company': company,
                'date_posted': date_posted,
                'location': location,
                'job_id': job_id,
                'job_url': job_url,
                'hiring_status': hiring_status,
                'created_at': datetime.utcnow().isoformat()  # Use UTC timestamp for consistency
            })

        time.sleep(pause)
    return jobs

def call_ollama_run(model: str, prompt: str) -> str:
    """
    Calls the Ollama CLI to run a specified model with the given prompt.

    Args:
        model (str): The name of the model to use (e.g., "mistral").
        prompt (str): The prompt to send to the model.

    Returns:
        str: The output from the model.

    Raises:
        RuntimeError: If the subprocess call fails.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        output = result.stdout.decode("utf-8").strip()
        logging.debug(f"Ollama response: {output}")
        return output
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode("utf-8")
        raise RuntimeError(f"Error calling Ollama: {error_message}")

def rank_job_posting(resume: str, job_posting: str, lim: int = 2000) -> int:
    """
    Uses the Ollama CLI with the Mistral model to score the relevance of a job posting
    to the provided resume on a scale from 1 (low relevance) to 10 (high relevance).

    The resume and job posting are truncated to a maximum of `lim` characters each
    to ensure prompt length constraints.

    Args:
        resume (str): The candidate's resume.
        job_posting (str): The job posting description or title.
        lim (int): Maximum number of characters to include from resume and posting.

    Returns:
        int: The relevance score, or 0 in case of error.
    """
    # Truncate inputs to avoid exceeding token limits
    truncated_resume = resume[:lim]
    truncated_posting = job_posting[:lim]

    prompt = (
        "You are a numeric AI job relevance scorer. For each input, output only a single number "
        "(from 1 to 10) on a single line, with no additional text, explanation, or formatting. "
        "The number represents the relevance of the job posting to the candidate's resume (10 means highly relevant, 1 means not relevant).\n\n"
        f"### Resume:\n{truncated_resume}\n\n"
        f"### Job Posting:\n{truncated_posting}\n"
    )
    # Ensure prompt does not exceed maximum allowed length (if applicable)
    prompt = prompt[:4000]
    try:
        response = call_ollama_run("mistral", prompt)
        score = float(response.strip())
    except Exception as e:
        logging.error(f"Error processing job posting: {job_posting[:50]}... Error: {e}")
        score = 0

    logging.info(f"Job Posting (truncated): {job_posting[:50]}... | Score: {score}")
    return int(score)

def write_cover_letter(resume: str, job_posting: str, lim: int = 2000) -> str:
    """
    Uses the Ollama CLI with the Mistral model to generate a cover letter tailored to the job posting,
    adapting the candidate's resume to the job requirements.

    Args:
        resume (str): The candidate's resume.
        job_posting (str): The job posting description or title.
        lim (int): Maximum number of characters to include from resume and posting.

    Returns:
        str: The generated cover letter.
    """
    truncated_resume = resume[:lim]
    truncated_posting = job_posting[:lim]

    prompt = (
        "You are an AI professional cover letter writer. For each input, return a cover letter that adapts "
        "the candidate's Resume to the Job Posting. Answer the question: why does David want to work here?\n\n"
        f"### Resume:\n{truncated_resume}\n\n"
        f"### Job Posting:\n{truncated_posting}\n"
    )
    prompt = prompt[:4000]
    try:
        response = call_ollama_run("mistral", prompt)
    except Exception as e:
        logging.error(f"Error generating cover letter for job posting: {job_posting[:50]}... Error: {e}")
        response = ""
    return response

def convert_to_days(date_posted: str) -> Optional[float]:
    """
    Converts a human-readable time string (e.g., '1 week ago', '2 months ago') into the equivalent number of days.

    Args:
        date_posted (str): The string indicating when the job was posted.

    Returns:
        Optional[float]: The number of days corresponding to the input string, or None if the format is unexpected.
    """
    time_mapping = {
        "minute": 1 / 1440,  # 1 minute = 1/1440 days
        "hour": 1 / 24,      # 1 hour = 1/24 days
        "day": 1,
        "week": 7,
        "month": 30.4,
        "year": 365
    }
    if not date_posted:
        return None

    parts = date_posted.split()
    if len(parts) < 2:
        return None

    try:
        number = float(parts[0])
        unit = parts[1].rstrip('s')  # Normalize unit (e.g., "weeks" -> "week")
        return number * time_mapping.get(unit, 1)
    except (ValueError, KeyError):
        return None

def split_location(location: str, mapping: Dict[str, Tuple[str, str]]) -> Tuple[str, str]:
    """
    Splits a location string into city and state using provided mapping for special cases.

    Args:
        location (str): Location string (e.g., "Chicago, IL" or "Greater San Francisco Area").
        mapping (Dict[str, Tuple[str, str]]): A dictionary mapping specific location strings to (city, state).

    Returns:
        Tuple[str, str]: A tuple (city, state) extracted from the location string.
    """
    if location in mapping:
        return mapping[location]

    if "," in location:
        parts = location.split(",")
        city = parts[0].strip()
        state = parts[1].strip()
        if state.lower() == "united states":
            state = abbreviation_mapping.get(city, "Unknown")
            city = "Unknown"
        return city, state

    return "Unknown", "Unknown"

# Mappings to standardize location data for job postings.
mapping: Dict[str, Tuple[str, str]] = {
    'San Francisco Bay Area': ("San Francisco", "CA"),
    'Greater Chicago Area': ("Chicago", "IL"),
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
    'Albany, New York Metropolitan Area': ("Albany", "NY"),
    'Columbus, Ohio Metropolitan Area': ("Columbus", "OH"),
}

# Additional mappings to standardize U.S. state abbreviations.
abbreviation_mapping: Dict[str, str] = {
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

def parse_job_id(job_id: str, url: str, resume: str, extract_html: bool = False) -> Dict[str, Any]:
    """
    Parses a job posting page to extract detailed job information and calculates a relevance score.

    Args:
        job_id (str): The job ID.
        url (str): URL of the job posting.
        resume (str): The candidate's resume for relevance scoring.
        extract_html (bool): Whether to keep the raw HTML response for storage. Default is False.

    Returns:
        Dict[str, Any]: A dictionary containing job details and metadata.
    """
    ans: Dict[str, Any] = {}
    dollar_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?'

    try:
        response = requests.get(url)
    except Exception as e:
        logging.error(f"Failed to fetch job posting at {url}: {e}")
        response = None

    # Only keep the raw response if extract_html is True
    if extract_html:
        ans['raw_response'] = response
    
    if response:
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract date posted
        date_posted_tag = soup.find('span', class_='posted-time-ago__text')
        if date_posted_tag:
            ans['date_posted'] = date_posted_tag.get_text(strip=True)

        # Extract number of applicants
        num_applicants_tag = soup.find('figcaption', class_='num-applicants__caption')
        if num_applicants_tag:
            ans['num_applicants'] = num_applicants_tag.get_text(strip=True)

        # Extract job description and compute relevance score
        job_description_div = soup.find('div', class_='show-more-less-html__markup')
        if job_description_div:
            job_description = job_description_div.get_text(strip=True)
            ans['job_description'] = job_description
            relevance = rank_job_posting(resume, job_description)
            ans['job_description_relevance'] = relevance

        # Extract additional header information
        for item in soup.find_all('li', class_='description__job-criteria-item'):
            header_elem = item.find('h3', class_='description__job-criteria-subheader')
            value_elem = item.find('span', class_='description__job-criteria-text')
            if header_elem and value_elem:
                header = header_elem.get_text(strip=True)
                value = value_elem.get_text(strip=True)
                ans[header] = value

        # Extract title and location information from meta tag
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title_content = og_title['content']
            ans['title_content'] = title_content
            if "in " in title_content:
                location = title_content.split("in ")[-1].split("|")[0].strip()
                ans['location'] = location

        # Extract salary information
        salary_div = soup.find('div', class_='compensation__salary')
        if salary_div:
            salary = salary_div.get_text(strip=True)
            if "/yr" in salary:
                salary_frequency = 'annual'
            elif "/hr" in salary:
                salary_frequency = 'hourly'
            elif "/mo" in salary:
                salary_frequency = 'monthly'
            else:
                salary_frequency = 'unknown'
            ans['salary_frequency'] = salary_frequency
            try:
                min_salary_str, max_salary_str = salary.replace("$", "").replace("/yr", "").replace("/hr", "").replace("/mo", "").split(" - ")
                min_salary = float(min_salary_str.replace(",", ""))
                max_salary = float(max_salary_str.replace(",", ""))
                ans.update({
                    'min_salary': min_salary,
                    'max_salary': max_salary,
                    'salary_range': salary
                })
            except Exception as e:
                logging.warning(f"Salary parsing issue for job {job_id}: {e}")
        else:
            # Fallback: extract salary from job description using regex
            if 'job_description' in ans:
                dollar_amounts = re.findall(dollar_pattern, ans['job_description'])
                dollar_amounts = [float(s.replace("$", "").replace(",", "")) for s in dollar_amounts]
                if dollar_amounts:
                    max_salary = max(dollar_amounts)
                    if max_salary > 500000:
                        max_salary = None
                    safe_min = [s for s in dollar_amounts if s > (max_salary * 0.1) if max_salary]
                    min_salary = min(safe_min) if safe_min else None
                    ans.update({
                        'min_salary': min_salary,
                        'max_salary': max_salary
                    })
    return ans

def compare_job_files(old_file: str, new_file: str, id_column: str = "job_id") -> Tuple[Set[Any], Set[Any]]:
    """
    Compares two CSV files containing job postings and identifies new and dropped job IDs.

    Args:
        old_file (str): Path to the old CSV file.
        new_file (str): Path to the new CSV file.
        id_column (str): Column name representing the unique job identifier.

    Returns:
        Tuple[Set[Any], Set[Any]]: A tuple containing sets of new job IDs and dropped job IDs.
    """
    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(new_file)

    old_jobs = set(old_df[id_column])
    new_jobs = set(new_df[id_column])

    common_jobs = old_jobs & new_jobs
    new_only_jobs = new_jobs - old_jobs
    dropped_jobs = old_jobs - new_jobs

    summary = {
        "Total Jobs in Old File": len(old_jobs),
        "Total Jobs in New File": len(new_jobs),
        "Common Jobs": len(common_jobs),
        "New Jobs Since Last Run": len(new_only_jobs),
        "Dropped Jobs": len(dropped_jobs),
    }

    logging.info("=== Job Comparison Summary ===")
    for key, value in summary.items():
        logging.info(f"{key}: {value}")

    return new_only_jobs, dropped_jobs



