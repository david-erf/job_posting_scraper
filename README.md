# Job Posting Scraper & AI Relevance Analyzer

A modular pipeline for automating job searches by scraping postings from LinkedIn, extracting full job descriptions, and leveraging an LLM (via ollama/mistral) to assess relevance based on a candidate’s resume. The system streamlines data collection and filtering, making it easier to identify high-priority opportunities.
## Overview

This project automates the process of:
1. Scraping job postings from LinkedIn.
2. Extracting full job descriptions.
3. Using an LLM to analyze and score job relevance based on predefined criteria.

By leveraging a candidate’s resume (augmented with target keywords), the system classifies and prioritizes job postings, making it easier to filter for roles that match your background.

## How It Works

1. **Web Scraping:**  
   Collects job postings from LinkedIn using libraries such as BeautifulSoup (or Selenium/Scrapy if needed). The scraper retrieves data for multiple keywords and handles pagination and duplicates.

2. **Job Description Extraction:**  
   For each posting, detailed job descriptions are extracted and saved for further analysis.

3. **LLM-Based Analysis:**  
   The extracted job titles and the candidate’s resume are passed to an LLM (ollama/mistral) to generate structured relevance scores. This scoring filters and prioritizes postings based on how well they match the candidate’s background.

## Tech Stack

- **Python:** Core programming language
- **BeautifulSoup / Selenium / Scrapy:** For web scraping
- **LLM (ollama/mistral):** For text classification and relevance scoring
- **Pandas & NumPy:** For data handling and analysis
- **SQLite:** For persistent data storage
- **Streamlit (Optional):** For an interactive dashboard

## Why This Is Useful

- **Automation:**  
  Streamlines the job search process by automatically filtering and scoring relevant job postings.
  
- **Structured Analysis:**  
  Provides clean, structured output (in CSVs, pickle files, and a SQLite database) that can be easily analyzed or visualized.

- **Extendable:**  
  The modular design allows for future enhancements such as user-specific ranking, additional AI-based filtering, and interactive data exploration via a dashboard.

## Project Structure

- **data/**: Contains generated output files such as CSVs, pickle files, and the SQLite database.
- **notebooks/**: Includes exploratory notebooks and reports for deeper analysis.
- **src/**: Houses the core application code:
  - **main.py:** The primary entry point that orchestrates scraping, processing, and database updates.
  - **utils.py:** Contains functions for web scraping, LLM relevance scoring, and data cleaning.
  - **db.py:** (Optional) A dedicated module for handling SQLite database operations.
- **streamlit_app/**: Contains code for an interactive Streamlit dashboard to visualize job data.

## Usage

### Setup Environment

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/project_name.git
   cd project_name
   ```
2. **Configure Environment Variables:**

Copy the example environment file and update the variables:
```bash
cp .env.example .env
```

Ensure that RESUME_PATH in the .env file points to your resume text file.

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
### Run the Pipeline
Execute the main application from the project root:
```bash
python src/main.py
```
Check the data/ folder for generated output files (e.g., CSVs, pickle files, and the SQLite database).

## Technical Details
### LLM Relevance Scorer
- **Context**:
The candidate’s resume (augmented with target keywords) is used as context for analysis.
- **Processing**:
Job titles are processed by the LLM (via ollama/mistral) to generate structured relevance scores.
- **Purpose**:
The relevance scores help filter and prioritize job postings, ensuring that only the most pertinent roles are highlighted.

### Web Scraping & Data Processing
- **Scraping**:
Retrieves job postings based on predefined keywords and handles pagination.
- **Cleaning**:
Utilizes Pandas to remove duplicates and clean the data.
- **Output**:
Saves processed data into CSV and pickle files for further analysis.

### Database Integration
- **SQLite**:
New job postings are inserted into a SQLite database, ensuring persistent storage.
- **Workflow**:
The system checks for duplicates before updating the database, maintaining data consistency.
Considerations
- **Legal & Ethical**:
Be mindful of LinkedIn's terms of service when scraping data. This project is intended for personal use and demonstration purposes, not for large-scale data extraction.
- **Performance**:
The LLM component is resource-intensive. Ensure you have the necessary API access and compute resources.

### Future Improvements
- **Enhanced Error Handling & Logging**:
Replace print statements with Python’s logging module for robust, production-level error handling.
- **Modular Database Operations**:
Further separate database functions into a dedicated module (db.py).
- **User-Specific Enhancements**:
Implement job ranking based on user preferences and integrate resume-matching features.
- **Optimized LLM Performance**:
Fine-tune the LLM configuration for improved classification accuracy.
- **Extended Dashboard Features**:
Enhance a Streamlit app with richer interactive data exploration and visual analytics.


This project offers an automated approach to job searching, integrating web scraping, data processing, and AI-driven analysis into a streamlined pipeline. Future enhancements will continue refining its functionality and adaptability.
