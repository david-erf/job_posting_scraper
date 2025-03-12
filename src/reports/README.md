# LinkedIn Job Search Reports

This directory contains reporting tools for analyzing job data collected from LinkedIn. These scripts help you extract insights from your job search data to make better career decisions.

## Available Reports

### Recent Jobs Report (`recent_jobs.py`)

Query and display recent job postings based on various criteria like posting date, application status, and relevance scores.

**Features:**
- Filter by date posted, relevance score, location, company, and more
- Focus on early application opportunities
- Save results to CSV for further analysis

**Example usage:**
```bash
# Display recent high-relevance jobs
python src/reports/recent_jobs.py --min-relevance 7 --days 5

# Find remote jobs at specific companies
python src/reports/recent_jobs.py --location "remote" --company "google"

# Early application opportunities only
python src/reports/recent_jobs.py --early-only --verbose

# Save results to CSV
python src/reports/recent_jobs.py --save
```

### Skill Analysis Report (`skill_analysis.py`)

Analyze job descriptions to identify the most requested skills for specific job titles or keywords.

**Features:**
- Count occurrences of skills by category (programming languages, databases, cloud technologies, etc.)
- Filter by job title, company, or search keyword
- Visualize results with bar charts
- Export data to CSV for further analysis

**Example usage:**
```bash
# Analyze skills for data scientist positions
python src/reports/skill_analysis.py --title "data scientist"

# See what skills top companies are looking for
python src/reports/skill_analysis.py --company "amazon" --plot

# Compare skills for different search keywords
python src/reports/skill_analysis.py --keyword "machine learning" --save
```

### Location Analysis Report (`location_analysis.py`)

Analyze job locations to identify hiring trends by geography, remote work opportunities, and location-based patterns.

**Features:**
- Identify top hiring cities, states, and countries
- Analyze remote work trends and hybrid opportunities
- Filter by job titles, keywords, or timeframes
- Visualize geographic distribution of jobs

**Example usage:**
```bash
# See remote work trends in recent jobs
python src/reports/location_analysis.py --days 30 --plot

# Analyze locations for specific job types
python src/reports/location_analysis.py --title "data engineer" --save
```

### Company Analysis Report (`company_analysis.py`)

Analyze companies that are hiring to identify trends, the most active employers, and patterns in job title variations.

**Features:**
- Identify companies with the most open positions
- Detect which companies offer the most remote work
- Analyze hiring timelines to spot hiring surges
- Standardize company names and detect duplicates
- Export detailed company hiring data

**Example usage:**
```bash
# See which companies are hiring the most
python src/reports/company_analysis.py --min-relevance 7 --top 30

# Find companies with remote opportunities
python src/reports/company_analysis.py --top 50 --plot

# Detect duplicate company entries
python src/reports/company_analysis.py --detect-duplicates --save
```

## Adding New Reports

When adding new reports to this directory:

1. Follow the existing pattern of command-line arguments and filtering options
2. Use type hints for better code readability
3. Implement both display and export functionality
4. Add documentation for your report in this README

## Common Functionality

All reports share similar command-line interfaces with these common options:

- Filtering options (title, company, timeframe, relevance score)
- Display options (verbose output, visualization)
- Export options (save to CSV)

The reports use the SQLite database created by the main job search tool. 