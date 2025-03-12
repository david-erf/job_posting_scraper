#!/usr/bin/env python3
"""
Simple examples for querying the jobs database in IPython.
Copy and paste these commands into your IPython session.
"""

# Import the database singleton instance
from src.db import db


# Complex query
query = '''
with all_jobs as(
SELECT 
    r.job_id,
    r.created_at,
    r.title,
    r.company,
    r.location,
    r.date_posted,
    r.hiring_status,
    f.description_text AS description,
    a.title_relevance,
    a.description_relevance,
    app.status AS application_status
FROM 
    raw_jobs r
LEFT JOIN 
    job_analysis a ON r.job_id = a.job_id
LEFT JOIN 
    formatted_descriptions f ON r.job_id = f.job_id
LEFT JOIN 
    job_applications app ON r.job_id = app.job_id
ORDER BY 
    r.created_at DESC
)

select 
sum(case when title_relevance is not null then 1 else 0 end)  as has_title_relevance,
sum(case when description_relevance is not null then 1 else 0 end)  as has_description_relevance,
sum(case when application_status is not null then 1 else 0 end)  as has_application_status,
count(*) as total_jobs
from all_jobs
    
'''


df = db.query_to_df(query)