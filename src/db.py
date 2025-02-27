#!/usr/bin/env python3
import sqlite3
import logging
import sys
import pandas as pd 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

DB_FILE = '../data/linkedin.db'

conn = sqlite3.connect("../data/linkedin.db")
cursor = conn.cursor()
# Create table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        created_at TIMESTAMP, 
        title TEXT,
        company TEXT,
        location TEXT,
        description TEXT,
        date_posted TEXT,
        hiring_status TEXT,
        searched_keyword TEXT,
        title_relevance TEXT
    )
""")
conn.commit()
conn.close()


q= ''' 
select date(created_at),
       location,
       case when hiring_status = 'Be an early applicant' then 1 else 0 end as be_early,
       count(*)
 from jobs 
 where created_at is not null 
 group by 1,2,3
 order by 1 desc, 4 desc
'''

query_database_to_df(DB_FILE,q)



#  find recently added, saturated jobs; what's going on?
q= ''' 
select date(created_at),
       location,
       case when hiring_status = 'Be an early applicant' then 1 else 0 end as be_early,
       count(*)
 from jobs 
 where created_at is not null and hiring_status != 'Be an early applicant'
 group by 1,2,3
 order by 1 desc, 4 desc
'''



def query_database_to_df(db_path, query):
    try:
        # Connect to the SQLite database with a timeout of 10 seconds
        conn = sqlite3.connect(db_path, timeout=10)
        logging.info("Connected to database: %s", db_path)
        
        # Execute the query and load results into a DataFrame
        df = pd.read_sql_query(query, conn)
        logging.info("Query executed successfully. Number of rows returned: %d", len(df))
        
        # Close the connection
        conn.close()
        logging.info("Database connection closed.")
        return df
    except Exception as e:
        logging.error("An error occurred: %s", e)
        sys.exit(1)