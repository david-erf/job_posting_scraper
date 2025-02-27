#!/usr/bin/env python3
import sqlite3
import logging
import sys
import pandas as pd 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

DB_FILE = '../data/linkedin.db'


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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} <database_path> <query>".format(sys.argv[0]))
        sys.exit(1)
    
    db_path = sys.argv[1]
    query = sys.argv[2]
    query_database(db_path, query)
