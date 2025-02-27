import sqlite3

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