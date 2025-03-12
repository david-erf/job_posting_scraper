#!/usr/bin/env python3
import sqlite3
import logging
import sys
import pandas as pd
from typing import List, Dict, Tuple, Any, Set, Optional
from datetime import datetime
import os
import threading
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ConnectionPool:
    """A simple connection pool for SQLite connections"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, max_connections=5, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConnectionPool, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, max_connections=5, db_path=None):
        if self._initialized:
            return
            
        self.db_path = db_path or os.getenv("DB_PATH", "../data/jobs.db")
        self.max_connections = max_connections
        self.connections = []
        self.in_use = {}
        self.lock = threading.Lock()
        self._initialized = True
        logger.debug(f"Initialized connection pool for {self.db_path} with max {max_connections} connections")
    
    def get_connection(self):
        """Get a connection from the pool or create a new one if needed"""
        with self.lock:
            # First try to find an existing unused connection
            for conn in self.connections:
                if conn not in self.in_use or not self.in_use[conn]:
                    self.in_use[conn] = True
                    return conn
                    
            # If we have room, create a new connection
            if len(self.connections) < self.max_connections:
                conn = sqlite3.connect(self.db_path, timeout=30)
                conn.row_factory = sqlite3.Row
                self.connections.append(conn)
                self.in_use[conn] = True
                return conn
                
            # If we're at max connections, wait for one to become available
            logger.warning("Max connections reached, waiting for a connection to become available")
            while True:
                for conn in self.connections:
                    if not self.in_use[conn]:
                        self.in_use[conn] = True
                        return conn
    
    def release_connection(self, conn):
        """Return a connection to the pool"""
        with self.lock:
            if conn in self.in_use:
                self.in_use[conn] = False
    
    def close_all(self):
        """Close all connections in the pool"""
        with self.lock:
            for conn in self.connections:
                conn.close()
            self.connections = []
            self.in_use = {}

class DatabaseManager:
    """
    A singleton class that manages all database operations.
    Follows the repository pattern to isolate database access.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, db_path=None):
        if self._initialized:
            return
            
        self.db_path = db_path or os.getenv("DB_PATH", "../data/jobs.db")
        logger.info(f"Initializing DatabaseManager with path: {self.db_path}")
        
        # Create the connection pool
        self.pool = ConnectionPool(max_connections=10, db_path=self.db_path)
        
        self._initialize_database()
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with context management."""
        conn = self.pool.get_connection()
        try:
            yield conn
        finally:
            self.pool.release_connection(conn)
    
    def _initialize_database(self):
        """Initialize the database schema if it doesn't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Raw jobs table - contains only the scraped data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP, 
                    title TEXT,
                    company TEXT,
                    location TEXT,
                    date_posted TEXT,
                    hiring_status TEXT,
                    searched_keyword TEXT,
                    job_url TEXT
                )
            """)
            
            # Job analysis table - contains analysis data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_analysis (
                    job_id TEXT PRIMARY KEY,
                    title_relevance REAL,
                    description_relevance REAL,
                    analyzed_at TIMESTAMP,
                    resume_version TEXT,
                    resume_hash TEXT,
                    FOREIGN KEY (job_id) REFERENCES raw_jobs(job_id)
                )
            """)
            
            # Raw descriptions table - contains raw HTML content
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_descriptions (
                    job_id TEXT PRIMARY KEY,
                    html_content TEXT,
                    response_status INTEGER,
                    fetched_at TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES raw_jobs(job_id)
                )
            """)
            
            # Formatted descriptions table - contains parsed text and metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS formatted_descriptions (
                    job_id TEXT PRIMARY KEY,
                    description_text TEXT,
                    seniority_level TEXT,
                    employment_type TEXT,
                    job_function TEXT,
                    industries TEXT,
                    num_applicants TEXT,
                    salary_range TEXT,
                    min_salary REAL,
                    max_salary REAL,
                    salary_frequency TEXT,
                    formatted_at TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES raw_descriptions(job_id)
                )
            """)
            
            # Job applications table - tracks application status and history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_applications (
                    job_id TEXT PRIMARY KEY,
                    status TEXT CHECK(status IN ('saved', 'applied', 'rejected', 'interviewing', 'offer_received', 'offer_accepted', 'offer_declined')),
                    applied_at TIMESTAMP,
                    last_updated_at TIMESTAMP,
                    notes TEXT,
                    follow_up_status TEXT,
                    company_response TEXT,
                    FOREIGN KEY (job_id) REFERENCES raw_jobs(job_id)
                )
            """)
            
            # Legacy table for backward compatibility
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
                    title_relevance TEXT,
                    job_url TEXT
                )
            """)
            
            # Manual rankings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS manual_rankings (
                    id TEXT PRIMARY KEY,  -- UUID for each ranking event
                    job_id TEXT,
                    item_type TEXT,       -- 'title' or 'description'
                    manual_score INTEGER,
                    timestamp TEXT,
                    session_id TEXT       -- To group rankings from the same session
                )
            """)
            
            conn.commit()
    
    def insert_raw_jobs(self, df: pd.DataFrame) -> int:
        """
        Inserts raw job postings from a DataFrame into the raw_jobs table, skipping duplicates.
        
        Args:
            df (pd.DataFrame): DataFrame containing raw job postings data
            
        Returns:
            int: Number of new raw jobs inserted
        """
        if df.empty:
            logging.info("No jobs to insert - DataFrame is empty")
            return 0
            
        # Ensure required columns exist; add default values if missing
        for col in ["created_at", "hiring_status", "location", "date_posted"]:
            if col not in df.columns:
                df[col] = None
        
        new_jobs = []
        total_rows = len(df)
        skipped = 0
        seen_job_ids = set()
        
        with self.get_connection() as conn:
            for _, row in df.iterrows():
                # Skip if job_id is missing
                if pd.isna(row.get('job_id')):
                    skipped += 1
                    continue
                    
                job_id = str(row['job_id'])
                
                if job_id in seen_job_ids or self.job_exists(job_id, conn, table='raw_jobs'):
                    skipped += 1
                else:
                    seen_job_ids.add(job_id)
                    # Create a tuple with all columns in the correct order
                    job_data = (
                        job_id,
                        row.get('created_at'),
                        row.get('title'),
                        row.get('company'),
                        row.get('location'),
                        row.get('date_posted'),
                        row.get('hiring_status'),
                        row.get('searched_keyword'),
                        row.get('job_url')
                    )
                    new_jobs.append(job_data)
            
            if new_jobs:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO raw_jobs (
                        job_id, created_at, title, company, location, 
                        date_posted, hiring_status, searched_keyword, job_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, new_jobs)
                conn.commit()
                logging.info(f"Inserted {len(new_jobs)} new raw jobs, skipped {skipped} duplicates (total {total_rows} rows).")
            else:
                logging.info(f"No new raw jobs inserted. Skipped {skipped} duplicates out of {total_rows} rows.")
        
        return len(new_jobs)
    
    def insert_raw_descriptions(self, descriptions_data: List[Dict[str, Any]]) -> int:
        """
        Insert raw HTML job descriptions into the raw_descriptions table.
        
        Args:
            descriptions_data (List[Dict]): List of dictionaries containing job_id and HTML content
            
        Returns:
            int: Number of descriptions inserted
        """
        if not descriptions_data:
            logging.info("No raw descriptions to insert")
            return 0
            
        inserted = 0
        updated = 0
        timestamp = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for desc in descriptions_data:
                job_id = desc.get('job_id')
                if not job_id:
                    continue
                    
                # Check if this job exists in raw_jobs
                if not self.job_exists(job_id, conn, 'raw_jobs'):
                    logging.warning(f"Job ID {job_id} not found in raw_jobs table, skipping raw description")
                    continue
                
                # Get HTML content and response status
                html_content = desc.get('html_content', '')
                response_status = desc.get('response_status', 0)
                
                # Check if description already exists
                cursor.execute("SELECT 1 FROM raw_descriptions WHERE job_id = ?", (job_id,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    cursor.execute("""
                        UPDATE raw_descriptions 
                        SET html_content = ?, response_status = ?, fetched_at = ?
                        WHERE job_id = ?
                    """, (html_content, response_status, timestamp, job_id))
                    updated += 1
                else:
                    cursor.execute("""
                        INSERT INTO raw_descriptions 
                        (job_id, html_content, response_status, fetched_at)
                        VALUES (?, ?, ?, ?)
                    """, (job_id, html_content, response_status, timestamp))
                    inserted += 1
                    
            conn.commit()
            
        logging.info(f"Raw descriptions: inserted {inserted} new records, updated {updated} existing records")
        return inserted + updated
    
    def insert_formatted_descriptions(self, formatted_data: List[Dict[str, Any]]) -> int:
        """
        Insert formatted job descriptions into the formatted_descriptions table.
        
        Args:
            formatted_data (List[Dict]): List of dictionaries containing parsed description data
            
        Returns:
            int: Number of formatted descriptions inserted/updated
        """
        if not formatted_data:
            logging.info("No formatted descriptions to insert")
            return 0
            
        inserted = 0
        updated = 0
        timestamp = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for data in formatted_data:
                job_id = data.get('job_id')
                if not job_id:
                    continue
                    
                # Check if raw description exists
                if not self.job_exists(job_id, conn, 'raw_descriptions'):
                    logging.warning(f"Job ID {job_id} not found in raw_descriptions table, skipping formatted description")
                    continue
                
                # Prepare data tuple with all fields
                formatted_data = (
                    data.get('description_text', ''),
                    data.get('seniority_level'),
                    data.get('employment_type'),
                    data.get('job_function'),
                    data.get('industries'),
                    data.get('num_applicants'),
                    data.get('salary_range'),
                    data.get('min_salary'),
                    data.get('max_salary'),
                    data.get('salary_frequency'),
                    timestamp,
                    job_id
                )
                
                # Check if formatted description already exists
                cursor.execute("SELECT 1 FROM formatted_descriptions WHERE job_id = ?", (job_id,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    cursor.execute("""
                        UPDATE formatted_descriptions 
                        SET description_text = ?, seniority_level = ?, employment_type = ?,
                            job_function = ?, industries = ?, num_applicants = ?,
                            salary_range = ?, min_salary = ?, max_salary = ?,
                            salary_frequency = ?, formatted_at = ?
                        WHERE job_id = ?
                    """, formatted_data)
                    updated += 1
                else:
                    cursor.execute("""
                        INSERT INTO formatted_descriptions 
                        (description_text, seniority_level, employment_type,
                         job_function, industries, num_applicants,
                         salary_range, min_salary, max_salary,
                         salary_frequency, formatted_at, job_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, formatted_data)
                    inserted += 1
                    
            conn.commit()
            
        logging.info(f"Formatted descriptions: inserted {inserted} new records, updated {updated} existing records")
        return inserted + updated
    
    def insert_job_analysis(self, df: pd.DataFrame) -> int:
        """
        Inserts job analysis data from a DataFrame into the job_analysis table.
        If a row already exists for a job_id, it will be updated.
        
        Args:
            df (pd.DataFrame): DataFrame containing job analysis data with job_id and relevance scores
            
        Returns:
            int: Number of job analyses inserted or updated
        """
        if df.empty:
            logging.info("No job analyses to insert - DataFrame is empty")
            return 0
        
        required_columns = ['job_id', 'title_relevance']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return 0
        
        inserted = 0
        updated = 0
        current_time = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                if pd.isna(row['job_id']):
                    continue
                
                job_id = str(row['job_id'])
                
                # Skip if the job doesn't exist in raw_jobs table
                if not self.job_exists(job_id, conn, table='raw_jobs'):
                    logging.warning(f"Job ID {job_id} not found in raw_jobs table, skipping analysis")
                    continue
                
                # Convert values to Python native types
                title_relevance = float(row['title_relevance']) if pd.notna(row['title_relevance']) else None
                description_relevance = float(row['description_relevance']) if pd.notna(row.get('description_relevance')) else None
                resume_version = str(row.get('resume_version', '')) if pd.notna(row.get('resume_version')) else ''
                resume_hash = str(row.get('resume_hash', '')) if pd.notna(row.get('resume_hash')) else ''
                analyzed_at = str(row.get('analyzed_at', current_time)) if pd.notna(row.get('analyzed_at')) else current_time
                
                # Check if analysis already exists
                cursor.execute("SELECT 1 FROM job_analysis WHERE job_id = ?", (job_id,))
                exists = cursor.fetchone() is not None
                
                try:
                    if exists:
                        cursor.execute("""
                            UPDATE job_analysis 
                            SET title_relevance = ?, 
                                description_relevance = ?,
                                analyzed_at = ?,
                                resume_version = ?,
                                resume_hash = ?
                            WHERE job_id = ?
                        """, (title_relevance, description_relevance, analyzed_at, 
                              resume_version, resume_hash, job_id))
                        updated += 1
                    else:
                        cursor.execute("""
                            INSERT INTO job_analysis 
                            (job_id, title_relevance, description_relevance, analyzed_at,
                             resume_version, resume_hash)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (job_id, title_relevance, description_relevance, analyzed_at,
                              resume_version, resume_hash))
                        inserted += 1
                except Exception as e:
                    logging.error(f"Error inserting/updating analysis for job {job_id}: {e}")
                    logging.error(f"Values: {(title_relevance, description_relevance, analyzed_at, resume_version, resume_hash)}")
                    continue
            
            conn.commit()
            
        logging.info(f"Job analysis: inserted {inserted} new records, updated {updated} existing records")
        return inserted + updated
    
    def job_exists(self, job_id: str, conn: sqlite3.Connection = None, table: str = 'raw_jobs') -> bool:
        """
        Check if a job ID already exists in the specified table.

        Args:
            job_id (str): The job ID to check.
            conn (sqlite3.Connection, optional): An active SQLite connection. If None, a new connection is created.
            table (str): The table to check. Default is 'raw_jobs'.

        Returns:
            bool: True if the job ID exists, False otherwise.
        """
        # If connection is provided, use it directly
        if conn is not None:
            cursor = conn.cursor()
            cursor.execute(f"SELECT 1 FROM {table} WHERE job_id = ?", (job_id,))
            return cursor.fetchone() is not None
        
        # Otherwise, use connection context manager
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT 1 FROM {table} WHERE job_id = ?", (job_id,))
            return cursor.fetchone() is not None
    
    def get_jobs_with_analysis(self, title_relevance_threshold: float = 0) -> pd.DataFrame:
        """
        Get jobs with their analysis data joined together, filtered by relevance threshold.
        
        Args:
            title_relevance_threshold (float): Minimum title relevance score to include
            
        Returns:
            pd.DataFrame: DataFrame with joined job data
        """
        query = f"""
            SELECT r.*, a.title_relevance, a.description_relevance, a.analyzed_at
            FROM raw_jobs r
            LEFT JOIN job_analysis a ON r.job_id = a.job_id
            WHERE a.title_relevance >= {title_relevance_threshold} OR a.title_relevance IS NULL
        """
        return self.query_to_df(query)
    
    def get_jobs_with_descriptions(self, relevance_threshold: float = 0) -> pd.DataFrame:
        """
        Get jobs with their descriptions and analysis data joined together.
        
        Args:
            relevance_threshold (float): Minimum relevance score to include
            
        Returns:
            pd.DataFrame: DataFrame with joined job and description data
        """
        query = f"""
            SELECT r.*, a.title_relevance, a.description_relevance, 
                   f.description_text, f.seniority_level, f.employment_type,
                   f.job_function, f.num_applicants, f.salary_range,
                   f.min_salary, f.max_salary, f.salary_frequency
            FROM raw_jobs r
            LEFT JOIN job_analysis a ON r.job_id = a.job_id
            LEFT JOIN formatted_descriptions f ON r.job_id = f.job_id
            WHERE (
                a.title_relevance >= {relevance_threshold}
                OR 
                (a.description_relevance IS NOT NULL AND a.description_relevance >= {relevance_threshold})
            )
            AND f.description_text IS NOT NULL
        """
        return self.query_to_df(query)
    
    def query_to_df(self, query: str, params=None) -> pd.DataFrame:
        """Execute a query and return the results as a pandas DataFrame.
        
        Args:
            query: SQL query to execute
            params: Parameters for the query
            
        Returns:
            pd.DataFrame: Query results as a DataFrame
        """
        try:
            with self.get_connection() as conn:
                if params:
                    return pd.read_sql_query(query, conn, params=params)
                else:
                    return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params}")
            return pd.DataFrame()
    
    # Legacy methods for backward compatibility
    def insert_jobs(self, df: pd.DataFrame) -> int:
        """Legacy method that inserts into both tables for backward compatibility."""
        raw_count = self.insert_raw_jobs(df)
        
        # If title_relevance exists, update job_analysis
        if 'title_relevance' in df.columns:
            analysis_df = df[['job_id', 'title_relevance']].copy()
            if 'description_relevance' in df.columns:
                analysis_df['description_relevance'] = df['description_relevance']
            self.insert_job_analysis(analysis_df)
            
        return raw_count
    
    def insert_new_jobs(self, csv_file: str) -> None:
        """Legacy method that inserts new job postings from a CSV file."""
        df = pd.read_csv(csv_file, dtype=str)
        self.insert_jobs(df)

    def update_application_status(self, job_id: str, status: str, notes: str = None, 
                                follow_up_status: str = None, company_response: str = None) -> bool:
        """
        Update or insert an application status for a job.
        
        Args:
            job_id (str): The job ID to update
            status (str): New application status
            notes (str, optional): Any notes about the application
            follow_up_status (str, optional): Status of follow-up communications
            company_response (str, optional): Response received from the company
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.job_exists(job_id):
            logging.error(f"Job ID {job_id} not found in raw_jobs table")
            return False
            
        current_time = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if application record exists
            cursor.execute("SELECT 1 FROM job_applications WHERE job_id = ?", (job_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                cursor.execute("""
                    UPDATE job_applications 
                    SET status = ?,
                        last_updated_at = ?,
                        notes = CASE 
                            WHEN ? IS NOT NULL THEN 
                                CASE 
                                    WHEN notes IS NULL THEN ?
                                    ELSE notes || CHAR(10) || '[' || ? || '] ' || ?
                                END
                            ELSE notes 
                        END,
                        follow_up_status = COALESCE(?, follow_up_status),
                        company_response = COALESCE(?, company_response)
                    WHERE job_id = ?
                """, (status, current_time, notes, notes, current_time, notes,
                      follow_up_status, company_response, job_id))
            else:
                cursor.execute("""
                    INSERT INTO job_applications 
                    (job_id, status, applied_at, last_updated_at, notes, 
                     follow_up_status, company_response)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (job_id, status, current_time, current_time, notes,
                      follow_up_status, company_response))
            
            conn.commit()
            return True
    
    def get_application_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current application status and history for a job.
        
        Args:
            job_id (str): The job ID to look up
            
        Returns:
            Optional[Dict[str, Any]]: Application status data or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.*, r.title, r.company, r.job_url
                FROM job_applications a
                JOIN raw_jobs r ON a.job_id = r.job_id
                WHERE a.job_id = ?
            """, (job_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'job_id': row[0],
                    'status': row[1],
                    'applied_at': row[2],
                    'last_updated_at': row[3],
                    'notes': row[4],
                    'follow_up_status': row[5],
                    'company_response': row[6],
                    'title': row[7],
                    'company': row[8],
                    'job_url': row[9]
                }
            return None
    
    def get_all_applications(self, status: Optional[str] = None) -> pd.DataFrame:
        """
        Get all job applications, optionally filtered by status.
        
        Args:
            status (str, optional): Filter by specific application status
            
        Returns:
            pd.DataFrame: DataFrame containing application data
        """
        query = """
            SELECT 
                a.*,
                r.title,
                r.company,
                r.location,
                r.job_url,
                f.salary_range,
                an.title_relevance,
                an.description_relevance
            FROM job_applications a
            JOIN raw_jobs r ON a.job_id = r.job_id
            LEFT JOIN formatted_descriptions f ON a.job_id = f.job_id
            LEFT JOIN job_analysis an ON a.job_id = an.job_id
        """
        
        if status:
            query += " WHERE a.status = ?"
            return self.query_to_df(query, [status])
        return self.query_to_df(query)

    def create_manual_rankings_table(self) -> None:
        """Create the manual rankings table if it doesn't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS manual_rankings (
                    id TEXT PRIMARY KEY,  -- UUID for each ranking event
                    job_id TEXT,
                    item_type TEXT,       -- 'title' or 'description'
                    manual_score INTEGER,
                    timestamp TEXT,
                    session_id TEXT       -- To group rankings from the same session
                )
            """)
            conn.commit()

    def insert_manual_ranking(self, ranking_data: Dict[str, Any]) -> bool:
        """Insert a manual ranking into the database
        
        Args:
            ranking_data: Dictionary with id, job_id, item_type, manual_score, timestamp, session_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO manual_rankings 
                    (id, job_id, item_type, manual_score, timestamp, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    ranking_data['id'],
                    ranking_data['job_id'],
                    ranking_data['item_type'],
                    ranking_data['manual_score'],
                    ranking_data['timestamp'],
                    ranking_data['session_id']
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error inserting manual ranking: {e}")
            return False

    def get_random_jobs(self, count=10, with_descriptions=False) -> pd.DataFrame:
        """Get random jobs from the database
        
        Args:
            count: Number of jobs to return
            with_descriptions: If True, only return jobs that have descriptions
            
        Returns:
            DataFrame with job_id, title, and has_description
        """
        if with_descriptions:
            query = """
                SELECT rj.job_id, rj.title, 1 as has_description
                FROM raw_jobs rj
                JOIN formatted_descriptions fd ON rj.job_id = fd.job_id
                ORDER BY RANDOM()
                LIMIT ?
            """
        else:
            query = """
                SELECT rj.job_id, rj.title, 
                    CASE WHEN fd.job_id IS NOT NULL THEN 1 ELSE 0 END as has_description
                FROM raw_jobs rj
                LEFT JOIN formatted_descriptions fd ON rj.job_id = fd.job_id
                ORDER BY RANDOM()
                LIMIT ?
            """
        
        return self.query_to_df(query, params=(count,))

    def get_job_description(self, job_id: str) -> str:
        """Get the formatted description for a job
        
        Args:
            job_id: The job ID
            
        Returns:
            The description text or empty string if not found
        """
        query = """
            SELECT description_text FROM formatted_descriptions
            WHERE job_id = ?
        """
        df = self.query_to_df(query, params=(job_id,))
        
        if not df.empty:
            return df.iloc[0]['description_text']
        return ""

    def get_available_tables(self) -> List[str]:
        """Get a list of all tables in the database
        
        Returns:
            List of table names
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [row[0] for row in cursor.fetchall()]

    def get_jobs_needing_analysis(self) -> pd.DataFrame:
        """Get jobs that haven't been analyzed yet
        
        Returns:
            DataFrame with job_id and title for jobs that need analysis
        """
        query = """
            SELECT rj.job_id, rj.title 
            FROM raw_jobs rj
            LEFT JOIN job_analysis ja ON rj.job_id = ja.job_id
            WHERE ja.job_id IS NULL
        """
        return self.query_to_df(query)

    def get_jobs_needing_descriptions(self, title_relevance_threshold=1.0) -> Dict[str, set]:
        """Get jobs that need description processing based on a relevance threshold
        
        Args:
            title_relevance_threshold: Minimum title relevance score to consider (default: 1.0)
            
        Returns:
            Dictionary with sets of job_ids for different processing statuses:
            - 'relevant': All jobs meeting the relevance threshold
            - 'need_raw': Jobs that need raw description fetching
            - 'need_formatted': Jobs that have raw descriptions but need formatting
            - 'fully_processed': Jobs that are completely processed
        """
        # Get all relevant jobs
        relevant_query = f"""
            SELECT job_id 
            FROM job_analysis 
            WHERE title_relevance >= {title_relevance_threshold}
        """
        relevant_df = self.query_to_df(relevant_query)
        relevant_jobs = set(relevant_df['job_id']) if not relevant_df.empty else set()
        
        # Get jobs with raw descriptions
        raw_query = "SELECT DISTINCT job_id FROM raw_descriptions"
        raw_df = self.query_to_df(raw_query)
        raw_jobs = set(raw_df['job_id']) if not raw_df.empty else set()
        
        # Get jobs with formatted descriptions
        formatted_query = "SELECT DISTINCT job_id FROM formatted_descriptions"
        formatted_df = self.query_to_df(formatted_query)
        formatted_jobs = set(formatted_df['job_id']) if not formatted_df.empty else set()
        
        # Calculate processing status
        fully_processed = raw_jobs.intersection(formatted_jobs)
        need_formatted_only = raw_jobs - formatted_jobs
        need_raw = relevant_jobs - raw_jobs
        
        return {
            'relevant': relevant_jobs,
            'need_raw': need_raw,
            'need_formatted': need_formatted_only,
            'fully_processed': fully_processed
        }

    def get_job_urls(self, job_ids: set, batch_size=500) -> Dict[str, str]:
        """Get job URLs for a set of job IDs
        
        Args:
            job_ids: Set of job IDs to get URLs for
            batch_size: Size of batches for processing large sets
            
        Returns:
            Dictionary mapping job_id to job_url
        """
        if not job_ids:
            return {}
            
        urls = {}
        # Process in batches to avoid SQL IN clause limitations
        job_ids_list = list(job_ids)
        
        for i in range(0, len(job_ids_list), batch_size):
            batch = job_ids_list[i:i+batch_size]
            query = f"""
                SELECT job_id, job_url 
                FROM raw_jobs 
                WHERE job_id IN ({','.join(str(id) for id in batch)})
            """
            batch_df = self.query_to_df(query)
            # Update the dictionary with this batch
            urls.update(dict(zip(batch_df['job_id'], batch_df['job_url'])))
            
        return urls

    def execute_query(self, query: str, params=None) -> sqlite3.Cursor:
        """Execute a query and return the cursor.
        
        Args:
            query: SQL query to execute
            params: Parameters for the query
            
        Returns:
            sqlite3.Cursor: The cursor after executing the query
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params}")
            raise

# Create a singleton instance for convenience
db = DatabaseManager()

# For backward compatibility
def insert_new_jobs(csv_file: str) -> None:
    """Wrapper for backwards compatibility"""
    db.insert_new_jobs(csv_file)

def job_exists(job_id: str, conn: sqlite3.Connection = None) -> bool:
    """Wrapper for backwards compatibility"""
    return db.job_exists(job_id, conn)

def query_database_to_df(db_path, query):
    """Wrapper for backwards compatibility"""
    temp_db = DatabaseManager(db_path)
    return temp_db.query_to_df(query)