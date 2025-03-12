#!/usr/bin/env python3
import os
import sys
import random
import argparse
import pandas as pd
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Add the src directory to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import the DatabaseManager
from db import DatabaseManager

load_dotenv()  # Load environment variables

def rank_items(items, item_type, session_id):
    """Prompt user to rank items and return the rankings"""
    rankings = []
    total = len(items)
    
    for idx, (_, item) in enumerate(items.iterrows(), 1):
        job_id = item['job_id']
        
        if item_type == 'title':
            # Display the title
            print(f"\n[{idx}/{total}] Rate this JOB TITLE on relevance (1-10):")
            print(f"Title: {item['title']}")
        else:
            # Display the description with title as context
            description = item['description_text']
            display_desc = description[:4000] + "..." if len(description) > 4000 else description
            print(f"\n[{idx}/{total}] Rate this JOB DESCRIPTION on relevance (1-10):")
            print(f"Title: {item['title']}")
            print(f"Description: {display_desc}")
        
        # Get the user's score
        while True:
            try:
                score = int(input("Your score (1-10, or 0 to skip): "))
                if 0 <= score <= 10:
                    break
                print("Please enter a number between 0 and 10.")
            except ValueError:
                print("Please enter a valid number.")
        
        if score > 0:  # Only record if not skipped
            rankings.append({
                'id': str(uuid.uuid4()),
                'job_id': job_id,
                'item_type': item_type,
                'manual_score': score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'session_id': session_id
            })
    
    return rankings

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Rank job titles and descriptions')
    parser.add_argument('--titles', type=int, default=10,
                        help='Number of titles to rank (default: 10)')
    parser.add_argument('--descriptions', type=int, default=10,
                        help='Number of descriptions to rank (default: 10)')
    parser.add_argument('--only-titles', action='store_true',
                        help='Only rank titles, no descriptions')
    parser.add_argument('--only-descriptions', action='store_true',
                        help='Only rank descriptions, no titles')
    parser.add_argument('--db-path', type=str,
                        help='Override the database path (defaults to DB_PATH env variable or "../data/jobs.db")')
    args = parser.parse_args()
    
    # Override database path if provided
    if args.db_path:
        os.environ["DB_PATH"] = args.db_path
        print(f"Using database path: {args.db_path}")
    
    # Initialize database connection
    db = DatabaseManager()
    
    try:
        # Check available tables in the database
        tables = db.get_available_tables()
        print(f"Available tables: {', '.join(tables)}")
        
        # Check if we found necessary tables
        required_tables = ['raw_jobs']
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            print(f"\nMissing required tables: {', '.join(missing_tables)}")
            print("Make sure the database path is correct and the database has been initialized.")
            print("You can specify the path with --db-path or set the DB_PATH environment variable.")
            return
            
        # Setup the rankings table
        db.create_manual_rankings_table()
        
        # Generate a session ID to group this set of rankings
        session_id = str(uuid.uuid4())
        print(f"Starting ranking session {session_id}")
        
        saved_count = 0
        title_success = False
        desc_success = False
        
        # Handle titles if needed
        if not args.only_descriptions:
            print(f"\n===== RANKING JOB TITLES =====")
            titles_df = db.get_random_jobs(args.titles, with_descriptions=False)
            
            if titles_df.empty:
                print("No job titles found in the database.")
            else:
                title_rankings = rank_items(titles_df, 'title', session_id)
                
                # Save title rankings
                for ranking in title_rankings:
                    db.insert_manual_ranking(ranking)
                
                saved_count += len(title_rankings)
                print(f"Saved {len(title_rankings)} title rankings.")
                title_success = len(title_rankings) > 0
        
        # Handle descriptions if needed
        if not args.only_titles:
            print(f"\n===== RANKING JOB DESCRIPTIONS =====")
            descriptions_df = db.get_random_jobs(args.descriptions, with_descriptions=True)
            
            if descriptions_df.empty:
                print("No job descriptions found in the database.")
            else:
                # Get the descriptions for each job
                for i, row in descriptions_df.iterrows():
                    job_id = row['job_id']
                    desc_text = db.get_job_description(job_id)
                    descriptions_df.at[i, 'description_text'] = desc_text
                
                desc_rankings = rank_items(descriptions_df, 'description', session_id)
                
                # Save description rankings
                for ranking in desc_rankings:
                    db.insert_manual_ranking(ranking)
                
                saved_count += len(desc_rankings)
                print(f"Saved {len(desc_rankings)} description rankings.")
                desc_success = len(desc_rankings) > 0
        
        # Final summary
        if saved_count > 0:
            print(f"\nRanking session complete! Saved {saved_count} rankings with session ID {session_id}")
        else:
            print("\nNo rankings were saved. Possible issues:")
            if not args.only_descriptions and not title_success:
                print("- No job titles were found or you skipped all titles")
            if not args.only_titles and not desc_success:
                print("- No job descriptions were found or you skipped all descriptions")
            print("\nTips:")
            print("- Make sure you're connecting to the correct database")
            print("- Verify that your database contains job data")
            print("- Try using --db-path to specify a different database location")
        
    except KeyboardInterrupt:
        print("\nRanking interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    main() 