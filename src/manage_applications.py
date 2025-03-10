#!/usr/bin/env python3
"""
Command-line interface for managing job applications.
This script provides an interactive way to:
- View all applications
- Add new applications
- Update application status
- Add notes and track responses
"""

import sys
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from db import db
import logging
import argparse
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def display_applications(applications: pd.DataFrame) -> None:
    """Display applications in a nicely formatted table."""
    if applications.empty:
        print("\nNo applications found.")
        return
        
    # Select and rename columns for display
    display_cols = [
        'status', 'company', 'title', 'location', 'applied_at', 
        'last_updated_at', 'title_relevance', 'description_relevance',
        'salary_range', 'notes'
    ]
    
    # Ensure all columns exist
    for col in display_cols:
        if col not in applications.columns:
            applications[col] = None
    
    display_df = applications[display_cols].copy()
    
    # Truncate long text fields
    if 'notes' in display_df.columns:
        display_df['notes'] = display_df['notes'].apply(
            lambda x: (str(x)[:50] + '...') if pd.notna(x) and len(str(x)) > 50 else x
        )
    
    # Format timestamps
    for col in ['applied_at', 'last_updated_at']:
        if col in display_df.columns:
            display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d')
    
    # Round relevance scores, handling None/NULL values
    for col in ['title_relevance', 'description_relevance']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(2)
    
    print("\nJob Applications:")
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

def add_or_update_application() -> None:
    """Interactive function to add or update an application."""
    job_id = input("\nEnter job ID: ").strip()
    
    # Check if job exists
    if not db.job_exists(job_id):
        print(f"Error: Job ID {job_id} not found in database.")
        return
    
    # Get current job details
    current_status = db.get_application_status(job_id)
    if current_status:
        print("\nCurrent application status:")
        for key, value in current_status.items():
            if value and key not in ['job_id']:
                print(f"{key}: {value}")
    
    # Get new status
    valid_statuses = [
        'saved', 'applied', 'rejected', 'interviewing',
        'offer_received', 'offer_accepted', 'offer_declined'
    ]
    
    print("\nValid status values:", ', '.join(valid_statuses))
    status = input("Enter new status: ").strip().lower()
    
    if status not in valid_statuses:
        print(f"Error: Invalid status. Must be one of: {', '.join(valid_statuses)}")
        return
    
    # Get optional fields
    notes = input("Enter notes (optional): ").strip()
    follow_up = input("Enter follow-up status (optional): ").strip()
    response = input("Enter company response (optional): ").strip()
    
    # Update the application
    success = db.update_application_status(
        job_id=job_id,
        status=status,
        notes=notes if notes else None,
        follow_up_status=follow_up if follow_up else None,
        company_response=response if response else None
    )
    
    if success:
        print("\nApplication updated successfully!")
        # Show the updated status
        updated = db.get_application_status(job_id)
        if updated:
            print("\nUpdated application status:")
            for key, value in updated.items():
                if value and key not in ['job_id']:
                    print(f"{key}: {value}")
    else:
        print("\nError updating application.")

def view_application_details() -> None:
    """View detailed information about a specific application."""
    job_id = input("\nEnter job ID: ").strip()
    
    status = db.get_application_status(job_id)
    if not status:
        print(f"No application found for job ID: {job_id}")
        return
    
    print("\nApplication Details:")
    print("=" * 50)
    
    # Print main details
    main_fields = ['title', 'company', 'status', 'job_url']
    for field in main_fields:
        if status.get(field):
            print(f"{field.replace('_', ' ').title()}: {status[field]}")
    
    # Print timestamps
    print(f"\nTimeline:")
    print(f"Applied: {status.get('applied_at', 'N/A')}")
    print(f"Last Updated: {status.get('last_updated_at', 'N/A')}")
    
    # Print notes with proper formatting
    if status.get('notes'):
        print("\nNotes:")
        print("-" * 50)
        print(status['notes'])
    
    # Print follow-up status and company response
    if status.get('follow_up_status'):
        print("\nFollow-up Status:")
        print("-" * 50)
        print(status['follow_up_status'])
    
    if status.get('company_response'):
        print("\nCompany Response:")
        print("-" * 50)
        print(status['company_response'])

def main() -> None:
    """Main function to run the application manager."""
    while True:
        print("\nJob Application Manager")
        print("=" * 50)
        print("1. View all applications")
        print("2. View applications by status")
        print("3. Add/Update application")
        print("4. View application details")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            applications = db.get_all_applications()
            display_applications(applications)
        
        elif choice == '2':
            valid_statuses = [
                'saved', 'applied', 'rejected', 'interviewing',
                'offer_received', 'offer_accepted', 'offer_declined'
            ]
            print("\nValid status values:", ', '.join(valid_statuses))
            status = input("Enter status to filter by: ").strip().lower()
            
            if status in valid_statuses:
                applications = db.get_all_applications(status=status)
                display_applications(applications)
            else:
                print(f"Error: Invalid status. Must be one of: {', '.join(valid_statuses)}")
        
        elif choice == '3':
            add_or_update_application()
        
        elif choice == '4':
            view_application_details()
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0) 