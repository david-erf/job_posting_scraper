#!/usr/bin/env python3
"""
Configuration management for LinkedIn job search tools.
This module provides a centralized place for configuration settings.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "database": {
        "path": "../data/jobs.db",
        "connection_pool_size": 10,
        "timeout": 30
    },
    "search": {
        "keywords": [
            "data scientist",
            "data science",
            "machine learning engineer",
            "ML engineer",
            "AI engineer"
        ],
        "pages_per_keyword": 20,
        "max_age_days": 30,
        "include_remote": True
    },
    "analysis": {
        "title_relevance_threshold": 1.0,
        "description_relevance_threshold": 1.0,
        "process_descriptions": True,
        "batch_size": 500
    },
    "resume": {
        "path": "../data/resume.txt",
    },
    "logging": {
        "level": "INFO",
        "file": None
    }
}

class Config:
    """Configuration manager for LinkedIn job search tools."""
    _instance = None
    
    def __new__(cls, config_file=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_file=None):
        if self._initialized:
            return
            
        self.config_file = config_file
        self.config = DEFAULT_CONFIG.copy()
        
        # Override with environment variables
        self._load_from_env()
        
        # Override with config file if provided
        if config_file:
            self._load_from_file(config_file)
            
        self._initialized = True
        
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Database settings
        if db_path := os.getenv("DB_PATH"):
            self.config["database"]["path"] = db_path
            
        # Resume path
        if resume_path := os.getenv("RESUME_PATH"):
            self.config["resume"]["path"] = resume_path
            
        # Logging level
        if log_level := os.getenv("LOG_LEVEL"):
            self.config["logging"]["level"] = log_level
            
    def _load_from_file(self, config_file):
        """Load configuration from a JSON file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                
            # Recursively update config, preserving nested structure
            self._update_config(self.config, file_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            
    def _update_config(self, target, source):
        """Recursively update nested dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config(target[key], value)
            else:
                target[key] = value
                
    def save_config(self, config_file=None):
        """Save the current configuration to a file."""
        file_path = config_file or self.config_file
        if not file_path:
            logger.error("No config file specified for saving")
            return False
            
        try:
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Saved configuration to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")
            return False
            
    def get(self, path, default=None):
        """Get a configuration value using dot notation path."""
        current = self.config
        for part in path.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, path, value):
        """Set a configuration value using dot notation path."""
        parts = path.split('.')
        current = self.config
        
        # Navigate to the parent of the leaf node
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the leaf node value
        current[parts[-1]] = value
        
    def __getitem__(self, key):
        return self.config[key]
        
    def __setitem__(self, key, value):
        self.config[key] = value

# Create a singleton instance
config = Config()

def load_config(config_file):
    """Load configuration from a file."""
    global config
    config = Config(config_file)
    return config

def get_config():
    """Get the current configuration."""
    return config 