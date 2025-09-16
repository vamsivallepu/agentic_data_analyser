"""
Main entry point for the Multi-CSV Agent application.
This file serves as the entry point for the Streamlit application.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import and run the main application
from main import main

if __name__ == "__main__":
    main()
