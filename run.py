#!/usr/bin/env python3
"""
Test runner script for the Data Analyst Agent API.
This script is used by the test framework to interact with the API.
"""

import subprocess
import sys

def main():
    """Main function to run the test"""
    if len(sys.argv) != 2:
        print("Usage: python run.py <api_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Check which files exist and build the curl command accordingly
    curl_command = ["curl", url]
    
    # Check for question.txt (note: singular, not plural)
    if Path("question.txt").exists():
        curl_command.extend(["-F", "questions.txt=@question.txt"])
    elif Path("questions.txt").exists():
        curl_command.extend(["-F", "questions.txt=@questions.txt"])
    else:
        print("Error: Neither question.txt nor questions.txt found")
        sys.exit(1)
    
    # Check for additional data files
    if Path("edges.csv").exists():
        curl_command.extend(["-F", "edges.csv=@edges.csv"])
    
    if Path("sample-sales.csv").exists():
        curl_command.extend(["-F", "sample-sales.csv=@sample-sales.csv"])
    
    if Path("weather_data.csv").exists():
        curl_command.extend(["-F", "weather_data.csv=@weather_data.csv"])
    
    try:
        print(f"Executing: {' '.join(curl_command)}")
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: curl command not found. Please install curl.")
        sys.exit(1)

if __name__ == "__main__":
    from pathlib import Path
    main()
