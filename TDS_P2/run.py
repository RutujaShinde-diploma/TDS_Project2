# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]

import subprocess
import sys
import os
from pathlib import Path

def get_test_files():
    """Automatically detect test files without hardcoding specific test cases"""
    current_dir = Path.cwd()
    
    print(f"Current directory: {current_dir}")
    
    # Always look for question.txt first (required)
    available_files = []
    
    # Check for questions file (could be question.txt, questions.txt, etc.)
    question_files = list(Path(".").glob("*question*.txt"))
    if question_files:
        available_files.append(question_files[0].name)
        print(f"Found questions file: {question_files[0].name}")
    else:
        print("Warning: No questions file found")
    
    # Look for any data files (CSV, JSON, etc.)
    data_files = []
    for ext in ["*.csv", "*.json", "*.xlsx", "*.xls", "*.txt"]:
        for file in Path(".").glob(ext):
            if file.name not in [f.name for f in question_files]:  # Don't include questions file
                data_files.append(file.name)
    
    if data_files:
        available_files.extend(data_files)
        print(f"Found data files: {data_files}")
    
    # If no specific data files found, just return questions file
    if not available_files:
        print("Error: No test files found in current directory")
        return []
    
    print(f"Total files to send: {available_files}")
    return available_files

def build_curl_command(url, files):
    """Build the curl command with the detected files"""
    curl_command = ["curl", url]
    
    # Add questions file first (always required)
    question_files = [f for f in files if "question" in f.lower()]
    if question_files:
        curl_command.extend(["-F", f"questions=@{question_files[0]}"])
    
    # Add other data files
    data_files = [f for f in files if "question" not in f.lower()]
    for file in data_files:
        curl_command.extend(["-F", f"files=@{file}"])
    
    return curl_command

def main():
    if len(sys.argv) != 2:
        print("Usage: python run.py <api_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Detect test files dynamically
    test_files = get_test_files()
    
    if not test_files:
        print("Error: No test files found in current directory")
        sys.exit(1)
    
    # Build and execute curl command
    curl_command = build_curl_command(url, test_files)
    print(f"Executing: {' '.join(curl_command)}")
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
