#!/usr/bin/env python3
"""
Test runner script for the Data Analyst Agent API.
This script is used by the test framework to interact with the API.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def get_test_files():
    """Automatically detect test files without hardcoding specific test cases"""
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}", file=sys.stderr)
    available_files = []
    question_files = list(Path(".").glob("*question*.txt"))
    if question_files:
        available_files.append(question_files[0].name)
        print(f"Found questions file: {question_files[0].name}", file=sys.stderr)
    else:
        print("Warning: No questions file found", file=sys.stderr)
    data_files = []
    for ext in ["*.csv", "*.json", "*.xlsx", "*.xls", "*.txt"]:
        for file in Path(".").glob(ext):
            if file.name not in [f.name for f in question_files]:
                data_files.append(file.name)
    if data_files:
        available_files.extend(data_files)
        print(f"Found data files: {data_files}", file=sys.stderr)
    if not available_files:
        print("Error: No test files found in current directory", file=sys.stderr)
        return []
    print(f"Total files to send: {available_files}", file=sys.stderr)
    return available_files

def build_curl_command(url, files):
    """Build the curl command with the detected files"""
    curl_command = ["curl", url]
    question_files = [f for f in files if "question" in f.lower()]
    if question_files:
        curl_command.extend(["-F", f"questions=@{question_files[0]}"])
    data_files = [f for f in files if "question" not in f.lower()]
    for file in data_files:
        curl_command.extend(["-F", f"files=@{file}"])
    return curl_command

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python run.py <api_url>"}))
        sys.exit(1)
    
    url = sys.argv[1]
    test_files = get_test_files()
    
    if not test_files:
        print(json.dumps({"error": "No test files found"}))
        sys.exit(1)
    
    curl_command = build_curl_command(url, test_files)
    print(f"Executing: {' '.join(curl_command)}", file=sys.stderr)
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print(result.stdout)
        else:
            print(json.dumps({"error": "API returned empty response"}))
    except subprocess.CalledProcessError as e:
        print(json.dumps({"error": f"Subprocess failed", "details": e.stderr.strip()}))
        sys.exit(1)

if __name__ == "__main__":
    main()
