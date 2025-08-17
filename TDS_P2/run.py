# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]
import subprocess
import sys
import os
import json
import argparse
from pathlib import Path

def get_test_files(verbose=False):
    """Automatically detect test files without hardcoding specific test cases"""
    current_dir = Path.cwd()
    if verbose:
        print(f"Current directory: {current_dir}", file=sys.stderr)
        sys.stderr.flush()
    
    available_files = []
    question_files = list(Path(".").glob("*question*.txt"))
    
    if question_files:
        available_files.append(question_files[0].name)
        if verbose:
            print(f"Found questions file: {question_files[0].name}", file=sys.stderr)
            sys.stderr.flush()
    else:
        print("Warning: No questions file found", file=sys.stderr)
        sys.stderr.flush()
    
    data_files = []
    for ext in ["*.csv", "*.json", "*.xlsx", "*.xls", "*.txt"]:
        for file in Path(".").glob(ext):
            if file.name not in [f.name for f in question_files]:
                data_files.append(file.name)
    
    if data_files:
        available_files.extend(data_files)
        if verbose:
            print(f"Found data files: {data_files}", file=sys.stderr)
            sys.stderr.flush()
    
    if not available_files:
        print("Error: No test files found in current directory", file=sys.stderr)
        sys.stderr.flush()
        return []
    
    if verbose:
        print(f"Total files to send: {available_files}", file=sys.stderr)
        sys.stderr.flush()
    
    return available_files

def validate_files(files, verbose=False):
    """Validate that all detected files exist and are readable"""
    valid_files = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File {file_path} does not exist", file=sys.stderr)
            sys.stderr.flush()
            continue
        
        if not path.is_file():
            print(f"Error: {file_path} is not a regular file", file=sys.stderr)
            sys.stderr.flush()
            continue
        
        if not os.access(path, os.R_OK):
            print(f"Error: File {file_path} is not readable", file=sys.stderr)
            sys.stderr.flush()
            continue
        
        valid_files.append(file_path)
        if verbose:
            print(f"Validated file: {file_path} (size: {path.stat().st_size} bytes)", file=sys.stderr)
            sys.stderr.flush()
    
    return valid_files

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests against the Data Analyst Agent API")
    parser.add_argument("api_url", help="The API endpoint URL to test against")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--timeout", "-t", type=int, default=30, help="Timeout for curl command in seconds")
    
    try:
        args = parser.parse_args()
    except SystemExit:
        # argparse already printed usage and exited
        sys.exit(1)
    
    verbose = args.verbose
    timeout = args.timeout
    
    # Validate API URL
    if not args.api_url.startswith(('http://', 'https://')):
        print("Error: API URL must start with http:// or https://", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
    
    # Detect test files
    test_files = get_test_files(verbose)
    if not test_files:
        print(json.dumps({"error": "No test files found"}))
        sys.exit(1)
    
    # Validate all files exist and are readable
    valid_files = validate_files(test_files, verbose)
    if not valid_files:
        print(json.dumps({"error": "No valid test files found"}))
        sys.exit(1)
    
    # Build and execute curl command
    curl_command = build_curl_command(args.api_url, valid_files)
    if verbose:
        print(f"Executing: {' '.join(curl_command)}", file=sys.stderr)
        print(f"Timeout: {timeout} seconds", file=sys.stderr)
        sys.stderr.flush()
    
    try:
        result = subprocess.run(
            curl_command, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=timeout
        )
        
        if result.stdout.strip():
            print(result.stdout)
        else:
            print(json.dumps({"error": "API returned empty response"}))
            
    except subprocess.TimeoutExpired:
        print("Error: curl command timed out", file=sys.stderr)
        print(json.dumps({"error": "Request timed out", "timeout_seconds": timeout}))
        sys.exit(1)
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Subprocess execution failed with return code {e.returncode}", file=sys.stderr)
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"STDERR:\n{e.stderr}", file=sys.stderr)
        print(json.dumps({
            "error": "Subprocess failed", 
            "return_code": e.returncode,
            "stdout": e.stdout,
            "stderr": e.stderr
        }))
        sys.stderr.flush()
        sys.exit(1)
        
    except FileNotFoundError:
        print("Error: curl command not found. Please install curl.", file=sys.stderr)
        print(json.dumps({"error": "curl command not found. Please install curl."}))
        sys.stderr.flush()
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: Unexpected error occurred: {str(e)}", file=sys.stderr)
        print(json.dumps({"error": "Unexpected error", "details": str(e)}))
        sys.stderr.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
