#!/usr/bin/env python3
"""
Test runner script for the Data Analyst Agent API.
This script acts as a bridge between the test framework and the API.
"""

import sys
import requests
import json
import base64
from pathlib import Path
import os

def main():
    """Main function to handle test requests"""
    if len(sys.argv) != 2:
        print("Usage: python run.py <api_url>")
        sys.exit(1)
    
    api_url = sys.argv[1].rstrip('/')  # Remove trailing slash if present
    
    try:
        # Read the question from stdin
        question = sys.stdin.read().strip()
        
        if not question:
            print(json.dumps({"error": "No question provided"}))
            sys.exit(1)
        
        # Prepare the request to the /api/ endpoint
        files = {
            'questions': ('questions.txt', question, 'text/plain')
        }
        
        # Make the POST request to the API
        response = requests.post(f"{api_url}/api/", files=files, timeout=300)
        
        if response.status_code == 200:
            # Return the successful response
            result = response.json()
            print(json.dumps(result))
        else:
            # Handle error responses
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get('detail', error_detail)
            except:
                pass
            
            print(json.dumps({
                "error": f"API request failed with status {response.status_code}",
                "detail": error_detail
            }))
            sys.exit(1)
            
    except requests.exceptions.Timeout:
        print(json.dumps({"error": "Request timeout - API took too long to respond"}))
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(json.dumps({"error": "Connection error - could not connect to API"}))
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(json.dumps({"error": f"Request failed: {str(e)}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
