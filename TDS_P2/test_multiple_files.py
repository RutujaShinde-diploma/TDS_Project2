#!/usr/bin/env python3
"""
Test multiple file upload functionality
"""

import requests
import json
import time
import tempfile
import os

def test_multiple_files():
    """Test the API with multiple file uploads"""
    
    print("📁 Testing Multiple File Upload...")
    print("=" * 50)
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("What is the sum of the numbers in the CSV file?")
        questions_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("numbers\n1\n2\n3\n4\n5")
        csv_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Additional context data")
        context_file = f.name
    
    try:
        # Test multiple file upload
        files = [
            ('questions', ('questions.txt', open(questions_file, 'rb'), 'text/plain')),
            ('files', ('data.csv', open(csv_file, 'rb'), 'text/csv')),
            ('files', ('context.txt', open(context_file, 'rb'), 'text/plain'))
        ]
        
        print(f"📤 Uploading files:")
        print(f"  - questions: {questions_file}")
        print(f"  - data.csv: {csv_file}")
        print(f"  - context.txt: {context_file}")
        
        response = requests.post('http://localhost:8000/api/', files=files)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Multiple file upload successful!")
            print(f"📋 Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Upload failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        
    finally:
        # Clean up temporary files
        for file_path in [questions_file, csv_file, context_file]:
            try:
                os.unlink(file_path)
            except:
                pass

if __name__ == "__main__":
    test_multiple_files()
