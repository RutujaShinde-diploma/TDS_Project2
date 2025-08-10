import requests
import json

def test_wikipedia_scraping():
    """Test the Wikipedia scraping functionality"""
    
    # Read the wiki_test.txt file
    with open('wiki_test.txt', 'r', encoding='utf-8') as f:
        questions_content = f.read()
    
    print("Testing Wikipedia scraping...")
    print(f"Questions content: {questions_content[:200]}...")
    
    # Prepare the request
    files = {
        'questions': ('wiki_test.txt', questions_content, 'text/plain'),
        'files': ('', '', 'text/plain')  # Empty file
    }
    
    try:
        # Make the request
        response = requests.post(
            'http://localhost:8000/api/',
            files=files,
            timeout=300  # 5 minutes timeout
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Check if we got proper answers
            if isinstance(result, dict):
                print(f"Number of answers: {len(result)}")
                for key, value in result.items():
                    print(f"{key}: {value[:100] if len(str(value)) > 100 else value}")
            else:
                print(f"Unexpected response format: {type(result)}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_wikipedia_scraping()

