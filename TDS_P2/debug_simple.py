#!/usr/bin/env python3
"""
Simple debug script to test LLM code generation directly
"""

import asyncio
import openai
import os
from config import config

async def test_llm_directly():
    """Test LLM code generation directly"""
    
    # Simple, direct prompt
    system_prompt = """You are a Python programmer. Generate ONLY Python code for data analysis. DO NOT include any explanations or generic messages."""
    
    user_prompt = """Generate Python code to:
1. Load a CSV file called 'sample-sales.csv'
2. Calculate the total sales from the 'sales' column
3. Print the result

Return ONLY the Python code, nothing else."""
    
    try:
        client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        code = response.choices[0].message.content
        print("=== GENERATED CODE ===")
        print(code)
        print("=== END CODE ===")
        
        # Check if it contains the problematic message
        if "Data was scraped successfully" in code:
            print("❌ PROBLEM: Code contains 'Data was scraped successfully'")
        else:
            print("✅ SUCCESS: Code looks clean")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_directly())
