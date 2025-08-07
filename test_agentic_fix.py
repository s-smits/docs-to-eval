#!/usr/bin/env python3
"""
Test script to validate the agentic API key fix
"""

import os
import requests
import json
from time import sleep

def test_agentic_without_api_key():
    """Test agentic evaluation without API key (should use mock LLMs)"""
    
    print("🧪 Testing agentic evaluation without API key...")
    
    # Ensure no API key is set for this test
    if 'OPENROUTER_API_KEY' in os.environ:
        print(f"⚠️  Found OPENROUTER_API_KEY in environment - temporarily removing for test")
        api_key_backup = os.environ.pop('OPENROUTER_API_KEY')
    else:
        api_key_backup = None
    
    # Test data
    test_request = {
        "corpus_text": "Machine learning algorithms analyze data to identify patterns and make predictions. Deep learning uses neural networks to process complex information.",
        "eval_type": "domain_knowledge", 
        "num_questions": 3,
        "use_agentic": True,
        "temperature": 0.7
    }
    
    try:
        # Test the start evaluation endpoint
        print("🚀 Sending agentic evaluation request...")
        response = requests.post(
            "http://localhost:8000/api/v1/evaluation/start",
            json=test_request,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Agentic evaluation started without API key!")
            
            result = response.json()
            run_id = result.get('run_id')
            
            if run_id:
                print(f"📊 Run ID: {run_id}")
                
                # Check status periodically
                for i in range(10):
                    sleep(2)
                    status_response = requests.get(f"http://localhost:8000/api/v1/evaluation/{run_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"Status check {i+1}: {status_data.get('status', 'unknown')}")
                        
                        if status_data.get('status') in ['completed', 'error']:
                            break
                    else:
                        print(f"Status check failed: {status_response.status_code}")
        
        elif response.status_code == 500 and "API key is required" in response.text:
            print("❌ FAILED: Still requires API key despite fix")
            print("The fix didn't work - API key is still being required")
        else:
            print(f"❌ FAILED: Unexpected response")
            print(f"Status: {response.status_code}")  
            print(f"Body: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Cannot connect to server at http://localhost:8000")
        print("Make sure the server is running with 'python run_server.py'")
        
    except Exception as e:
        print(f"❌ FAILED: Unexpected error - {e}")
        
    finally:
        # Restore API key if it was set
        if api_key_backup:
            os.environ['OPENROUTER_API_KEY'] = api_key_backup
            print("🔄 Restored OPENROUTER_API_KEY to environment")


def test_debug_endpoints():
    """Test debug endpoints"""
    print("\\n🔍 Testing debug endpoints...")
    
    try:
        # Test env endpoint
        env_response = requests.get("http://localhost:8000/api/v1/debug/env", timeout=10)
        print(f"Debug env endpoint: {env_response.status_code}")
        if env_response.status_code == 200:
            print("✅ Debug /env endpoint working")
        
        # Test env-check endpoint  
        check_response = requests.get("http://localhost:8000/api/v1/debug/env-check", timeout=10)
        print(f"Debug env-check endpoint: {check_response.status_code}")
        if check_response.status_code == 200:
            print("✅ Debug /env-check endpoint working")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
    except Exception as e:
        print(f"❌ Debug endpoint test failed: {e}")


if __name__ == "__main__":
    print("🧪 Testing Agentic API Key Fix")
    print("=" * 40)
    
    test_agentic_without_api_key()
    test_debug_endpoints()
    
    print("\\n🎉 Test completed!")