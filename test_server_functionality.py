#!/usr/bin/env python3
"""
Test script to verify server functionality end-to-end
"""

import asyncio
import json
import time
import requests
import websockets
from pathlib import Path

BASE_URL = "http://localhost:8080"

def test_health_check():
    """Test basic server health"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_environment_debug():
    """Test environment variable loading"""
    print("🔍 Testing environment debug endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/debug/env", timeout=10)
        assert response.status_code == 200
        data = response.json()
        print(f"📊 Environment status: {data}")
        
        if data.get("openrouter_key_set"):
            print("✅ API key is loaded")
        else:
            print("⚠️  API key not loaded - agentic evaluation will fail")
        
        return True
    except Exception as e:
        print(f"❌ Environment debug failed: {e}")
        return False

def test_corpus_upload():
    """Test corpus upload functionality"""
    print("🔍 Testing corpus upload...")
    try:
        corpus_data = {
            "text": "The Etruscan civilization was an ancient civilization created by the Etruscans. They inhabited Etruria in ancient Italy and developed a sophisticated culture with unique art, architecture, and religious practices. The Etruscans were skilled metalworkers and traders.",
            "name": "test_etruscan_corpus",
            "description": "Test corpus about Etruscan civilization"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/corpus/upload",
            json=corpus_data,
            timeout=30
        )
        assert response.status_code == 200
        data = response.json()
        
        corpus_id = data.get("corpus_id")
        assert corpus_id is not None
        print(f"✅ Corpus uploaded successfully: {corpus_id}")
        print(f"📈 Classification: {data.get('classification', {}).get('primary_type')}")
        
        return corpus_id
    except Exception as e:
        print(f"❌ Corpus upload failed: {e}")
        return None

def test_evaluation_start(corpus_id=None):
    """Test evaluation start functionality"""
    print("🔍 Testing evaluation start...")
    try:
        eval_data = {
            "corpus_text": "The Etruscan civilization was an ancient civilization created by the Etruscans. They inhabited Etruria in ancient Italy and developed a sophisticated culture with unique art, architecture, and religious practices. The Etruscans were skilled metalworkers and traders who influenced Roman culture significantly.",
            "num_questions": 3,
            "eval_type": "domain_knowledge", 
            "use_agentic": True,
            "temperature": 0.7,
            "token_threshold": 1000,
            "run_name": "Test Evaluation Run"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/evaluation/start",
            json=eval_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            run_id = data.get("run_id")
            print(f"✅ Evaluation started successfully: {run_id}")
            return run_id
        else:
            print(f"❌ Evaluation start failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Evaluation start failed: {e}")
        return None

def test_evaluation_status(run_id):
    """Test evaluation status checking"""
    print(f"🔍 Testing evaluation status for run: {run_id}")
    try:
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            response = requests.get(f"{BASE_URL}/api/v1/evaluation/status/{run_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                phase = data.get("phase")
                progress = data.get("progress_percent", 0)
                
                print(f"📊 Status: {status}, Phase: {phase}, Progress: {progress}%")
                
                if status == "completed":
                    print("✅ Evaluation completed successfully")
                    return True
                elif status == "error":
                    error = data.get("error", "Unknown error")
                    print(f"❌ Evaluation failed: {error}")
                    return False
                    
                time.sleep(2)
                attempt += 1
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
        
        print("⚠️  Evaluation timeout after 60 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False

async def test_websocket_connection(run_id):
    """Test WebSocket connection for real-time updates"""
    print(f"🔍 Testing WebSocket connection for run: {run_id}")
    try:
        uri = f"ws://localhost:8080/api/v1/ws/{run_id}"
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Listen for a few messages
            message_count = 0
            while message_count < 5:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    print(f"📨 Received: {data.get('type')} - {data.get('message', '')[:50]}...")
                    message_count += 1
                except asyncio.TimeoutError:
                    print("⚠️  WebSocket timeout - no more messages")
                    break
            
            print("✅ WebSocket communication working")
            return True
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting docs-to-eval Server Functionality Tests")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Health check
    results["health"] = test_health_check()
    
    # Test 2: Environment debug
    results["environment"] = test_environment_debug()
    
    # Test 3: Corpus upload
    corpus_id = test_corpus_upload()
    results["corpus_upload"] = corpus_id is not None
    
    # Test 4: Evaluation start
    run_id = test_evaluation_start(corpus_id)
    results["evaluation_start"] = run_id is not None
    
    if run_id:
        # Test 5: WebSocket connection
        try:
            results["websocket"] = asyncio.run(test_websocket_connection(run_id))
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            results["websocket"] = False
        
        # Test 6: Evaluation status (this takes time)
        results["evaluation_complete"] = test_evaluation_status(run_id)
    else:
        results["websocket"] = False
        results["evaluation_complete"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Server is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the server configuration.")
        return 1

if __name__ == "__main__":
    exit(main())