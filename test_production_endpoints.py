"""
Test script for production endpoints.

Tests:
1. Enhanced GET /api/chat/history with pagination
2. GET /api/sessions
3. GET /api/conversations/rollup
4. POST /api/conversations/summarize

Run: python test_production_endpoints.py
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8080"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_chat_history_pagination():
    """Test enhanced chat history with pagination and filters."""
    print_section("1. Testing Enhanced Chat History (Pagination)")
    
    # Test 1: Basic history (no filters)
    print("→ GET /api/chat/history (basic)")
    response = requests.get(f"{BASE_URL}/api/chat/history", params={"limit": 10})
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Messages: {data.get('count', 0)}")
    print(f"Has more: {data.get('has_more', False)}")
    print(f"Next cursor: {data.get('next_cursor', 'None')[:50] if data.get('next_cursor') else 'None'}")
    
    # Test 2: Pagination with cursor
    if data.get('next_cursor'):
        print("\n→ GET /api/chat/history (with cursor)")
        response = requests.get(
            f"{BASE_URL}/api/chat/history",
            params={"limit": 10, "cursor": data['next_cursor']}
        )
        page2 = response.json()
        print(f"Status: {response.status_code}")
        print(f"Page 2 messages: {page2.get('count', 0)}")
    
    # Test 3: Filter by session_id
    print("\n→ GET /api/chat/history (filter by session_id)")
    response = requests.get(
        f"{BASE_URL}/api/chat/history",
        params={"session_id": "test-session", "limit": 5}
    )
    filtered = response.json()
    print(f"Status: {response.status_code}")
    print(f"Filtered messages: {filtered.get('count', 0)}")
    
    # Test 4: Filter by role
    print("\n→ GET /api/chat/history (filter by role=user)")
    response = requests.get(
        f"{BASE_URL}/api/chat/history",
        params={"role": "user", "limit": 5}
    )
    role_filtered = response.json()
    print(f"Status: {response.status_code}")
    print(f"User messages: {role_filtered.get('count', 0)}")
    
    # Test 5: Filter by timestamp
    print("\n→ GET /api/chat/history (filter by since)")
    since_time = datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat() + "Z"
    response = requests.get(
        f"{BASE_URL}/api/chat/history",
        params={"since": since_time, "limit": 10}
    )
    time_filtered = response.json()
    print(f"Status: {response.status_code}")
    print(f"Messages since {since_time}: {time_filtered.get('count', 0)}")

def test_sessions():
    """Test session listing endpoint."""
    print_section("2. Testing Sessions Endpoint")
    
    print("→ GET /api/sessions")
    response = requests.get(f"{BASE_URL}/api/sessions", params={"limit": 10})
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Total sessions: {data.get('total', 0)}")
    print(f"Sessions returned: {data.get('count', 0)}")
    
    if data.get('sessions'):
        print("\nSample session:")
        session = data['sessions'][0]
        print(f"  Session ID: {session.get('session_id')}")
        print(f"  Last activity: {session.get('last_message_time')}")
        print(f"  Message count: {session.get('message_count')}")
        print(f"  Summary preview: {session.get('summary_preview', 'None')[:100] if session.get('summary_preview') else 'None'}")

def test_conversation_rollup(session_id=None):
    """Test conversation rollup endpoint."""
    print_section("3. Testing Conversation Rollup")
    
    # Get a session_id first if not provided
    if not session_id:
        sessions_response = requests.get(f"{BASE_URL}/api/sessions", params={"limit": 1})
        sessions_data = sessions_response.json()
        if sessions_data.get('sessions'):
            session_id = sessions_data['sessions'][0]['session_id']
        else:
            print("⚠️  No sessions found. Skipping rollup test.")
            return
    
    print(f"→ GET /api/conversations/rollup?session_id={session_id}")
    response = requests.get(
        f"{BASE_URL}/api/conversations/rollup",
        params={"session_id": session_id}
    )
    
    if response.status_code == 404:
        print(f"Status: {response.status_code}")
        print("⚠️  No rollup found for this session (expected if not yet summarized)")
        return None
    
    data = response.json()
    print(f"Status: {response.status_code}")
    
    if data.get('rollup'):
        rollup = data['rollup']
        print(f"\nRollup summary:")
        print(f"  Summary: {rollup.get('summary', '')[:200]}")
        print(f"  Key topics: {rollup.get('key_topics', [])[:5]}")
        print(f"  Sentiment: {rollup.get('sentiment')} ({rollup.get('sentiment_score')})")
        print(f"  Message count: {rollup.get('message_count')}")
        print(f"  Updated at: {rollup.get('updated_at')}")
    
    return session_id

def test_conversation_summarize(session_id=None):
    """Test conversation summarization endpoint."""
    print_section("4. Testing Conversation Summarization")
    
    # Get a session_id first if not provided
    if not session_id:
        sessions_response = requests.get(f"{BASE_URL}/api/sessions", params={"limit": 1})
        sessions_data = sessions_response.json()
        if sessions_data.get('sessions'):
            session_id = sessions_data['sessions'][0]['session_id']
        else:
            print("⚠️  No sessions found. Skipping summarization test.")
            return
    
    print(f"→ POST /api/conversations/summarize (session_id={session_id})")
    response = requests.post(
        f"{BASE_URL}/api/conversations/summarize",
        params={
            "session_id": session_id,
            "message_limit": 20,
            "force": False  # Don't force if recent
        }
    )
    
    data = response.json()
    print(f"Status: {response.status_code}")
    
    if data.get('rollup'):
        rollup = data['rollup']
        print(f"\nGenerated rollup:")
        print(f"  Summary: {rollup.get('summary', '')[:200]}")
        print(f"  Key topics: {rollup.get('key_topics', [])[:5]}")
        print(f"  Sentiment: {rollup.get('sentiment')} ({rollup.get('sentiment_score')})")
        print(f"  Messages summarized: {data.get('messages_summarized', 0)}")
    elif data.get('message'):
        print(f"\nInfo: {data.get('message')}")

def test_auto_summarization():
    """Test auto-summarization trigger (requires 20 messages)."""
    print_section("5. Testing Auto-Summarization (Background)")
    
    print("Auto-summarization is triggered automatically after:")
    print("  • 20 new messages in a session")
    print("  • OR 10 minutes of activity")
    print("\nThis happens in the background when you send chat messages.")
    print("Check server logs for: 'Auto-triggering summarization...'")

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  ROBOTO SAI 2026 - Production Endpoints Test")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    
    try:
        # Test 1: Chat history with pagination
        test_chat_history_pagination()
        time.sleep(1)
        
        # Test 2: Sessions endpoint
        test_sessions()
        time.sleep(1)
        
        # Test 3: Get conversation rollup
        session_id = test_conversation_rollup()
        time.sleep(1)
        
        # Test 4: Create/update conversation summary
        test_conversation_summarize(session_id)
        time.sleep(1)
        
        # Test 5: Auto-summarization info
        test_auto_summarization()
        
        print_section("✅ All Tests Complete")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to backend server")
        print(f"   Make sure the server is running at {BASE_URL}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
