"""Migrate legacy Roboto SAI memories from JSON backups to Supabase messages table."""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from datetime import datetime

from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Fixed user_id for Roboto SAI (create this in auth.users table first)
ROBOTO_USER_ID = "roboto-sai-user-uuid"  # TODO: Set to actual UUID

def generate_fingerprint(content: str, emotion: str, timestamp: str) -> str:
    """Generate fingerprint for deduplication."""
    data = f"{content}|{emotion}|{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()

def transform_chat_history(chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform chat_history to messages format."""
    messages = []
    session_counter = 0

    for item in chat_history:
        if not item.get('permanent', True):
            continue  # Skip non-permanent memories

        message = item['message']
        response = item['response']
        emotion = item.get('emotion', '')
        timestamp = item['timestamp']

        # Generate session_id from fingerprint group
        session_id = f"legacy_session_{session_counter}"
        session_counter += 1

        # User message
        user_msg = {
            'user_id': ROBOTO_USER_ID,
            'session_id': session_id,
            'role': 'user',
            'content': message,
            'emotion': emotion if emotion else None,
            'created_at': timestamp,
            'fingerprint': generate_fingerprint(message, emotion, timestamp)
        }

        # Roboto response
        roboto_msg = {
            'user_id': ROBOTO_USER_ID,
            'session_id': session_id,
            'role': 'roboto',
            'content': response,
            'emotion': emotion if emotion else None,
            'created_at': timestamp,
            'fingerprint': generate_fingerprint(response, emotion, timestamp)
        }

        messages.extend([user_msg, roboto_msg])

    return messages

async def migrate_legacy_memory():
    """Main migration function."""
    # Initialize Supabase client
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required")

    supabase: Client = create_client(supabase_url, supabase_key)

    # Path to sai-memory directory
    sai_memory_path = Path(__file__).parent / 'sai-memory'

    if not sai_memory_path.exists():
        # Try absolute path
        sai_memory_path = Path('sai-memory')
        if not sai_memory_path.exists():
            raise FileNotFoundError(f"sai-memory directory not found at {sai_memory_path}")

    total_processed = 0
    total_inserted = 0

    # Process each backup file
    for json_file in sai_memory_path.glob('roboto_backup_*.json'):
        print(f"Processing {json_file.name}...")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chat_history = data.get('chat_history', [])
        if not chat_history:
            print(f"No chat_history in {json_file.name}, skipping")
            continue

        # Transform to messages
        messages = transform_chat_history(chat_history)

        # Check for existing fingerprints
        fingerprints = [msg['fingerprint'] for msg in messages]
        existing_query = supabase.table('messages').select('fingerprint').in_('fingerprint', fingerprints)
        existing = await asyncio.to_thread(lambda: existing_query.execute())
        existing_fps = {row['fingerprint'] for row in existing.data}

        # Filter out duplicates
        new_messages = [msg for msg in messages if msg['fingerprint'] not in existing_fps]

        if new_messages:
            # Insert new messages
            insert_data = [
                {k: v for k, v in msg.items() if k != 'fingerprint'}
                for msg in new_messages
            ]

            result = await asyncio.to_thread(
                lambda: supabase.table('messages').insert(insert_data).execute()
            )

            inserted_count = len(result.data) if result.data else 0
            total_inserted += inserted_count
            print(f"Inserted {inserted_count} messages from {json_file.name}")

        total_processed += len(messages)
        print(f"Processed {len(messages)} total messages from {json_file.name}")

    print(f"\nMigration complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total inserted: {total_inserted}")

    # Verify counts
    total_count = await asyncio.to_thread(
        lambda: supabase.table('messages').select('id', count='exact').eq('user_id', ROBOTO_USER_ID).execute()
    )

    print(f"Total messages for Roboto user: {total_count.count}")

if __name__ == '__main__':
    asyncio.run(migrate_legacy_memory())