#!/usr/bin/env python3
"""
Recover messages from Supabase and local SQLite, export them, and rehydrate the QuantumEnhancedMemorySystem.
This script attempts both sources and is idempotent (uses DB dedupe).
"""
import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# Ensure repository root is on sys.path so imports like `backend.*` work when this
# script is executed directly (python backend/scripts/recover_and_rehydrate.py).
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))
except Exception:
    pass

OUT_DIR = Path("exports")
OUT_DIR.mkdir(exist_ok=True)


def export_supabase():
    try:
        from backend.utils.supabase_client import get_supabase_client
    except Exception as e:
        print("Supabase client import failed:", e)
        return []

    client = None
    try:
        client = get_supabase_client()
    except Exception as e:
        print("get_supabase_client() failed:", e)
        client = None

    if not client:
        print("Supabase not configured or unavailable.")
        return []

    try:
        resp = client.table('messages').select('*').order('created_at').execute()
        rows = resp.data or []
        out = OUT_DIR / 'messages_supabase_export.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Supabase: exported {len(rows)} rows to {out}")
        return rows
    except Exception as e:
        print('Supabase export error:', e)
        traceback.print_exc()
        return []


def export_sqlite():
    try:
        from backend.persistent_memory_store import get_persistent_store
    except Exception as e:
        print('PersistentMemoryStore import failed:', e)
        return []

    try:
        store = get_persistent_store()
        if not store:
            print('No persistent store available')
            return []
        exported = store.export_to_json()
        print('SQLite exported files:', exported)
        # also dump row data to a single file
        cnt = store.get_conversation_count()
        rows = store.list_recent_conversations(limit=cnt or 10000)
        out = OUT_DIR / 'messages_sqlite_export.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"SQLite: exported {len(rows)} conversations to {out}")
        return rows
    except Exception as e:
        print('SQLite export error:', e)
        traceback.print_exc()
        return []


def rehydrate_from_sqlite(rows, memsys):
    count = 0
    for c in rows:
        try:
            user_input = c.get('user_input') or c.get('message') or ''
            response = c.get('response') or c.get('bot') or ''
            emotion = c.get('emotion') or 'neutral'
            mem_id = memsys.add_episodic_memory(user_input, response, emotion)
            if mem_id:
                count += 1
        except Exception:
            print('Failed to add sqlite conv', traceback.format_exc())
    print(f'Rehydrated {count} conversations from SQLite into memory')
    return count


def rehydrate_from_supabase(rows, memsys):
    # Group by session_id, pair user -> next assistant
    sessions = defaultdict(list)
    for r in rows:
        sid = r.get('session_id') or 'global'
        sessions[sid].append(r)

    count = 0
    for sid, msglist in sessions.items():
        try:
            msglist.sort(key=lambda x: x.get('created_at') or '')
        except Exception:
            pass
        i = 0
        while i < len(msglist) - 1:
            a = msglist[i]
            b = msglist[i+1]
            if (a.get('role') == 'user') and (b.get('role') in ('roboto', 'assistant')):
                try:
                    memsys.add_episodic_memory(a.get('content',''), b.get('content',''), a.get('emotion') or 'neutral')
                    count += 1
                except Exception:
                    print('Failed adding supabase pair', traceback.format_exc())
                i += 2
            else:
                i += 1
    print(f'Rehydrated {count} user->assistant pairs from Supabase into memory')
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', action='store_true', help='Perform writes (will save memory file and DB writes)')
    args = parser.parse_args()

    summary = {
        'supabase_rows': 0,
        'sqlite_rows': 0,
        'rehydrated_from_supabase': 0,
        'rehydrated_from_sqlite': 0,
    }

    rows_supabase = []
    rows_sqlite = []

    # Export both sources (best-effort)
    try:
        rows_supabase = export_supabase()
        summary['supabase_rows'] = len(rows_supabase)
    except Exception as e:
        print('Supabase export step failed:', e)

    try:
        rows_sqlite = export_sqlite()
        summary['sqlite_rows'] = len(rows_sqlite)
    except Exception as e:
        print('SQLite export step failed:', e)

    # Attempt to rehydrate using memory system
    try:
        from backend.memory_system import QuantumEnhancedMemorySystem
        memsys = QuantumEnhancedMemorySystem()
        # Rehydrate from sqlite first
        if rows_sqlite:
            summary['rehydrated_from_sqlite'] = rehydrate_from_sqlite(rows_sqlite, memsys)
        # Then from supabase
        if rows_supabase:
            summary['rehydrated_from_supabase'] = rehydrate_from_supabase(rows_supabase, memsys)
        # Save memory (if commit)
        if args.commit:
            try:
                memsys.save_memory()
                print('Memory saved to', memsys.memory_file)
            except Exception:
                print('Failed to save memory:', traceback.format_exc())
    except Exception as e:
        print('Memory system unavailable or rehydration failed:', e)
        traceback.print_exc()

    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
