"""Quick check of FIM pilot data quality."""
import json
import sys
from pathlib import Path

data_path = Path(__file__).parent.parent / "data" / "fim_pilot.jsonl"
if not data_path.exists():
    print(f"ERROR: {data_path} not found")
    sys.exit(1)

lines = [json.loads(l) for l in open(data_path, encoding="utf-8")]
print(f"Total examples: {len(lines)}")
print(f"Keys: {list(lines[0].keys())}")

py_count = sum(1 for l in lines if l.get("language") == "python")
ts_count = sum(1 for l in lines if l.get("language") == "typescript")
print(f"Languages: python={py_count}, typescript={ts_count}")

text_lens = [len(l["text"]) for l in lines]
print(f"Avg text length: {sum(text_lens) // len(text_lens)} chars")
print(f"Min/Max text length: {min(text_lens)} / {max(text_lens)}")

# Check FIM token presence
fim_ok = sum(1 for l in lines if "<|fim_prefix|>" in l["text"] and "<|fim_suffix|>" in l["text"] and "<|fim_middle|>" in l["text"])
print(f"Valid FIM tokens: {fim_ok}/{len(lines)} ({100*fim_ok//len(lines)}%)")

# Sample
print(f"\nSample (first 200 chars): {lines[0]['text'][:200]}")
