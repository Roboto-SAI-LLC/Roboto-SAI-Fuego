"""Quick validation of generated FIM data."""
import json
from pathlib import Path

data = Path("apps/model-training/data/fim_pilot.jsonl").read_text("utf-8").strip().split("\n")
records = [json.loads(line) for line in data]

py = [r for r in records if r["language"] == "python"]
ts = [r for r in records if r["language"] == "typescript"]
print(f"Total: {len(records)} | Python: {len(py)} | TypeScript: {len(ts)}")

avg_len = sum(len(r["text"]) for r in records) / len(records)
print(f"Avg text length: {avg_len:.0f} chars")

FP = "<|fim_prefix|>"
FS = "<|fim_suffix|>"
FM = "<|fim_middle|>"
EOT = "<|endoftext|>"

has_all = sum(1 for r in records if FP in r["text"] and FS in r["text"] and FM in r["text"])
print(f"Records with all FIM tokens: {has_all}/{len(records)}")

# Show first example
first = records[0]
print(f"\nFirst example:")
print(f"  language: {first['language']}")
print(f"  path: {first['path']}")
print(f"  repo: {first['repo_name']}")

text = first["text"]
prefix_end = text.index(FS)
prefix = text[len(FP):prefix_end]
middle_start = text.index(FM) + len(FM)
middle_end = text.index(EOT)
middle = text[middle_start:middle_end]

print(f"  prefix (first 200): {prefix[:200]}...")
print(f"  middle (first 200): {middle[:200]}...")

# Unique repos and paths
repos = set(r["repo_name"] for r in records)
print(f"\nUnique repos: {repos}")
print(f"Unique paths: {len(set(r['path'] for r in records))}")
