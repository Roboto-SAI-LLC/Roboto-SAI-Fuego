"""Quick check of FIM pilot data."""
import json
from pathlib import Path

data_path = Path("apps/model-training/data/fim_pilot.jsonl")
if not data_path.exists():
    print(f"NOT FOUND: {data_path}")
    raise SystemExit(1)

records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"FIM examples: {len(records)}")
print(f"Keys: {list(records[0].keys())}")
has_fim = sum(1 for r in records if "<|fim_prefix|>" in r.get("text", ""))
print(f"With Qwen FIM tokens: {has_fim}/{len(records)}")
langs = {}
for r in records:
    lang = r.get("language", "unknown")
    langs[lang] = langs.get(lang, 0) + 1
print(f"Languages: {langs}")
print("OK")
