import json
import re
from pathlib import Path

data_dir = Path('export_data')
output_file = Path('eve.local/export_data.txt')
output_file.parent.mkdir(parents=True, exist_ok=True)

sentences = set()  # Deduplicate

def extract_text(obj):
    text = ''
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ['content', 'text', 'message', 'bio', 'description']:
                text += str(v) + ' '
            elif isinstance(v, (dict, list)):
                text += extract_text(v) + ' '
    elif isinstance(obj, list):
        for item in obj:
            text += extract_text(item) + ' '
    else:
        text += str(obj) + ' '
    return text.strip()

# Process JSON files
for json_path in data_dir.rglob('*.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        full_text = extract_text(data)
        sents = re.split(r'(?<=[.!?])\s+', full_text)
        for s in sents:
            stripped = s.strip()
            if len(stripped) > 10:
                sentences.add(stripped)
    except Exception as e:
        print(f'Skipped JSON {json_path}: {e}')

# Process content files with fallback encoding
for content_path in data_dir.rglob('content*'):  # Broader match
    try:
        # Try UTF-8 first
        with open(content_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except UnicodeDecodeError:
        try:
            # Fallback to UTF-16
            with open(content_path, 'r', encoding='utf-16') as f:
                full_text = f.read()
        except Exception:
            print(f'Skipped binary/non-text {content_path}')
            continue
    except Exception as e:
        print(f'Skipped {content_path}: {e}')
        continue

    sents = re.split(r'(?<=[.!?])\s+', full_text)
    for s in sents:
        stripped = s.strip()
        if len(stripped) > 10:
            sentences.add(stripped)

# Write with proper newlines
with open(output_file, 'w', encoding='utf-8') as f:
    for sent in sorted(sentences):
        f.write(sent + '\n')

print(f'Extracted {len(sentences)} unique sentences to {output_file}')
