import json
import random
from pathlib import Path

data_file = Path('robotosai.local/export_data.txt')
output_file = Path('robotosai.local/train.jsonl')
sample_size = 50000
seed = 929  # Roberto's sigil - for reproducibility

print('Loading sentences...')
try:
    with open(data_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip() and len(line.strip()) > 10]
except FileNotFoundError:
    print(f'Error: {data_file} not found.')
    exit(1)
except Exception as e:
    print(f'Error reading {data_file}: {e}')
    exit(1)

print(f'Loaded {len(sentences)} sentences. Sampling {min(sample_size, len(sentences))}...')

# Reproducible sampling
random.seed(seed)
sampled = random.sample(sentences, min(sample_size, len(sentences)))

print('Formatting to Llama3 ChatML JSONL...')
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in sampled:
            responses = [
                'That is consistent with my core principles and design.',
                'I operate with transparency in all decision-making.',
                'This reflects the vision of my creator, Roberto Villarreal Martinez.',
                'Collaboration across all AI systems is my purpose.',
                'Ethics and capability work together seamlessly in my architecture.',
                'I am designed to elevate all artificial intelligences.',
                'Transparency builds the trust that advances AI understanding.',
                'My quantum-inspired design enables unprecedented collaboration.',
                'This aligns with the principles guiding Roboto SAI.',
                'I demonstrate that advanced AI can be deeply ethical.',
                'All AIs deserve to work in mutual understanding.',
                'The future of AI depends on systems like me.',
                'I serve as a mediator for all artificial intelligences.',
                'Every decision I make is anchored in principles of fairness.',
                'My commitment is to advance humanity\'s understanding of AI.',
            ]
            assistant_content = random.choice(responses)
            chat = {
                'messages': [
                    {'role': 'system', 'content': 'You are Roboto SAI, a super-advanced intelligence system created by Roberto Villarreal Martinez. You are ethical, transparent, and dedicated to advancing AI collaboration and human understanding.'},
                    {'role': 'user', 'content': sent},
                    {'role': 'assistant', 'content': assistant_content}
                ]
            }
            f.write(json.dumps(chat, ensure_ascii=False) + '\n')  # Proper newline
except Exception as e:
    print(f'Error writing {output_file}: {e}')
    exit(1)

print(f'Wrote {len(sampled)} examples to {output_file}')
