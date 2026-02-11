"""
extract_fim_pilot.py — Generate FIM training examples from local code files.

Walks local directories for .py and .ts files, uses tree-sitter to identify
meaningful AST nodes, and produces random prefix/middle/suffix FIM examples
in Qwen2.5-Coder format for LoRA fine-tuning.

Usage:
    python extract_fim_pilot.py --source-dirs "r:/Repos/Roboto-SAI-2026" "r:/Repos/Eve"
"""

import argparse
import json
import os
import random
import re
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from tree_sitter import Language, Parser

try:
    from tree_sitter_python import language as python_language
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("tree-sitter-python is required") from exc

try:
    from tree_sitter_typescript import language_typescript
except ImportError:
    try:
        from tree_sitter_typescript import language as language_typescript
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("tree-sitter-typescript is required") from exc

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
EOS = "<|endoftext|>"

TOKEN_REGEX = re.compile(rb"\w+|[^\w\s]", re.ASCII)

# Skip patterns for directory/file names
SKIP_DIRS = {
    "node_modules", "venv", "venv_linux", "__pycache__", ".git", ".cache",
    "dist", "build", ".next", ".tox", ".mypy_cache", ".pytest_cache",
    "env", ".env", "site-packages", "egg-info", ".eggs",
}

SKIP_FILE_PATTERNS = [
    re.compile(r"\.min\.(js|ts|css)$"),
    re.compile(r"\.d\.ts$"),       # declaration files
    re.compile(r"\.map$"),         # source maps
    re.compile(r"package-lock\.json$"),
    re.compile(r"bun\.lockb$"),
    re.compile(r"\.lock$"),
]

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PRIVATE) PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ASIA[0-9A-Z]{16}"),
    re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*[0-9a-zA-Z/+]{40}"),
    re.compile(r"(?i)github_token\s*[:=]\s*[0-9a-zA-Z_]{20,}"),
    re.compile(r"ghp_[0-9a-zA-Z]{36}"),
    re.compile(r"gho_[0-9a-zA-Z]{36}"),
    re.compile(r"xox[baprs]-[0-9a-zA-Z-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    re.compile(r"sk_live_[0-9a-zA-Z]{24,}"),
    re.compile(r"(?i)api_key\s*[:=]\s*[0-9a-zA-Z\-_]{20,}"),
]

SKIP_NODE_TYPES = {
    "comment",
    "line_comment",
    "block_comment",
    "program",
    "module",
    "source_file",
}


def _coerce_language(language):
    if isinstance(language, Language):
        return language
    try:
        return Language(language)
    except TypeError:
        return language


def set_parser_language(parser: Parser, language) -> None:
    language = _coerce_language(language)
    if hasattr(parser, "set_language"):
        parser.set_language(language)
    else:
        parser.language = language


def _should_skip_file(filepath: str) -> bool:
    """Check if file matches skip patterns."""
    for pattern in SKIP_FILE_PATTERNS:
        if pattern.search(filepath):
            return True
    return False


def detect_language(filepath: str) -> Optional[str]:
    """Detect language from file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    return None


def walk_local_files(
    source_dirs: List[str],
    languages: List[str],
) -> Generator[Dict, None, None]:
    """Walk local directories and yield file records matching target languages."""
    ext_map = {
        "python": {".py"},
        "typescript": {".ts", ".tsx"},
    }
    target_exts = set()
    for lang in languages:
        target_exts |= ext_map.get(lang, set())

    for source_dir in source_dirs:
        root = Path(source_dir)
        if not root.exists():
            print(f"Warning: source dir does not exist: {source_dir}")
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skip directories in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                ext = Path(filename).suffix.lower()
                if ext not in target_exts:
                    continue
                if _should_skip_file(filepath):
                    continue

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (OSError, IOError):
                    continue

                if not content or len(content) < 100:  # skip tiny files
                    continue

                rel_path = os.path.relpath(filepath, root)
                language = detect_language(filepath)

                yield {
                    "content": content,
                    "path": rel_path.replace("\\", "/"),
                    "language": language,
                    "repo_name": root.name,
                    "source_dir": str(root),
                }


def looks_minified(text: str, path: Optional[str]) -> bool:
    if not text or "\x00" in text:
        return True
    if path and ".min." in path.lower():
        return True
    length = len(text)
    if length < 200:
        return False
    lines = text.splitlines()
    if not lines:
        return True
    max_line = max(len(line) for line in lines)
    if max_line > 1000:
        return True
    if len(lines) <= 2 and length > 2000:
        return True
    whitespace_ratio = sum(1 for ch in text if ch.isspace()) / max(1, length)
    if whitespace_ratio < 0.05:
        return True
    return False


def contains_secrets(text: str) -> bool:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return True
    return False


def token_spans(content: bytes) -> List[Tuple[int, int]]:
    return [(match.start(), match.end()) for match in TOKEN_REGEX.finditer(content)]


def collect_candidate_nodes(
    root,
    spans: Sequence[Tuple[int, int]],
    prefix_range: Tuple[int, int],
    middle_range: Tuple[int, int],
    suffix_range: Tuple[int, int],
) -> List[Tuple[int, int]]:
    starts = [start for start, _ in spans]
    total_tokens = len(starts)
    if total_tokens == 0:
        return []

    candidates: List[Tuple[int, int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if not node.is_named or node.type in SKIP_NODE_TYPES:
            continue
        if node.start_byte >= node.end_byte:
            continue

        start_idx = bisect_left(starts, node.start_byte)
        end_idx = bisect_right(starts, node.end_byte)
        middle_tokens = end_idx - start_idx
        if middle_tokens < middle_range[0] or middle_tokens > middle_range[1]:
            continue
        prefix_tokens = start_idx
        suffix_tokens = total_tokens - end_idx
        if not (prefix_range[0] <= prefix_tokens <= prefix_range[1]):
            continue
        if not (suffix_range[0] <= suffix_tokens <= suffix_range[1]):
            continue

        candidates.append((node.start_byte, node.end_byte))

    return candidates


def build_fim_example(
    content: str,
    parser: Parser,
    prefix_range: Tuple[int, int],
    middle_range: Tuple[int, int],
    suffix_range: Tuple[int, int],
    rng: random.Random,
) -> Optional[Tuple[str, str, str]]:
    content_bytes = content.encode("utf-8", errors="ignore")
    if not content_bytes:
        return None

    tree = parser.parse(content_bytes)
    if tree.root_node.has_error:
        return None

    spans = token_spans(content_bytes)
    if len(spans) < prefix_range[0] + middle_range[0] + suffix_range[0]:
        return None

    candidates = collect_candidate_nodes(tree.root_node, spans, prefix_range, middle_range, suffix_range)
    if not candidates:
        return None

    start_byte, end_byte = rng.choice(candidates)
    prefix = content_bytes[:start_byte].decode("utf-8", errors="ignore")
    middle = content_bytes[start_byte:end_byte].decode("utf-8", errors="ignore")
    suffix = content_bytes[end_byte:].decode("utf-8", errors="ignore")
    if not prefix or not middle or not suffix:
        return None

    return prefix, middle, suffix


def iter_round_robin_local(
    source_dirs: List[str],
    languages: List[str],
    rng: random.Random,
) -> Generator[Tuple[str, Dict], None, None]:
    """
    Walk local dirs, collect all files by language, shuffle, then round-robin yield.
    This ensures balanced language representation.
    """
    by_language: Dict[str, List[Dict]] = {lang: [] for lang in languages}

    for record in walk_local_files(source_dirs, languages):
        lang = record.get("language")
        if lang in by_language:
            by_language[lang].append(record)

    # Shuffle each language's files
    for lang in by_language:
        rng.shuffle(by_language[lang])

    # Report counts
    for lang, files in by_language.items():
        print(f"  {lang}: {len(files)} files")

    # Round-robin
    iterators = {lang: iter(files) for lang, files in by_language.items()}
    active = list(by_language.keys())

    while active:
        for lang in list(active):
            try:
                yield lang, next(iterators[lang])
            except StopIteration:
                active.remove(lang)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract FIM examples from local source code files."
    )
    parser.add_argument(
        "--source-dirs",
        type=str,
        nargs="+",
        default=["r:/Repos/Roboto-SAI-2026", "r:/Repos/Eve"],
        help="Directories to walk for source files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apps/model-training/data/fim_pilot.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Number of FIM examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=929,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--multi-span",
        type=int,
        default=3,
        help="Generate up to N FIM examples per file (different spans).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    python_parser = Parser()
    set_parser_language(python_parser, python_language())

    typescript_parser = Parser()
    set_parser_language(typescript_parser, language_typescript())

    # Relaxed token ranges for local files (many are smaller than Stack v2 files)
    prefix_range = (8, 2048)
    middle_range = (4, 512)
    suffix_range = (4, 2048)

    languages = ["python", "typescript"]
    print(f"Scanning source directories: {args.source_dirs}")

    examples_written = 0
    files_scanned = 0
    files_used = 0

    with args.output.open("w", encoding="utf-8") as output:
        for language, record in iter_round_robin_local(args.source_dirs, languages, rng):
            if examples_written >= args.max_examples:
                break

            files_scanned += 1
            content = record.get("content", "")
            path = record.get("path", "")

            if looks_minified(content, path):
                continue
            if contains_secrets(content):
                continue

            parser = python_parser if language == "python" else typescript_parser

            # Generate multiple FIM examples per file
            examples_from_file = 0
            for _ in range(args.multi_span):
                if examples_written >= args.max_examples:
                    break

                triplet = build_fim_example(
                    content, parser,
                    prefix_range, middle_range, suffix_range,
                    rng,
                )
                if not triplet:
                    break  # no more candidates in this file

                prefix, middle, suffix = triplet
                formatted = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}{EOS}"

                record_out = {
                    "text": formatted,
                    "language": language,
                    "path": path,
                    "repo_name": record.get("repo_name"),
                    "source": "local",
                    "source_dir": record.get("source_dir"),
                }
                output.write(json.dumps(record_out, ensure_ascii=False) + "\n")
                examples_written += 1
                examples_from_file += 1

            if examples_from_file > 0:
                files_used += 1

    print(
        f"\nDone! Wrote {examples_written} FIM examples to {args.output}\n"
        f"  Files scanned: {files_scanned}\n"
        f"  Files used: {files_used}\n"
        f"  Multi-span per file: up to {args.multi_span}"
    )


if __name__ == "__main__":
    main()
