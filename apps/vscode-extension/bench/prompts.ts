export type PromptSize = "short" | "medium" | "long";
export type PromptLanguage = "typescript" | "python";

export interface PromptCase {
  id: string;
  language: PromptLanguage;
  size: PromptSize;
  approxTokens: number;
  prompt: string;
}

const FIM_PREFIX = "<|fim_prefix|>";
const FIM_SUFFIX = "<|fim_suffix|>";
const FIM_MIDDLE = "<|fim_middle|>";

const SIZE_CONFIG: Record<PromptSize, { repeat: number; approxTokens: number }> = {
  short: { repeat: 6, approxTokens: 500 },
  medium: { repeat: 24, approxTokens: 2000 },
  long: { repeat: 48, approxTokens: 4000 }
};

const repeatBlock = (block: string, count: number): string =>
  Array.from({ length: count }, () => block).join("");

const buildFIMPrompt = (prefix: string, suffix: string, middle = ""): string =>
  `${FIM_PREFIX}${prefix}${FIM_SUFFIX}${suffix}${FIM_MIDDLE}${middle}`;

const buildTypeScriptPrompt = (size: PromptSize): PromptCase => {
  const { repeat, approxTokens } = SIZE_CONFIG[size];

  const header = `// File: src/indexer.ts\n` +
    `export interface SymbolInfo {\n` +
    `  name: string;\n` +
    `  kind: "class" | "function" | "interface" | "type" | "enum";\n` +
    `  location: string;\n` +
    `}\n\n` +
    `export interface IndexResult {\n` +
    `  root: string;\n` +
    `  symbols: SymbolInfo[];\n` +
    `  filesIndexed: number;\n` +
    `  elapsedMs: number;\n` +
    `}\n\n` +
    `export class ProjectIndexer {\n` +
    `  constructor(private readonly ignoreGlobs: string[]) {}\n\n` +
    `  public async indexWorkspace(root: string): Promise<IndexResult> {\n` +
    `    const started = Date.now();\n` +
    `    const files = this.collectFiles(root);\n` +
    `    const symbols = this.extractSymbols(files);\n` +
    `    const elapsedMs = Date.now() - started;\n`;

  const filler =
    `\n  private collectFiles(root: string): string[] {\n` +
    `    // Pretend we are walking the file system and applying ignore globs.\n` +
    `    return [\"src/indexer.ts\", \"src/parser.ts\", \"src/cache.ts\"];\n` +
    `  }\n\n` +
    `  private extractSymbols(files: string[]): SymbolInfo[] {\n` +
    `    const symbols: SymbolInfo[] = [];\n` +
    `    for (const file of files) {\n` +
    `      symbols.push({ name: \"ProjectIndexer\", kind: \"class\", location: file });\n` +
    `    }\n` +
    `    return symbols;\n` +
    `  }\n`;

  const holePrefix = `\n    const result: IndexResult = { root, symbols, filesIndexed: files.length, elapsedMs };\n` +
    `    `;

  const holeSuffix =
    `\n    return result;\n` +
    `  }\n\n` +
    `  private buildReport(result: IndexResult): string {\n` +
    `    return [\"Indexed\", result.filesIndexed, \"files in\", result.elapsedMs, \"ms\"].join(\" \");\n` +
    `  }\n` +
    `}\n`;

  const prefix = header + repeatBlock(filler, repeat) + holePrefix;
  const suffix = holeSuffix + repeatBlock(filler, Math.max(1, Math.floor(repeat / 3)));

  return {
    id: `ts-${size}`,
    language: "typescript",
    size,
    approxTokens,
    prompt: buildFIMPrompt(prefix, suffix)
  };
};

const buildPythonPrompt = (size: PromptSize): PromptCase => {
  const { repeat, approxTokens } = SIZE_CONFIG[size];

  const header = `# File: src/indexer.py\n` +
    `from dataclasses import dataclass\n` +
    `from typing import List\n\n` +
    `@dataclass\n` +
    `class SymbolInfo:\n` +
    `    name: str\n` +
    `    kind: str\n` +
    `    location: str\n\n` +
    `@dataclass\n` +
    `class IndexResult:\n` +
    `    root: str\n` +
    `    symbols: List[SymbolInfo]\n` +
    `    files_indexed: int\n` +
    `    elapsed_ms: int\n\n` +
    `class ProjectIndexer:\n` +
    `    def __init__(self, ignore_globs: List[str]) -> None:\n` +
    `        self.ignore_globs = ignore_globs\n\n` +
    `    async def index_workspace(self, root: str) -> IndexResult:\n` +
    `        started = self._now_ms()\n` +
    `        files = self._collect_files(root)\n` +
    `        symbols = self._extract_symbols(files)\n` +
    `        elapsed_ms = self._now_ms() - started\n`;

  const filler =
    `\n    def _collect_files(self, root: str) -> List[str]:\n` +
    `        # Pretend we are walking the file system and applying ignore globs.\n` +
    `        return [\"src/indexer.py\", \"src/parser.py\", \"src/cache.py\"]\n\n` +
    `    def _extract_symbols(self, files: List[str]) -> List[SymbolInfo]:\n` +
    `        symbols: List[SymbolInfo] = []\n` +
    `        for file in files:\n` +
    `            symbols.append(SymbolInfo(name=\"ProjectIndexer\", kind=\"class\", location=file))\n` +
    `        return symbols\n`;

  const holePrefix = `\n        result = IndexResult(root=root, symbols=symbols, files_indexed=len(files), elapsed_ms=elapsed_ms)\n` +
    `        `;

  const holeSuffix =
    `\n        return result\n\n` +
    `    def build_report(self, result: IndexResult) -> str:\n` +
    `        return f\"Indexed {result.files_indexed} files in {result.elapsed_ms} ms\"\n\n` +
    `    def _now_ms(self) -> int:\n` +
    `        import time\n` +
    `        return int(time.time() * 1000)\n`;

  const prefix = header + repeatBlock(filler, repeat) + holePrefix;
  const suffix = holeSuffix + repeatBlock(filler, Math.max(1, Math.floor(repeat / 3)));

  return {
    id: `py-${size}`,
    language: "python",
    size,
    approxTokens,
    prompt: buildFIMPrompt(prefix, suffix)
  };
};

export const prompts: PromptCase[] = [
  buildTypeScriptPrompt("short"),
  buildTypeScriptPrompt("medium"),
  buildTypeScriptPrompt("long"),
  buildPythonPrompt("short"),
  buildPythonPrompt("medium"),
  buildPythonPrompt("long")
];
