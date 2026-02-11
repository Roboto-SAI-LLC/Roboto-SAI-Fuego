import http from 'http';
import path from 'path';
import fs from 'fs';
import Module from 'module';

type TestCase = {
  name: string;
  run: () => Promise<void> | void;
};

type TestResult = {
  name: string;
  passed: boolean;
  error?: string;
};

const vscodeMock = {
  window: {
    createOutputChannel: (_name: string) => ({
      appendLine: (_message: string) => undefined,
      dispose: () => undefined
    }),
    showWarningMessage: async () => 'Send redacted'
  },
  commands: {
    executeCommand: async () => [
      {
        name: 'doThing',
        kind: 12,
        range: {
          start: { line: 1, character: 2 }
        },
        children: []
      }
    ]
  },
  languages: {
    getDiagnostics: () => [
      {
        range: { start: { line: 4, character: 0 } },
        severity: 0,
        code: 'E100',
        message: 'Example error'
      }
    ]
  },
  DiagnosticSeverity: {
    0: 'Error',
    1: 'Warning',
    2: 'Information',
    3: 'Hint',
    Error: 0,
    Warning: 1,
    Information: 2,
    Hint: 3
  },
  SymbolKind: {
    12: 'Function',
    5: 'Class',
    Function: 12,
    Class: 5
  }
};

// Hook Module._resolveFilename to redirect 'vscode' to our mock.
// This works on Node v24+ where _load is read-only.
const originalResolveFilename = (Module as any)._resolveFilename;
(Module as any)._resolveFilename = function (
  request: string,
  parent: any,
  isMain: boolean,
  options: any
) {
  if (request === 'vscode') {
    // Return a sentinel path that we intercept in the require cache
    const mockPath = path.join(__dirname, '__vscode_mock__.js');
    return mockPath;
  }
  return originalResolveFilename.call(this, request, parent, isMain, options);
};

// Inject the mock into require.cache under our sentinel path
const mockPath = path.join(__dirname, '__vscode_mock__.js');
const mockModule = new Module(mockPath) as any;
mockModule.exports = vscodeMock;
mockModule.loaded = true;
require.cache[mockPath] = mockModule;

const controllerModulePath = path.join(__dirname, '..', '..', 'dist', 'daemon', 'controller');
const clientModulePath = path.join(__dirname, '..', '..', 'dist', 'llm', 'client');
const fimModulePath = path.join(__dirname, '..', '..', 'dist', 'prompt', 'fim');
const contextModulePath = path.join(__dirname, '..', '..', 'dist', 'context', 'builder');
const secretModulePath = path.join(__dirname, '..', '..', 'dist', 'security', 'secret-scanner');

let LlamaDaemonController: new (options: {
  modelPath: string;
  port: number;
  serverArgs: string[];
  serverPath?: string;
}) => { ensureReady: () => Promise<void>; dispose: () => void };
let LlmClient: new (baseUrl: string) => {
  complete: (request: {
    prompt: string;
    maxTokens: number;
    temperature: number;
    stop?: string[];
  }) => Promise<string>;
};
let renderFimPrompt: (input: {
  prefix: string;
  suffix: string;
  languageId: string;
  context: {
    languageId: string;
    fileName: string;
    selection: string;
    cursor: { line: number; character: number };
    symbols: string[];
    diagnostics: string[];
  };
}) => string;
let buildEditorContext: (editor: unknown) => Promise<{
  languageId: string;
  fileName: string;
  selection: string;
  cursor: { line: number; character: number };
  symbols: string[];
  diagnostics: string[];
}>;
let scanTextForSecrets: (text: string, filePath?: string) => {
  safe: boolean;
  redactedText: string;
  findings: Array<{ type: string; matchCount: number }>;
};

async function loadModules(): Promise<void> {
  // Use require() so the vscode mock resolution hook is active
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const controllerModule = require(controllerModulePath);
  const clientModule = require(clientModulePath);
  const fimModule = require(fimModulePath);
  const contextModule = require(contextModulePath);
  const secretModule = require(secretModulePath);

  LlamaDaemonController = controllerModule.LlamaDaemonController;
  LlmClient = clientModule.LlmClient;
  renderFimPrompt = fimModule.renderFimPrompt;
  buildEditorContext = contextModule.buildEditorContext;
  scanTextForSecrets = secretModule.scanTextForSecrets;
}

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function requestJson(
  method: 'GET' | 'POST',
  pathName: string,
  body?: Record<string, unknown>
): Promise<{ status: number; json: unknown }> {
  return new Promise((resolve, reject) => {
    const payload = body ? JSON.stringify(body) : '';
    const req = http.request(
      {
        hostname: '127.0.0.1',
        port: 8787,
        method,
        path: pathName,
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload)
        }
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk.toString();
        });
        res.on('end', () => {
          const json = data.length ? JSON.parse(data) : null;
          resolve({ status: res.statusCode ?? 0, json });
        });
      }
    );

    req.on('error', reject);
    if (payload.length > 0) {
      req.write(payload);
    }
    req.end();
  });
}

async function waitForServer(timeoutMs: number): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const response = await requestJson('GET', '/v1/models');
      if (response.status >= 200 && response.status < 300) {
        return;
      }
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
  }
  throw new Error('Mock server did not respond in time.');
}

async function runTests(testCases: TestCase[]): Promise<TestResult[]> {
  const results: TestResult[] = [];
  for (const testCase of testCases) {
    try {
      await testCase.run();
      results.push({ name: testCase.name, passed: true });
      process.stdout.write(`PASS: ${testCase.name}\n`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      results.push({ name: testCase.name, passed: false, error: message });
      process.stdout.write(`FAIL: ${testCase.name} - ${message}\n`);
    }
  }
  return results;
}

async function main(): Promise<void> {
  await waitForServer(5000);
  await loadModules();

  const tests: TestCase[] = [
    {
      name: 'Daemon controller health check (models ok)',
      run: async () => {
        await requestJson('POST', '/__mock__/config', { modelsStatus: 200, healthStatus: 200 });
        const controller = new LlamaDaemonController({
          modelPath: '',
          port: 8787,
          serverArgs: []
        });
        await controller.ensureReady();
        controller.dispose();
      }
    },
    {
      name: 'Daemon controller health fallback (/health)',
      run: async () => {
        await requestJson('POST', '/__mock__/config', { modelsStatus: 500, healthStatus: 200 });
        const controller = new LlamaDaemonController({
          modelPath: '',
          port: 8787,
          serverArgs: []
        });
        await controller.ensureReady();
        controller.dispose();
        await requestJson('POST', '/__mock__/config', { modelsStatus: 200, healthStatus: 200 });
      }
    },
    {
      name: 'LLM client completion response',
      run: async () => {
        await requestJson('POST', '/__mock__/config', { completionText: '/* ok */' });
        const client = new LlmClient('http://127.0.0.1:8787');
        const result = await client.complete({
          prompt: 'hello',
          maxTokens: 4,
          temperature: 0
        });
        assert(result === '/* ok */', 'Unexpected completion text.');
      }
    },
    {
      name: 'FIM prompt formatting',
      run: () => {
        const prompt = renderFimPrompt({
          prefix: 'const a =',
          suffix: '42;',
          languageId: 'typescript',
          context: {
            languageId: 'typescript',
            fileName: 'sample.ts',
            selection: '',
            cursor: { line: 1, character: 1 },
            symbols: ['foo (Function) [1:1]'],
            diagnostics: []
          }
        });

        assert(prompt.includes('Language: typescript'), 'Missing language header.');
        assert(prompt.includes('File: sample.ts'), 'Missing file header.');
        assert(prompt.includes('<|fim_prefix|>const a =<|fim_suffix|>42;<|fim_middle|>'), 'FIM tokens not ordered.');
      }
    },
    {
      name: 'Secret scanner allows clean text',
      run: () => {
        const input = 'const value = 123;';
        const scan = scanTextForSecrets(input, 'sample.ts');
        assert(scan.safe, 'Expected clean text to be safe.');
        assert(scan.redactedText === input, 'Clean text should remain unchanged.');
      }
    },
    {
      name: 'Secret scanner redacts AWS key',
      run: () => {
        const input = 'AKIA0123456789ABCDEF should be hidden';
        const scan = scanTextForSecrets(input, 'sample.ts');
        assert(!scan.safe, 'Expected AWS key to be flagged.');
        assert(
          scan.redactedText.includes('AKIA****************'),
          'AWS key should be redacted.'
        );
      }
    },
    {
      name: 'Context builder output shape',
      run: async () => {
        const editor = {
          document: {
            languageId: 'typescript',
            fileName: 'sample.ts',
            uri: { fsPath: 'sample.ts' },
            getText: (_selection?: unknown) => 'selected text'
          },
          selection: {
            isEmpty: false,
            active: { line: 2, character: 4 }
          }
        };

        const context = await buildEditorContext(editor);
        assert(context.languageId === 'typescript', 'Incorrect languageId.');
        assert(context.fileName === 'sample.ts', 'Incorrect fileName.');
        assert(context.selection === 'selected text', 'Incorrect selection text.');
        assert(context.cursor.line === 3 && context.cursor.character === 5, 'Incorrect cursor.');
        assert(context.symbols.length === 1, 'Expected one symbol.');
        assert(context.diagnostics.length === 1, 'Expected one diagnostic.');
        assert(
          context.symbols[0].includes('doThing (Function) [2:3]'),
          'Symbol formatting mismatch.'
        );
        assert(
          context.diagnostics[0].includes('[5:1] Error code=E100 Example error'),
          'Diagnostics formatting mismatch.'
        );
      }
    }
  ];

  const results = await runTests(tests);
  const failed = results.filter((result) => !result.passed);
  const passed = results.length - failed.length;

  process.stdout.write(`\n${passed}/${results.length} tests passed.\n`);
  if (failed.length > 0) {
    process.exitCode = 1;
  }
}

void main();
