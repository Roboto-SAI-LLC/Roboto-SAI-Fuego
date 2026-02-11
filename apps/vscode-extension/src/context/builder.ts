import * as vscode from 'vscode';

export interface EditorContext {
  languageId: string;
  fileName: string;
  selection: string;
  cursor: {
    line: number;
    character: number;
  };
  symbols: string[];
  diagnostics: string[];
}

export async function buildEditorContext(editor: vscode.TextEditor): Promise<EditorContext> {
  const document = editor.document;
  const selectionText = editor.selection.isEmpty
    ? ''
    : document.getText(editor.selection);

  const symbols = await getSymbols(document);
  const diagnostics = formatDiagnostics(document.uri);

  return {
    languageId: document.languageId,
    fileName: document.fileName,
    selection: selectionText,
    cursor: {
      line: editor.selection.active.line + 1,
      character: editor.selection.active.character + 1
    },
    symbols,
    diagnostics
  };
}

async function getSymbols(document: vscode.TextDocument): Promise<string[]> {
  const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
    'vscode.executeDocumentSymbolProvider',
    document.uri
  );

  if (!symbols || symbols.length === 0) {
    return [];
  }

  const lines: string[] = [];
  const walk = (symbol: vscode.DocumentSymbol, depth: number): void => {
    const kindName = vscode.SymbolKind[symbol.kind] ?? 'Symbol';
    const start = symbol.range.start;
    lines.push(
      `${' '.repeat(depth * 2)}${symbol.name} (${kindName}) [${start.line + 1}:${start.character + 1}]`
    );
    for (const child of symbol.children) {
      walk(child, depth + 1);
    }
  };

  for (const symbol of symbols) {
    walk(symbol, 0);
  }

  return lines;
}

function formatDiagnostics(uri: vscode.Uri): string[] {
  const diagnostics = vscode.languages.getDiagnostics(uri);
  return diagnostics.map((diagnostic) => {
    const start = diagnostic.range.start;
    const severity = vscode.DiagnosticSeverity[diagnostic.severity];
    const code = diagnostic.code ? ` code=${diagnostic.code}` : '';
    return `[${start.line + 1}:${start.character + 1}] ${severity}${code} ${diagnostic.message}`;
  });
}
