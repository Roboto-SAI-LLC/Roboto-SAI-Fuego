import * as vscode from 'vscode';
import { buildEditorContext } from '../context/builder';
import { LlmClient } from '../llm/client';
import { LlamaDaemonController } from '../daemon/controller';
import { confirmSafeToSend, scanTextForSecrets } from '../security/secret-scanner';

export interface CommandDeps {
  client: LlmClient;
  controller: LlamaDaemonController;
}

export function registerCommands(
  context: vscode.ExtensionContext,
  deps: CommandDeps
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('roboto-sai.suggest', () =>
      runCommand('Suggest', deps)
    ),
    vscode.commands.registerCommand('roboto-sai.refactor', () =>
      runCommand('Refactor', deps)
    ),
    vscode.commands.registerCommand('roboto-sai.generateDocs', () =>
      runCommand('Generate Docs', deps)
    )
  );
}

async function runCommand(
  mode: 'Suggest' | 'Refactor' | 'Generate Docs',
  deps: CommandDeps
): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    void vscode.window.showInformationMessage('Roboto SAI: No active editor.');
    return;
  }

  const document = editor.document;
  const selectionText = editor.selection.isEmpty
    ? document.lineAt(editor.selection.active.line).text
    : document.getText(editor.selection);

  const editorContext = await buildEditorContext(editor);
  const prompt = buildCommandPrompt(mode, editorContext, selectionText);

  const scanResult = scanTextForSecrets(prompt, document.fileName);
  const confirmation = await confirmSafeToSend(scanResult, `${mode} command`);
  if (!confirmation.allowed) {
    return;
  }

  try {
    await deps.controller.ensureReady();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    void vscode.window.showErrorMessage(`Roboto SAI: ${message}`);
    return;
  }

  const result = await deps.client.complete({
    prompt: confirmation.redactedText,
    maxTokens: 256,
    temperature: 0.2
  });

  if (!result.trim()) {
    void vscode.window.showInformationMessage('Roboto SAI: No response returned.');
    return;
  }

  const outputDoc = await vscode.workspace.openTextDocument({
    content: result.trim(),
    language: document.languageId
  });

  await vscode.window.showTextDocument(outputDoc, {
    viewColumn: vscode.ViewColumn.Beside,
    preview: true
  });
}

function buildCommandPrompt(
  mode: 'Suggest' | 'Refactor' | 'Generate Docs',
  context: Awaited<ReturnType<typeof buildEditorContext>>,
  selection: string
): string {
  const directive =
    mode === 'Suggest'
      ? 'Suggest an improvement or completion for the code.'
      : mode === 'Refactor'
        ? 'Refactor the code to improve readability and maintain behavior.'
        : 'Generate concise documentation comments for the code.';

  const scope = selection.trim().length > 0 ? selection : 'No selection provided.';

  return [
    'You are Roboto SAI, a local-only code assistant.',
    `Task: ${directive}`,
    `Language: ${context.languageId}`,
    `File: ${context.fileName}`,
    context.diagnostics.length > 0
      ? `Diagnostics:\n${context.diagnostics.join('\n')}`
      : 'Diagnostics: none',
    context.symbols.length > 0
      ? `Symbols:\n${context.symbols.join('\n')}`
      : 'Symbols: none',
    'Return only the result without markdown.',
    '---',
    scope,
    '---'
  ].join('\n');
}
