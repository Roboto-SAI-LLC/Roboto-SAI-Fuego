import * as vscode from 'vscode';
import { buildEditorContext } from '../context/builder';
import { renderFimPrompt } from '../prompt/fim';
import { LlmClient } from '../llm/client';
import { LlamaDaemonController } from '../daemon/controller';
import { confirmSafeToSend, scanTextForSecrets } from '../security/secret-scanner';

const MAX_PREFIX = 4000;
const MAX_SUFFIX = 2000;
const MAX_TOKENS = 128;

export class InlineProvider implements vscode.InlineCompletionItemProvider {
  constructor(
    private controller: LlamaDaemonController,
    private client: LlmClient
  ) {}

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionItem[] | vscode.InlineCompletionList> {
    if (token.isCancellationRequested) {
      return [];
    }

    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.uri.toString() !== document.uri.toString()) {
      return [];
    }

    const { prefix, suffix } = sliceDocument(document, position);
    const editorContext = await buildEditorContext(editor);
    let prompt = renderFimPrompt({
      prefix,
      suffix,
      context: editorContext,
      languageId: document.languageId
    });

    if (token.isCancellationRequested) {
      return [];
    }

    const scanResult = scanTextForSecrets(prompt, document.fileName);
    const confirmation = await confirmSafeToSend(scanResult, 'inline completion');
    if (!confirmation.allowed) {
      return [];
    }
    prompt = confirmation.redactedText;

    try {
      await this.controller.ensureReady();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      void vscode.window.showErrorMessage(`Roboto SAI: ${message}`);
      return [];
    }

    const completion = await this.client.complete({
      prompt,
      maxTokens: MAX_TOKENS,
      temperature: 0.2,
      stop: ['<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>']
    });

    if (token.isCancellationRequested || completion.trim().length === 0) {
      return [];
    }

    const range = editor.selection.isEmpty
      ? new vscode.Range(position, position)
      : editor.selection;

    return [new vscode.InlineCompletionItem(completion, range)];
  }
}

function sliceDocument(
  document: vscode.TextDocument,
  position: vscode.Position
): { prefix: string; suffix: string } {
  const fullText = document.getText();
  const offset = document.offsetAt(position);
  const rawPrefix = fullText.slice(0, offset);
  const rawSuffix = fullText.slice(offset);

  const prefix = rawPrefix.length > MAX_PREFIX
    ? rawPrefix.slice(-MAX_PREFIX)
    : rawPrefix;
  const suffix = rawSuffix.length > MAX_SUFFIX
    ? rawSuffix.slice(0, MAX_SUFFIX)
    : rawSuffix;

  return { prefix, suffix };
}
