import * as vscode from 'vscode';
import { InlineProvider } from './inline/provider';
import { LlamaDaemonController } from './daemon/controller';
import { LlmClient } from './llm/client';
import { registerCommands } from './commands';

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration('roboto-sai');
  const port = config.get<number>('port', 8787);
  const modelPath = config.get<string>('modelPath', '');
  const serverPath = config.get<string>('serverPath', '');
  const serverArgs = config.get<string[]>('serverArgs', []);

  const controller = new LlamaDaemonController({
    modelPath,
    port,
    serverPath,
    serverArgs
  });

  const client = new LlmClient(`http://127.0.0.1:${port}`);
  const inlineProvider = new InlineProvider(controller, client);

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(
      [{ language: 'typescript' }, { language: 'python' }],
      inlineProvider
    )
  );

  registerCommands(context, { client, controller });
  context.subscriptions.push(controller);
}

export function deactivate(): void {
  // Disposed by subscriptions.
}
