import * as vscode from 'vscode';
import axios from 'axios';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';

export interface LlamaDaemonOptions {
  modelPath: string;
  port: number;
  serverArgs: string[];
  serverPath?: string;
}

export class LlamaDaemonController implements vscode.Disposable {
  private process?: ChildProcessWithoutNullStreams;
  private output: vscode.OutputChannel;
  private starting?: Promise<void>;
  private disposed = false;

  constructor(private options: LlamaDaemonOptions) {
    this.output = vscode.window.createOutputChannel('Roboto SAI');
  }

  get baseUrl(): string {
    return `http://127.0.0.1:${this.options.port}`;
  }

  async ensureReady(): Promise<void> {
    if (this.disposed) {
      throw new Error('Roboto SAI controller is disposed.');
    }

    if (await this.isHealthy()) {
      return;
    }

    if (!this.starting) {
      this.starting = this.startAndWait().finally(() => {
        this.starting = undefined;
      });
    }

    await this.starting;
  }

  private async startAndWait(): Promise<void> {
    if (!(await this.isHealthy())) {
      this.spawnProcess();
      await this.waitForHealthy();
      await this.warmUp();
    }
  }

  private spawnProcess(): void {
    if (this.process) {
      return;
    }

    if (!this.options.modelPath) {
      throw new Error('Set roboto-sai.modelPath to start llama-server.exe.');
    }

    const serverPath = this.options.serverPath?.trim() || 'llama-server.exe';
    const args = [
      '--host',
      '127.0.0.1',
      '--port',
      String(this.options.port),
      '--model',
      this.options.modelPath,
      ...this.options.serverArgs
    ];

    this.output.appendLine(`Starting llama-server.exe: ${serverPath} ${args.join(' ')}`);
    this.process = spawn(serverPath, args, { windowsHide: true });

    this.process.stdout.on('data', (data) => {
      this.output.appendLine(data.toString());
    });

    this.process.stderr.on('data', (data) => {
      this.output.appendLine(data.toString());
    });

    this.process.on('exit', (code) => {
      this.output.appendLine(`llama-server.exe exited with code ${code ?? 'unknown'}`);
      this.process = undefined;
    });
  }

  private async isHealthy(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseUrl}/v1/models`, { timeout: 800 });
      return response.status >= 200 && response.status < 300;
    } catch (error) {
      try {
        const response = await axios.get(`${this.baseUrl}/health`, { timeout: 800 });
        return response.status >= 200 && response.status < 300;
      } catch {
        return false;
      }
    }
  }

  private async waitForHealthy(): Promise<void> {
    const timeoutMs = 15000;
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
      if (await this.isHealthy()) {
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 350));
    }

    throw new Error('llama-server.exe did not become healthy in time.');
  }

  private async warmUp(): Promise<void> {
    try {
      await axios.post(
        `${this.baseUrl}/v1/completions`,
        {
          prompt: 'ping',
          max_tokens: 1,
          temperature: 0
        },
        { timeout: 3000 }
      );
    } catch (error) {
      this.output.appendLine('Warm-up request failed; continuing without cache.');
    }
  }

  dispose(): void {
    this.disposed = true;

    if (this.process) {
      this.process.kill();
      this.process = undefined;
    }

    this.output.dispose();
  }
}
