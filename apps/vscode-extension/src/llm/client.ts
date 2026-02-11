import axios, { AxiosInstance } from 'axios';

export interface CompletionRequest {
  prompt: string;
  maxTokens: number;
  temperature: number;
  stop?: string[];
}

export class LlmClient {
  private http: AxiosInstance;

  constructor(baseUrl: string) {
    this.http = axios.create({
      baseURL: baseUrl,
      timeout: 30000
    });
  }

  async complete(request: CompletionRequest): Promise<string> {
    const response = await this.http.post('/v1/completions', {
      prompt: request.prompt,
      max_tokens: request.maxTokens,
      temperature: request.temperature,
      stop: request.stop,
      model: 'local'
    });

    const data = response.data as { choices?: Array<{ text?: string }> };
    return data.choices?.[0]?.text ?? '';
  }
}
