import { performance } from "node:perf_hooks";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { prompts, PromptCase, PromptLanguage, PromptSize } from "./prompts";

type MetricStats = {
  min: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
};

type Sample = {
  requestIndex: number;
  promptId: string;
  language: PromptLanguage;
  size: PromptSize;
  promptChars: number;
  ttfbMs: number;
  firstTokenMs: number;
  totalMs: number;
  completionChars: number;
};

type RunGroup = {
  ttfbMs: MetricStats;
  firstTokenMs: MetricStats;
  totalMs: MetricStats;
  samples: Sample[];
};

type BenchmarkOutput = {
  meta: {
    url: string;
    requests: number;
    maxTokens: number;
    temperature: number;
    stopSequences: string[];
    promptFilter: {
      sizes: PromptSize[];
      languages: PromptLanguage[];
    };
    startedAt: string;
    nodeVersion: string;
  };
  cold: RunGroup;
  warm: RunGroup;
  all: RunGroup;
};

type CliOptions = {
  url: string;
  requests: number;
  maxTokens: number;
  temperature: number;
  outputPath: string | null;
  tablePath: string | null;
  sizes: PromptSize[];
  languages: PromptLanguage[];
};

const DEFAULT_URL = "http://127.0.0.1:8787/v1/completions";
const DEFAULT_REQUESTS = 50;
const DEFAULT_MAX_TOKENS = 128;
const DEFAULT_TEMPERATURE = 0.2;
const STOP_SEQUENCES = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"];

const parseArgs = (argv: string[]): CliOptions => {
  const options: CliOptions = {
    url: DEFAULT_URL,
    requests: DEFAULT_REQUESTS,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: DEFAULT_TEMPERATURE,
    outputPath: null,
    tablePath: null,
    sizes: ["short", "medium", "long"],
    languages: ["typescript", "python"]
  };

  const readNext = (args: string[], index: number): string => {
    if (index + 1 >= args.length) {
      throw new Error(`Missing value for ${args[index]}`);
    }
    return args[index + 1];
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--url":
        options.url = readNext(argv, i);
        i += 1;
        break;
      case "--requests":
      case "-n":
        options.requests = Number(readNext(argv, i));
        i += 1;
        break;
      case "--max-tokens":
        options.maxTokens = Number(readNext(argv, i));
        i += 1;
        break;
      case "--temperature":
        options.temperature = Number(readNext(argv, i));
        i += 1;
        break;
      case "--output":
        options.outputPath = readNext(argv, i);
        i += 1;
        break;
      case "--table":
        options.tablePath = readNext(argv, i);
        i += 1;
        break;
      case "--sizes": {
        const raw = readNext(argv, i);
        options.sizes = raw.split(",").map((value) => value.trim()) as PromptSize[];
        i += 1;
        break;
      }
      case "--languages": {
        const raw = readNext(argv, i);
        options.languages = raw.split(",").map((value) => value.trim()) as PromptLanguage[];
        i += 1;
        break;
      }
      case "--help":
      case "-h":
        console.log("Usage: node bench/dist/benchmark.js [--requests N] [--url URL]");
        process.exit(0);
      default:
        break;
    }
  }

  if (!Number.isFinite(options.requests) || options.requests <= 0) {
    throw new Error("--requests must be a positive number");
  }

  return options;
};

const percentile = (values: number[], pct: number): number => {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil((pct / 100) * sorted.length) - 1;
  const index = Math.min(sorted.length - 1, Math.max(0, rank));
  return sorted[index];
};

const computeStats = (values: number[]): MetricStats => {
  if (values.length === 0) {
    return { min: 0, p50: 0, p95: 0, p99: 0, max: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  return {
    min: sorted[0],
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
    p99: percentile(sorted, 99),
    max: sorted[sorted.length - 1]
  };
};

const formatMs = (value: number): string => value.toFixed(1);

const renderGroupTable = (label: string, group: RunGroup): string => {
  const lines: string[] = [];
  const header = `${label} metrics (ms)`;
  lines.push(header);
  lines.push("Metric           min     p50     p95     p99     max");
  const rows: Array<[string, MetricStats]> = [
    ["TTFB", group.ttfbMs],
    ["First token", group.firstTokenMs],
    ["Total", group.totalMs]
  ];
  for (const [name, stats] of rows) {
    const paddedName = name.padEnd(14, " ");
    lines.push(
      `${paddedName}` +
        `${formatMs(stats.min).padStart(7, " ")}` +
        `${formatMs(stats.p50).padStart(8, " ")}` +
        `${formatMs(stats.p95).padStart(8, " ")}` +
        `${formatMs(stats.p99).padStart(8, " ")}` +
        `${formatMs(stats.max).padStart(8, " ")}`
    );
  }
  return lines.join("\n");
};

const formatTable = (output: BenchmarkOutput): string => {
  const lines: string[] = [];
  lines.push("Roboto SAI llama-server latency benchmark");
  lines.push(
    `URL: ${output.meta.url} | requests: ${output.meta.requests} | max_tokens: ${output.meta.maxTokens} | temp: ${output.meta.temperature}`
  );
  lines.push(
    `Prompt sizes: ${output.meta.promptFilter.sizes.join(", ")} | languages: ${output.meta.promptFilter.languages.join(", ")}`
  );
  lines.push("");
  lines.push(renderGroupTable("Cold (first request)", output.cold));
  lines.push("");
  lines.push(renderGroupTable("Warm (subsequent)", output.warm));
  lines.push("");
  lines.push(renderGroupTable("All requests", output.all));
  return lines.join("\n");
};

const summarizeGroup = (samples: Sample[]): RunGroup => {
  const ttfbMs = samples.map((sample) => sample.ttfbMs);
  const firstTokenMs = samples.map((sample) => sample.firstTokenMs);
  const totalMs = samples.map((sample) => sample.totalMs);
  return {
    ttfbMs: computeStats(ttfbMs),
    firstTokenMs: computeStats(firstTokenMs),
    totalMs: computeStats(totalMs),
    samples
  };
};

const selectPromptCases = (sizes: PromptSize[], languages: PromptLanguage[]): PromptCase[] => {
  const filtered = prompts.filter(
    (prompt) => sizes.includes(prompt.size) && languages.includes(prompt.language)
  );
  if (filtered.length === 0) {
    throw new Error("Prompt filter produced no cases. Check --sizes and --languages.");
  }
  return filtered;
};

const readStreamMetrics = async (response: Response, startTime: number): Promise<{
  ttfbMs: number;
  firstTokenMs: number;
  totalMs: number;
  completionChars: number;
}> => {
  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let firstByteMs = 0;
  let firstTokenMs = 0;
  let completionChars = 0;
  let ttfbRecorded = false;
  let tokenRecorded = false;
  let doneStreaming = false;

  if (!reader) {
    const text = await response.text();
    completionChars = text.length;
    const totalMs = performance.now() - startTime;
    return {
      ttfbMs: totalMs,
      firstTokenMs: totalMs,
      totalMs,
      completionChars
    };
  }

  while (!doneStreaming) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    if (!ttfbRecorded) {
      firstByteMs = performance.now() - startTime;
      ttfbRecorded = true;
    }
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const eventEnd = buffer.indexOf("\n\n");
      if (eventEnd === -1) {
        break;
      }
      const eventBlock = buffer.slice(0, eventEnd);
      buffer = buffer.slice(eventEnd + 2);
      const lines = eventBlock.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) {
          continue;
        }
        const payload = trimmed.slice("data:".length).trim();
        if (payload === "[DONE]") {
          doneStreaming = true;
          break;
        }
        try {
          const parsed = JSON.parse(payload) as {
            choices?: Array<{ text?: string }>;
          };
          const text = parsed.choices?.[0]?.text ?? "";
          if (text.length > 0 && !tokenRecorded) {
            firstTokenMs = performance.now() - startTime;
            tokenRecorded = true;
          }
          completionChars += text.length;
        } catch (error) {
          continue;
        }
      }
      if (doneStreaming) {
        break;
      }
    }
  }

  const totalMs = performance.now() - startTime;
  if (!ttfbRecorded) {
    firstByteMs = totalMs;
  }
  if (!tokenRecorded) {
    firstTokenMs = totalMs;
  }

  return {
    ttfbMs: firstByteMs,
    firstTokenMs,
    totalMs,
    completionChars
  };
};

const runSingleRequest = async (
  url: string,
  promptCase: PromptCase,
  requestIndex: number,
  maxTokens: number,
  temperature: number
): Promise<Sample> => {
  const payload = {
    model: "local-llama",
    prompt: promptCase.prompt,
    max_tokens: maxTokens,
    temperature,
    stop: STOP_SEQUENCES,
    stream: true
  };

  const startTime = performance.now();
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Request failed (${response.status}): ${errorBody}`);
  }

  const metrics = await readStreamMetrics(response, startTime);

  return {
    requestIndex,
    promptId: promptCase.id,
    language: promptCase.language,
    size: promptCase.size,
    promptChars: promptCase.prompt.length,
    ttfbMs: metrics.ttfbMs,
    firstTokenMs: metrics.firstTokenMs,
    totalMs: metrics.totalMs,
    completionChars: metrics.completionChars
  };
};

const main = async (): Promise<void> => {
  const options = parseArgs(process.argv.slice(2));
  const promptCases = selectPromptCases(options.sizes, options.languages);

  const samples: Sample[] = [];
  const startedAt = new Date().toISOString();

  for (let i = 0; i < options.requests; i += 1) {
    const promptCase = promptCases[i % promptCases.length];
    process.stderr.write(`Request ${i + 1}/${options.requests} -> ${promptCase.id}\n`);
    const sample = await runSingleRequest(
      options.url,
      promptCase,
      i,
      options.maxTokens,
      options.temperature
    );
    samples.push(sample);
  }

  const coldSamples = samples.slice(0, 1);
  const warmSamples = samples.slice(1);

  const output: BenchmarkOutput = {
    meta: {
      url: options.url,
      requests: options.requests,
      maxTokens: options.maxTokens,
      temperature: options.temperature,
      stopSequences: STOP_SEQUENCES,
      promptFilter: {
        sizes: options.sizes,
        languages: options.languages
      },
      startedAt,
      nodeVersion: process.version
    },
    cold: summarizeGroup(coldSamples),
    warm: summarizeGroup(warmSamples.length > 0 ? warmSamples : coldSamples),
    all: summarizeGroup(samples)
  };

  const table = formatTable(output);
  console.log(table);
  console.log("\nJSON:\n" + JSON.stringify(output, null, 2));

  if (options.outputPath) {
    const outPath = resolve(options.outputPath);
    writeFileSync(outPath, JSON.stringify(output, null, 2), "utf8");
  }

  if (options.tablePath) {
    const tablePath = resolve(options.tablePath);
    writeFileSync(tablePath, table + "\n", "utf8");
  }
};

main().catch((error) => {
  console.error("Benchmark failed:", error);
  process.exit(1);
});
