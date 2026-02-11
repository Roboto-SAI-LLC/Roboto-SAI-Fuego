import * as vscode from 'vscode';

export interface SecretFinding {
  type: string;
  matchCount: number;
}

export interface SecretScanResult {
  safe: boolean;
  redactedText: string;
  findings: SecretFinding[];
}

const DENYLIST_PATTERNS: RegExp[] = [
  /(^|[\\/])\.env(\.|$)/i,
  /\.pem$/i,
  /\.key$/i,
  /(^|[\\/])id_rsa$/i,
  /\.p12$/i
];

const SECRET_PATTERNS: Array<{
  name: string;
  regex: RegExp;
  redact: (match: string, ...groups: string[]) => string;
}> = [
  {
    name: 'AWS access key id',
    regex: /\b(AKIA|ASIA)[0-9A-Z]{16}\b/g,
    redact: (match) => `${match.slice(0, 4)}****************`
  },
  {
    name: 'AWS secret access key',
    regex: /\baws_secret_access_key\s*[:=]\s*([A-Za-z0-9/+=]{40})/gi,
    redact: (match, key) => match.replace(key, 'REDACTED')
  },
  {
    name: 'GitHub token',
    regex: /\b(gh[opsu]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})\b/g,
    redact: (match) => `${match.slice(0, 6)}...REDACTED`
  },
  {
    name: 'JWT',
    regex: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g,
    redact: () => 'REDACTED_JWT'
  },
  {
    name: 'Private key',
    regex: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----/g,
    redact: () => '-----BEGIN PRIVATE KEY-----\nREDACTED\n-----END PRIVATE KEY-----'
  }
];

const ENV_LINE_REGEX = /^(?!\s*(?:export|set)\s+)[A-Z][A-Z0-9_]{2,}\s*=\s*.+$/gm;
const FIM_TOKENS = {
  prefix: '<fim_prefix>',
  suffix: '<fim_suffix>',
  middle: '<fim_middle>'
};

export function scanTextForSecrets(text: string, filePath?: string): SecretScanResult {
  if (filePath && isDenylistedFile(filePath)) {
    return {
      safe: false,
      redactedText: redactEntireText(text),
      findings: [{ type: 'Denylisted file type', matchCount: 1 }]
    };
  }

  let redactedText = text;
  const findings: SecretFinding[] = [];

  for (const pattern of SECRET_PATTERNS) {
    let matchCount = 0;
    redactedText = redactedText.replace(pattern.regex, (...args) => {
      matchCount += 1;
      return pattern.redact(args[0], ...(args.slice(1) as string[]));
    });

    if (matchCount > 0) {
      findings.push({ type: pattern.name, matchCount });
    }
  }

  const envLines = redactedText.match(ENV_LINE_REGEX) ?? [];
  if (envLines.length >= 2) {
    findings.push({ type: '.env content', matchCount: envLines.length });
    redactedText = redactedText.replace(ENV_LINE_REGEX, (line) => {
      const eqIndex = line.indexOf('=');
      if (eqIndex === -1) {
        return line;
      }
      return `${line.slice(0, eqIndex + 1)} REDACTED`;
    });
  }

  return {
    safe: findings.length === 0,
    redactedText,
    findings
  };
}

export async function confirmSafeToSend(
  scanResult: SecretScanResult,
  sourceLabel: string
): Promise<{ allowed: boolean; redactedText: string }> {
  if (scanResult.safe) {
    return { allowed: true, redactedText: scanResult.redactedText };
  }

  const summary = formatFindingSummary(scanResult.findings);
  const choice = await vscode.window.showWarningMessage(
    `Roboto SAI: Potential secrets detected in ${sourceLabel} (${summary}).`,
    { modal: true },
    'Send redacted',
    'Cancel'
  );

  if (choice !== 'Send redacted') {
    return { allowed: false, redactedText: scanResult.redactedText };
  }

  return { allowed: true, redactedText: scanResult.redactedText };
}

function isDenylistedFile(filePath: string): boolean {
  return DENYLIST_PATTERNS.some((pattern) => pattern.test(filePath));
}

function formatFindingSummary(findings: SecretFinding[]): string {
  const uniqueTypes = Array.from(new Set(findings.map((finding) => finding.type)));
  return uniqueTypes.join(', ');
}

function redactEntireText(text: string): string {
  if (
    text.includes(FIM_TOKENS.prefix) &&
    text.includes(FIM_TOKENS.suffix) &&
    text.includes(FIM_TOKENS.middle)
  ) {
    const headerEnd = text.indexOf(FIM_TOKENS.prefix);
    const header = headerEnd >= 0 ? text.slice(0, headerEnd) : '';
    return `${header}${FIM_TOKENS.prefix}[REDACTED]${FIM_TOKENS.suffix}[REDACTED]${FIM_TOKENS.middle}`;
  }

  if (/---\n[\s\S]+?\n---/m.test(text)) {
    return text.replace(/---\n[\s\S]+?\n---/m, '---\n[REDACTED]\n---');
  }

  return '[REDACTED]';
}
