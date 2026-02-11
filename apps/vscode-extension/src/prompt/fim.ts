import { EditorContext } from '../context/builder';

export interface FimInput {
  prefix: string;
  suffix: string;
  languageId: string;
  context: EditorContext;
}

const FIM_TOKENS = {
  prefix: '<|fim_prefix|>',
  suffix: '<|fim_suffix|>',
  middle: '<|fim_middle|>'
};

export function renderFimPrompt(input: FimInput): string {
  // For base code models (e.g. Qwen2.5-Coder), keep the prompt minimal:
  // just a language/file comment followed by raw FIM tokens.
  const header = `// Language: ${input.languageId} | File: ${input.context.fileName}`;

  return `${header}\n${FIM_TOKENS.prefix}${input.prefix}${FIM_TOKENS.suffix}${input.suffix}${FIM_TOKENS.middle}`;
}
