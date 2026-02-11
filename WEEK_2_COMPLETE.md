# Week 2 Day 1 - Execution Complete

**Date:** February 10, 2026  
**Status:** Core pipeline verified. Manual firewall step remains.

---

## ✅ Completed Tasks

### 1. Model Switch: 8B → 3B ✅

| Metric | Before (8B) | After (3B) | Target |
|--------|------------|-----------|--------|
| **Model** | Llama-3-8B-Instruct | **Qwen2.5-Coder-3B** | - |
| **Size** | 4,693 MB | **1,841 MB** | - |
| **Cold start** | ~20 sec | **~2 sec** | - |
| **First-token p50** | 384 ms | **158 ms** | **< 300 ms ✅** |
| **First-token p95** | 605 ms | **191 ms** | **< 300 ms ✅** |

**Result:** Target met. 158ms median is **47% under the 300ms requirement**.

#### Files Modified (5):
- [apps/vscode-extension/src/prompt/fim.ts](apps/vscode-extension/src/prompt/fim.ts) — Qwen FIM tokens + simplified prompt
- [apps/vscode-extension/src/inline/provider.ts](apps/vscode-extension/src/inline/provider.ts) — Stop sequences
- [apps/vscode-extension/bench/benchmark.ts](apps/vscode-extension/bench/benchmark.ts) — Stop sequences
- [apps/vscode-extension/bench/prompts.ts](apps/vscode-extension/bench/prompts.ts) — FIM token constants
- [apps/vscode-extension/test/smoke.ts](apps/vscode-extension/test/smoke.ts) — FIM assertion

#### Quality Verified:
```typescript
// Input:
function add(a: number, b: number): number {
  <|fim_middle|>
}

// Output:
return a + b;
```

#### Tests:
- Extension build: **PASS**
- Smoke tests: **7/7 PASS**
- Benchmark: **5 requests, all < 300ms first-token**

---

### 2. FIM Training Data Generation ✅

**Script:** [apps/model-training/pipeline/extract_fim_pilot.py](apps/model-training/pipeline/extract_fim_pilot.py)

**Output:** `apps/model-training/data/fim_pilot.jsonl`

| Metric | Value |
|--------|-------|
| **Total examples** | 1,000 |
| **Python** | 570 |
| **TypeScript** | 430 |
| **Avg chars/example** | 4,561 |
| **Unique source files** | 200 |
| **FIM token format** | Qwen (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) |
| **Validation** | 100% valid |

**Sources:**
- `Roboto-SAI-2026`: 122 Python + 42 TypeScript files
- `Eve`: 101 Python + 127 TypeScript files

**Sample:**
```json
{
  "text": "# File: backend/services/grok_client.py\n<|fim_prefix|>class GrokClient:\n    def __init__(self, api_key: str):\n        self.api_key = api_key\n    <|fim_suffix|>\n    def chat(self, messages: list) -> str:\n        return self._request('/chat', messages)<|fim_middle|>\n    def _request(self, endpoint: str, payload: dict) -> dict:\n        # Make HTTP request\n        pass",
  "language": "python",
  "source": "backend/services/grok_client.py"
}
```

---

### 3. LoRA Fine-Tuning ⚠️ DEFERRED (GPU Required)

**Status:** Pipeline created but execution deferred.

**Reason:** CPU-only training with 3B model + LoRA on 16GB RAM is impractical:
- Model load: 6.18 GB (bfloat16)
- Training overhead: 4-6 GB
- Speed: ~10-20 min per training step
- Risk: OOM crashes

**Script created:** [apps/model-training/pipeline/train_lora_fim.py](apps/model-training/pipeline/train_lora_fim.py)

**Recommendation:** Run on GPU instance (DigitalOcean GPU droplet, RunPod, Lambda Labs)

**When you have GPU access:**
```powershell
cd r:\Repos\Roboto-SAI-2026
python apps/model-training/pipeline/train_lora_fim.py
```

This will:
- Load Qwen2.5-Coder-3B base model
- Apply LoRA adapters (r=16, α=32, dropout=0.05)
- Train on 1000 FIM examples for 100 steps (~30 min on GPU)
- Save to `apps/model-training/models/roboto-sai-fim-lora/`

**For beta:** Ship with the base Qwen2.5-Coder-3B model (158ms already beats target)

---

## 🔒 Remaining Task: Firewall Setup (CRITICAL)

**Script:** [scripts/setup-firewall.ps1](scripts/setup-firewall.ps1)

**Security requirement:** Before using the extension, the firewall must be configured to:
- ✅ Allow llama-server on **127.0.0.1:8787 only**
- ❌ Block all external network access from Code.exe + child processes

### How to Execute (Run Once):

1. **Open PowerShell as Administrator:**
   - Press Win+X → "Windows PowerShell (Admin)"

2. **Navigate to scripts directory:**
   ```powershell
   cd r:\Repos\Roboto-SAI-2026\scripts
   ```

3. **Audit current state (optional):**
   ```powershell
   .\setup-firewall.ps1 -Audit
   ```

4. **Apply firewall rules:**
   ```powershell
   .\setup-firewall.ps1
   ```

5. **Verify rules were created:**
   ```powershell
   .\verify-firewall.ps1
   ```

**What it does:**
- Creates 28 firewall rules (4 per program × 7 programs)
- Programs: Code.exe, Code - Insiders.exe, llama-server.exe, node.exe, git.exe, python.exe, powershell.exe
- Rules: Inbound Allow (loopback), Inbound Block (non-loopback), Outbound Allow (loopback), Outbound Block (non-loopback)

**To rollback (if needed):**
```powershell
.\setup-firewall.ps1 -Rollback
```

---

## 📦 Week 2 Deliverables Summary

| Item | Status | Location |
|------|--------|----------|
| VS Code extension | ✅ Built + tested | [apps/vscode-extension/](apps/vscode-extension/) |
| 3B GGUF model | ✅ Downloaded | [apps/model-training/models/Qwen2.5-Coder-3B-Q4_K_M.gguf](apps/model-training/models/Qwen2.5-Coder-3B-Q4_K_M.gguf) (1.84 GB) |
| FIM training data | ✅ Generated | [apps/model-training/data/fim_pilot.jsonl](apps/model-training/data/fim_pilot.jsonl) (1000 examples) |
| Latency benchmark | ✅ Passed | 158ms p50 (target: <300ms) |
| Smoke tests | ✅ Passed | 7/7 tests |
| Landing page | ✅ Built | [apps/landing/dist/](apps/landing/dist/) |
| Firewall script | ✅ Ready | [scripts/setup-firewall.ps1](scripts/setup-firewall.ps1) — **NEEDS MANUAL RUN** |
| LoRA pipeline | ⚠️ Deferred | [apps/model-training/pipeline/train_lora_fim.py](apps/model-training/pipeline/train_lora_fim.py) — requires GPU |

---

## 🚀 Next Steps (Week 3)

### For Private Beta Launch:

1. **Run firewall setup** (manual, see above)
2. **Package extension:**
   ```powershell
   cd apps/vscode-extension
   npm install -g @vscode/vsce
   vsce package
   ```
   This creates `roboto-sai-0.0.1.vsix`

3. **Deploy landing page to GitHub Pages:**
   ```powershell
   cd apps/landing
   # Copy dist/ contents to gh-pages branch
   ```

4. **Create private GitHub release:**
   - Upload `roboto-sai-0.0.1.vsix`
   - Upload `Qwen2.5-Coder-3B-Q4_K_M.gguf` (chunked if needed)
   - Upload `llama-server.exe` binaries
   - Write installation README

5. **Beta tester instructions:**
   - Download `.vsix` + `llama-server.exe` + model GGUF
   - Install extension in VS Code
   - Configure settings:
     ```json
     {
       "roboto-sai.modelPath": "C:\\path\\to\\Qwen2.5-Coder-3B-Q4_K_M.gguf",
       "roboto-sai.serverPath": "C:\\path\\to\\llama-server.exe"
     }
     ```
   - Run firewall script (Admin PS)
   - Test in TypeScript/Python files

### Optional (Post-Beta):

- Fine-tune Qwen2.5-Coder-3B with LoRA on GPU (1000 FIM examples)
- Add medium/long prompt benchmarks
- Implement prompt caching (`--cache-prompt`) for repeat prefixes
- Reduce `--ctx-size` from 4096 → 2048 for faster TTFB
- Add support for more languages (Go, Rust, etc.)

---

## ⚠️ Risk Items

1. **Qwen License:** qwen-research license (not standard OSS). Review before commercial distribution. Fallback: `stable-code-3b`.

2. **Firewall gaps:** Script covers 7 programs but not:
   - `pwsh.exe` (PowerShell Core)
   - `curl.exe` / `wget.exe`
   - WSL processes
   - Container runtimes

3. **HuggingFace gated datasets:** `bigcode/the-stack-v2` requires manual approval. Used local code instead for FIM pilot.

---

## 🎯 CEO Decision Record

- **D1-A:** HuggingFace auth first ✅
- **D2-A:** Managed form provider for waitlist ✅ (Formspree placeholder in landing)
- **D3-A:** Firewall rewrite critical ✅ (script ready, needs manual run)
- **D4-B:** Drop to 3B model for <300ms target ✅ (Qwen2.5-Coder-3B selected)

---

## 📊 Final Metrics

| Milestone | Target | Actual | Status |
|-----------|--------|--------|--------|
| First-token latency | < 300ms | **158ms** | ✅ 47% under |
| Model size | < 5GB | **1.84 GB** | ✅ 63% under |
| Cold start | < 30s | **2s** | ✅ 93% under |
| Extension build | Pass | **Pass** | ✅ |
| Smoke tests | 7/7 | **7/7** | ✅ |
| FIM data | 100+ examples | **1000** | ✅ 10× target |

---

**Status:** Ready for beta after manual firewall setup.

**Orchestrator sign-off:** All automated tasks complete. Firewall requires elevated permissions (manual CEO action).
