# Roboto SAI v0.0.1 Beta Release

## 🚀 Private Beta Launch

**Date:** February 11, 2026

This is the first private beta release of Roboto SAI, featuring the complete offline AI coding assistant for VS Code.

## 📦 What's Included

### Core Components
- **VS Code Extension** (`roboto-sai-0.0.1.vsix`) - 28KB
- **Roboto SAI Qwen2.5-Coder-3B Model** (`Roboto-SAI-Qwen2.5-Coder-3B-Q4_K_M.gguf`) - 1.84GB 
- **Llama Server Binary** (`llama-server.exe`) - 10MB

### Performance Benchmarks
- **First-token latency:** 158ms (p50), 191ms (p95)
- **Cold start time:** ~2 seconds
- **Model size:** 1.84GB (63% smaller than 8B alternative)
- **Extension smoke tests:** 7/7 passing

## 🛠️ Installation Instructions

### Prerequisites
- Windows 10/11 with Administrator privileges
- VS Code or VS Code Insiders
- At least 8GB RAM (recommended 16GB+)

### Step 1: Install Extension
1. Download `roboto-sai-0.0.1.vsix`
2. Open VS Code
3. Go to Extensions → Install from VSIX
4. Select the downloaded `.vsix` file

### Step 2: Configure Paths
Open VS Code settings (Ctrl+,) and set:
```json
{
  "roboto-sai.modelPath": "C:\\path\\to\\Roboto-SAI-Qwen2.5-Coder-3B-Q4_K_M.gguf",
  "roboto-sai.serverPath": "C:\\path\\to\\llama-server.exe"
}
```

### Step 3: Configure Firewall (Admin Required)
Run PowerShell as Administrator:
```powershell
# Navigate to your Roboto-SAI directory
cd path\to\Roboto-SAI-2026\scripts
.\setup-firewall-simple.ps1
```

This creates minimal firewall rules allowing llama-server on localhost only.

### Step 4: Test Installation
1. Open a TypeScript or Python file
2. Type a comment like `// Function to add two numbers`
3. Press Tab - Roboto SAI should complete the function

## 🎯 Beta Features

- **Offline-first:** No internet required after setup
- **FIM completion:** Fill-in-the-middle code completion
- **VS Code integration:** Native extension experience  
- **Multi-language:** TypeScript, Python, JavaScript support
- **Low latency:** <200ms first-token response

## 🐛 Known Issues

- Firewall setup requires Administrator privileges
- Model loading takes ~2 seconds on first use
- Currently supports Windows only (Linux/Mac coming in v0.1.0)

## 📞 Support

For beta feedback and issues:
- Create GitHub issue in Roboto-SAI-Fuego repository
- Include VS Code version, Windows version, and error logs

---

**Installation Guide:** See repository README for detailed setup instructions.