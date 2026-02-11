"""Quick package availability check — writes results to _pkg_check.json"""
import json, sys, os

result = {}
for pkg in ["torch", "transformers", "peft", "accelerate", "trl", "bitsandbytes"]:
    try:
        m = __import__(pkg)
        result[pkg] = m.__version__
    except Exception as e:
        result[pkg] = f"MISSING: {type(e).__name__}: {e}"

if "torch" in result and not result["torch"].startswith("MISSING"):
    import torch
    result["torch_cuda"] = torch.cuda.is_available()
    result["torch_bf16"] = hasattr(torch, "bfloat16")
    result["torch_ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1) if hasattr(os, "sysconf") else "N/A (Windows)"

result["python"] = sys.version
out = os.path.join(os.path.dirname(__file__), "_pkg_check.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"Results written to {out}")
