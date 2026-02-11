"""Install required ML packages for LoRA pilot training."""
import subprocess
import sys

packages = [
    "transformers>=4.45.0",
    "peft>=0.13.0",
    "accelerate>=1.0.0",
    "trl>=0.12.0",
]

print(f"Python: {sys.executable}")
print(f"Installing: {', '.join(packages)}")

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + packages,
    capture_output=False,
    text=True,
)
sys.exit(result.returncode)
