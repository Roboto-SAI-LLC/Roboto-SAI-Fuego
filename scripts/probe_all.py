import urllib.request
import json
import ssl

BACKEND_URL = "https://roboto-sai-backend.onrender.com"
ENDPOINTS = [
    "/api/health",
    "/api/mcp/config",
    "/api/agent/chat"
]

def main():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for ep in ENDPOINTS:
        url = f"{BACKEND_URL}{ep}"
        print(f"Probing {url}...")
        try:
            # For POST endpoints like agent/chat, we just check if it exists (405 or 422 vs 404)
            method = "POST" if "chat" in ep else "GET"
            req = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(req, context=ctx) as response:
                print(f"Status: {response.status} - OK")
        except urllib.error.HTTPError as e:
            print(f"Status: {e.code}")
            if e.code == 404:
                print(f"FAILED: {ep} NOT FOUND")
            elif e.code in [401, 405, 422]:
                print(f"PASSED: {ep} exists (returned {e.code})")
            else:
                print(f"WARNING: {ep} returned {e.code}")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
