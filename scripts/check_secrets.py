"""Scan tracked/untracked files for leaked tokens/API keys before committing.

Run:  python scripts/check_secrets.py
Exits with code 1 if a token-like string is found (Hugging Face, GitHub, Anthropic, AWS).
"""
import re, subprocess, sys
from pathlib import Path

PATTERNS = {
    "Hugging Face": r"hf_[A-Za-z0-9]{30,}",
    "GitHub":       r"(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}",
    "Anthropic":    r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "AWS":          r"AKIA[0-9A-Z]{16}",
}
MAX_BYTES = 5_000_000

def files():
    try:
        out = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"],
                                      stderr=subprocess.DEVNULL, text=True)
        return [Path(p) for p in out.splitlines()]
    except Exception:
        return [p for p in Path(".").rglob("*") if p.is_file() and ".git" not in p.parts]

def main():
    bad = 0
    for p in files():
        try:
            if not p.is_file() or p.stat().st_size > MAX_BYTES:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name, pat in PATTERNS.items():
                m = re.search(pat, line)
                if m:
                    bad += 1
                    print(f"❌ {p}:{i}  looks like a {name} token: {m.group(0)[:6]}***")
    if bad:
        print(f"\nFound {bad} suspicious match(es). REVOKE the token, remove it from the file (and notebook output), then commit again.")
        sys.exit(1)
    print("✅ No tokens found in the files to be committed.")

if __name__ == "__main__":
    main()
