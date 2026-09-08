"""Lightweight security posture checks; not a complete security audit."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks: list[tuple[str, str, str]] = []
gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
checks.append(("PASS" if ".env" in gitignore else "FAIL", ".env is gitignored", ""))
env_example = ROOT / ".env.example"
checks.append(("PASS" if env_example.exists() else "FAIL", "security configuration example exists", ""))
main = (ROOT / "backend/app/main.py").read_text(encoding="utf-8")
checks.append(("PASS" if "CORSMiddleware" in main and 'allow_origins=["*"]' not in main else "FAIL", "CORS is not wildcard", ""))
checks.append(("PASS" if "security_headers_middleware" in main else "FAIL", "security headers middleware configured", ""))
validator = ROOT / "backend/app/security/input_validator.py"
checks.append(("PASS" if validator.exists() else "FAIL", "zero-trust image validator exists", ""))
requirements = (ROOT / "backend/requirements.txt").read_text(encoding="utf-8")
checks.append(("WARN" if "slowapi" not in requirements else "PASS", "rate limiting is in-memory MVP mode", "distributed deployments need shared limiting"))
for status, label, note in checks:
    print(f"{status}: {label}{f' ({note})' if note else ''}")
if any(status == "FAIL" for status, _, _ in checks): raise SystemExit(1)