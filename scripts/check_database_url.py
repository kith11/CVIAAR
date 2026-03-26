from __future__ import annotations

import os
from urllib.parse import urlparse


def normalize(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def main() -> int:
    raw = normalize(os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL"))
    if not raw:
        print("DATABASE_URL is missing.")
        return 1

    parsed = urlparse(raw)
    username = parsed.username or ""
    host = parsed.hostname or ""

    print(f"scheme: {parsed.scheme}")
    print(f"host: {host}")
    print(f"port: {parsed.port}")
    print(f"database: {parsed.path.lstrip('/') or '(missing)'}")
    print(f"username_present: {'yes' if username else 'no'}")
    print(f"username_preview: {username[:24]}{'...' if len(username) > 24 else ''}")

    if ".pooler.supabase.com" in host:
        print("mode: Supabase pooler")
        if "." not in username:
            print("hint: pooler usernames are usually project-specific, for example postgres.[PROJECT_REF]")
        else:
            print("hint: verify the project ref in the username matches the current Supabase project")
    elif host.startswith("db.") and host.endswith(".supabase.co"):
        print("mode: direct database host")
        print("hint: ensure this exact host/password pair still matches the Supabase Connect panel")
    else:
        print("mode: generic postgres")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
