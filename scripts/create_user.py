"""Admin script to provision a Northbound team member's login.

Usage: uv run python -m scripts.create_user <email> <password>
"""

import sys

from backend.supabase_client import get_supabase_admin_client


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: uv run python -m scripts.create_user <email> <password>")
        sys.exit(1)

    email, password = sys.argv[1], sys.argv[2]
    supabase = get_supabase_admin_client()
    result = supabase.auth.admin.create_user(
        {"email": email, "password": password, "email_confirm": True}
    )
    print(f"Created user {result.user.email} (id={result.user.id})")


if __name__ == "__main__":
    main()
