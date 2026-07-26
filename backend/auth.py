from fastapi import Header, HTTPException

from backend.supabase_client import get_supabase_admin_client


def get_current_user(authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        response = get_supabase_admin_client().auth.get_user(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    if not response or not response.user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return response.user
