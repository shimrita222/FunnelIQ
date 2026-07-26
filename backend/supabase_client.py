from functools import lru_cache

from supabase import Client, create_client

from backend.config import SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL


@lru_cache
def get_supabase_client() -> Client:
    """Anon-key client — safe for operations scoped by RLS."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_ANON_KEY are not set")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


@lru_cache
def get_supabase_admin_client() -> Client:
    """Service-role client — server-side only, bypasses RLS. Never expose to the browser."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
