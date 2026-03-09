"""Utility functions for interacting with a Supabase database."""
import logging
from typing import Any, Dict, List, Union

from config import SUPABASE_URL, SUPABASE_KEY

try:
    from supabase import create_client, Client
except ImportError:
    raise ImportError("supabase package not installed. Please run 'pip install supabase' or update requirements.txt")

logger = logging.getLogger(__name__)

_client: Union[Client, None] = None

def get_supabase_client() -> Client:
    """Return a cached Supabase client, initializing if necessary."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    return _client

def insert_rows(table: str, data: List[Dict[str, Any]]) -> None:
    """Insert rows into a Supabase table."""
    client = get_supabase_client()
    try:
        response = client.table(table).insert(data).execute()
        logger.info(f"Inserted {len(data)} rows into {table}")
    except Exception as e:
        logger.error(f"Failed to insert rows: {e}")
        raise