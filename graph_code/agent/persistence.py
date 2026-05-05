"""LangGraph persistence backend factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from ..config import Config


def create_checkpointer(config: Config) -> Any:
    """Create a LangGraph checkpointer from config."""
    backend = config.checkpoint_backend.lower()
    if backend == "memory":
        return InMemorySaver()
    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointing requires installing langgraph-checkpoint-sqlite."
            ) from exc
        uri = config.checkpoint_uri or str(Path(config.working_path) / ".agent" / "checkpoints.sqlite")
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(uri)
    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointing requires installing langgraph-checkpoint-postgres."
            ) from exc
        if not config.checkpoint_uri:
            raise RuntimeError("CHECKPOINT_URI is required for Postgres checkpointing.")
        return PostgresSaver.from_conn_string(config.checkpoint_uri)
    raise ValueError(f"Unsupported checkpoint backend: {config.checkpoint_backend}")


def create_store(config: Config) -> Any:
    """Create a LangGraph store from config."""
    backend = config.store_backend.lower()
    if backend == "memory":
        return InMemoryStore()
    if backend == "postgres":
        try:
            from langgraph.store.postgres import PostgresStore
        except ImportError as exc:
            raise RuntimeError(
                "Postgres store requires installing the LangGraph Postgres store package."
            ) from exc
        if not config.store_uri:
            raise RuntimeError("STORE_URI is required for Postgres store.")
        return PostgresStore.from_conn_string(config.store_uri)
    if backend == "sqlite":
        raise RuntimeError(
            "LangGraph does not provide a stable SQLite store in this environment; "
            "use memory for development or Postgres for production store."
        )
    raise ValueError(f"Unsupported store backend: {config.store_backend}")
