"""Central configuration: reads settings from .env into one typed, immutable object.

Nothing runs at import time. Call `load_settings()` when you need the config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """All runtime settings for KnowledgeLens."""

    gemini_api_key: str
    gen_model: str
    embed_model: str
    embed_dim: int
    price_in_per_m: float
    price_out_per_m: float
    price_embed_per_m: float
    chunk_size: int
    chunk_overlap: int
    gen_rpm: int
    embed_rpm: int
    docs_dir: Path
    tables_dir: Path
    table_file: Path
    chroma_dir: Path
    cache_dir: Path
    logs_dir: Path

    @property
    def default_collection(self) -> str:
        """Name of the Chroma collection for the configured chunking settings."""
        return collection_name(self.chunk_size, self.chunk_overlap)


def collection_name(chunk_size: int, overlap: int) -> str:
    """Collection names encode the chunking config, e.g. kb_cs1000_ov200."""
    return f"kb_cs{chunk_size}_ov{overlap}"


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from the .env file (if present) and environment variables.

    Raises:
        RuntimeError: if GEMINI_API_KEY is missing or still the placeholder.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    tables_dir = PROJECT_ROOT / "data" / "tables"
    return Settings(
        gemini_api_key=key,
        gen_model=os.getenv("GEN_MODEL", "gemini-3.5-flash-lite"),
        embed_model=os.getenv("EMBED_MODEL", "gemini-embedding-2"),
        embed_dim=int(os.getenv("EMBED_DIM", "768")),
        price_in_per_m=float(os.getenv("PRICE_IN_PER_M", "0.30")),
        price_out_per_m=float(os.getenv("PRICE_OUT_PER_M", "2.50")),
        price_embed_per_m=float(os.getenv("PRICE_EMBED_PER_M", "0.20")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        gen_rpm=int(os.getenv("GEN_RPM", "10")),
        embed_rpm=int(os.getenv("EMBED_RPM", "60")),
        docs_dir=PROJECT_ROOT / "data" / "docs",
        tables_dir=tables_dir,
        table_file=tables_dir / os.getenv("TABLE_FILE", "employees.csv"),
        chroma_dir=PROJECT_ROOT / "chroma_db",
        cache_dir=PROJECT_ROOT / ".cache",
        logs_dir=PROJECT_ROOT / "logs",
    )


def estimate_cost(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of one generation call using paid-tier prices.

    On the free tier you pay nothing; this number tells you what it WOULD cost in production.
    """
    return (
        input_tokens * settings.price_in_per_m + output_tokens * settings.price_out_per_m
    ) / 1_000_000


def estimate_embed_cost(settings: Settings, input_tokens: int) -> float:
    """Estimate the USD cost of embedding `input_tokens` tokens (paid-tier price)."""
    return input_tokens * settings.price_embed_per_m / 1_000_000
