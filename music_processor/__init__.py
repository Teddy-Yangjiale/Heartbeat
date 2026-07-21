"""Heartbeat-driven song alignment and region-editing pipeline."""

from .core import (
    MixParams,
    RegionEdit,
    analyze_song_bytes,
    process_music_bytes,
)
from .sync_adapter import (
    SyncServiceConfig,
    adapt_sync_result,
    build_sync_command,
    discover_sync_service,
    run_sync_cli,
)

__all__ = [
    "MixParams",
    "RegionEdit",
    "analyze_song_bytes",
    "process_music_bytes",
    "SyncServiceConfig",
    "adapt_sync_result",
    "build_sync_command",
    "discover_sync_service",
    "run_sync_cli",
]
