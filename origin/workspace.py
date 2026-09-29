"""Workspaces: what each user owns (memory, curator, tools, engine, calibration).

Without authentication there is one workspace, "" (the one Origin always had). With it,
every user gets their own; the first user (id 1) keeps "", so turning authentication on
loses nothing. The chat models are shared: only data and the objects bound to it are
per workspace, built lazily on a user's first request.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from origin.config import Settings
from origin.core import LLMEngine
from origin.integrations import ToolContext, ToolRegistry
from origin.memory import VectorMemory
from origin.memory.calibration import CalibrationStore, MemoryCalibrator, Thresholds
from origin.memory.curator import MemoryCurator
from origin.retry import RetryPolicy

logger = logging.getLogger(__name__)

DEFAULT = ""


@dataclass
class Workspace:
    id: str
    memory: VectorMemory
    curator: MemoryCurator
    engine: LLMEngine
    calibrator: MemoryCalibrator


def collection_name(settings: Settings, space: str) -> str:
    return (
        settings.memory_collection if space == DEFAULT else f"{settings.memory_collection}-{space}"
    )


def calibration_path(settings: Settings, space: str) -> Path:
    path = Path(settings.calibration_path)
    return path if space == DEFAULT else path.with_name(f"{path.stem}-{space}{path.suffix}")


def backup_dir(settings: Settings, space: str) -> Path:
    path = Path(settings.memory_backup_dir)
    return path if space == DEFAULT else path / space


class Workspaces:
    def __init__(
        self,
        settings: Settings,
        memory: VectorMemory,
        judge: BaseChatModel,
        engine: LLMEngine,
    ) -> None:
        """`memory` is the default workspace's (its client and embeddings are shared);
        `engine` a template with the shared chat models."""
        self.settings = settings
        self.base_memory = memory
        self.judge = judge
        self.template = engine
        self._spaces: dict[str, Workspace] = {}
        self._lock = asyncio.Lock()

    async def get(self, space: str = DEFAULT, probe: bool = True) -> Workspace:
        if space in self._spaces:
            return self._spaces[space]
        async with self._lock:
            if space not in self._spaces:
                workspace = self._build(space)
                # Stored calibrated thresholds apply to this workspace's components.
                await workspace.calibrator.bootstrap(probe=probe)
                self._spaces[space] = workspace
                logger.info(
                    "Workspace %r ready (collection %s)", space, workspace.memory.collection
                )
        return self._spaces[space]

    def get_loaded(self, space: str = DEFAULT) -> Workspace:
        """An already built workspace (the default one is built at startup)."""
        return self._spaces[space]

    @property
    def loaded(self) -> list[Workspace]:
        return list(self._spaces.values())

    def _build(self, space: str) -> Workspace:
        s = self.settings
        memory = (
            self.base_memory
            if space == DEFAULT
            else self.base_memory.sibling(collection_name(s, space))
        )
        curator = MemoryCurator(
            memory, self.judge, s.memory_conflict_threshold, retry=RetryPolicy.from_settings(s)
        )
        registry = (
            ToolRegistry.discover(
                ToolContext(s, memory, llm=self.judge, curator=curator, workspace=space),
                disabled=s.tools_disabled,
            )
            if s.tools_enabled
            else ToolRegistry()
        )
        engine = self.template.with_memory(memory, registry.tools)

        def apply_thresholds(thresholds: Thresholds) -> None:
            # Every component that reads a memory threshold, updated in place.
            memory.dedup_threshold = thresholds.dedup
            curator.conflict_threshold = thresholds.conflict
            engine.memory_min_score = thresholds.min_score

        calibrator = MemoryCalibrator(
            memory,
            judge=curator,
            store=CalibrationStore(calibration_path(s, space)),
            embed_model=s.ollama_embed_model,
            chat_model=s.ollama_model,
            defaults=Thresholds(
                dedup=s.memory_dedup_threshold,
                conflict=s.memory_conflict_threshold,
                min_score=s.memory_min_score,
            ),
            apply=apply_thresholds,
            backup_dir=backup_dir(s, space),
        )
        return Workspace(space, memory, curator, engine, calibrator)
