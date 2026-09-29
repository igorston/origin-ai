"""Plugin discovery for tools.

Any module inside `origin.integrations.tools` that exposes
`get_tools(ctx: ToolContext) -> list[BaseTool]` is loaded automatically.
Modules whose name starts with `_` are skipped. Installed packages add their own through
the "origin.tools" entry point (see `origin.plugins`).
"""

import importlib
import logging
import pkgutil
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from origin.config import Settings
from origin.memory import VectorMemory
from origin.memory.curator import MemoryCurator
from origin.plugins import plugin_tool_factories

logger = logging.getLogger(__name__)

TOOLS_PACKAGE = "origin.integrations.tools"


@dataclass(frozen=True)
class ToolContext:
    """Shared resources a tool plugin may depend on."""

    settings: Settings
    memory: VectorMemory | None = None
    # Deterministic (temperature 0) model for small classification calls made inside tools.
    llm: BaseChatModel | None = None
    # Shared with the API, so calibrated thresholds apply to tools too.
    curator: MemoryCurator | None = None


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    @property
    def tools(self) -> dict[str, BaseTool]:
        return dict(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    @classmethod
    def discover(
        cls, ctx: ToolContext, package: str = TOOLS_PACKAGE, disabled: Iterable[str] = ()
    ) -> "ToolRegistry":
        disabled = set(disabled)
        registry = cls()
        pkg = importlib.import_module(package)
        for info in pkgutil.iter_modules(pkg.__path__, f"{package}."):
            if info.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            factory = getattr(importlib.import_module(info.name), "get_tools", None)
            if factory is None:
                continue
            for tool in factory(ctx):
                if tool.name not in disabled:
                    registry.register(tool)
        if package == TOOLS_PACKAGE:
            # Installed plugins (the "origin.tools" entry points), after the built-ins: a
            # plugin tool with a built-in's name is refused as a duplicate, not swapped in.
            for _, factory in plugin_tool_factories():
                for tool in factory(ctx):
                    if tool.name not in disabled:
                        registry.register(tool)
        logger.debug("Discovered tools: %s", sorted(registry._tools))
        return registry
