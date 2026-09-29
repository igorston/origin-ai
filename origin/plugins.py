"""Extension points for packages installed next to Origin, declared as entry points.

In the extending package's pyproject.toml:

    [project.entry-points."origin.tools"]
    crm = "acme_origin.tools.crm"          # a module with get_tools(ctx), like the built-ins

    [project.entry-points."origin.app"]
    acme = "acme_origin.app:register"       # register(app, settings): routes, middleware...

A plugin that fails to load stops the app with its name in the error: it was installed
on purpose, and running without it would hide the failure.
"""

import logging
from collections.abc import Callable, Iterator
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from origin.config import Settings

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from origin.integrations.registry import ToolContext

logger = logging.getLogger(__name__)

TOOLS_GROUP = "origin.tools"
APP_GROUP = "origin.app"


class PluginError(RuntimeError):
    pass


def _load(group: str) -> Iterator[tuple[EntryPoint, Any]]:
    for entry_point in sorted(entry_points(group=group), key=lambda ep: ep.name):
        try:
            yield entry_point, entry_point.load()
        except Exception as exc:
            raise PluginError(f"Plugin {entry_point.name!r} ({entry_point.value}): {exc}") from exc


def plugin_tool_factories() -> Iterator[tuple[str, Callable[["ToolContext"], list["BaseTool"]]]]:
    """`(plugin name, get_tools)` for each `origin.tools` entry point."""
    for entry_point, target in _load(TOOLS_GROUP):
        factory = getattr(target, "get_tools", None) if isinstance(target, ModuleType) else target
        if not callable(factory):
            raise PluginError(
                f"Plugin {entry_point.name!r} ({entry_point.value}) has no get_tools(ctx)"
            )
        yield entry_point.name, factory


def load_app_plugins(app: FastAPI, settings: Settings) -> list[str]:
    """Call each `origin.app` plugin's register(app, settings); returns their names."""
    names = []
    for entry_point, register in _load(APP_GROUP):
        try:
            register(app, settings)
        except Exception as exc:
            raise PluginError(f"Plugin {entry_point.name!r} failed to register: {exc}") from exc
        names.append(entry_point.name)
    if names:
        logger.info("App plugins: %s", ", ".join(names))
    return names
