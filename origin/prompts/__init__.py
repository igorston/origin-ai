from functools import cache
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"


@cache
def load_prompt(name: str) -> str:
    """Load a prompt template from `templates/<name>.md`."""
    return (TEMPLATES_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


__all__ = ["TEMPLATES_DIR", "load_prompt"]
