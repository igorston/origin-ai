"""White-label branding: product and assistant names, logo, colors, persona, texts.

Everything lives in one JSON file (ORIGIN_BRAND_PATH, default ./brand/brand.json) next to
its images, so a fork or a deployment rebrands the app without touching code. Missing
file or fields fall back to Origin's own defaults. See brand/brand.example.json.
"""

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from origin.config import Settings
from origin.i18n import Translator, resolve_locale

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".gif"}
# CSS custom properties a brand may set (see the :root tokens in styles.css).
THEME_TOKENS = {
    "bg", "surface", "surface-2", "border", "text", "muted", "accent", "accent-contrast",
    "accent-soft", "danger", "ok", "warn", "user-bubble", "user-text", "code-bg", "radius",
}  # fmt: skip
# Values end up inside a <style> block: allow colors, lengths and color functions only.
SAFE_CSS_VALUE = re.compile(r"^[#\w\s().,%-]{1,64}$")


class Theme(BaseModel):
    light: dict[str, str] = {}
    dark: dict[str, str] = {}

    @field_validator("light", "dark")
    @classmethod
    def _safe(cls, tokens: dict[str, str]) -> dict[str, str]:
        unknown = set(tokens) - THEME_TOKENS
        if unknown:
            raise ValueError(f"unknown theme tokens {sorted(unknown)}; use {sorted(THEME_TOKENS)}")
        bad = [f"{k}: {v}" for k, v in tokens.items() if not SAFE_CSS_VALUE.match(v)]
        if bad:
            raise ValueError(f"unsafe theme values: {bad}")
        return tokens


class Welcome(BaseModel):
    """Empty-chat screen. Unset fields use the interface language's defaults."""

    title: str | None = None
    text: str | None = None
    suggestions: list[str] | None = None


class Brand(BaseModel):
    product_name: str = "Origin"
    assistant_name: str = "Origin"
    description: str = "Local-first, modular personal AI assistant."
    # An emoji / short text, or an image file in the brand folder ("logo.svg").
    logo: str = "🧬"
    favicon: str | None = None  # image file in the brand folder; defaults to the logo
    locale: str | None = None  # interface language; defaults to ORIGIN_LOCALE
    # Extra instructions for the assistant: tone, audience, what it is for. Appended to
    # the system prompt (the rules that make the agent work stay in place).
    persona: str = ""
    theme: Theme = Theme()
    welcome: Welcome = Welcome()
    # label -> URL (sidebar footer): http(s), or a path on this site ("/knowledge", a
    # plugin's page), which opens in the same tab.
    links: dict[str, str] = Field(default_factory=dict)

    @field_validator("product_name", "assistant_name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("links")
    @classmethod
    def _http_links(cls, links: dict[str, str]) -> dict[str, str]:
        # "//host" is another site; "javascript:" and friends are refused like any scheme.
        bad = [url for url in links.values() if not re.match(r"^(https?://|/(?!/))", url)]
        if bad:
            raise ValueError(f"links must be http(s) URLs or paths on this site: {bad}")
        return links

    @staticmethod
    def is_image(value: str | None) -> bool:
        return bool(value) and Path(value).suffix.lower() in IMAGE_SUFFIXES

    def asset_url(self, value: str | None) -> str | None:
        return f"/brand/{Path(value).name}" if self.is_image(value) else None


def system_prompt(brand: Brand) -> str:
    """The assistant's system prompt under this brand: its name, plus the persona."""
    from origin.prompts import load_prompt

    prompt = load_prompt("system").format(assistant_name=brand.assistant_name)
    if brand.persona.strip():
        persona = load_prompt("persona").format(persona=brand.persona.strip())
        prompt = f"{prompt}\n\n{persona}"
    return prompt


def app_locale(settings: Settings) -> str:
    """Interface and server-message language: the brand's, else ORIGIN_LOCALE."""
    return resolve_locale(get_brand(settings).locale or settings.origin_locale)


def translator(settings: Settings | None = None) -> Translator:
    if settings is None:
        from origin.config import get_settings

        settings = get_settings()
    return Translator(app_locale(settings))


def brand_dir(settings: Settings) -> Path:
    return Path(settings.origin_brand_path).parent


def load_brand(settings: Settings) -> Brand:
    """The configured brand; an unreadable or invalid file falls back to the defaults
    (with a warning) rather than keeping the app from starting."""
    path = Path(settings.origin_brand_path)
    if not path.exists():
        return Brand()
    try:
        brand = Brand.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring brand file %s: %s", path, exc)
        return Brand()
    for name in (brand.logo, brand.favicon):
        if brand.is_image(name) and not (path.parent / Path(name).name).is_file():
            logger.warning("Brand image %s not found in %s", name, path.parent)
    return brand


_cache: dict[str, Brand] = {}


def get_brand(settings: Settings) -> Brand:
    """Loaded once per brand file (restart to apply edits, like the rest of the config)."""
    if settings.origin_brand_path not in _cache:
        _cache[settings.origin_brand_path] = load_brand(settings)
    return _cache[settings.origin_brand_path]
