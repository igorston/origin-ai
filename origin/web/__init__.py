"""Browser UI: plain HTML/CSS/JS, no build step, no CDN (fully local).

The page is rendered per brand: language, title, favicon and theme colors are written
into index.html, and a JSON block carries the brand's texts and every translation
catalog, so the first paint already shows the right product (no request, no flash).
"""

import html
import json
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from origin import __version__
from origin.branding import Brand, app_locale, brand_dir, get_brand, product_version
from origin.config import Settings, get_settings
from origin.i18n import catalogs, languages

STATIC_DIR = Path(__file__).parent / "static"


def public_brand(brand: Brand) -> dict:
    """What the interface (and GET /api/brand) may see: not the persona prompt."""
    return {
        "product_name": brand.product_name,
        "assistant_name": brand.assistant_name,
        "description": brand.description,
        "logo": {"url": brand.asset_url(brand.logo)}
        if brand.is_image(brand.logo)
        else {"text": brand.logo},
        "welcome": brand.welcome.model_dump(exclude_none=True),
        "links": brand.links,
    }


def _favicon(brand: Brand) -> str:
    for name in (brand.favicon, brand.logo):
        if brand.is_image(name):
            return brand.asset_url(name)
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
        f"<text y='.9em' font-size='90'>{html.escape(brand.logo[:4])}</text></svg>"
    )
    return "data:image/svg+xml," + quote(svg)


def _theme_css(brand: Brand) -> str:
    def block(tokens: dict[str, str]) -> str:
        return " ".join(f"--{name}: {value};" for name, value in tokens.items())

    css = []
    if brand.theme.light:
        css.append(f":root {{ {block(brand.theme.light)} }}")
    if brand.theme.dark:
        css.append(
            f"@media (prefers-color-scheme: dark) {{ :root {{ {block(brand.theme.dark)} }} }}"
        )
    return "\n".join(css)


def render_index(
    settings: Settings,
    user: dict | None = None,
    page: str = "index.html",
    template: Path | None = None,
) -> str:
    """A page of the interface with the brand, the language and the boot data filled in.
    `template` renders a page from another package (a plugin's) the same way: it gets the
    placeholders {{lang}}, {{title}}, {{favicon}}, {{theme}} and {{boot}}, and can import
    /static/i18n.js and the other modules."""
    brand = get_brand(settings)
    locale = app_locale(settings)
    boot = {
        "version": product_version(settings),
        "core_version": __version__,
        "brand": public_brand(brand),
        "locale": locale,
        "languages": languages(),
        "messages": {name: catalog["web"] for name, catalog in catalogs().items()},
        "auth": settings.origin_auth,
        # Internet tools available here (each viewer still turns them on).
        "web": {
            "available": settings.web_access and settings.tools_enabled,
            "provider": {"duckduckgo": "DuckDuckGo", "searxng": "SearXNG", "brave": "Brave Search"}[
                settings.web_search_provider
            ],
        },
        "user": user,
    }
    # Inside <script type="application/json">: "</" must not close the tag.
    boot_json = json.dumps(boot, ensure_ascii=False).replace("</", "<\\/")
    html_page = (template or STATIC_DIR / page).read_text(encoding="utf-8")
    replacements = {
        "{{lang}}": html.escape(locale),
        "{{title}}": html.escape(brand.product_name),
        "{{favicon}}": html.escape(_favicon(brand)),
        "{{theme}}": _theme_css(brand),
        "{{boot}}": boot_json,
    }
    for placeholder, value in replacements.items():
        html_page = html_page.replace(placeholder, value)
    return html_page


def mount_web(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/brand/{name}", include_in_schema=False)
    async def brand_asset(name: str) -> FileResponse:
        # Only images, only by bare file name: the folder also holds brand.json.
        path = brand_dir(get_settings()) / name
        if Path(name).name != name or not Brand.is_image(name) or not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.get("/", include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        # no-cache: always revalidate, so UI edits show up on a plain reload.
        user = getattr(request.state, "user", None)
        page = render_index(get_settings(), user.public() if user else None)
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    @app.get("/login", include_in_schema=False, response_model=None)
    async def login_page() -> HTMLResponse | RedirectResponse:
        settings = get_settings()
        if settings.origin_auth == "off":
            return RedirectResponse("/", status_code=303)
        page = render_index(settings, page="login.html")
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    @app.get("/api/brand", tags=["system"])
    async def brand() -> dict:
        """The white-label brand in use (names, logo, texts, links) and the locale."""
        settings = get_settings()
        return {**public_brand(get_brand(settings)), "locale": app_locale(settings)}
