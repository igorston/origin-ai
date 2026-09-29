import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from origin.branding import Brand, get_brand, load_brand, system_prompt
from origin.config import Settings
from origin.web import render_index

BRAND = {
    "product_name": "Aurora",
    "assistant_name": "Aurora",
    "description": "Assistente do escritório Silva.",
    "logo": "logo.svg",
    "locale": "en",
    "persona": "Você é a assistente jurídica do escritório Silva.",
    "theme": {"light": {"accent": "#0f766e"}, "dark": {"accent": "#5eead4"}},
    "welcome": {"title": "Olá, sou a Aurora", "suggestions": ["Qual o prazo do processo?"]},
    "links": {"Suporte": "https://example.com/suporte"},
}


@pytest.fixture
def brand_settings(tmp_path: Path) -> Settings:
    (tmp_path / "brand.json").write_text(json.dumps(BRAND), encoding="utf-8")
    (tmp_path / "logo.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
    )
    return Settings(origin_brand_path=str(tmp_path / "brand.json"))


def boot(page: str) -> dict:
    raw = re.search(r'<script id="boot" type="application/json">(.*?)</script>', page, re.S)[1]
    return json.loads(raw)


def test_defaults_without_a_brand_file(tmp_path: Path) -> None:
    brand = load_brand(Settings(origin_brand_path=str(tmp_path / "missing.json")))
    assert (brand.product_name, brand.logo) == ("Origin", "🧬")
    assert system_prompt(brand).startswith("You are Origin, a personal AI assistant")
    assert "About you" not in system_prompt(brand)


def test_brand_names_and_persona_reach_the_system_prompt(brand_settings: Settings) -> None:
    prompt = system_prompt(get_brand(brand_settings))
    assert prompt.startswith("You are Aurora, a personal AI assistant")
    # The persona is added after the rules that make the agent work, not instead of them.
    assert prompt.index("Answer in the same language") < prompt.index("escritório Silva")


@pytest.mark.parametrize(
    "bad",
    [
        "{not json",
        json.dumps({"theme": {"light": {"accent": "red; } body { display: none"}}}),
        json.dumps({"theme": {"light": {"font": "Comic Sans"}}}),  # not a theme token
        json.dumps({"product_name": "  "}),
        json.dumps({"links": {"x": "javascript:alert(1)"}}),
        json.dumps({"links": {"x": "//evil.example/"}}),  # another site, scheme-relative
    ],
)
def test_invalid_brand_files_fall_back_to_the_defaults(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "brand.json"
    path.write_text(bad, encoding="utf-8")
    assert load_brand(Settings(origin_brand_path=str(path))) == Brand()


def test_links_may_point_at_pages_of_this_site(tmp_path: Path) -> None:
    path = tmp_path / "brand.json"
    links = {"Base de conhecimento": "/knowledge", "Site": "https://example.com"}
    path.write_text(json.dumps({"links": links}), encoding="utf-8")
    assert load_brand(Settings(origin_brand_path=str(path))).links == links


def test_the_page_is_rendered_with_the_brand(brand_settings: Settings) -> None:
    page = render_index(brand_settings)
    assert "<title>Aurora</title>" in page and '<html lang="en">' in page
    assert '<link rel="icon" href="/brand/logo.svg" />' in page
    assert "--accent: #0f766e;" in page and "--accent: #5eead4;" in page
    assert "Origin" not in page  # nothing of the default brand leaks into the page
    data = boot(page)
    assert data["brand"]["logo"] == {"url": "/brand/logo.svg"}
    assert data["brand"]["welcome"]["title"] == "Olá, sou a Aurora"
    assert "persona" not in data["brand"]  # the prompt is not sent to browsers
    assert set(data["messages"]) >= {"pt-BR", "en"}


def test_boot_json_cannot_close_its_script_tag(tmp_path: Path) -> None:
    path = tmp_path / "brand.json"
    path.write_text(json.dumps({"product_name": "X</script><script>alert(1)"}), encoding="utf-8")
    page = render_index(Settings(origin_brand_path=str(path)))
    assert "</script><script>alert(1)" not in page
    assert boot(page)["brand"]["product_name"] == "X</script><script>alert(1)"


def test_brand_routes(client: TestClient, brand_settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr("origin.web.get_settings", lambda: brand_settings)
    brand = client.get("/api/brand").json()
    assert brand["product_name"] == "Aurora" and brand["locale"] == "en"
    assert "persona" not in brand
    assert client.get("/brand/logo.svg").status_code == 200
    # Only images, only by name: the folder also holds brand.json (with the persona).
    assert client.get("/brand/brand.json").status_code == 404
    assert client.get("/brand/..%2Fbrand.json").status_code == 404
    assert "<title>Aurora</title>" in client.get("/").text
