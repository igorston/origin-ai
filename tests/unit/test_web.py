import re

import pytest
from fastapi.testclient import TestClient

from origin.web import STATIC_DIR


def test_index_serves_the_ui(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    assert '<script type="module" src="/static/app.js">' in response.text


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/static/app.js", "javascript"),
        ("/static/api.js", "javascript"),
        ("/static/markdown.js", "javascript"),
        ("/static/styles.css", "text/css"),
    ],
)
def test_static_assets_are_served(client: TestClient, path: str, content_type: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]


def test_ui_is_fully_local() -> None:
    """No CDN or remote asset: Origin must work offline."""
    for file in STATIC_DIR.iterdir():
        text = file.read_text(encoding="utf-8")
        remote = re.findall(r"""(?:src|href)=["']https?://""", text)
        imports = re.findall(r"""^import .* from ["']https?://""", text, re.MULTILINE)
        assert not remote and not imports, f"{file.name} loads remote assets"


def test_ui_does_not_shadow_the_api(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/sessions").status_code == 200
