import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from origin.api.routes import auth as auth_routes
from origin.auth import UserError, UserStore, tokens
from origin.auth.__main__ import main as cli
from origin.auth.users import hash_password, verify_password
from origin.config import Settings
from origin.workspace import collection_name


def test_passwords() -> None:
    stored = hash_password("correct horse")
    assert stored.startswith("scrypt$") and "correct horse" not in stored
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong horse", stored)
    assert not verify_password("x", "garbage")
    assert hash_password("same") != hash_password("same")  # salted


def test_user_store(tmp_path: Path) -> None:
    users = UserStore(tmp_path / "db.sqlite")
    alice = users.add("alice", "s3cret-pass", is_admin=True)
    assert alice.id == 1 and alice.workspace == ""  # the first user keeps the default data
    bob = users.add("bob", "another-pass")
    assert bob.workspace == "u2"
    with pytest.raises(UserError):
        users.add("ALICE", "whatever-pass")  # names are case-insensitive
    with pytest.raises(UserError):
        users.add("carol", "short")
    assert users.authenticate("alice", "s3cret-pass") == alice
    assert users.authenticate("alice", "nope-nope") is None
    assert users.authenticate("nobody", "s3cret-pass") is None
    users.remove("bob")
    assert [u.username for u in users.all()] == ["alice"]


def test_tokens(tmp_path: Path) -> None:
    users = UserStore(tmp_path / "db.sqlite")
    user = users.add("alice", "s3cret-pass")
    key = b"k" * 32
    token = tokens.issue(user, key, days=1)
    payload = tokens.read(token, key)
    assert payload["uid"] == user.id and tokens.still_valid(payload, user)
    assert tokens.read(token, b"other-key" * 4) is None  # forged
    assert tokens.read(token[:-2] + "xx", key) is None  # tampered
    assert tokens.read(tokens.issue(user, key, days=-1), key) is None  # expired
    users.set_password("alice", "brand-new-pass")
    assert not tokens.still_valid(payload, users.by_id(user.id))  # signed out everywhere


def test_secret_key_is_generated_once(tmp_path: Path) -> None:
    settings = Settings(sqlite_path=str(tmp_path / "origin.db"))
    first = tokens.secret_key(settings)
    assert len(first) > 40 and tokens.secret_key(settings) == first
    assert tokens.secret_key(Settings(origin_secret_key="fixed")) == b"fixed"


def test_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "cli.db"))
    from origin.config import get_settings

    get_settings.cache_clear()
    try:
        monkeypatch.setattr("sys.stdin", io.StringIO("s3cret-pass\n"))
        assert cli(["add-user", "alice", "--admin", "--password-stdin"]) == 0
        assert cli(["list"]) == 0
        assert "alice" in capsys.readouterr().out
        monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
        assert cli(["add-user", "bob", "--password-stdin"]) == 1
        # From a PowerShell pipe: a BOM and \r\n, neither part of the password.
        monkeypatch.setattr("sys.stdin", io.StringIO("\ufeffwindows-pass\r\n"))
        assert cli(["add-user", "carol", "--password-stdin"]) == 0
        assert UserStore(tmp_path / "cli.db").authenticate("carol", "windows-pass")
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------- the API with accounts


@pytest.fixture
def accounts(client: TestClient, tmp_path: Path, monkeypatch) -> UserStore:
    settings = Settings(origin_auth="password", sqlite_path=str(tmp_path / "auth.db"))
    for module in ("origin.auth.middleware", "origin.api.routes.auth", "origin.web"):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings)
    users = UserStore(settings.sqlite_path)
    users.add("alice", "alice-password", is_admin=True)
    users.add("bob", "bob-password")
    monkeypatch.setattr(app.state, "users", users, raising=False)
    monkeypatch.setattr(app.state, "auth_key", b"test-key" * 4, raising=False)
    auth_routes._failures.clear()
    return users


def login(client: TestClient, name: str) -> None:
    response = client.post(
        "/api/auth/login", json={"username": name, "password": f"{name}-password"}
    )
    assert response.status_code == 200, response.text


def test_everything_but_the_login_needs_a_session(client: TestClient, accounts) -> None:
    assert client.get("/sessions").status_code == 401
    assert client.post("/chat", json={"message": "oi"}).status_code == 401
    page = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/login"
    assert 'id="login-form"' in client.get("/login").text
    assert client.get("/api/brand").status_code == 200
    assert client.get("/api/auth/me").json() == {"auth": "password", "user": None}


def test_login_logout(client: TestClient, accounts) -> None:
    assert (
        client.post("/api/auth/login", json={"username": "alice", "password": "x"}).status_code
        == 401
    )
    login(client, "alice")
    cookie = client.cookies.get(tokens.COOKIE)
    assert cookie and client.get("/sessions").status_code == 200
    me = client.get("/api/auth/me").json()["user"]
    assert me == {"id": 1, "username": "alice", "is_admin": True}
    assert '"username": "alice"' in client.get("/").text  # the page knows who is signed in
    client.post("/api/auth/logout")
    assert client.get("/sessions").status_code == 401


def test_brute_force_is_throttled(client: TestClient, accounts) -> None:
    for _ in range(auth_routes.MAX_FAILURES):
        client.post("/api/auth/login", json={"username": "alice", "password": "guess"})
    blocked = client.post(
        "/api/auth/login", json={"username": "alice", "password": "alice-password"}
    )
    assert blocked.status_code == 429


def test_users_only_see_their_own_conversations(client: TestClient, accounts) -> None:
    login(client, "alice")
    mine = client.post("/sessions").json()
    assert mine["owner"] == ""  # alice, the first user, keeps the default workspace
    client.post("/api/auth/logout")

    login(client, "bob")
    assert client.get("/sessions").json() == []
    assert client.get(f"/sessions/{mine['id']}").status_code == 404
    assert client.delete(f"/sessions/{mine['id']}").status_code == 404
    chat = client.post("/chat", json={"message": "oi", "session_id": mine["id"]})
    assert chat.status_code == 404
    bobs = client.post("/sessions").json()
    assert bobs["owner"] == "u2" and [s["id"] for s in client.get("/sessions").json()] == [
        bobs["id"]
    ]


def test_cross_site_requests_and_admin_routes(client: TestClient, accounts) -> None:
    login(client, "bob")
    evil = client.post("/sessions", headers={"origin": "https://evil.example"})
    assert evil.status_code == 403
    assert client.post("/sessions", headers={"origin": "http://testserver"}).status_code == 201
    assert client.post("/setup/pull").status_code == 403  # not an admin


def test_changing_a_password_signs_out(client: TestClient, accounts) -> None:
    login(client, "bob")
    accounts.set_password("bob", "a-new-password")
    assert client.get("/sessions").status_code == 401


async def test_each_user_has_their_own_memory_collection() -> None:
    workspaces = app.state.workspaces
    default, bob = await workspaces.get(""), await workspaces.get("u2")
    assert default.memory.collection == Settings().memory_collection
    assert bob.memory.collection == collection_name(Settings(), "u2") != default.memory.collection
    assert bob.engine.memory is bob.memory and default.engine.memory is default.memory
    assert bob.engine.model is default.engine.model  # chat models are shared
    assert set(bob.engine.tools) == set(default.engine.tools)
    assert bob.engine.tools["remember"] is not default.engine.tools["remember"]


def test_the_login_form_works_without_javascript(client: TestClient, accounts) -> None:
    # A plain form post: the password goes in the body, never in a URL.
    wrong = client.post(
        "/login", data={"username": "alice", "password": "nope"}, follow_redirects=False
    )
    assert wrong.status_code == 303 and wrong.headers["location"] == "/login?error=failed"
    ok = client.post(
        "/login",
        data={"username": "alice", "password": "alice-password"},
        follow_redirects=False,
    )
    assert ok.status_code == 303 and ok.headers["location"] == "/"
    assert client.get("/sessions").status_code == 200
    assert 'method="post" action="/login"' in client.get("/login").text
