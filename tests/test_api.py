"""API-тесты через FastAPI TestClient (LLM/эмбеддер замоканы в conftest)."""

AUTH = ("tester", "secretpass123")


def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_requires_auth(client):
    assert client.get("/").status_code == 401


def test_root_wrong_password(client):
    assert client.get("/", auth=("tester", "wrong")).status_code == 401


def test_root_ok_with_auth(client):
    r = client.get("/", auth=AUTH)
    assert r.status_code == 200


def test_chat_requires_auth(client):
    assert client.post("/chat", json={"message": "привет"}).status_code == 401


def test_chat_ok(client):
    r = client.post("/chat", json={"message": "Что такое труд?"}, auth=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert "Толстого" in data["reply"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "Письмо №1, 1880"
    # оригинальный текст фрагмента возвращается
    assert "text" in data["sources"][0]
    # возраст Толстого для показа в «Источниках» (перед cos φ)
    assert data["sources"][0]["age"] == 52
    # id полного текста для модалки /doc
    assert data["sources"][0]["doc"] == "0"


def test_doc_requires_auth(client):
    assert client.get("/doc/0").status_code == 401


def test_doc_404_when_docs_json_missing(client, monkeypatch, tmp_path):
    # нет index/docs.json → 404 с подсказкой (а не падение)
    app = client._app_module
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_docs_cache", None)
    r = client.get("/doc/0", auth=AUTH)
    assert r.status_code == 404
    assert "docs.json" in r.json()["detail"]


def test_doc_ok(client, monkeypatch):
    app = client._app_module
    monkeypatch.setattr(app, "_docs_cache", [
        {"doc": "0", "label": "Письмо №1, 1880",
         "text": "Труд есть условие жизни.",
         "meta": {"type": "письмо", "year": "1880"}},
    ])
    r = client.get("/doc/0", auth=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "Письмо №1, 1880"
    assert data["text"] == "Труд есть условие жизни."
    # несуществующий id
    assert client.get("/doc/5", auth=AUTH).status_code == 404


def test_chat_empty_message_rejected(client):
    # пустое → 422 (min_length=1)
    assert client.post("/chat", json={"message": ""}, auth=AUTH).status_code == 422


def test_chat_too_long_rejected(client):
    big = "а" * 5000
    assert client.post("/chat", json={"message": big}, auth=AUTH).status_code == 422


def test_chat_missing_body(client):
    assert client.post("/chat", auth=AUTH).status_code == 422


def test_favicon_no_auth(client):
    # без файла отдаёт 204, с файлом 200 — оба без авторизации (не 401)
    assert client.get("/favicon.ico").status_code in (200, 204)


def test_display_name_local_from_env(client, monkeypatch):
    """Имя локальной модели в UI берётся из конфига (EMBED_MODEL), а не из metadata."""
    app = client._app_module

    monkeypatch.setenv(
        "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert app._display_name("local", {}) == "paraphrase-multilingual-MiniLM-L12-v2"

    # без env — fallback на model из metadata
    monkeypatch.delenv("EMBED_MODEL")
    assert (
        app._display_name("local", {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"})
        == "paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert app._display_name("local", {}) == "Локальная модель"

    # внешние модели — как раньше, из display_name
    assert app._display_name("qwen3-embedding-8b", {"display_name": "Qwen3 Embedding 8B"}) == "Qwen3 Embedding 8B"
