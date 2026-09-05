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
