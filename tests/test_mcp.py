"""Тесты MCP-сервера (/mcp, streamable HTTP): протокол, авторизация по токену,
инструменты tolstoy_* (эмбеддер/индекс замоканы в conftest)."""
import json
import sys

MCP_AUTH = {"Authorization": "Bearer test-mcp-token"}


def _headers(session_id=None):
    headers = {**MCP_AUTH, "Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if session_id:
        headers["mcp-session-id"] = session_id
    return headers


def _parse(r):
    """Ответ может быть JSON или SSE (text/event-stream) — разбираем оба."""
    if "text/event-stream" in r.headers.get("content-type", ""):
        data = [line[5:].strip() for line in r.text.splitlines()
                if line.startswith("data:")]
        return json.loads(data[-1]) if data else {}
    return r.json()


def _init_mcp(client):
    r = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0.0"},
        },
    }, headers=_headers())
    assert r.status_code == 200, r.text
    body = _parse(r)
    assert body["result"]["serverInfo"]["name"] == "tolstoy"
    session = r.headers.get("mcp-session-id")
    assert session
    # протокольное уведомление после initialize
    client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=_headers(session))
    return session


def _call_tool(client, session, tool, arguments):
    r = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }, headers=_headers(session))
    assert r.status_code == 200, r.text
    return _parse(r)["result"]


def _tool_data(result):
    if result.get("structuredContent"):
        return result["structuredContent"]
    text = result["content"][0]["text"] if result.get("content") else ""
    return json.loads(text) if text else {}


# --- авторизация --------------------------------------------------------------

def test_mcp_requires_token(client):
    assert client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                     "params": {}}, headers={"Content-Type": "application/json"}).status_code == 401
    bad = {**MCP_AUTH, "Authorization": "Bearer wrong-token"}
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                  "params": {}}, headers={**bad, "Content-Type": "application/json"})
    assert r.status_code == 401


def test_mcp_handshake(client):
    session = _init_mcp(client)
    assert session


def test_mcp_tools_list(client):
    session = _init_mcp(client)
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/list",
                                  "params": {}}, headers=_headers(session))
    assert r.status_code == 200, r.text
    names = {t["name"] for t in _parse(r)["result"]["tools"]}
    # ровно 5 инструментов фактуры; инструментов, зовущих LLM, нет
    assert names == {"tolstoy_search", "tolstoy_browse", "tolstoy_facets",
                     "tolstoy_doc", "tolstoy_grep"}


# --- инструменты ---------------------------------------------------------------

def test_mcp_search(client):
    session = _init_mcp(client)
    data = _tool_data(_call_tool(client, session, "tolstoy_search",
                                 {"query": "труд", "top_k": 2}))
    assert data["count"] == 1
    res = data["results"][0]
    assert res["text"] == "Труд есть условие жизни."
    assert res["score"] == 0.7
    assert res["meta"]["type"] == "письмо"
    assert res["meta"]["doc"] == "0"


def test_mcp_search_unknown_type(client):
    session = _init_mcp(client)
    result = _call_tool(client, session, "tolstoy_search",
                        {"query": "труд", "types": ["сонет"]})
    assert result["isError"] is True
    assert "сонет" in result["content"][0]["text"]


def test_mcp_browse(client):
    session = _init_mcp(client)
    data = _tool_data(_call_tool(client, session, "tolstoy_browse", {}))
    assert data["total"] == 1
    assert data["records"][0]["doc"] == "0"
    assert data["records"][0]["addressee"] == "Страхов Николай Николаевич"


def test_mcp_facets(client):
    session = _init_mcp(client)
    data = _tool_data(_call_tool(client, session, "tolstoy_facets", {}))
    assert data["types"]["письмо"] == 1
    assert data["years"] == {"min": "1880", "max": "1880"}


def test_mcp_grep(client):
    session = _init_mcp(client)
    data = _tool_data(_call_tool(client, session, "tolstoy_grep",
                                 {"pattern": "труд"}))
    assert data["count"] == 1
    assert data["results"][0]["matches"] == 1
    assert data["results"][0]["chunk_id"] == "v63_001#0"
    assert data["results"][0]["spans"] == [[0, 4]]
    assert data["results"][0]["text"] == "Труд есть…"
    # full=True пробрасывается в retriever — чанк целиком
    data = _tool_data(_call_tool(client, session, "tolstoy_grep",
                                 {"pattern": "труд", "full": True}))
    assert data["results"][0]["text"] == "Труд есть условие жизни."
    # whole_word: «труд есть» как целое слово не находится (два слова)
    data = _tool_data(_call_tool(client, session, "tolstoy_grep",
                                 {"pattern": "труд есть", "whole_word": True}))
    assert data["count"] == 0


def test_mcp_doc(client, monkeypatch):
    session = _init_mcp(client)
    app = client._app_module
    monkeypatch.setattr(app, "_docs_cache", [
        {"doc": "0", "label": "Письмо №1, 1880",
         "text": "Труд есть условие жизни.",
         "meta": {"type": "письмо", "year": "1880"}},
    ])
    data = _tool_data(_call_tool(client, session, "tolstoy_doc", {"doc_id": 0}))
    assert data["label"] == "Письмо №1, 1880"
    assert data["text"] == "Труд есть условие жизни."
    # несуществующий id → isError с понятным текстом
    result = _call_tool(client, session, "tolstoy_doc", {"doc_id": 5})
    assert result["isError"] is True
    assert "не найден" in result["content"][0]["text"]


# --- выключенный /mcp ----------------------------------------------------------
# Без MCP_API_KEY маршрут не монтируется вообще (404, а не 401) — этот тест
# делает свежий импорт app с пустым токеном; фикстура client его не видит.
def test_mcp_disabled_without_key(monkeypatch):
    monkeypatch.setenv("APP_USER", "tester")
    monkeypatch.setenv("APP_PASS", "secretpass123")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("AUTH_FAIL_DELAY", "0")
    monkeypatch.setenv("MCP_API_KEY", "")

    for m in ("app", "embed", "retriever", "embedders"):
        sys.modules.pop(m, None)

    import app as app_module
    from fastapi.testclient import TestClient
    try:
        c = TestClient(app_module.app)  # без контекст-менеджера lifespan не запускается
        r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                 "params": {}}, headers={"Content-Type": "application/json"})
        assert r.status_code == 404
    finally:
        # не оставляем app с пустым токеном другим тестам (fixture client попнет сам,
        # но чистый выход надёжнее)
        for m in ("app", "embed", "retriever", "embedders"):
            sys.modules.pop(m, None)


def test_mcp_search_min_score(client):
    session = _init_mcp(client)
    # у заглушки единственный фрагмент со score 0.7
    data = _tool_data(_call_tool(client, session, "tolstoy_search",
                                 {"query": "труд", "min_score": 0.6}))
    assert data["count"] == 1
    data = _tool_data(_call_tool(client, session, "tolstoy_search",
                                 {"query": "труд", "min_score": 0.8}))
    assert data["count"] == 0


def test_mcp_grep_offset(client):
    session = _init_mcp(client)
    data = _tool_data(_call_tool(client, session, "tolstoy_grep",
                                 {"pattern": "труд", "offset": 5, "limit": 10}))
    assert data["offset"] == 5 and data["limit"] == 10
