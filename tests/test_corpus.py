"""Тесты парсинга/чанкинга корпуса (scripts/corpus.py) — без сети и модели."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import corpus  # noqa: E402


def test_parse_meta_basic():
    line = "### META | vol=18 | type=письмо | year=1849 | num=4 | to=С. Н. Толстому"
    meta = corpus._parse_meta(line)
    assert meta["vol"] == "18"
    assert meta["type"] == "письмо"
    assert meta["year"] == "1849"
    assert meta["num"] == "4"
    assert meta["to"] == "С. Н. Толстому"


def test_parse_meta_ignores_garbage():
    meta = corpus._parse_meta("### META | vol=1 | brokenpart | k=v")
    assert meta["vol"] == "1"
    assert meta["k"] == "v"
    assert "brokenpart" not in meta


def test_chunk_paragraphs_respects_size():
    paras = ["к" * 500, "о" * 500, "т" * 500, "!" * 500]
    chunks = corpus._chunk_paragraphs(paras)
    assert len(chunks) >= 2  # 2000 симв не влезут в один чанк ~1200
    # ни один чанк не должен быть абсурдно большим
    assert all(len(c) <= corpus.CHUNK_CHARS * 2 for c in chunks)


def test_chunk_single_paragraph():
    chunks = corpus._chunk_paragraphs(["короткий абзац"])
    assert chunks == ["короткий абзац"]


def test_split_long_paragraph():
    para = ". ".join(["Предложение номер %d" % i for i in range(200)]) + "."
    parts = corpus._split_long(para)
    assert len(parts) > 1
    assert all(len(p) <= corpus.CHUNK_CHARS * 1.2 for p in parts)


def test_split_long_overlap_between_parts():
    # перекрытие: последнее предложение куска повторяется первым в следующем
    para = ". ".join(["Предложение номер %d" % i for i in range(100)]) + "."
    parts = corpus._split_long(para)
    assert len(parts) > 1
    assert all(len(p) <= corpus.CHUNK_CHARS * 1.2 for p in parts)
    for a, b in zip(parts, parts[1:]):
        last_a = a.split(". ")[-1].strip().rstrip(".")
        first_b = b.split(". ")[0].strip().rstrip(".")
        assert last_a == first_b
    # ничего не потеряно: каждое предложение есть хотя бы в одном куске
    joined = ". ".join(parts)
    for i in range(100):
        assert "Предложение номер %d" % i in joined


def test_split_long_giant_sentence_not_lost():
    # предложение длиннее чанка целиком попадает в один кусок
    giant = "А" * (corpus.CHUNK_CHARS + 200)
    parts = corpus._split_long(giant + ". И короткое предложение.")
    assert any(giant in p for p in parts)
    assert any("короткое" in p for p in parts)


def test_chunk_paragraphs_overlap_no_oversize():
    # абзацы размером с чанк: перекрытие не влезает — чанки не должны
    # разрастаться за счёт дубля перекрытия
    paras = ["п" * 400, "р" * 400, "о" * 400]
    chunks = corpus._chunk_paragraphs(paras)
    assert all(len(c) <= corpus.CHUNK_CHARS * 1.5 for c in chunks)
    # текст не потерян
    assert all(p in "|".join(chunks) for p in paras)


def test_source_label_letter():
    meta = {"type": "письмо", "year": "1849", "num": "4", "to": "С. Н. Толстому"}
    label = corpus.source_label(meta)
    assert "Письмо" in label
    assert "4" in label
    assert "С. Н. Толстому" in label
    assert "1849" in label


def test_source_label_diary():
    meta = {"type": "дневник", "year": "1886"}
    label = corpus.source_label(meta)
    assert "Дневник" in label
    assert "1886" in label


def test_is_junk_record_filters_reference_matter():
    # справочный аппарат изданий — мусор для RAG
    assert corpus._is_junk_record({"entry": "УКАЗАТЕЛЬ СОБСТВЕННЫХ ИМЕН", "type": "дневник"})
    assert corpus._is_junk_record({"title": "Содержание (из 48-го тома)", "type": "дневник"})
    assert corpus._is_junk_record({"to": "ОГЛАВЛЕНИЕ", "type": "дневник"})
    assert corpus._is_junk_record({"entry": "указатель имён", "type": "дневник"})
    # обычные записи не фильтруются
    assert not corpus._is_junk_record({"entry": "1877 г.", "to": "1877 г.", "type": "дневник"})
    assert not corpus._is_junk_record({"to": "С. Н. Толстому", "type": "письмо"})
    assert not corpus._is_junk_record({"entry": "Примечания", "type": "дневник"})
    assert not corpus._is_junk_record({})
