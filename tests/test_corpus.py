"""Тесты парсинга/чанкинга корпуса (scripts/corpus.py) — без сети и модели."""
import os
import sys

import pytest

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
    # 1 абзац = 1 чанк: четыре абзаца по 500 символов (меньше порога сплита
    # CHUNK_CHARS*1.5) дают ровно 4 чанка, каждый равен своему абзацу
    paras = ["к" * 500, "о" * 500, "т" * 500, "!" * 500]
    chunks = corpus._chunk_paragraphs(paras)
    assert chunks == paras


def test_chunk_one_para_per_chunk_no_duplicates():
    # абзацы не склеиваются и не дублируются (раньше было перекрытие)
    paras = ["п" * 400, "р" * 400, "о" * 400]
    chunks = corpus._chunk_paragraphs(paras)
    assert chunks == paras


def test_chunk_single_paragraph():
    chunks = corpus._chunk_paragraphs(["короткий абзац"])
    assert chunks == ["короткий абзац"]


def test_chunk_short_para_merged_with_next():
    # абзац короче MIN_CHARS приклеивается к следующему, чтобы не выпал
    chunks = corpus._chunk_paragraphs(["короткий", "д" * 300])
    assert chunks == ["короткий\n\n" + "д" * 300]


def test_chunk_short_para_at_end_merged_with_previous():
    chunks = corpus._chunk_paragraphs(["д" * 300, "короткий"])
    assert chunks == ["д" * 300 + "\n\nкороткий"]


def test_chunk_long_tail_of_shorts_own_chunk():
    # хвост из коротких ≥ MIN_CHARS не раздувает предыдущий чанк сверх окна
    chunks = corpus._chunk_paragraphs(["к" * 650, "х" * 20, "у" * 20])
    assert chunks == ["к" * 650, "х" * 20 + "\n\n" + "у" * 20]


def test_chunk_run_of_short_paras_merged():
    chunks = corpus._chunk_paragraphs(["а" * 10, "б" * 10, "в" * 300])
    assert chunks == ["а" * 10 + "\n\n" + "б" * 10 + "\n\n" + "в" * 300]


def test_chunk_merge_with_long_para_gets_split_no_loss():
    # короткий + длинный (режется по предложениям): ничего не потеряно,
    # ни одно предложение не дублируется
    short = "короткий"
    long_para = ". ".join(["Предложение номер %d" % i for i in range(80)]) + "."
    chunks = corpus._chunk_paragraphs([short, long_para])
    assert len(chunks) >= 2
    assert short in " ".join(chunks)
    for i in range(80):
        sent = "Предложение номер %d." % i
        assert sum(1 for c in chunks if sent in c) == 1


def test_chunk_all_chunks_at_least_min_chars():
    paras = ["д" * 300, "х" * 10, "ж" * 300, "у" * 15, "з" * 100, "ц" * 5]
    chunks = corpus._chunk_paragraphs(paras)
    assert all(len(c) >= corpus.MIN_CHARS for c in chunks)


def test_split_long_paragraph():
    para = ". ".join(["Предложение номер %d" % i for i in range(200)]) + "."
    parts = corpus._split_long(para)
    assert len(parts) > 1
    assert all(len(p) <= corpus.CHUNK_CHARS * 1.2 for p in parts)


def test_split_long_no_overlap_between_parts():
    # без перекрытия: каждое предложение попадает ровно в один кусок,
    # порядок сохраняется
    para = ". ".join(["Предложение номер %d" % i for i in range(100)]) + "."
    parts = corpus._split_long(para)
    assert len(parts) > 1
    assert all(len(p) <= corpus.CHUNK_CHARS * 1.2 for p in parts)
    for i in range(100):
        sent = "Предложение номер %d." % i
        assert sum(1 for p in parts if sent in p) == 1
    joined = " ".join(parts)
    assert joined.find("Предложение номер 0.") < joined.find("Предложение номер 99.")


def test_split_long_giant_sentence_not_lost():
    # предложение длиннее чанка целиком попадает в один кусок
    giant = "А" * (corpus.CHUNK_CHARS + 200)
    parts = corpus._split_long(giant + ". И короткое предложение.")
    assert any(giant in p for p in parts)
    assert any("короткое" in p for p in parts)


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


# --- фильтр подписей/дат -----------------------------------------------------

@pytest.mark.parametrize("sig", [
    "Л. Толстой.",
    "Граф Лев Толстой.",
    "Л. Т.",
    "Léon Tolstoï.",
    "Ваш Гр. Л. Толстой.",
    "Твой Л. Т.",
    "31 окт. 1909.",
    "Апреля 12-го дня 1847 года.",
    "Л. Толстой. 1844 Мая 30.",
    "С совершенным почтением имею честь быть ваш покорный слуга гр. Л. Толстой.",
    "Искренне уважающий вас Л. Толстой.",
])
def test_is_signature_or_date_matches(sig):
    assert corpus._is_signature_or_date(sig)


@pytest.mark.parametrize("content", [
    # текст и подпись в одной строке (телеграммы) — это контент, не трогаем
    "Ждешь меня или едешь сама. Что больной. Толстой.",
    "Convalescent. Reçu dernière lettre. Repondu precedente. Tolstoy.",
    "Невыразимо радуюсь. Давайте поручения. Располагайте нами. Толстой.",
    # прочие короткие строки — не подписи и не даты
    "Ясная Поляна.",
    "1 марта.",
])
def test_is_signature_or_date_keeps_content(content):
    assert not corpus._is_signature_or_date(content)


def test_filter_letter_signature_drops_trailing():
    paras = ["тело письма", "Л. Толстой.", "31 окт. 1909."]
    out = corpus._filter_letter_signature({"type": "письмо"}, paras)
    assert out == ["тело письма"]


def test_filter_letter_signature_stops_at_first_nonmatch():
    # режем только хвост: «подпись» в начале записи (цитата) не трогается
    paras = ["Лев Толстой.", "тело письма"]
    out = corpus._filter_letter_signature({"type": "письмо"}, paras)
    assert out == paras


def test_filter_letter_signature_only_for_letters():
    paras = ["текст", "31 окт. 1909."]
    assert corpus._filter_letter_signature({"type": "дневник"}, paras) == paras


def test_filter_letter_signature_length_cap():
    # длинная строка с фамилией — не подпись, не трогаем
    assert not corpus._is_signature_or_date("Толстой " + "и" * 90)


def test_filter_letter_signature_envelope_kept():
    # почтовые пометки — не подпись
    assert not corpus._is_signature_or_date("На конверте: Л. Толстой.")


def test_filter_letter_signature_whole_record():
    # письмо из одной подписи → пустой список чанков (но останется в load_docs)
    assert corpus._filter_letter_signature({"type": "письмо"}, ["Л. Толстой."]) == []


# --- doc_id и полные тексты --------------------------------------------------

def _write_corpus_files(tmp_path):
    (tmp_path / "tei_letters.txt").write_text(
        "### META | vol=59 | type=письмо | num=1 | to=Лобачевский | year=1844\n"
        "тело письма А с достаточно длинным содержанием\n\nЛ. Толстой.\n\n"
        "### META | vol=59 | type=письмо | num=2 | year=1844\n"
        "тело письма Б с достаточно длинным содержанием\n\n"
        "### META | vol=59 | type=письмо | entry=УКАЗАТЕЛЬ СОБСТВЕННЫХ ИМЕН\n"
        "мусорная запись\n\n",
        encoding="utf-8")
    (tmp_path / "tei_diaries.txt").write_text(
        "### META | vol=46 | type=дневник | date=1847-03-17 | year=1847\n"
        "запись дневника Ц с достаточно длинным содержанием\n\n"
        "### META | vol=46 | type=дневник | date=1847-03-18 | year=1847\n"
        "запись дневника Д с достаточно длинным содержанием\n\n31 окт. 1909.\n\n",
        encoding="utf-8")
    (tmp_path / "tom01.txt").write_text(
        "### META | vol=1 | type=проза | title=Проза\n"
        "прозаическая запись с достаточно длинным содержанием\n\n",
        encoding="utf-8")


def test_load_chunks_docs_doc_ids_align(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "DATA_DIR", str(tmp_path))
    _write_corpus_files(tmp_path)

    include = {"письмо", "дневник"}
    chunks = corpus.load_chunks(include_types=include)
    docs = corpus.load_docs(include_types=include)

    # doc_id — сквозной счётчик записей, совпадает у чанков и docs
    chunk_docs = {c.meta["doc"] for c in chunks}
    assert chunk_docs == {d["doc"] for d in docs} == {"0", "1", "2", "3"}
    # подпись отфильтрована из чанков письма, но есть в полном тексте
    letter_a_doc = next(d for d in docs if d["doc"] == "2")
    assert "Л. Толстой." in letter_a_doc["text"]
    assert all("Л. Толстой." not in c.text for c in chunks)
    # дневник не фильтруем: дата-хвост остаётся (склеена с предыдущим абзацем)
    assert any("31 окт. 1909." in c.text for c in chunks)
    # мусор и проза исключены из обоих
    assert all(c.meta.get("type") in ("письмо", "дневник") for c in chunks)
    assert all(d["meta"].get("type") in ("письмо", "дневник") for d in docs)


def test_load_docs_label_and_full_text(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "DATA_DIR", str(tmp_path))
    (tmp_path / "tei_letters.txt").write_text(
        "### META | vol=59 | type=письмо | num=1 | to=Лобачевский | year=1844\n"
        "тело письма А с достаточно длинным содержанием\n\nЛ. Толстой.\n\n",
        encoding="utf-8")

    docs = corpus.load_docs(include_types={"письмо"})
    assert len(docs) == 1
    d = docs[0]
    assert d["doc"] == "0"
    # полный текст БЕЗ фильтра подписей — для модалки /doc
    assert d["text"] == "тело письма А с достаточно длинным содержанием\n\nЛ. Толстой."
    assert d["label"] == corpus.source_label(d["meta"])
