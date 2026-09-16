"""Тесты парсера TEI/XML (scripts/tei_to_txt.py) — без сети и модели.

Сэмплы — упрощённые копии реальных файлов tolstoydigital/TEI
(texts/letters/vNN_NNN_*.xml, texts/diaries/vNN_p1_p2_YYYY_MM_DD.xml).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import tei_to_txt as tei  # noqa: E402


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


LETTER_XML = """<?xml version='1.0' encoding='UTF-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:xi="http://www.w3.org/2001/XInclude">
 <teiHeader>
  <fileDesc>
   <titleStmt>
    <title xml:id="v90_189_B_O_Goldenblatu">Письма и документы. Tolstoy Digital</title>
    <title type="main">Письмо Б. О. Гольденблату .... 1900—1910 гг.</title>
    <title type="bibl">Толстой Л. Н. Письмо Б. О. Гольденблату.... // Полное собрание сочинений: в 90 тт. Т. 90. М.: 1958. С. 343.</title>
   </titleStmt>
   <sourceDesc>
    <biblStruct>
     <monogr>
      <title level="m">Полное собрание сочинений. Том 90</title>
      <imprint><date when="1958"/></imprint>
      <biblScope unit="vol">90</biblScope>
      <biblScope unit="page">343</biblScope>
     </monogr>
    </biblStruct>
    <bibl>Публикуется впервые</bibl>
   </sourceDesc>
  </fileDesc>
  <profileDesc>
   <correspDesc>
    <correspAction type="sending">
     <persName ref="13883">Л. Н. Толстой</persName>
     <placeName>Ясная Поляна</placeName>
     <date when="1910-11-10"/>
    </correspAction>
    <correspAction type="receiving">
     <persName ref="3588">Гольденблат Борис Осипович</persName>
    </correspAction>
   </correspDesc>
   <langUsage><language ident="ru"/><language ident="ru-petr1708"/></langUsage>
  </profileDesc>
 </teiHeader>
 <xenoData><number type="in a volume">189</number></xenoData>
 <text xmlns="http://www.tei-c.org/ns/1.0" xml:id="B.O.Goldenblatu...._189">
  <body>
   <div>
    <pb n="343"/>
    <opener>* 189. Б. О. Гольденблату. 1900—1910 гг. Я. П.</opener>
    <p>Прошу помочь подательнице. <ref target="#note1">1</ref></p>
    <note index="false" resp="volume_editor" xml:id="note1">О ней сведений не имеется.</note>
    <p><choice><orig>Императорскаго</orig><reg>Императорского</reg></choice> общества.</p>
    <p rend="right"><name placement="text" ref="13883" type="person">Л. Толстой</name>.</p>
    <noteGrp type="comments"><note><p>Написано на визитной карточке.</p></note></noteGrp>
   </div>
  </body>
 </text>
</TEI>
"""


def test_letter_meta(tmp_path):
    path = _write(tmp_path, "v90_189_B_O_Goldenblatu.xml", LETTER_XML)
    meta, paras = tei.parse_letter(path)

    assert meta["type"] == "письмо"
    assert meta["vol"] == "90"
    assert meta["num"] == "189"
    assert meta["to"] == "Гольденблат Борис Осипович"
    assert meta["place"] == "Ясная Поляна"
    assert meta["date"] == "1910-11-10"
    assert meta["year"] == "1910"
    assert "Б. О. Гольденблату" in meta["title"]
    assert meta["bibl"].startswith("Толстой Л. Н. Письмо")
    assert meta["pages"] == "343"
    assert meta["ed_year"] == "1958"
    assert meta["first_pub"] == "Публикуется впервые"
    assert meta["old_orth"] == "да"
    assert meta["entry"] == "189. Гольденблат Борис Осипович, 1910, Ясная Поляна, с. 343"


def test_letter_body_choice_reg_notes_dropped(tmp_path):
    path = _write(tmp_path, "v90_189_B_O_Goldenblatu.xml", LETTER_XML)
    _, paras = tei.parse_letter(path)

    assert paras[0] == "Прошу помочь подательнице."          # маркер сноски <ref> выброшен
    assert paras[1] == "Императорского общества."            # choice -> reg, не orig
    assert paras[2] == "Л. Толстой."                         # <name> сохраняем
    # редакторские сноски и комментарии не попали в текст
    assert "сведений не имеется" not in "\n\n".join(paras)
    assert "визитной карточке" not in "\n\n".join(paras)
    assert "1900—1910" not in "\n\n".join(paras)             # opener пропущен


def test_letter_num_leading_zeros_normalized(tmp_path):
    path = _write(tmp_path, "v59_001_Lobachevskomu.xml", LETTER_XML)
    meta, _ = tei.parse_letter(path)
    assert meta["num"] == "1"          # 001 -> 1
    assert meta["entry"].startswith("1. ")


def test_letter_undated_range(tmp_path):
    xml = LETTER_XML.replace(
        '<date when="1910-11-10"/>',
        '<date notBefore="1900-01-01" notAfter="1910-12-31"/>',
    )
    path = _write(tmp_path, "v90_189_B_O_Goldenblatu.xml", xml)
    meta, _ = tei.parse_letter(path)
    assert meta["date"] == "1900-01-01—1910-12-31"
    assert meta["year"] == "1900–1910"


def test_letter_without_addressee_falls_back_to_title(tmp_path):
    xml = LETTER_XML.replace(
        '<correspAction type="receiving">\n     <persName ref="3588">Гольденблат Борис Осипович</persName>\n    </correspAction>',
        '<correspAction type="receiving"/>',
    )
    path = _write(tmp_path, "v90_189_B_O_Goldenblatu.xml", xml)
    meta, _ = tei.parse_letter(path)
    assert "to" not in meta
    assert meta["entry"] == meta["title"]  # fallback на title type="main"


def test_letter_empty_body_skipped(tmp_path):
    # убираем все абзацы — остаются только pb/opener, текста нет
    xml = LETTER_XML
    for marker in (
        "<p>Прошу помочь подательнице. <ref target=\"#note1\">1</ref></p>",
        "<p><choice><orig>Императорскаго</orig><reg>Императорского</reg></choice> общества.</p>",
        '<p rend="right"><name placement="text" ref="13883" type="person">Л. Толстой</name>.</p>',
    ):
        xml = xml.replace(marker, "")
    path = _write(tmp_path, "v90_189_B_O_Goldenblatu.xml", xml)
    assert tei.parse_letter(path) is None


def test_letter_broken_xml_raises(tmp_path):
    path = _write(tmp_path, "v59_005_broken.xml", "<TEI><teiHeader>")
    with pytest.raises(tei.ET.ParseError):
        tei.parse_letter(path)


def test_format_record_escapes_pipe():
    rec = tei.format_record({"vol": "59", "type": "письмо", "to": "Б|В"}, ["абзац"])
    assert "to=Б/В" in rec
    assert "to=Б | В" not in rec
    # и разбор обратно в corpus._parse_meta проходит
    import corpus
    meta = corpus._parse_meta(rec.splitlines()[0])
    assert meta["to"] == "Б/В"


DIARY_XML = """<?xml version='1.0' encoding='UTF-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
 <teiHeader>
  <fileDesc>
   <titleStmt>
    <title type="main">Дневник для одного себя</title>
    <title type="bibl">Толстой Л. Н. Дневник для одного себя // Полное собрание сочинений: в 90 тт. Т. 58. М.: 1934. С. 143.</title>
   </titleStmt>
   <sourceDesc>
    <biblStruct>
     <monogr>
      <title level="m">Полное собрание сочинений. Том 58</title>
      <imprint><date when="1934"/></imprint>
      <biblScope unit="vol">58</biblScope>
      <biblScope unit="page">143</biblScope>
     </monogr>
    </biblStruct>
   </sourceDesc>
  </fileDesc>
  <profileDesc>
   <creation><date from="1910-10-27" to="1910-10-28"/></creation>
  </profileDesc>
 </teiHeader>
 <text xmlns="http://www.tei-c.org/ns/1.0">
  <body>
   <div n="v58_n381" type="entry">
    <opener><dateline><date resp="editor" when="1910-10-27">27 октября 1910 г.</date></dateline></opener>
    <p>26-го Окт. Ничего особенного не было. Только росло чувство стыда.</p>
    <note exclude="true" index="false" resp="volume_editor">Зачеркнуто: текст из рукописи</note>
   </div>
  </body>
 </text>
</TEI>
"""


def test_diary_meta(tmp_path):
    path = _write(tmp_path, "v58_143_143_1910_10_27_2.xml", DIARY_XML)
    meta, paras = tei.parse_diary(path)

    assert meta["type"] == "дневник"
    assert meta["vol"] == "58"
    assert meta["date"] == "1910-10-27"
    assert meta["year"] == "1910"
    assert meta["pages"] == "143–143"
    assert meta["entry"] == "Дневник для одного себя"
    assert meta["ed_year"] == "1934"
    assert len(paras) == 1
    assert paras[0].startswith("26-го Окт.")
    assert "Зачеркнуто" not in paras[0]  # редакторские пометы выброшены


def test_diary_plain_title_no_entry(tmp_path):
    xml = DIARY_XML.replace("Дневник для одного себя", "Дневник", 1)
    path = _write(tmp_path, "v46_003_004_1847_03_17.xml", xml)
    meta, _ = tei.parse_diary(path)
    assert "entry" not in meta  # «Дневник» сам по себе не несёт информации


def test_load_chunks_reads_tei_files(tmp_path, monkeypatch):
    import corpus
    monkeypatch.setattr(corpus, "DATA_DIR", str(tmp_path))

    (tmp_path / "tom01.txt").write_text(
        "### META | vol=1 | type=проза\n\nАбзац прозы.\n\n", encoding="utf-8")
    (tmp_path / "tei_letters.txt").write_text(
        "### META | vol=59 | type=письмо | to=Адресат\n\nАбзац письма.\n\n",
        encoding="utf-8")

    chunks = corpus.load_chunks()
    types = {c.meta["type"] for c in chunks}
    assert types == {"проза", "письмо"}
    assert sum(1 for c in chunks if c.meta["type"] == "письмо") == 1
