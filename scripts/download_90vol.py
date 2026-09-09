#!/usr/bin/env python3
"""
Скрипт для скачивания 90-томного собрания сочинений Толстого с tolstoy.ru.

Структура сайта: https://tolstoy.ru/creativity/90-volume-collection-of-the-works/
Страницы томов имеют URL вида: https://tolstoy.ru/creativity/90-volume-collection-of-the-works/tom-NN/

Зависимости: requests, beautifulsoup4 (установить через uv)
"""
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Требуются зависимости: requests, beautifulsoup4")
    print("Установите: uv pip install requests beautifulsoup4")
    sys.exit(1)

BASE_URL = "https://tolstoy.ru"
COLLECTION_URL = f"{BASE_URL}/creativity/90-volume-collection-of-the-works/"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "90 томов Толстого"

# User-Agent чтобы выглядеть как браузер
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_page(url: str) -> str:
    """Скачать HTML страницы с retry."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"  Ошибка загрузки {url}: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                raise
    return ""


def find_volume_links(html: str) -> list[tuple[int, str]]:
    """Извлечь ссылки на страницы томов из главной страницы. Возвращает [(том_номер, URL)]."""
    soup = BeautifulSoup(html, "html.parser")
    volumes = []

    # Ищем выпадающий список "Выбрать том"
    # Структура: <li><a href="/creativity/90-volume-collection-of-the-works/664/">01</a></li>
    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(strip=True)

        # Проверяем, что это ссылка на том (цифры в тексте ссылки)
        if "/90-volume-collection-of-the-works/" in href and text.isdigit():
            vol_num = int(text)
            if 1 <= vol_num <= 90:
                full_url = BASE_URL + href if href.startswith("/") else href
                volumes.append((vol_num, full_url))

    return sorted(volumes)


def extract_volume_number(url: str) -> int | None:
    """Извлечь номер тома из URL."""
    # Паттерны: tom-5, volume-5, /5/, и т.п.
    match = re.search(r"(?:tom-|volume-)(\d+)", url, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"/(\d+)/?$", url)
    if match:
        return int(match.group(1))
    return None


def find_epub_link(html: str, volume_num: int) -> str | None:
    """Найти ссылку на EPUB файл на странице тома."""
    soup = BeautifulSoup(html, "html.parser")

    # Структура: <a href="/upload/iblock/416/01_tom.epub" ... >Скачать EPUB ...</a>
    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(strip=True).lower()

        if ".epub" in href.lower() and ("скачать" in text or "download" in text):
            if href.startswith("http"):
                return href
            elif href.startswith("/"):
                return BASE_URL + href
            else:
                return BASE_URL + "/" + href.lstrip("./")

    return None


def download_file(url: str, output_path: Path) -> bool:
    """Скачать файл по URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"    Ошибка скачивания: {e}")
        return False


def main() -> int:
    print(f"Скачивание 90-томного собрания сочинений Л.Н. Толстого")
    print(f"Источник: {COLLECTION_URL}")
    print(f"Целевая директория: {OUTPUT_DIR}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Загрузить главную страницу
    print("Загрузка главной страницы...")
    try:
        main_html = fetch_page(COLLECTION_URL)
    except Exception as e:
        print(f"Не удалось загрузить главную страницу: {e}")
        return 1

    # 2. Найти ссылки на тома
    print("Поиск страниц томов...")
    volume_links = find_volume_links(main_html)

    if not volume_links:
        print("Не найдено ссылок на тома. Возможно, структура сайта изменилась.")
        print("Попробуйте вручную проверить страницу и обновить скрипт.")
        return 1

    print(f"Найдено томов: {len(volume_links)}\n")

    # 3. Для каждого тома: найти EPUB и скачать
    downloaded = 0
    skipped = 0
    errors = 0

    for vol_num, vol_url in volume_links:
        output_file = OUTPUT_DIR / f"Толстой_Том_{vol_num:02d}.epub"

        if output_file.exists():
            print(f"Том {vol_num:2d}: уже скачан, пропускаем")
            skipped += 1
            continue

        print(f"Том {vol_num:2d}: загрузка страницы {vol_url}")

        try:
            vol_html = fetch_page(vol_url)
            epub_url = find_epub_link(vol_html, vol_num)

            if not epub_url:
                print(f"  ⚠️  EPUB не найден на странице")
                errors += 1
                continue

            print(f"  Найден EPUB: {epub_url}")
            print(f"  Скачивание в {output_file.name}...")

            if download_file(epub_url, output_file):
                size_mb = output_file.stat().st_size / 1024 / 1024
                print(f"  ✅ Скачано ({size_mb:.1f} МБ)")
                downloaded += 1
            else:
                print(f"  ❌ Ошибка скачивания")
                errors += 1

            # Небольшая пауза между запросами
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            errors += 1

    print(f"\n{'='*60}")
    print(f"Скачано:   {downloaded}")
    print(f"Пропущено: {skipped} (уже были)")
    print(f"Ошибок:    {errors}")
    print(f"{'='*60}")

    if errors > 0:
        print("\n⚠️  Некоторые тома не удалось скачать.")
        print("Возможные причины:")
        print("  - Структура сайта изменилась (нужно обновить скрипт)")
        print("  - EPUB недоступен для некоторых томов")
        print("  - Проблемы с сетью")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
