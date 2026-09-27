#!/usr/bin/env python3
"""
Сборка индекса эмбеддингов через OpenRouter API (быстро, платно).

Использует внешнюю модель для векторизации всех документов.
Результат сохраняется в index.<model-slug>/ с metadata.json.

Пример:
    EXTERNAL_EMBEDDER_MODEL=qwen/qwen3-embedding-8b \
    EXTERNAL_EMBEDDER_API_KEY=sk-or-v1-... \
    uv run python scripts/build_embeddings_api.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

# Загрузка .env перед чтением переменных окружения
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from corpus import load_chunks, source_label  # noqa: E402

# Ленивый импорт embedders — он тянет httpx, который не нужен для локальной сборки
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIN_CHARS = 40  # чанки короче — мусор
CHECKPOINT_EVERY_BATCHES = 10  # сохраняем прогресс каждые N батчей


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = n % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many


def _fmt_eta(eta_min: float) -> str:
    """ETA для прогресс-бара: до часа — только минуты, иначе «X час(а/ов) Y минут(ы)»."""
    total = max(1, round(eta_min))
    if total < 60:
        return f"ETA: {total} {_plural(total, 'минута', 'минуты', 'минут')}"
    h, m = divmod(total, 60)
    out = f"ETA: {h} {_plural(h, 'час', 'часа', 'часов')}"
    if m:
        out += f" {m} {_plural(m, 'минута', 'минуты', 'минут')}"
    return out


def _checkpoint_paths(index_dir: str) -> tuple[str, str]:
    return (
        os.path.join(index_dir, ".partial_vectors.npy"),
        os.path.join(index_dir, ".partial_meta.json"),
    )


def save_checkpoint(
    index_dir: str,
    processed: int,
    total: int,
    vectors: np.ndarray,
    *,
    model: str | None = None,
    provider: str | None = None,
    n_batches: int | None = None,
) -> None:
    """Сохранить промежуточный результат, чтобы перезапуск не начинался с нуля.

    В meta пишется fingerprint сборки (model/provider/dims): resume допустим
    только при полном совпадении, иначе векторы от разных провайдеров/моделей
    смешались бы в одном индексе.
    """
    vec_path, meta_path = _checkpoint_paths(index_dir)
    np.save(vec_path, vectors)
    meta: dict = {"processed": processed, "total": total, "dims": int(vectors.shape[1])}
    if model:
        meta["model"] = model
    if provider:
        meta["provider"] = provider.lower()
    if n_batches is not None:
        meta["n_batches"] = n_batches
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_checkpoint(
    index_dir: str,
    total: int,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> np.ndarray | None:
    """Вернуть промежуточные векторы, если чекпоинт валиден.

    Валидность: тот же корпус (total), та же модель и провайдер (fingerprint).
    Оба поля пустые с обеих сторон = легаси-чекпоинт без пина, принимаем.
    """
    vec_path, meta_path = _checkpoint_paths(index_dir)
    if not (os.path.exists(vec_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("total") != total:
            return None
        if model and meta.get("model") and meta["model"] != model:
            return None
        if (provider or "").lower() != (meta.get("provider") or "").lower():
            return None
        vecs = np.load(vec_path)
        if vecs.ndim != 2 or vecs.shape[0] > total or vecs.shape[0] != meta.get("processed"):
            return None
        if meta.get("dims") and vecs.shape[1] != meta["dims"]:
            return None
        return vecs
    except (ValueError, OSError, json.JSONDecodeError):
        return None


def _validate_batch(vecs: np.ndarray, expected_dims: int | None = None) -> None:
    """Проверить ответ API: конечность, отсутствие нулевых строк, стабильность dims.

    NaN-вектор при касте в int8 превращается в мусор (или вектор-константу,
    который «притягивается» к любым запросам) — фейлим громко, а не молча.
    """
    if not np.isfinite(vecs).all() or np.any(np.all(vecs == 0, axis=1)):
        raise RuntimeError("API вернул NaN/inf или нулевые векторы — ответ провайдера повреждён")
    if expected_dims is not None and vecs.shape[1] != expected_dims:
        raise RuntimeError(
            f"Размерность векторов сменилась ({expected_dims} -> {vecs.shape[1]}) — "
            f"провайдер нестабилен или подменил модель"
        )


def clear_checkpoint(index_dir: str) -> None:
    vec_path, meta_path = _checkpoint_paths(index_dir)
    for p in (vec_path, meta_path):
        if os.path.exists(p):
            os.remove(p)


def main() -> int:
    # Конфигурация из окружения
    model = os.environ.get("EXTERNAL_EMBEDDER_MODEL", "qwen/qwen3-embedding-8b")
    api_key = os.environ.get("EXTERNAL_EMBEDDER_API_KEY")
    base_url = os.environ.get("EXTERNAL_EMBEDDER_ENDPOINT", "https://openrouter.ai/api/v1")
    # Пин провайдера: дефолт nebius (неквантизованный апстрим Qwen3-8B).
    # Для qwen3-embedding-4b единственный провайдер — deepinfra.
    # Пустая строка = авто-роутинг (НЕ рекомендуется: батчи разъедутся по
    # провайдерам, а один из них (siliconflow) квантизует модель в fp8).
    provider = os.environ.get("EXTERNAL_EMBEDDER_PROVIDER", "nebius").strip() or None

    if not api_key:
        print("ERROR: EXTERNAL_EMBEDDER_API_KEY не задан в .env", file=sys.stderr)
        return 1

    # Ограничение типов (опционально)
    include = None
    env_types = os.environ.get("EMBED_TYPES", "").strip()
    if env_types:
        include = {t.strip() for t in env_types.split(",") if t.strip()}
        print(f"Ограничение по типу: {include}")

    # Загрузка чанков
    chunks = load_chunks(include_types=include)
    before = len(chunks)
    chunks = [c for c in chunks if len(c.text) >= MIN_CHARS]
    print(f"Чанков: {len(chunks)} (отброшено коротких: {before - len(chunks)})")
    if not chunks:
        print("Нечего эмбеддить. Сначала: python3 scripts/epub_to_txt.py", file=sys.stderr)
        return 1

    texts = [c.text for c in chunks]

    # Каталог индекса нужен уже сейчас — для чекпоинтов прогресса
    # model_id для фронтенда: "qwen/qwen3-embedding-8b" → "qwen3-embedding-8b" (только последняя часть)
    model_id = model.split("/")[-1] if "/" in model else model
    index_dir = os.path.join(ROOT, f"index.{model_id}")
    os.makedirs(index_dir, exist_ok=True)

    # Импорт эмбеддера (здесь, чтобы не грузить лишнее если нет чанков)
    from embedders import OpenRouterEmbedder  # noqa: E402

    print(f"\nЭмбеддинг {len(texts)} чанков через {model}...")
    print(f"API: {base_url}")
    print(f"Провайдер: {provider or 'авто (НЕ РЕКОМЕНДУЕТСЯ)'}")
    sys.stdout.flush()

    # Таймаут 120с: батч из 100 текстов (~25k токенов) под нагрузкой OpenRouter
    # может думать дольше 30с. 5 попыток с экспоненциальным backoff.
    embedder = OpenRouterEmbedder(
        model=model, api_key=api_key, base_url=base_url, timeout=120.0, max_retries=5,
        provider=provider,
    )

    t0 = time.time()

    # Обёртка для embed_passages с прогресс-баром
    batch_size = 100
    all_vecs = []

    # Возобновление с чекпоинта (если прошлый запуск упал) — не начинаем с нуля
    resume = load_checkpoint(index_dir, len(texts), model=model, provider=provider)
    if resume is not None and resume.shape[0] > 0:
        print(f"Найден чекпоинт: {resume.shape[0]}/{len(texts)} чанков уже готово — "
              f"продолжаем с этого места")
        all_vecs.append(resume)
        start = resume.shape[0]
        batches_done = resume.shape[0] // batch_size
    else:
        # Чекпоинт отклонён (модель/провайдер/корпус изменились) или отсутствует.
        # Подчистим, чтобы устаревший чекпоинт не всплыл позже.
        vec_path, _ = _checkpoint_paths(index_dir)
        if os.path.exists(vec_path):
            print("Чекпоинт отклонён (модель/провайдер/корпус изменились) — "
                  "начинаем заново")
        clear_checkpoint(index_dir)
        start = 0
        batches_done = 0

    try:
        for i in range(start, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vecs = embedder._call_api(batch)
            _validate_batch(batch_vecs, expected_dims=all_vecs[0].shape[1] if all_vecs else None)
            all_vecs.append(batch_vecs)
            batches_done += 1

            # Прогресс
            processed = min(i + batch_size, len(texts))
            percent = 100 * processed / len(texts)
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta_sec = (len(texts) - processed) / rate if rate > 0 else 0
            eta_min = eta_sec / 60

            # Чекпоинт каждые N батчей — при обрыве продолжаем с места обрыва
            if batches_done % CHECKPOINT_EVERY_BATCHES == 0:
                save_checkpoint(
                    index_dir, processed, len(texts), np.vstack(all_vecs),
                    model=model, provider=provider, n_batches=batches_done,
                )

            # Прогресс-бар
            bar_width = 40
            filled = int(bar_width * processed / len(texts))
            bar = '█' * filled + '░' * (bar_width - filled)

            print(f"\r[{bar}] {percent:.1f}% | {processed}/{len(texts)} | "
                  f"{rate:.1f} чанков/с | {_fmt_eta(eta_min)}", end='', flush=True)

        print()  # новая строка после прогресс-бара
    except Exception as e:
        print(f"\n\nОшибка при вызове API: {e}", file=sys.stderr)
        print(f"Прогресс сохранён в чекпоинте — перезапустите `make embeddings-qwen`, "
              f"сборка продолжится с места обрыва.", file=sys.stderr)
        return 1

    vectors = np.vstack(all_vecs)
    # Нормализация
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    # Запоминаем размерность
    embedder._dimensions = vectors.shape[1]

    dt = time.time() - t0
    print(f"\nГотово за {dt:.1f}с ({len(texts)/dt:.1f} чанков/с). Форма: {vectors.shape}")

    # Формат индекса: float32 по умолчанию (точная точность, поиск через BLAS;
    # на сервере 4 ГБ RAM индекс ~1.4 ГБ влезает с запасом).
    # float16 — опция экономии (716 МБ, качество ≈ float32).
    # int8 больше НЕ используется: усечение (не округление) обнуляло ~39%
    # компонент и давало ~40% расхождений top-10 с float32.
    index_dtype = os.environ.get("EMBED_INDEX_DTYPE", "float32").strip()
    if index_dtype == "float32":
        vectors_out = vectors
    elif index_dtype == "float16":
        vectors_out = vectors.astype(np.float16)
    elif index_dtype == "int8":
        # Легаси-вариант (совместим с int8-путём поиска в retriever — та же
        # схема усечения и перенормировки, менять нельзя без синхронного
        # изменения рантайма)
        print(f"\nВНИМАНИЕ: int8-индекс искажает ранжирование — используйте float32")
        vectors_int8 = (vectors * 127).astype(np.int8)
        norms_int8 = np.linalg.norm(vectors_int8.astype(np.float32), axis=1, keepdims=True)
        norms_int8[norms_int8 == 0] = 1.0
        vectors_out = (vectors_int8.astype(np.float32) / norms_int8 * 127).astype(np.int8)
    else:
        print(f"ERROR: EMBED_INDEX_DTYPE={index_dtype} — допустимы только "
              f"float32, float16 или int8", file=sys.stderr)
        return 1

    print(f"\nФормат индекса: {index_dtype}")
    sample_norm = np.linalg.norm(vectors_out[0].astype(np.float32))
    print(f"  Форма: {vectors_out.shape}, размер: {vectors_out.nbytes / 1e6:.1f} МБ, "
          f"норма[0]: {sample_norm:.4f}")

    # Оценка стоимости (приблизительная, ~250 токенов на чанк)
    est_tokens = len(texts) * 250
    est_cost_per_1m = 0.01  # для qwen3-embedding-8b
    if "qwen3-embedding-8b" in model:
        est_cost_per_1m = 0.01
    elif "qwen3-embedding-4b" in model:
        est_cost_per_1m = 0.02
    elif "nemotron" in model:
        est_cost_per_1m = 0.0
    est_cost = (est_tokens / 1_000_000) * est_cost_per_1m
    print(f"Оценка: ~{est_tokens:,} токенов, ~${est_cost:.2f} ({model})")

    # Сохранение vectors.npy (атомарно: tmp + os.replace — при падении между
    # записями vectors.npy и metadata.json старый файл остаётся целым).
    # ВАЖНО: tmp-имя должно заканчиваться на .npy, иначе np.save допишет
    # расширение сам и os.replace не найдёт файл.
    vectors_path = os.path.join(index_dir, "vectors.npy")
    tmp_vec_path = os.path.join(index_dir, "vectors.tmp.npy")
    np.save(tmp_vec_path, vectors_out)
    os.replace(tmp_vec_path, vectors_path)
    print(f"\nСохранено: vectors.npy ({index_dtype}) {os.path.getsize(vectors_path) / 1e6:.1f} МБ")

    # Полная float32-версия — только для диагностики или будущего re-rank
    if os.environ.get("EMBED_SAVE_F32", "").strip() in ("1", "true", "yes"):
        vectors_f32_path = os.path.join(index_dir, "vectors_float32.npy")
        np.save(vectors_f32_path, vectors)
        print(f"  vectors_float32.npy:      {os.path.getsize(vectors_f32_path) / 1e6:.1f} МБ "
              f"(диагностика, поиск не использует)")

    # Сохранение chunks.json (тот же, что для локальной модели)
    # Можно использовать symlink для экономии памяти и диска
    chunks_path = os.path.join(index_dir, "chunks.json")
    local_chunks_path = os.path.join(ROOT, "index", "chunks.json")

    if os.path.exists(local_chunks_path) and not os.path.exists(chunks_path):
        # Создаём symlink вместо копирования (экономия ~227 МБ)
        try:
            # Относительный путь: работает и на хосте, и внутри Docker-контейнера
            # (в контейнере index* смонтированы в /app/index*, путь хоста там не виден).
            rel_target = os.path.relpath(local_chunks_path, index_dir)
            os.symlink(rel_target, chunks_path)
            print(f"\nСоздан symlink для chunks.json ({rel_target}) → экономия ~227 МБ")
        except OSError:
            # На Windows может не работать — копируем
            print(f"\nSymlink не поддерживается, копируем chunks.json...")
            with open(local_chunks_path, encoding="utf-8") as f:
                local_chunks = json.load(f)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(local_chunks, f, ensure_ascii=False)
    elif not os.path.exists(chunks_path):
        # Локального индекса нет — сохраняем chunks заново
        meta_out = []
        for c in chunks:
            meta_out.append({
                "text": c.text,
                "source": source_label(c.meta),
                "meta": c.meta,
            })
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False)
    else:
        print(f"\nchunks.json уже существует, пропускаем")

    # Сохранение metadata.json (атомарно, как vectors.npy выше)
    metadata = {
        "model": model,
        "dimensions": int(vectors.shape[1]),
        "api": True,
        "quantized": index_dtype == "int8",  # флаг оставлен для совместимости чтения
        "dtype": index_dtype,                # источник истины для retriever
        "provider": provider,                # апстрим сборки — для сверки в рантайме
        "display_name": _display_name(model),
        "latency_ms": 350,  # типичная latency для API
        "cost_per_1k": est_cost_per_1m * 1000 / 1_000_000,
        "description": f"Внешняя модель через OpenRouter API ({index_dtype})",
    }
    metadata_path = os.path.join(index_dir, "metadata.json")
    tmp_meta_path = metadata_path + ".tmp"
    with open(tmp_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(tmp_meta_path, metadata_path)

    # Статистика
    vec_mb = os.path.getsize(vectors_path) / 1e6
    json_mb = os.path.getsize(chunks_path) / 1e6
    print(f"\nСохранено в {index_dir}/:")
    print(f"  vectors.npy     {vec_mb:.1f} МБ")
    print(f"  chunks.json     {json_mb:.1f} МБ")
    print(f"  metadata.json   {os.path.getsize(metadata_path) / 1e3:.1f} КБ")

    # Сборка завершилась — чекпоинт больше не нужен
    clear_checkpoint(index_dir)

    print(f"\nИндексы rsync'ом не синкаются (.rsyncignore: index*/) — сборка")
    print("выполняется там же, где индекс используется.")
    return 0


def _display_name(model: str) -> str:
    """Красивое имя модели для UI."""
    if "qwen3-embedding-8b" in model:
        return "Qwen3 Embedding 8B"
    elif "qwen3-embedding-4b" in model:
        return "Qwen3 Embedding 4B"
    elif "nemotron" in model:
        return "NVIDIA Nemotron 3B"
    elif "openai" in model and "text-embedding-3-large" in model:
        return "OpenAI Text Embedding 3 Large"
    elif "openai" in model and "text-embedding-3-small" in model:
        return "OpenAI Text Embedding 3 Small"
    else:
        return model.replace("/", " ").replace("-", " ").title()


if __name__ == "__main__":
    raise SystemExit(main())
