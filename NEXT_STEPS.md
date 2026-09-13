# Финальные шаги для перехода на Caddy

## Что уже сделано

✅ Обновлён `docker-compose.yml` - контейнер слушает HTTP на localhost:8000
✅ Обновлён healthcheck - использует HTTP вместо HTTPS
✅ Создан `Caddyfile` - конфигурация reverse proxy
✅ Создан `scripts/setup_caddy.sh` - скрипт установки
✅ Создана документация `CADDY_SETUP.md`
✅ Обновлены `README.md` и `CLAUDE.md`
✅ Обновлён `.rsyncignore` - исключены старые сертификаты

## Что нужно сделать

### 1. Закоммитить изменения (когда классификатор заработает)

```bash
git commit -m "feat: add Caddy reverse proxy with Let's Encrypt for verbatis.ru

- Replace self-signed TLS in Docker with Caddy reverse proxy
- Docker container now listens on localhost:8000 HTTP only
- Caddy handles TLS termination with automatic Let's Encrypt certs
- Add Caddyfile with security headers and logging
- Add setup_caddy.sh script for automated installation
- Add CADDY_SETUP.md with detailed deployment instructions
- Update docker-compose.yml: remove TLS env vars, bind to 127.0.0.1:8000
- Update healthcheck to use HTTP instead of HTTPS
- Remove certs/ volume mount (no longer needed)
- Update .rsyncignore to exclude certs/ directory
- Update documentation in README.md and CLAUDE.md

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### 2. Настроить DNS (перед деплоем!)

В панели управления DNS провайдера для домена verbatis.ru:

```
Тип: A
Имя: @ (или verbatis.ru)
Значение: 168.113.209.202
TTL: 300
```

Проверить пропагацию:
```bash
dig verbatis.ru
# Должно вернуть: 168.113.209.202
```

### 3. Деплой

```bash
# На сервере: остановить текущий контейнер
ssh root@168.113.209.202
cd /opt/leotolstoy
docker compose down
exit

# Локально: деплой
./deploy.sh

# На сервере: установить и настроить Caddy
ssh root@168.113.209.202
cd /opt/leotolstoy
chmod +x scripts/setup_caddy.sh
sudo ./scripts/setup_caddy.sh
```

### 4. Проверка

```bash
# Проверить статус
sudo systemctl status caddy
docker compose ps

# Проверить доступность
curl http://localhost:8000/health
curl https://verbatis.ru/health

# Логи
sudo journalctl -u caddy -f
docker compose logs -f
```

### 5. Если что-то не работает

См. раздел "Проблемы и решения" в [CADDY_SETUP.md](CADDY_SETUP.md)

## Важно

- DNS должен быть настроен ДО запуска Caddy (иначе не сможет получить Let's Encrypt)
- Порты 80 и 443 должны быть открыты
- Старый порт 8443 больше не используется
