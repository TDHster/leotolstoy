# Инструкция по переходу на Caddy + домен verbatis.ru

## Подготовка (выполнить ДО деплоя)

### 1. Настройка DNS

Добавьте A-запись в DNS провайдере для домена verbatis.ru:

```
Тип: A
Имя: @ (или verbatis.ru)
Значение: 168.113.209.202
TTL: 300 (5 минут) или авто
```

Проверьте пропагацию DNS (может занять до 24 часов, обычно 5-30 минут):

```bash
# Локально
dig verbatis.ru
# или
nslookup verbatis.ru

# Должно показать: 168.113.209.202
```

### 2. Проверка портов на сервере

Подключитесь к серверу и проверьте что порты 80 и 443 свободны:

```bash
ssh root@168.113.209.202

# Проверка портов
sudo netstat -tulpn | grep ':80\|:443'

# Если порты заняты nginx или apache:
sudo systemctl stop nginx
sudo systemctl disable nginx
# или
sudo systemctl stop apache2
sudo systemctl disable apache2
```

Проверьте firewall (если используется):

```bash
sudo ufw status
# Если активен, убедитесь что открыты порты 80 и 443:
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

## Деплой

### Шаг 1: Остановить текущий контейнер на сервере

```bash
ssh root@168.113.209.202
cd /opt/leotolstoy
docker compose down
```

### Шаг 2: Деплой обновлённой конфигурации (локально)

```bash
# В директории проекта
./deploy.sh
```

Это синхронизирует:
- Обновлённый `docker-compose.yml` (HTTP вместо HTTPS)
- Новый файл `Caddyfile`
- Скрипт установки `scripts/setup_caddy.sh`

### Шаг 3: Установка и настройка Caddy (на сервере)

```bash
ssh root@168.113.209.202
cd /opt/leotolstoy

# Сделать скрипт исполняемым (если нужно)
chmod +x scripts/setup_caddy.sh

# Запустить установку Caddy
sudo ./scripts/setup_caddy.sh
```

Скрипт выполнит:
- Установку Caddy из официального репозитория
- Проверку DNS и портов
- Копирование конфигурации в `/etc/caddy/Caddyfile`
- Валидацию конфигурации
- Запуск и включение Caddy в автозагрузку

### Шаг 4: Проверка работы

**На сервере:**

```bash
# Проверка статуса Caddy
sudo systemctl status caddy

# Проверка логов Caddy
sudo journalctl -u caddy -f

# Проверка что Docker контейнер работает
docker compose ps
curl http://localhost:8000/health
# Должно вернуть: {"status":"ok"}
```

**Локально или в браузере:**

```bash
# Проверка через Caddy
curl https://verbatis.ru/health
# Должно вернуть: {"status":"ok"}

# Проверка сертификата
curl -I https://verbatis.ru
# Должен быть Let's Encrypt сертификат

# Проверка HTTP→HTTPS редиректа
curl -I http://verbatis.ru
# Должен быть 301/308 редирект на https://
```

**В браузере:**
Откройте https://verbatis.ru - должен быть валидный зелёный замочек (Let's Encrypt).

## Что изменилось

| До | После |
|---|---|
| Docker контейнер слушает HTTPS на 8000 внутри | Docker контейнер слушает HTTP на 8000 внутри |
| Самоподписанный сертификат в `certs/` | Let's Encrypt сертификат (управляется Caddy) |
| Доступ через `https://168.113.209.202:8443` | Доступ через `https://verbatis.ru` |
| Порт 8443 публично открыт | Порт 8000 доступен только localhost |
| Предупреждение браузера о сертификате | Валидный сертификат без предупреждений |

## Добавление других сервисов на поддоменах

Отредактируйте `/etc/caddy/Caddyfile` на сервере:

```bash
sudo nano /etc/caddy/Caddyfile
```

Добавьте новые блоки:

```
api.verbatis.ru {
    reverse_proxy localhost:9000
}

admin.verbatis.ru {
    reverse_proxy localhost:9001
    basicauth {
        # Генерировать хеш: caddy hash-password
        admin $2a$14$...
    }
}
```

Перезапустите Caddy:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy автоматически получит сертификаты для всех доменов.

## Откат в случае проблем

Если что-то пошло не так:

```bash
# На сервере
ssh root@168.113.209.202
cd /opt/leotolstoy

# Остановить Caddy
sudo systemctl stop caddy
sudo systemctl disable caddy

# Откатить docker-compose.yml
git checkout docker-compose.yml

# Запустить контейнер по-старому
docker compose up -d

# Доступ по старому адресу
# https://168.113.209.202:8443
```

## Мониторинг

### Логи Caddy

```bash
# Systemd логи
sudo journalctl -u caddy -f

# JSON логи приложения
sudo tail -f /var/log/caddy/verbatis.log | jq
```

### Логи Docker контейнера

```bash
docker compose logs -f
```

### Автоматическое обновление сертификатов

Caddy автоматически обновляет Let's Encrypt сертификаты за 30 дней до истечения. Никаких cron задач не требуется.

Проверить текущий сертификат:

```bash
echo | openssl s_client -connect verbatis.ru:443 -servername verbatis.ru 2>/dev/null | openssl x509 -noout -dates
```

## Проблемы и решения

### Caddy не получает сертификат

Проверьте:
1. DNS правильно настроен: `dig verbatis.ru` → 168.113.209.202
2. Порт 80 открыт (нужен для ACME challenge): `sudo netstat -tulpn | grep :80`
3. Firewall не блокирует: `sudo ufw status`
4. Логи Caddy: `sudo journalctl -u caddy -n 50`

### Сертификат показывает "Caddy Local Authority"

Это значит Caddy работает в тестовом режиме (не может получить Let's Encrypt).
Проверьте DNS и доступность порта 80 из интернета.

### "Connection refused" при доступе к сайту

Проверьте что Docker контейнер запущен:
```bash
docker compose ps
curl http://localhost:8000/health
```

Если контейнер работает, проверьте Caddy:
```bash
sudo systemctl status caddy
```

## Обновление конфигурации Caddy после первого деплоя

В будущем, чтобы обновить `Caddyfile`:

```bash
# Локально отредактировать Caddyfile
nano Caddyfile

# Задеплоить
./deploy.sh

# На сервере
ssh root@168.113.209.202
cd /opt/leotolstoy
sudo cp Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

## Контакты и помощь

Документация Caddy: https://caddyserver.com/docs/
