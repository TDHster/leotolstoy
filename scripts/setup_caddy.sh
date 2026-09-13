#!/bin/bash
# Скрипт установки и настройки Caddy на сервере Ubuntu
set -euo pipefail

echo "=== Установка и настройка Caddy для verbatis.ru ==="

# Проверка что мы на сервере (не локально)
if [ ! -f /etc/os-release ]; then
    echo "Не могу определить ОС. Запускайте на Ubuntu сервере." >&2
    exit 1
fi

# Проверка DNS (опционально - может быть ещё не пропагировалось)
echo "Проверка DNS для verbatis.ru..."
if command -v dig &> /dev/null; then
    DNS_IP=$(dig +short verbatis.ru | head -n 1)
    if [ -n "$DNS_IP" ]; then
        echo "✅ DNS verbatis.ru → $DNS_IP"
    else
        echo "⚠️  DNS для verbatis.ru не найден. Проверьте настройки DNS перед запуском."
        read -p "Продолжить установку? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Проверка занятости портов 80/443
echo "Проверка портов 80 и 443..."
if sudo netstat -tulpn 2>/dev/null | grep -E ':80\s|:443\s' | grep -v caddy; then
    echo "⚠️  Порты 80 или 443 уже заняты другим процессом:"
    sudo netstat -tulpn | grep -E ':80\s|:443\s'
    echo ""
    echo "Остановите nginx/apache перед установкой Caddy:"
    echo "  sudo systemctl stop nginx"
    echo "  sudo systemctl disable nginx"
    exit 1
fi

# Установка Caddy
if command -v caddy &> /dev/null; then
    echo "✅ Caddy уже установлен: $(caddy version)"
else
    echo "Установка Caddy..."
    sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
    sudo apt update
    sudo apt install -y caddy
    echo "✅ Caddy установлен: $(caddy version)"
fi

# Создание директорий для логов
echo "Создание директорий..."
sudo mkdir -p /var/log/caddy
sudo chown caddy:caddy /var/log/caddy

# Копирование конфигурации
if [ ! -f Caddyfile ]; then
    echo "❌ Файл Caddyfile не найден в текущей директории" >&2
    exit 1
fi

echo "Копирование конфигурации Caddy..."
sudo cp Caddyfile /etc/caddy/Caddyfile
sudo chown root:root /etc/caddy/Caddyfile
sudo chmod 644 /etc/caddy/Caddyfile

# Проверка конфигурации
echo "Проверка конфигурации..."
if sudo caddy validate --config /etc/caddy/Caddyfile; then
    echo "✅ Конфигурация валидна"
else
    echo "❌ Ошибка в конфигурации Caddy" >&2
    exit 1
fi

# Включение и запуск Caddy
echo "Запуск Caddy..."
sudo systemctl enable caddy
sudo systemctl restart caddy

# Ждём немного для запуска
sleep 2

# Проверка статуса
if sudo systemctl is-active --quiet caddy; then
    echo "✅ Caddy запущен и работает"
    sudo systemctl status caddy --no-pager -l
else
    echo "❌ Caddy не запустился. Логи:" >&2
    sudo journalctl -u caddy -n 50 --no-pager
    exit 1
fi

echo ""
echo "=== Установка завершена ==="
echo ""
echo "Проверьте что Docker контейнер работает:"
echo "  docker compose ps"
echo "  curl http://localhost:8000/health"
echo ""
echo "Проверьте доступность через Caddy:"
echo "  curl https://verbatis.ru/health"
echo "  curl -I https://verbatis.ru"
echo ""
echo "Логи Caddy:"
echo "  sudo journalctl -u caddy -f"
echo "  sudo tail -f /var/log/caddy/verbatis.log"
echo ""
echo "Если нужно перезапустить:"
echo "  sudo systemctl restart caddy"
