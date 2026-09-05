#!/bin/bash
# Генерация самоподписанного TLS-сертификата для HTTPS на голом IP (без домена).
# Сертификат шифрует трафик; браузер покажет предупреждение один раз
# ("не защищён" → принять исключение). Ставить что-либо на клиенты НЕ нужно.
#
# Использование:
#   scripts/gen_cert.sh 203.0.113.10        # ← IP вашего сервера
#   scripts/gen_cert.sh                     # возьмёт IP из .env (SERVER_IP=...)
#
# Кладёт cert.pem + key.pem в certs/. Эти файлы монтируются в контейнер (см. compose).
set -euo pipefail
cd "$(dirname "$0")/.."

IP="${1:-}"
if [ -z "$IP" ] && [ -f .env ]; then
    IP="$(sed -n 's/^[[:space:]]*SERVER_IP=//p' .env | tail -n 1)"
fi
if [ -z "$IP" ]; then
    echo "Укажите IP: scripts/gen_cert.sh <IP>  (или SERVER_IP=<IP> в .env)" >&2
    exit 1
fi

mkdir -p certs
DAYS=3650   # 10 лет — самоподписанный не продлевается автоматически

echo "Генерирую самоподписанный сертификат для IP $IP (срок $DAYS дней)..."
openssl req -x509 -nodes -newkey rsa:2048 \
    -keyout certs/key.pem \
    -out certs/cert.pem \
    -days "$DAYS" \
    -subj "/CN=$IP" \
    -addext "subjectAltName=IP:$IP"

chmod 600 certs/key.pem
echo "Готово:"
echo "  certs/cert.pem  (сертификат)"
echo "  certs/key.pem   (приватный ключ — секрет, не коммитить)"
echo ""
echo "Заход после деплоя: https://$IP:8443"
