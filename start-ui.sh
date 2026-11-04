#!/bin/sh
set -eu

PORT="${PORT:-10000}"
export PORT

envsubst '${PORT}' \
    < /etc/nginx/templates/default.conf.template \
    > /etc/nginx/conf.d/default.conf

streamlit run UI.py --server.address=127.0.0.1 --server.port=8501 &
STREAMLIT_PID=$!

nginx -g "daemon off;" &
NGINX_PID=$!

stop_services() {
    kill "$STREAMLIT_PID" "$NGINX_PID" 2>/dev/null || true
    wait "$STREAMLIT_PID" "$NGINX_PID" 2>/dev/null || true
}

trap stop_services INT TERM

while kill -0 "$STREAMLIT_PID" 2>/dev/null \
    && kill -0 "$NGINX_PID" 2>/dev/null; do
    sleep 1
done

stop_services
exit 1
