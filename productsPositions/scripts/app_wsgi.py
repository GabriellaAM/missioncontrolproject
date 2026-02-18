"""
App WSGI para rodar o dashboard com Gunicorn (ou outro servidor WSGI).
Encaminha todas as requisições para o servidor do dashboard em uma thread interna.
"""
import os
import threading
import urllib.request
import urllib.error

# Flask só é importado quando for usar Gunicorn (evita dependência no run direto)
try:
    from flask import Flask, Response, request
except ImportError:
    Flask = Response = request = None

_DASHBOARD_PORT = None
_DASHBOARD_LOCK = threading.Lock()


def _get_dashboard_port():
    global _DASHBOARD_PORT
    if _DASHBOARD_PORT is not None:
        return _DASHBOARD_PORT
    with _DASHBOARD_LOCK:
        if _DASHBOARD_PORT is not None:
            return _DASHBOARD_PORT
        from productsPositions.scripts.servidor_dashboard import iniciar_servidor_background
        _DASHBOARD_PORT = iniciar_servidor_background(host='127.0.0.1', porta=0, silent=True)
    return _DASHBOARD_PORT


def _proxy_to_dashboard():
    port = _get_dashboard_port()
    path = (request.full_path.rstrip('?') or '/')
    if not path.startswith('/'):
        path = '/' + path
    url = f"http://127.0.0.1:{port}{path}"

    headers = {}
    for k, v in request.headers:
        if k.lower() in ('host', 'connection', 'transfer-encoding'):
            continue
        headers[k] = v

    try:
        if request.method == 'GET':
            req = urllib.request.Request(url, headers=headers, method='GET')
        elif request.method == 'POST':
            req = urllib.request.Request(url, data=request.get_data(), headers=headers, method='POST')
        elif request.method == 'OPTIONS':
            req = urllib.request.Request(url, headers=headers, method='OPTIONS')
        else:
            req = urllib.request.Request(url, data=request.get_data(), headers=headers, method=request.method)

        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_headers = [(k, v) for k, v in resp.headers.items()]
            return Response(resp.read(), status=resp.status, headers=resp_headers)
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b''
        return Response(body, status=e.code, headers=[(k, v) for k, v in e.headers.items()])
    except Exception as e:
        return Response(str(e).encode(), status=502)


def create_app():
    if Flask is None:
        raise RuntimeError("Instale flask: pip install flask")
    app = Flask(__name__)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>", methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"])
    def catch_all(path):
        return _proxy_to_dashboard()

    return app


app = create_app()
