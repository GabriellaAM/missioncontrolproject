"""
Vercel Serverless Function - Dashboard Web completo.

Serve o dashboard de Products & Positions como serverless function.
Todas as rotas (GET, POST) sao tratadas pelo DashboardHandler.
"""
import sys
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler

# Adicionar productsPositions ao path para imports do projeto
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "productsPositions"))
sys.path.insert(0, str(_root / "productsPositions" / "scripts"))

try:
    from scripts.servidor_dashboard import DashboardHandler
    handler = DashboardHandler
except Exception as e:
    # Se o import falhar, criar um handler de erro que mostra o problema
    _import_error = traceback.format_exc()

    class handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            msg = f"Erro ao importar DashboardHandler:\n\n{_import_error}"
            self.wfile.write(msg.encode("utf-8"))

        def do_POST(self):
            self.do_GET()
