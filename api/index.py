"""
Vercel Serverless Function - Dashboard Web completo.

Serve o dashboard de Products & Positions como serverless function.
Todas as rotas (GET, POST) sao tratadas pelo DashboardHandler.
"""
import sys
from pathlib import Path

# Adicionar productsPositions ao path para imports do projeto
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "productsPositions"))

from scripts.servidor_dashboard import DashboardHandler

# Vercel espera a classe handler (lowercase) herdando BaseHTTPRequestHandler
handler = DashboardHandler
