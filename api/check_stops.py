"""
Vercel Serverless Function - Check ATR Trailing Stops

Cron job que verifica stops de posicoes abertas a cada 4 horas.
Envia notificacao via Telegram quando um stop e atingido.

Endpoint: GET /api/check_stops
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler

# Adicionar productsPositions ao path para imports
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "productsPositions"))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Importar dependencias do projeto
            from storage.sqlite_repo import SQLiteRepo
            from services.atr_stop_service import atualizar_stops_posicoes_abertas

            # Criar repo (conecta ao Supabase via SUPABASE_DB_URL)
            repo = SQLiteRepo()

            # Executar check de stops
            resultado = atualizar_stops_posicoes_abertas(repo, verbose=False)

            response = {
                "status": "ok",
                "timestamp": datetime.utcnow().isoformat(),
                "resultado": {
                    "updated": resultado.get("updated", 0),
                    "skipped": resultado.get("skipped", 0),
                    "unchanged": resultado.get("unchanged", 0),
                    "breached": resultado.get("breached", 0),
                    "errors": resultado.get("errors", [])
                }
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))

        except Exception as e:
            error_response = {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode("utf-8"))
