"""
Script para verificacao automatica de stops via cron/agendador.

Verifica todos os stops ATR de posicoes abertas e envia
notificacao via Telegram quando um stop e atingido.

Uso:
    python verificar_stops_cron.py
    python verificar_stops_cron.py --verbose

Agendar no Windows Task Scheduler para rodar a cada 4 horas.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Garantir que o diretorio productsPositions esta no path
script_dir = Path(__file__).parent
products_dir = script_dir.parent
sys.path.insert(0, str(products_dir))

from storage.sqlite_repo import SQLiteRepo
from services.atr_stop_service import atualizar_stops_posicoes_abertas

# Log file para acompanhar execucoes
LOG_FILE = script_dir / "cron_stops.log"


def log(msg):
    """Escreve mensagem no log e no stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    log("=" * 50)
    log("VERIFICACAO AUTOMATICA DE STOPS")
    log("=" * 50)

    try:
        repo = SQLiteRepo()
        log("Conectado ao Supabase.")
    except Exception as e:
        log(f"ERRO ao conectar: {e}")
        sys.exit(1)

    try:
        resultado = atualizar_stops_posicoes_abertas(repo, verbose=verbose)

        log(f"Resultado:")
        log(f"  Atualizados: {resultado.get('updated', 0)}")
        log(f"  Sem alteracao: {resultado.get('unchanged', 0)}")
        log(f"  Pulados (sem ATR): {resultado.get('skipped', 0)}")
        log(f"  STOPS ATINGIDOS: {resultado.get('breached', 0)}")

        erros = resultado.get("errors", [])
        if erros:
            log(f"  Erros ({len(erros)}):")
            for e in erros[:10]:
                log(f"    - {e}")

        if resultado.get("breached", 0) > 0:
            log(">>> Notificacoes Telegram enviadas para stops atingidos!")

    except Exception as e:
        log(f"ERRO durante verificacao: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

    log("Verificacao concluida.")
    log("")


if __name__ == "__main__":
    main()
