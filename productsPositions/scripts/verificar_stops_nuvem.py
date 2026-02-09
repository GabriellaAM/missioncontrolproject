"""
Script para verificação automática de stops na nuvem (Bitbucket Pipelines).

Roda a cada hora via schedule, mas só executa o check efetivo a cada 4 horas
(00h, 04h, 08h, 12h, 16h, 20h UTC). Nas outras horas, faz exit imediato.

Quando um stop é atingido:
  - Envia notificação no Telegram (via notificacao_service)
  - Salva -1 no DB (para evitar re-notificação no próximo run)

Uso (chamado pelo bitbucket-pipelines.yml):
    python verificar_stops_nuvem.py
    python verificar_stops_nuvem.py --forcar   # Ignora filtro de hora (para testes)
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from services.atr_stop_service import atualizar_stops_posicoes_abertas


def main():
    parser = argparse.ArgumentParser(description='Verificação de stops na nuvem')
    parser.add_argument('--forcar', '-f', action='store_true', help='Forçar execução (ignorar filtro de hora)')
    args = parser.parse_args()

    agora = datetime.now(timezone.utc)
    hora_utc = agora.hour
    print(f"[Nuvem] {agora.strftime('%Y-%m-%d %H:%M UTC')}", flush=True)

    # Só roda a cada 4 horas (00, 04, 08, 12, 16, 20 UTC)
    if not args.forcar and hora_utc % 4 != 0:
        print(f"[Nuvem] Hora {hora_utc}h UTC - fora do intervalo de 4h. Saindo.", flush=True)
        sys.exit(0)

    print(f"[Nuvem] Hora {hora_utc}h UTC - executando verificação de stops...", flush=True)

    repo = SQLiteRepo()

    # Atualizar stops de TODAS as posições abertas (sem filtro de produto)
    resultado = atualizar_stops_posicoes_abertas(repo=repo, produto_id=None, verbose=True)

    # Resumo
    print(flush=True)
    print("=" * 40, flush=True)
    print("RESUMO", flush=True)
    print("=" * 40, flush=True)
    print(f"  Atualizados:  {resultado['updated']}", flush=True)
    print(f"  Inalterados:  {resultado['unchanged']}", flush=True)
    print(f"  Pulados:      {resultado['skipped']}", flush=True)
    print(f"  BREACHED:     {resultado['breached']}", flush=True)

    if resultado['errors']:
        print(f"\n  Erros ({len(resultado['errors'])}):", flush=True)
        for erro in resultado['errors']:
            print(f"    - {erro}", flush=True)

    # Exit code
    if resultado['errors']:
        print("\n[Nuvem] Finalizado com erros.", flush=True)
        sys.exit(1)
    elif resultado['breached'] > 0:
        print(f"\n[Nuvem] {resultado['breached']} stop(s) atingido(s)! Notificações enviadas.", flush=True)
    else:
        print("\n[Nuvem] Nenhum stop atingido. Tudo OK.", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
