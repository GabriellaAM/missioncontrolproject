"""
Script para atualizar stops automáticos baseados em ATR Trailing Stop.

Atualiza stops apenas para posições que têm atr_multiplier configurado.
Posições sem atr_multiplier são ignoradas (mantêm gestão manual de stops).

Uso:
    python atualizar_stops_atr.py              # Atualiza todas as posições abertas
    python atualizar_stops_atr.py --produto 1  # Atualiza apenas produto específico
    python atualizar_stops_atr.py --quiet      # Modo silencioso (sem output)
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from services.atr_stop_service import atualizar_stops_posicoes_abertas
from utils.cli_utils import imprimir_titulo, imprimir_secao


def main():
    parser = argparse.ArgumentParser(
        description='Atualiza stops automáticos baseados em ATR Trailing Stop'
    )
    parser.add_argument(
        '--produto', '-p',
        type=int,
        help='ID do produto (opcional - se não especificado, atualiza todos)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Modo silencioso (sem output detalhado)'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        imprimir_titulo("ATUALIZAÇÃO DE STOPS ATR")

    repo = SQLiteRepo()

    if verbose:
        if args.produto:
            imprimir_secao(f"ATUALIZANDO PRODUTO {args.produto}")
        else:
            imprimir_secao("ATUALIZANDO TODAS AS POSIÇÕES ABERTAS")
        print()

    resultado = atualizar_stops_posicoes_abertas(
        repo=repo,
        produto_id=args.produto,
        verbose=verbose
    )

    if verbose:
        print()
        imprimir_secao("RESUMO")
        print(f"  Atualizados: {resultado['updated']}")
        print(f"  Inalterados: {resultado['unchanged']}")
        print(f"  Pulados (sem ATR config): {resultado['skipped']}")
        print(f"  Stops breached: {resultado['breached']}")

        if resultado['errors']:
            print(f"\n  Erros ({len(resultado['errors'])}):")
            for erro in resultado['errors']:
                print(f"    - {erro}")

        print()

    # Exit code: 0 se sucesso, 1 se houve erros
    sys.exit(0 if not resultado['errors'] else 1)


if __name__ == "__main__":
    main()
