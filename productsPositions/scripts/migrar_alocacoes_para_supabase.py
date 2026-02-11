"""
Migração one-shot: atualiza alocações no Supabase a partir dos CSVs (AC, EXC, HB, LC).

Uso: no Render, o startCommand chama este script antes do dashboard para que,
na primeira subida (ou quando você fizer deploy), o Supabase seja atualizado.

Quando o Supabase estiver atualizado, você pode:
  1. Remover a chamada a este script do startCommand no render.yaml
  2. Deletar este arquivo (migrar_alocacoes_para_supabase.py)
"""
from pathlib import Path
import os
import sys

# Incluir scripts/ e productsPositions para o import
_scripts = Path(__file__).resolve().parent
_root_pp = _scripts.parent
sys.path.insert(0, str(_scripts))
sys.path.insert(0, str(_root_pp))

from atualizar_alocacoes_csv import main, DIR_CSV, PRODUTOS_CSV

if __name__ == "__main__":
    # Diagnóstico para Render: env e CSVs
    has_url = bool((os.getenv("SUPABASE_DB_URL") or "").strip())
    print("[MIGRAÇÃO ALOCAÇÕES] SUPABASE_DB_URL definido:", has_url)
    if not has_url:
        print("[MIGRAÇÃO ALOCAÇÕES] AVISO: Sem SUPABASE_DB_URL o app usa SQLite local (dados efêmeros no Render).")
    print("[MIGRAÇÃO ALOCAÇÕES] Diretório CSVs:", DIR_CSV)
    for nome in PRODUTOS_CSV:
        p = DIR_CSV / f"{nome}.csv"
        print(f"  {nome}.csv existe: {p.exists()}")
    print("[MIGRAÇÃO ALOCAÇÕES] Iniciando atualização no banco...")
    try:
        n = main(dry_run=False)
        print("[MIGRAÇÃO ALOCAÇÕES] Concluída com sucesso. Total de alocações inseridas:", n)
        print("[MIGRAÇÃO ALOCAÇÕES] Iniciando dashboard em seguida.")
    except Exception as e:
        print("[MIGRAÇÃO ALOCAÇÕES] ERRO:", e)
        raise
