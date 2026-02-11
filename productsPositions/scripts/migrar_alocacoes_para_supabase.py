"""
Migração one-shot: atualiza alocações no Supabase a partir dos CSVs (AC, EXC, HB, LC).

Uso: no Render, o startCommand chama este script antes do dashboard para que,
na primeira subida (ou quando você fizer deploy), o Supabase seja atualizado.

Quando o Supabase estiver atualizado, você pode:
  1. Remover a chamada a este script do startCommand no render.yaml
  2. Deletar este arquivo (migrar_alocacoes_para_supabase.py)
"""
from pathlib import Path
import sys

# Incluir scripts/ e productsPositions para o import
_scripts = Path(__file__).resolve().parent
_root_pp = _scripts.parent
sys.path.insert(0, str(_scripts))
sys.path.insert(0, str(_root_pp))

from atualizar_alocacoes_csv import main

if __name__ == "__main__":
    print("[MIGRAÇÃO ALOCAÇÕES] Iniciando atualização no banco (Supabase se SUPABASE_DB_URL estiver definido)...")
    try:
        n = main(dry_run=False)
        print("[MIGRAÇÃO ALOCAÇÕES] Concluída com sucesso. Total de alocações inseridas:", n)
        print("[MIGRAÇÃO ALOCAÇÕES] Iniciando dashboard em seguida.")
    except Exception as e:
        print("[MIGRAÇÃO ALOCAÇÕES] ERRO:", e)
        raise
