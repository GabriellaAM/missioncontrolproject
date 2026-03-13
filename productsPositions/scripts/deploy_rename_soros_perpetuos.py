"""
Deploy migration: Renomeia o produto "Soros Perpétuos 1" para "Soros Perpétuos"
e suas turmas para incluir "Perpétuos" no nome, renumerando de 2→1, 3→2, etc.

Idempotente: verifica se o produto e as turmas já foram alterados antes de executar.
Turmas já no formato "Soros Perpétuos Turma N" são ignoradas (evita duplicatas em deploys).

Uso:
  python -m productsPositions.scripts.deploy_rename_soros_perpetuos [--dry-run]
"""
import os
import sys
import re

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))
sys.path.insert(0, _here)

from dotenv import load_dotenv
load_dotenv(os.path.join(_here, '..', '..', '.env'))

db_url = os.getenv('SUPABASE_DB_URL', '').strip()
if not db_url or db_url.startswith('sqlite://'):
    print("[RENAME-PERP] Pulando: sem SUPABASE_DB_URL configurado")
    sys.exit(0)

DRY_RUN = '--dry-run' in sys.argv

TAG = "[RENAME-PERP]"

try:
    from storage.sqlite_repo import connect_pg
    conn = connect_pg(db_url)
    cursor = conn.cursor()

    # 1) Encontrar o produto "Soros Perpétuos 1"
    cursor.execute("SELECT id, nome FROM produtos WHERE nome LIKE 'Soros Perp%%'")
    rows = cursor.fetchall()

    if not rows:
        print(f"{TAG} Nenhum produto 'Soros Perp...' encontrado.")
        conn.close()
        sys.exit(0)

    print(f"{TAG} Produtos encontrados:")
    for r in rows:
        pid = r[0] if isinstance(r, (list, tuple)) else r['id']
        pnome = r[1] if isinstance(r, (list, tuple)) else r['nome']
        print(f"  id={pid}, nome='{pnome}'")

    # Buscar o produto que é "Soros Perpétuos 1" (com acento ou sem)
    prod_row = None
    for r in rows:
        pnome = (r[1] if isinstance(r, (list, tuple)) else r['nome']).strip()
        if re.match(r'^Soros Perp[eé]tuos\s*1$', pnome):
            prod_row = r
            break

    if not prod_row:
        # Verificar se já foi renomeado
        for r in rows:
            pnome = (r[1] if isinstance(r, (list, tuple)) else r['nome']).strip()
            if re.match(r'^Soros Perp[eé]tuos$', pnome):
                print(f"{TAG} Produto já renomeado para '{pnome}'. Verificando turmas...")
                prod_row = r
                break

    if not prod_row:
        print(f"{TAG} Produto 'Soros Perpétuos 1' não encontrado entre os resultados.")
        conn.close()
        sys.exit(0)

    prod_id = prod_row[0] if isinstance(prod_row, (list, tuple)) else prod_row['id']
    prod_nome_atual = (prod_row[1] if isinstance(prod_row, (list, tuple)) else prod_row['nome']).strip()

    # 2) Renomear o produto (se ainda não foi)
    novo_nome_produto = 'Soros Perpétuos'
    if prod_nome_atual != novo_nome_produto:
        print(f"{TAG} Renomeando produto: '{prod_nome_atual}' -> '{novo_nome_produto}'")
        if not DRY_RUN:
            cursor.execute("UPDATE produtos SET nome = %s WHERE id = %s", (novo_nome_produto, prod_id))
    else:
        print(f"{TAG} Produto já com nome correto: '{novo_nome_produto}'")

    # 3) Buscar turmas do produto
    cursor.execute(
        "SELECT id, nome FROM turmas WHERE produto_id = %s ORDER BY id",
        (prod_id,)
    )
    turmas = cursor.fetchall()

    if not turmas:
        print(f"{TAG} Nenhuma turma encontrada para produto_id={prod_id}")
    else:
        print(f"\n{TAG} Turmas atuais ({len(turmas)}):")
        for t in turmas:
            tid = t[0] if isinstance(t, (list, tuple)) else t['id']
            tnome = t[1] if isinstance(t, (list, tuple)) else t['nome']
            print(f"  id={tid}, nome='{tnome}'")

        # 4) Renomear turmas: extrair o número, decrementar 1, e incluir "Perpétuos"
        # IMPORTANTE: Só aplicar em turmas com formato ANTIGO. Turmas já no formato
        # "Soros Perpétuos Turma N" devem ser ignoradas, senão o script (rodado a cada
        # deploy) reaplicaria o decremento e criaria duplicatas (Turma 2→1, Turma 3→2...).
        PADRAO_JA_MIGRADO = re.compile(r'^Soros Perp[eé]tuos Turma \d+$')
        print(f"\n{TAG} Renumerando turmas:")
        for t in turmas:
            tid = t[0] if isinstance(t, (list, tuple)) else t['id']
            tnome = (t[1] if isinstance(t, (list, tuple)) else t['nome']).strip()

            # Pular turmas já no formato final (evita reaplicar em cada deploy)
            if PADRAO_JA_MIGRADO.match(tnome):
                print(f"  SKIP id={tid}: '{tnome}' -> já no formato final")
                continue

            # Extrair o número da turma atual
            # Padrões possíveis: "Soros Turma 2", "Soros Perpétuos Turma 2", "Turma 2", etc.
            match = re.search(r'[Tt]urma\s+(\d+)', tnome)
            if match:
                num_atual = int(match.group(1))
                novo_num = num_atual - 1
                if novo_num < 1:
                    print(f"  SKIP id={tid}: '{tnome}' -> número resultante seria {novo_num}, pulando")
                    continue
                novo_nome_turma = f"Soros Perpétuos Turma {novo_num}"
            else:
                # Se não tem número, tentar adicionar "Perpétuos" se não tem
                if 'perp' not in tnome.lower():
                    novo_nome_turma = tnome.replace('Soros', 'Soros Perpétuos')
                else:
                    novo_nome_turma = tnome
                    print(f"  SKIP id={tid}: '{tnome}' -> sem número de turma detectado, mantendo")
                    continue

            if tnome != novo_nome_turma:
                print(f"  id={tid}: '{tnome}' -> '{novo_nome_turma}'")
                if not DRY_RUN:
                    cursor.execute("UPDATE turmas SET nome = %s WHERE id = %s", (novo_nome_turma, tid))
            else:
                print(f"  id={tid}: '{tnome}' -> já correto")

    if DRY_RUN:
        print(f"\n{TAG} DRY-RUN: nenhuma alteração salva. Remova --dry-run para aplicar.")
        conn.rollback()
    else:
        conn.commit()
        print(f"\n{TAG} Alterações salvas com sucesso!")

    conn.close()

except Exception as e:
    print(f"{TAG} Erro: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
