"""
Script para configurar ATR Trailing Stop em posições existentes.

Permite definir atr_period e atr_multiplier por posição.
Posições com atr_multiplier configurado terão stops calculados automaticamente.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao
from services.atr_stop_service import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_ATR_MULTIPLIER,
    calcular_stop_para_posicao
)
import psycopg2
import pandas as pd


def main():
    imprimir_titulo("CONFIGURAR ATR TRAILING STOP")

    repo = SQLiteRepo()

    imprimir_secao("POSICOES ABERTAS (por produto)")

    from storage.sqlite_repo import connect_pg
    conn = connect_pg(repo.db_url)
    df_posicoes = pd.read_sql_query('''
        SELECT p.id, p.ativo, p.side, p.data_entrada, p.preco_entrada,
               p.coingecko_id, p.atr_period, p.atr_multiplier,
               pr.nome as produto_nome
        FROM posicoes p
        JOIN produtos pr ON p.produto_id = pr.id
        WHERE p.status = 'open'
        ORDER BY pr.nome, p.data_entrada
    ''', conn)
    conn.close()

    if df_posicoes.empty:
        print("Nenhuma posição aberta encontrada.")
        return

    for produto in df_posicoes['produto_nome'].unique():
        print(f"\n+ {produto}:")
        prod_pos = df_posicoes[df_posicoes['produto_nome'] == produto]

        for _, row in prod_pos.iterrows():
            atr_period = row['atr_period'] if pd.notna(row['atr_period']) else None
            atr_mult = row['atr_multiplier'] if pd.notna(row['atr_multiplier']) else None
            atr_status = f"ATR({int(atr_period)}, {atr_mult})" if atr_mult else "Manual"

            print(f"    ID: {row['id']} | {row['ativo']} | {row['side']} | "
                  f"{row['data_entrada']} @ ${row['preco_entrada']:.2f} | {atr_status}")

    print()

    # Selecionar posição
    posicao_id = obter_input("ID da posição para configurar: ", tipo=int, obrigatorio=True)

    # Verificar se posição existe
    posicao = repo.carregar_posicao(posicao_id)
    if not posicao:
        print(f"Posição com ID {posicao_id} não encontrada!")
        return

    imprimir_secao("POSIÇÃO SELECIONADA")
    print(f"  Ativo: {posicao['ativo']}")
    print(f"  Side: {posicao['side']}")
    print(f"  CoinGecko ID: {posicao.get('coingecko_id', 'N/A')}")
    print(f"  Data entrada: {posicao['data_entrada']}")
    print(f"  Preco entrada: ${posicao['preco_entrada']:.2f}")

    # Carregar config ATR atual (agora está na tabela posicoes)
    current_period = posicao.get('atr_period')
    current_mult = posicao.get('atr_multiplier')

    print(f"\n  Config ATR atual:")
    print(f"    Period: {current_period or 'Não configurado'}")
    print(f"    Multiplier: {current_mult or 'Não configurado'}")

    # Opções
    imprimir_secao("AÇÕES")
    print("  1. Configurar/Atualizar ATR params")
    print("  2. Remover config ATR (voltar para manual)")
    print("  3. Simular cálculo de stop")
    print("  4. Cancelar")

    opcao = obter_input("\nEscolha: ", tipo=int, obrigatorio=True)

    if opcao == 1:
        # Configurar ATR
        print(f"\n(Enter para usar default)")
        period_input = obter_input(
            f"ATR Period [{DEFAULT_ATR_PERIOD}]: ",
            tipo=int,
            obrigatorio=False
        )
        atr_period = period_input if period_input else DEFAULT_ATR_PERIOD

        mult_input = obter_input(
            f"ATR Multiplier [{DEFAULT_ATR_MULTIPLIER}]: ",
            tipo=float,
            obrigatorio=False
        )
        atr_multiplier = mult_input if mult_input else DEFAULT_ATR_MULTIPLIER

        # Simular antes de salvar
        print(f"\nSimulando com ATR({atr_period}, {atr_multiplier})...")

        coingecko_id = posicao.get('coingecko_id')
        if coingecko_id:
            stop, breached, erro = calcular_stop_para_posicao(
                coingecko_id=coingecko_id,
                side=posicao['side'],
                data_entrada=posicao['data_entrada'],
                atr_period=atr_period,
                atr_multiplier=atr_multiplier
            )

            if erro:
                print(f"  Erro: {erro}")
            elif breached:
                print(f"  AVISO: Stop já foi breached com esses parâmetros!")
            else:
                print(f"  Stop calculado: ${stop:.4f}")
        else:
            print("  Não foi possível simular (sem coingecko_id)")

        confirmar = obter_input(
            f"\nSalvar configuração ATR({atr_period}, {atr_multiplier})? (s/n): ",
            opcoes=["s", "n", "S", "N"],
            obrigatorio=True
        ).lower()

        if confirmar == "s":
            repo.atualizar_posicao(
                posicao_id,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier
            )
            print(f"\nConfiguração salva! O stop será calculado automaticamente ao visualizar.")
        else:
            print("\nOperação cancelada.")

    elif opcao == 2:
        # Remover config ATR
        confirmar = obter_input(
            "Remover configuração ATR (voltar para gestão manual)? (s/n): ",
            opcoes=["s", "n", "S", "N"],
            obrigatorio=True
        ).lower()

        if confirmar == "s":
            repo.atualizar_posicao(
                posicao_id,
                atr_period=None,
                atr_multiplier=None
            )
            print("\nConfiguração ATR removida. Posição agora usa stops manuais.")
        else:
            print("\nOperação cancelada.")

    elif opcao == 3:
        # Simular
        period_input = obter_input(
            f"ATR Period [{current_period or DEFAULT_ATR_PERIOD}]: ",
            tipo=int,
            obrigatorio=False
        )
        atr_period = period_input if period_input else (current_period or DEFAULT_ATR_PERIOD)

        mult_input = obter_input(
            f"ATR Multiplier [{current_mult or DEFAULT_ATR_MULTIPLIER}]: ",
            tipo=float,
            obrigatorio=False
        )
        atr_multiplier = mult_input if mult_input else (current_mult or DEFAULT_ATR_MULTIPLIER)

        coingecko_id = posicao.get('coingecko_id')
        if not coingecko_id:
            print("Não é possível simular sem coingecko_id")
            return

        print(f"\nSimulando ATR({atr_period}, {atr_multiplier})...")

        stop, breached, erro = calcular_stop_para_posicao(
            coingecko_id=coingecko_id,
            side=posicao['side'],
            data_entrada=posicao['data_entrada'],
            atr_period=atr_period,
            atr_multiplier=atr_multiplier
        )

        if erro:
            print(f"Erro: {erro}")
        elif breached:
            print(f"AVISO: Stop já foi breached!")
        else:
            print(f"Stop calculado: ${stop:.4f}")

            # Comparar com preço atual
            from services.valor_diario_service import obter_preco_atual
            preco_atual = obter_preco_atual(coingecko_id)
            if preco_atual:
                distancia_pct = abs(preco_atual - stop) / preco_atual * 100
                print(f"Preço atual: ${preco_atual:.4f}")
                print(f"Distância: {distancia_pct:.2f}%")

    else:
        print("\nOperação cancelada.")


if __name__ == "__main__":
    main()
