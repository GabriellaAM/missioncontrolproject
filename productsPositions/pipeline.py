from domain.tipo import Tipo
from domain.produto import Produto
from domain.valor_diario import ValorDiario
from services.posicao_service import PosicaoService
from services.alocacao_service import AlocacaoService
from services.carteira_service import CarteiraService
from services.valor_diario_service import ValorDiarioService
from storage.parquet_repo import ParquetRepo
from analytics.queries import posicoes_abertas, posicoes_fechadas, alocacoes_do_produto, resumo_alocacoes, carteira_do_produto, resumo_completo_produto, valores_da_posicao, valores_do_ativo

# Registrar tipo (se não existir) e criar produto
repo = ParquetRepo()
repo.registrar_tipo("Cripto", "Criptomoedas")  # Garante que o tipo existe

tipo = Tipo("Cripto")
produto = Produto("SOROS_1", "2025-01-01", tipo)

# Abrir posição (com coingecko_id)
p = PosicaoService.abrir(
    produto,
    ativo="BTC",
    side="long",
    data="2025-01-10",
    preco=100000,
    coingecko_id="bitcoin"  # ID do CoinGecko para buscar preços
)

# Persistir no Parquet (valida que tipo existe)
produto_id = repo.salvar_produto(produto)
posicao_id = repo.salvar_posicao(produto_id, p)

# Consultas rápidas
print("\nPOSIÇÕES ABERTAS:\n", posicoes_abertas(produto_id))
print("\nHISTÓRICO:\n", posicoes_fechadas(produto_id))

# Valores diários por ATIVO (não por posição - evita duplicação)
# Exemplo: garantir histórico do BTC
def fonte_precos_exemplo(ativo, data_inicio, data_fim):
    """Exemplo de função que retorna preços históricos"""
    # Em produção, isso viria de uma API (CoinGecko, Binance, etc.)
    return [
        {"data": "2025-01-10", "preco": 100000, "valor_usd": 1000},
        {"data": "2025-01-11", "preco": 102000, "valor_usd": 1020},
    ]

# Garantir que o ativo BTC tenha histórico (só baixa se não existir)
resultado = ValorDiarioService.garantir_historico_ativo(
    ativo="BTC",
    fonte_precos=fonte_precos_exemplo
)
print(f"\nHISTÓRICO BTC: {resultado}")

# Consultar valores diários do ativo
print("\nVALORES DIÁRIOS DO ATIVO BTC:\n", valores_do_ativo("BTC"))
print("\nVALORES DIÁRIOS DA POSIÇÃO (via JOIN):\n", valores_da_posicao(posicao_id))

# Exemplo de alocações
# Criar alocação manual para a posição
alocacao = AlocacaoService.criar_alocacao(
    produto_id=produto_id,
    posicao_id=posicao_id,
    percentual=100.0,  # 100% do capital nesta posição
    valor_usd=1000,
    data="2025-01-10"
)
alocacao_id = repo.salvar_alocacao(produto_id, alocacao)

# Exemplo: alocação equitativa entre múltiplas posições
# alocacoes = AlocacaoService.alocar_equitativamente(
#     produto_id=produto_id,
#     posicoes_ids=[posicao_id],
#     total_capital=1000
# )
# repo.salvar_alocacoes(produto_id, alocacoes)

print("\nALOCAÇÕES DO PRODUTO:\n", alocacoes_do_produto(produto_id))
print("\nRESUMO DE ALOCAÇÕES:\n", resumo_alocacoes(produto_id))

# Exemplo de carteira
# Opção 1: Criar carteira manualmente
carteira_manual = CarteiraService.criar_carteira(
    produto_id=produto_id,
    valor_disponivel=5000.0,  # Capital disponível
    valor_investido=1000.0,   # Valor já investido
    pnl_nao_realizado=50.0    # PnL não realizado
)
repo.salvar_carteira(carteira_manual)

# Opção 2: Preencher carteira automaticamente a partir das alocações
# (requer preços atuais para calcular PnL não realizado)
preco_atual_btc = 102000  # Preço atual do BTC
carteira_auto = CarteiraService.preencher_carteira(
    produto_id=produto_id,
    preco_atual_por_posicao={posicao_id: preco_atual_btc}
)
repo.salvar_carteira(carteira_auto)

# Opção 3: Criar carteira com capital inicial e recalcular tudo
carteira_completa = CarteiraService.atualizar_carteira_com_capital_inicial(
    produto_id=produto_id,
    capital_inicial=10000.0,
    preco_atual_por_posicao={posicao_id: preco_atual_btc}
)
repo.salvar_carteira(carteira_completa)

# Carregar carteira do banco
carteira_carregada = repo.carregar_carteira(produto_id)
if carteira_carregada:
    print(f"\nCARTEIRA DO PRODUTO:\n{carteira_carregada}")
    print(f"\nValor Total: ${carteira_carregada.valor_total:.2f}")

print("\nCARTEIRA DO PRODUTO:\n", carteira_do_produto(produto_id))
print("\nRESUMO COMPLETO DO PRODUTO:\n", resumo_completo_produto(produto_id))
