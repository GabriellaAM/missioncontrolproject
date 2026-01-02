"""
Script para criar uma nova posição em um produto existente
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.posicao_service import PosicaoService
from services.valor_diario_service import ValorDiarioService
from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, validar_data, imprimir_titulo, imprimir_secao

def main():
    imprimir_titulo("CRIAR POSIÇÃO")
    
    repo = SQLiteRepo()
    
    # Listar produtos disponíveis
    produtos = repo.listar_produtos()
    if not produtos:
        print("❌ Nenhum produto encontrado. Crie um produto primeiro usando criar_produto.py")
        return None
    
    print("Produtos disponíveis:")
    for produto_dict in produtos:
        print(f"  ID: {produto_dict['id']} - {produto_dict['nome']} ({produto_dict['tipo']})")
    
    produto_id = obter_input("\nID do produto: ", tipo=int, obrigatorio=True)
    
    # Verificar se produto existe e carregar como objeto
    produto = repo.carregar_produto_objeto(produto_id)
    if not produto:
        print(f"❌ Produto com ID {produto_id} não encontrado!")
        return None
    
    imprimir_secao("DADOS DA POSIÇÃO")
    
    ativo = obter_input("Ativo (ex: BTC, ETH): ", obrigatorio=True).upper()

    # Para produtos Spot, não faz sentido pedir long/short.
    # Nesses casos, definimos internamente como 'long' apenas para satisfazer o schema.
    tipo_valor = produto.tipo
    if isinstance(tipo_valor, str):
        tipo_nome = tipo_valor
    else:
        # Quando carregado via SQLiteRepo, tipo é um objeto Tipo com atributo .nome
        tipo_nome = getattr(tipo_valor, "nome", "")
    tipo_produto = (tipo_nome or "").lower()

    if "spot" in tipo_produto:
        side = "long"
        print("\nℹ️ Produto do tipo Spot detectado: side será definido internamente como 'long'.")
    else:
        side = obter_input("Side (long/short): ", opcoes=["long", "short"], obrigatorio=True)
    data_entrada = obter_input("Data de entrada (YYYY-MM-DD): ", obrigatorio=True)
    data_entrada = validar_data(data_entrada, "Data de entrada")
    
    preco_entrada = obter_input("Preco de entrada: ", tipo=float, obrigatorio=True)

    # Carregar configuração de atributos do produto
    atributos_config = repo.carregar_atributos_config(produto_id)
    atributos_nomes = {c['atributo_nome'] for c in atributos_config}

    # Verificar colunas usadas nas visualizações do produto
    # e adicionar atributos que estão nas visualizações mas não na config
    visualizacoes = repo.listar_visualizacoes(produto_id)
    colunas_viz = set()
    for viz in visualizacoes:
        colunas_viz.update(viz.get('colunas', []))

    # Colunas que são atributos de posição (não calculados nem de posição básica)
    colunas_posicao_basicas = {'id', 'ativo', 'side', 'data_entrada', 'preco_entrada',
                               'data_saida', 'preco_saida', 'status', 'coingecko_id', 'produto_id'}
    colunas_calculadas = {'pnl', 'pnl_valor', 'preco_atual', 'preco_atual_total',
                          'preco_saida_total', 'preco_entrada_total', 'stop_atual',
                          'risco_stop', 'rr', 'alocacao'}

    # Atributos necessários que estão nas visualizações
    atributos_necessarios = colunas_viz - colunas_posicao_basicas - colunas_calculadas

    # Adicionar atributos que estão nas visualizações mas não na config
    for atrib in atributos_necessarios:
        if atrib not in atributos_nomes:
            # Determinar tipo baseado no nome do atributo
            if atrib in ['quantidade', 'alvo1', 'alvo2']:
                tipo_atrib = 'float'
            elif atrib in ['perfil', 'motivo']:
                tipo_atrib = 'text'
            else:
                tipo_atrib = 'text'  # Default

            # Adicionar ao config do produto
            try:
                repo.adicionar_atributo_config(produto_id, atrib, tipo_atrib)
                print(f"   Atributo '{atrib}' adicionado automaticamente (usado nas visualizacoes)")
            except Exception:
                pass  # Ignorar se já existe

    # Recarregar configuração após adicionar atributos
    atributos_config = repo.carregar_atributos_config(produto_id)

    # Mapear tipos para obter_input
    tipo_map = {
        'text': str,
        'float': float,
        'int': int,
        'date': str
    }

    # Coletar atributos dinamicamente
    atributos_valores = {}

    for config in atributos_config:
        nome = config['atributo_nome']
        label = config['atributo_label'] or nome.replace('_', ' ').title()
        tipo = tipo_map.get(config['atributo_tipo'], str)
        obrigatorio = bool(config['obrigatorio'])

        # Pular preco_entrada_total (será calculado automaticamente se tiver quantidade)
        if nome == 'preco_entrada_total':
            continue

        valor = obter_input(f"{label}: ", tipo=tipo, obrigatorio=obrigatorio)
        if valor is not None:
            atributos_valores[nome] = valor

    # Calcular preco_entrada_total se tiver quantidade
    if 'quantidade' in atributos_valores and atributos_valores['quantidade']:
        preco_entrada_total = atributos_valores['quantidade'] * preco_entrada
        atributos_valores['preco_entrada_total'] = preco_entrada_total
        print(f"   Preco de entrada total: ${preco_entrada_total:,.2f}")

    coingecko_id = obter_input("CoinGecko ID (ex: bitcoin, ethereum): ", obrigatorio=False)
    
    # Abrir posição
    p = PosicaoService.abrir(
        produto,
        ativo=ativo,
        side=side,
        data=data_entrada,
        preco=preco_entrada,
        coingecko_id=coingecko_id if coingecko_id else None
    )
    
    # Adicionar stop inicial (opcional)
    adicionar_stop = obter_input("\nDeseja adicionar um stop inicial? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True).lower()
    
    if adicionar_stop == "s":
        data_stop = obter_input("Data do stop (YYYY-MM-DD): ", obrigatorio=True)
        data_stop = validar_data(data_stop, "Data do stop")
        valor_stop = obter_input("Valor do stop: ", tipo=float, obrigatorio=True)
        p.adicionar_stop(data_stop, valor_stop)
        print(f"✅ Stop adicionado: {data_stop} - ${valor_stop:.2f}")
    
    # Salvar posição
    posicao_id = repo.salvar_posicao(produto_id, p)

    # Salvar atributos específicos do produto (se houver algum)
    if atributos_valores:
        try:
            repo.salvar_atributos_posicao(
                posicao_id,
                produto_id,
                **atributos_valores
            )
        except Exception as e:
            print(f"⚠️  Não foi possível salvar atributos da posição: {e}")

    print(f"\n✅ Posição criada com ID: {posicao_id}")
    print(f"   Ativo: {ativo}")
    print(f"   Side: {side}")
    print(f"   Preço de entrada: ${preco_entrada:.2f}")

    # Exibir atributos dinâmicos
    for config in atributos_config:
        nome = config['atributo_nome']
        label = config['atributo_label'] or nome.replace('_', ' ').title()
        if nome in atributos_valores and atributos_valores[nome] is not None:
            valor = atributos_valores[nome]
            if isinstance(valor, float):
                if nome in ['quantidade']:
                    print(f"   {label}: {valor:,.4f}")
                else:
                    print(f"   {label}: ${valor:,.2f}")
            else:
                print(f"   {label}: {valor}")
    
    # Verificar se dados do CoinGecko estão disponíveis
    if coingecko_id:
        imprimir_secao("VERIFICANDO DADOS DO COINGECKO")
        print(f"📥 Verificando disponibilidade de dados do CoinGecko para {ativo}...")
        try:
            valores = ValorDiarioService.ler_valores_do_coingecko(coingecko_id)
            if valores:
                print(f"✅ {len(valores)} valores disponíveis no CoinGecko")
                print(f"   Os valores serão lidos diretamente da fonte quando necessário")
            else:
                print(f"⚠️  Nenhum dado encontrado para {coingecko_id}")
                print(f"   Verifique se o arquivo existe em: data_parquet/crypto_data/coingecko/{coingecko_id}/data.parquet")
        except Exception as e:
            print(f"⚠️  Erro ao verificar dados do CoinGecko: {str(e)}")
    
    return posicao_id

if __name__ == "__main__":
    main()

