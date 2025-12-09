from domain.valor_diario import ValorDiario
from storage.parquet_repo import ParquetRepo
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Serviço responsável por gerenciar valores diários de ativos
class ValorDiarioService:
    
    @staticmethod
    def _obter_caminho_coingecko(coingecko_id):
        """Obtém o caminho do arquivo Parquet do CoinGecko"""
        # Assumir que estamos em productsPositions/
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        caminho = project_root / "data_parquet" / "crypto_data" / "coingecko" / coingecko_id / "data.parquet"
        return caminho
    
    @staticmethod
    def ler_valores_do_coingecko(coingecko_id, data_inicio=None, data_fim=None):
        """
        Lê valores diários do diretório CoinGecko
        
        Args:
            coingecko_id: ID do CoinGecko (ex: "bitcoin")
            data_inicio: Data inicial (YYYY-MM-DD) - opcional
            data_fim: Data final (YYYY-MM-DD) - opcional
        
        Returns:
            list: Lista de dicionários [{"data": "2025-01-10", "preco": 100000}, ...]
        """
        caminho = ValorDiarioService._obter_caminho_coingecko(coingecko_id)
        
        if not caminho.exists():
            return []
        
        try:
            df = pd.read_parquet(caminho)
            
            # Converter timestamp para string de data
            if 'timestamp' in df.columns:
                # Converter para datetime se necessário e depois para string
                df['data'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
            elif 'date' in df.columns:
                df['data'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            else:
                # Tentar usar o índice se for datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    df['data'] = df.index.strftime('%Y-%m-%d')
                else:
                    raise ValueError("Não foi possível encontrar coluna de data no arquivo")
            
            # Usar 'close' como preço (ou 'open' se não houver close)
            if 'close' in df.columns:
                df['preco'] = df['close']
            elif 'open' in df.columns:
                df['preco'] = df['open']
            else:
                raise ValueError("Não foi possível encontrar coluna de preço no arquivo")
            
            # Filtrar por data se fornecido
            if data_inicio:
                df = df[df['data'] >= data_inicio]
            if data_fim:
                df = df[df['data'] <= data_fim]
            
            # Ordenar por data
            df = df.sort_values('data')
            
            # Converter para lista de dicionários
            valores = []
            for _, row in df.iterrows():
                valores.append({
                    'data': str(row['data']),
                    'preco': float(row['preco'])
                })
            
            return valores
            
        except Exception as e:
            # Retornar lista vazia em caso de erro (não quebrar o fluxo)
            return []
    
    @staticmethod
    def importar_do_coingecko(ativo, coingecko_id, data_inicio=None, data_fim=None):
        """
        Importa valores diários do diretório CoinGecko para o sistema
        
        Args:
            ativo: Nome do ativo (ex: "BTC")
            coingecko_id: ID do CoinGecko (ex: "bitcoin")
            data_inicio: Data inicial (YYYY-MM-DD) - opcional
            data_fim: Data final (YYYY-MM-DD) - opcional
        
        Returns:
            dict: Resultado da operação {'inseridos': int, 'ignorados': int}
        """
        repo = ParquetRepo()
        
        # Ler valores do CoinGecko
        valores_coingecko = ValorDiarioService.ler_valores_do_coingecko(
            coingecko_id, data_inicio, data_fim
        )
        
        if not valores_coingecko:
            return {
                'inseridos': 0,
                'ignorados': 0,
                'mensagem': f'Nenhum dado encontrado para {coingecko_id}'
            }
        
        # Converter para objetos ValorDiario
        valores_diarios = []
        for item in valores_coingecko:
            valor = ValorDiario(
                ativo=ativo,
                data=item['data'],
                preco=item['preco']
            )
            valores_diarios.append(valor)
        
        # Salvar no sistema
        resultado = repo.salvar_valores_diarios_ativo(ativo, valores_diarios)
        
        return resultado

    @staticmethod
    def verificar_se_ativo_ja_rastreado(ativo):
        """
        Verifica se um ativo já tem valores diários no banco
        
        Args:
            ativo: Nome do ativo
        
        Returns:
            bool: True se já está rastreado, False caso contrário
        """
        repo = ParquetRepo()
        rastreado = repo.verificar_ativo_rastreado(ativo)
        return rastreado is not None

    @staticmethod
    def baixar_historico_ativo(ativo, data_inicio=None, data_fim=None, fonte_precos=None):
        """
        Baixa histórico de valores diários de um ativo
        
        Args:
            ativo: Nome do ativo
            data_inicio: Data inicial do histórico (opcional, padrão: 1 ano atrás)
            data_fim: Data final do histórico (opcional, padrão: hoje)
            fonte_precos: Função que retorna preços (ativo, data_inicio, data_fim) -> list
        
        Returns:
            dict: Resultado da operação {'inseridos': int, 'ignorados': int}
        
        Note:
            Esta função deve ser implementada com uma fonte real de preços
            (ex: API de criptomoedas, banco de dados externo, etc.)
        """
        repo = ParquetRepo()
        
        # Verificar se já está rastreado
        rastreado = repo.verificar_ativo_rastreado(ativo)
        
        if rastreado:
            # Se já está rastreado, só atualizar valores recentes
            # (não baixar tudo de novo)
            data_ultima = rastreado.get('data_ultima_atualizacao')
            if data_ultima:
                data_inicio = data_ultima
            else:
                # Se não tem data de atualização, usar data do último valor
                valores = repo.obter_valores_diarios_ativo(ativo)
                if valores:
                    data_inicio = valores[-1][0]  # Última data
                else:
                    data_inicio = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            # Primeira vez - baixar histórico completo
            if data_inicio is None:
                data_inicio = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        if data_fim is None:
            data_fim = datetime.now().strftime("%Y-%m-%d")
        
        # Se não há fonte de preços, retornar erro
        if fonte_precos is None:
            raise ValueError("fonte_precos é obrigatório. Implemente uma função que retorna preços históricos.")
        
        # Buscar preços da fonte
        precos_historicos = fonte_precos(ativo, data_inicio, data_fim)
        
        # Converter para objetos ValorDiario
        valores_diarios = []
        for item in precos_historicos:
            if isinstance(item, dict):
                valor = ValorDiario(
                    ativo=ativo,
                    data=item.get('data'),
                    preco=item.get('preco')
                )
            else:
                # Assumir formato (data, preco) ou (data, preco, valor_usd) para compatibilidade
                valor = ValorDiario(
                    ativo=ativo,
                    data=item[0],
                    preco=item[1]
                )
            valores_diarios.append(valor)
        
        # Salvar no banco (com verificação de duplicação)
        resultado = repo.salvar_valores_diarios_ativo(ativo, valores_diarios)
        
        return resultado

    @staticmethod
    def garantir_historico_ativo(ativo, fonte_precos=None):
        """
        Garante que um ativo tenha histórico no banco.
        Se já existe, não baixa novamente. Se não existe, baixa tudo.
        
        Args:
            ativo: Nome do ativo
            fonte_precos: Função que retorna preços históricos
        
        Returns:
            dict: Resultado da operação
        """
        repo = ParquetRepo()
        
        # Verificar se já está rastreado
        if ValorDiarioService.verificar_se_ativo_ja_rastreado(ativo):
            # Já está rastreado - apenas atualizar valores recentes se necessário
            valores = repo.obter_valores_diarios_ativo(ativo)
            if valores:
                return {
                    'status': 'ja_rastreado',
                    'mensagem': f'Ativo {ativo} já está rastreado com {len(valores)} valores',
                    'total_valores': len(valores)
                }
        
        # Não está rastreado - baixar histórico completo
        if fonte_precos is None:
            return {
                'status': 'erro',
                'mensagem': 'fonte_precos é obrigatório para baixar histórico'
            }
        
        resultado = ValorDiarioService.baixar_historico_ativo(
            ativo=ativo,
            fonte_precos=fonte_precos
        )
        
        return {
            'status': 'baixado',
            'inseridos': resultado['inseridos'],
            'ignorados': resultado['ignorados']
        }

