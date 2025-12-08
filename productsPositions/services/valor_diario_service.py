from domain.valor_diario import ValorDiario
from storage.parquet_repo import ParquetRepo
from datetime import datetime, timedelta

# Serviço responsável por gerenciar valores diários de ativos
class ValorDiarioService:

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
                    preco=item.get('preco'),
                    valor_usd=item.get('valor_usd')
                )
            else:
                # Assumir formato (data, preco, valor_usd)
                valor = ValorDiario(
                    ativo=ativo,
                    data=item[0],
                    preco=item[1],
                    valor_usd=item[2] if len(item) > 2 else None
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

