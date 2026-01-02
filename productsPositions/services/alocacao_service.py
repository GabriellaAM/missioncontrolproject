from domain.alocacao import Alocacao

# Serviço responsável por gerenciar alocações de capital
class AlocacaoService:

    @staticmethod
    def criar_alocacao(produto_id, posicao_id, percentual, valor_usd=None, data=None):
        """
        Cria uma alocação manual para uma posição
        
        Args:
            produto_id: ID do produto
            posicao_id: ID da posição
            percentual: Percentual alocado (0-100)
            valor_usd: Valor em USD (opcional)
            data: Data da alocação (opcional)
        
        Returns:
            Alocacao: Objeto de alocação criado
        """
        # Validações
        if not isinstance(produto_id, int) or produto_id <= 0:
            raise ValueError("produto_id deve ser um inteiro positivo")
        
        if not isinstance(posicao_id, int) or posicao_id <= 0:
            raise ValueError("posicao_id deve ser um inteiro positivo")
        
        if not isinstance(percentual, (int, float)) or percentual < 0 or percentual > 100:
            raise ValueError("percentual deve ser um número entre 0 e 100")
        
        if valor_usd is not None and (not isinstance(valor_usd, (int, float)) or valor_usd < 0):
            raise ValueError("valor_usd deve ser um número positivo ou None")
        
        if data is not None and (not isinstance(data, str) or not data.strip()):
            raise ValueError("data deve ser uma string não vazia ou None")
        
        return Alocacao(produto_id, posicao_id, percentual, valor_usd, data)

