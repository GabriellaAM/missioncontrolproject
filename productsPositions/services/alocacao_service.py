from domain.alocacao import Alocacao
from domain.produto import Produto
from domain.posicao import Posicao

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

    @staticmethod
    def alocar_equitativamente(produto_id, posicoes_ids, total_capital=None):
        """
        Aloca capital de forma equitativa entre múltiplas posições
        
        Args:
            produto_id: ID do produto
            posicoes_ids: Lista de IDs das posições
            total_capital: Capital total em USD (opcional)
        
        Returns:
            list: Lista de objetos Alocacao criados
        """
        if not isinstance(produto_id, int) or produto_id <= 0:
            raise ValueError("produto_id deve ser um inteiro positivo")
        
        if not isinstance(posicoes_ids, list) or len(posicoes_ids) == 0:
            raise ValueError("posicoes_ids deve ser uma lista não vazia")
        
        if total_capital is not None and (not isinstance(total_capital, (int, float)) or total_capital <= 0):
            raise ValueError("total_capital deve ser um número positivo ou None")
        
        num_posicoes = len(posicoes_ids)
        percentual_por_posicao = 100.0 / num_posicoes
        
        alocacoes = []
        for posicao_id in posicoes_ids:
            valor_usd = (total_capital / num_posicoes) if total_capital else None
            alocacao = Alocacao(
                produto_id=produto_id,
                posicao_id=posicao_id,
                percentual=percentual_por_posicao,
                valor_usd=valor_usd
            )
            alocacoes.append(alocacao)
        
        return alocacoes

    @staticmethod
    def alocar_por_peso(produto_id, posicoes_pesos, total_capital=None):
        """
        Aloca capital proporcionalmente aos pesos fornecidos
        
        Args:
            produto_id: ID do produto
            posicoes_pesos: Dicionário {posicao_id: peso}
            total_capital: Capital total em USD (opcional)
        
        Returns:
            list: Lista de objetos Alocacao criados
        """
        if not isinstance(produto_id, int) or produto_id <= 0:
            raise ValueError("produto_id deve ser um inteiro positivo")
        
        if not isinstance(posicoes_pesos, dict) or len(posicoes_pesos) == 0:
            raise ValueError("posicoes_pesos deve ser um dicionário não vazio")
        
        if total_capital is not None and (not isinstance(total_capital, (int, float)) or total_capital <= 0):
            raise ValueError("total_capital deve ser um número positivo ou None")
        
        # Calcular soma total dos pesos
        soma_pesos = sum(posicoes_pesos.values())
        if soma_pesos <= 0:
            raise ValueError("A soma dos pesos deve ser maior que zero")
        
        alocacoes = []
        for posicao_id, peso in posicoes_pesos.items():
            if not isinstance(peso, (int, float)) or peso < 0:
                raise ValueError(f"Peso para posicao_id {posicao_id} deve ser um número positivo")
            
            percentual = (peso / soma_pesos) * 100
            valor_usd = (total_capital * peso / soma_pesos) if total_capital else None
            
            alocacao = Alocacao(
                produto_id=produto_id,
                posicao_id=posicao_id,
                percentual=percentual,
                valor_usd=valor_usd
            )
            alocacoes.append(alocacao)
        
        return alocacoes

