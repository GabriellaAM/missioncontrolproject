from domain.posicao import Posicao
from domain.produto import Produto

# Serviço responsável por abrir posições
class PosicaoService:

    @staticmethod
    def abrir(produto, ativo, side, data, preco, coingecko_id=None):
        """
        Abre uma nova posição
        
        Args:
            produto: Objeto Produto
            ativo: Nome do ativo (ex: "BTC")
            side: "long" ou "short"
            data: Data de entrada (YYYY-MM-DD)
            preco: Preço de entrada
            coingecko_id: ID do CoinGecko (ex: "bitcoin") - opcional
        
        Returns:
            Posicao: Objeto Posicao criado
        """
        # Validação de entrada
        if not isinstance(produto, Produto):
            raise TypeError("produto deve ser uma instância de Produto")
        
        if not isinstance(ativo, str) or not ativo.strip():
            raise ValueError("ativo deve ser uma string não vazia")
        
        if side not in ["long", "short"]:
            raise ValueError("side deve ser 'long' ou 'short'")
        
        if not isinstance(data, str) or not data.strip():
            raise ValueError("data deve ser uma string não vazia")
        
        try:
            # Validação básica de formato de data (YYYY-MM-DD)
            if len(data) != 10 or data[4] != '-' or data[7] != '-':
                raise ValueError("data deve estar no formato YYYY-MM-DD")
        except (IndexError, TypeError):
            raise ValueError("data deve estar no formato YYYY-MM-DD")
        
        if not isinstance(preco, (int, float)) or preco <= 0:
            raise ValueError("preco deve ser um número positivo")
        
        if coingecko_id is not None and (not isinstance(coingecko_id, str) or not coingecko_id.strip()):
            raise ValueError("coingecko_id deve ser uma string não vazia ou None")
        
        # Criar e adicionar posição
        pos = Posicao(ativo, side, data, preco, coingecko_id)
        produto.posicoes.append(pos)
        return pos
