# Representa a alocação de capital em uma posição dentro de um produto
class Alocacao:
    def __init__(self, produto_id, posicao_id, percentual, valor_usd=None, data=None):
        self.produto_id = produto_id      # ID do produto
        self.posicao_id = posicao_id      # ID da posição
        self.percentual = percentual      # Percentual alocado (0-100)
        self.valor_usd = valor_usd        # Valor em USD (opcional)
        self.data = data                  # Data da alocação
        self.status = "active"            # active/inactive

    def desativar(self):
        """Desativa uma alocação"""
        self.status = "inactive"

