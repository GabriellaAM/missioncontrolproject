# Representa a carteira de um produto (visão agregada do estado financeiro)
class Carteira:
    def __init__(self, produto_id, valor_disponivel=0.0, valor_investido=0.0, 
                 pnl_nao_realizado=0.0, valor_total=None, data_atualizacao=None):
        self.produto_id = produto_id          # ID do produto (relacionamento 1:1 opcional)
        self.valor_disponivel = valor_disponivel      # Capital disponível para novos investimentos
        self.valor_investido = valor_investido        # Valor total investido em posições abertas
        self.pnl_nao_realizado = pnl_nao_realizado    # Lucro/prejuízo não realizado das posições abertas
        self.data_atualizacao = data_atualizacao      # Data da última atualização
        
        # Valor total = disponível + investido + PnL não realizado
        if valor_total is None:
            self.valor_total = valor_disponivel + valor_investido + pnl_nao_realizado
        else:
            self.valor_total = valor_total

    def recalcular_valor_total(self):
        """Recalcula o valor total da carteira"""
        self.valor_total = self.valor_disponivel + self.valor_investido + self.pnl_nao_realizado
        return self.valor_total

    def __repr__(self):
        return (f"Carteira(produto_id={self.produto_id}, "
                f"disponivel=${self.valor_disponivel:.2f}, "
                f"investido=${self.valor_investido:.2f}, "
                f"pnl_nao_realizado=${self.pnl_nao_realizado:.2f}, "
                f"total=${self.valor_total:.2f})")

