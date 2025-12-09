# Representa o valor diário de um ativo (para gráficos e análise)
# Valores diários são por ATIVO, não por posição, para evitar duplicação
class ValorDiario:
    def __init__(self, ativo, data, preco):
        self.ativo = ativo      # Nome do ativo (ex: "BTC")
        self.data = data        # Data do valor
        self.preco = preco      # Preço do ativo na data (em USD)
