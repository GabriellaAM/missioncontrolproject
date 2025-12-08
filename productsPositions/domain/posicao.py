# Representa uma operação: abertura, fechamento, preço, status
class Posicao:
    def __init__(self, ativo, side, data_entrada, preco_entrada, coingecko_id=None):
        self.ativo = ativo           # Nome do ativo (ex.: "BTC")
        self.coingecko_id = coingecko_id  # ID do CoinGecko (ex.: "bitcoin")
        self.side = side             # long/short
        self.data_entrada = data_entrada
        self.preco_entrada = preco_entrada
        self.data_saida = None
        self.preco_saida = None
        self.status = "open"
        # Nota: valores_diarios são armazenados por ATIVO (não por posição)
        # para evitar duplicação. Use queries para obter valores do ativo.

    def fechar(self, preco_saida, data_saida):
        self.preco_saida = preco_saida
        self.data_saida = data_saida
        self.status = "closed"
