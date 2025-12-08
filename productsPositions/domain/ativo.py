# Representa um ativo (ex.: BTC, ETH)
class Ativo:
    def __init__(self, nome, coingecko_id):
        self.nome = nome              # Nome do ativo (ex: "BTC")
        self.coingecko_id = coingecko_id  # ID do CoinGecko (ex: "bitcoin")

