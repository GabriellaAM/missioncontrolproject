# Representa um produto (ex.: SOROS_1)
class Produto:
    def __init__(self, nome, data_inicio, tipo, capital_inicial=0.0):
        self.nome = nome
        self.data_inicio = data_inicio
        self.tipo = tipo
        self.capital_inicial = capital_inicial  # Valor investido inicialmente em USD
        self.posicoes = []  # Lista de objetos Posição
        self.carteira = None  # Carteira opcional do produto
