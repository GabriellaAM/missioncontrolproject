# Representa um produto (ex.: SOROS_1)
class Produto:
    def __init__(self, nome, data_inicio, tipo):
        self.nome = nome
        self.data_inicio = data_inicio
        self.tipo = tipo
        self.posicoes = []  # Lista de objetos Posição
        self.carteira = None  # Carteira opcional do produto
