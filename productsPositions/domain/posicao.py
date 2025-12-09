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
        self.stops = []  # Lista de dicionários: [{"data": "2025-01-10", "valor": 95000}, ...]
        # Nota: valores_diarios são armazenados por ATIVO (não por posição)
        # para evitar duplicação. Use queries para obter valores do ativo.

    def adicionar_stop(self, data, valor):
        """
        Adiciona um novo stop ao histórico
        
        Args:
            data: Data do stop (YYYY-MM-DD)
            valor: Valor do stop
        """
        if not isinstance(data, str) or len(data) != 10 or data[4] != '-' or data[7] != '-':
            raise ValueError("data deve estar no formato YYYY-MM-DD")
        
        if not isinstance(valor, (int, float)) or valor <= 0:
            raise ValueError("valor deve ser um número positivo")
        
        self.stops.append({
            "data": data,
            "valor": float(valor)
        })
        # Ordenar por data
        self.stops.sort(key=lambda x: x["data"])
    
    def obter_stop_atual(self):
        """Retorna o stop mais recente (último adicionado)"""
        if not self.stops:
            return None
        return self.stops[-1]
    
    def obter_stop_por_data(self, data):
        """Retorna o stop de uma data específica"""
        for stop in self.stops:
            if stop["data"] == data:
                return stop
        return None

    def fechar(self, preco_saida, data_saida):
        self.preco_saida = preco_saida
        self.data_saida = data_saida
        self.status = "closed"
