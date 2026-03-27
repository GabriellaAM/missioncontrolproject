from .okx import OKXExchange
from .bitget import BitgetExchange


class ExchangeFactory:

    @staticmethod
    def get(produto_nome: str, credentials: dict):

        nome = produto_nome.lower()

        if "okx" in nome:
            return OKXExchange(credentials)

        elif "bitget" in nome or "soros" in nome:
            return BitgetExchange(credentials)

        raise ValueError(f"Exchange não suportada: {produto_nome}")