# Calculadora simples para PnL, taxa etc.
class Calculadora:

    @staticmethod
    def pnl(posicao):
        """Calcula PnL realizado de uma posição fechada"""
        if posicao.status == "open":
            return None
        return posicao.preco_saida - posicao.preco_entrada

    @staticmethod
    def pnl_nao_realizado(posicao, preco_atual, valor_investido):
        """
        Calcula PnL não realizado de uma posição aberta
        
        Args:
            posicao: Objeto Posicao
            preco_atual: Preço atual do ativo
            valor_investido: Valor investido nesta posição
        
        Returns:
            float: PnL não realizado em USD
        """
        if posicao.status != "open":
            return 0.0
        
        if valor_investido <= 0:
            return 0.0
        
        # Calcular variação percentual
        variacao_pct = (preco_atual - posicao.preco_entrada) / posicao.preco_entrada
        
        # Para short, inverter o sinal
        if posicao.side == "short":
            variacao_pct = -variacao_pct
        
        return valor_investido * variacao_pct

    @staticmethod
    def calcular_valor_investido(alocacoes):
        """
        Calcula o valor total investido a partir das alocações
        
        Args:
            alocacoes: Lista de tuplas (posicao_id, valor_usd, percentual) ou lista de objetos Alocacao
        
        Returns:
            float: Valor total investido
        """
        if not alocacoes:
            return 0.0
        
        # Se for lista de tuplas (do banco)
        if isinstance(alocacoes[0], tuple):
            return sum(aloc[1] for aloc in alocacoes if aloc[1] is not None)
        
        # Se for lista de objetos Alocacao
        return sum(aloc.valor_usd for aloc in alocacoes if aloc.valor_usd is not None)

    @staticmethod
    def calcular_pnl_nao_realizado_total(posicoes_abertas, alocacoes, preco_atual_por_posicao):
        """
        Calcula o PnL não realizado total de múltiplas posições
        
        Args:
            posicoes_abertas: Lista de tuplas (id, ativo, side, preco_entrada) ou lista de objetos Posicao
            alocacoes: Lista de tuplas (posicao_id, valor_usd, percentual) ou lista de objetos Alocacao
            preco_atual_por_posicao: Dicionário {posicao_id: preco_atual}
        
        Returns:
            float: PnL não realizado total
        """
        if not preco_atual_por_posicao or not posicoes_abertas:
            return 0.0
        
        pnl_total = 0.0
        
        # Criar dicionário de alocações por posição_id para busca rápida
        alocacoes_dict = {}
        if isinstance(alocacoes[0], tuple):
            # Se for lista de tuplas (do banco)
            alocacoes_dict = {aloc[0]: aloc[1] for aloc in alocacoes if aloc[1] is not None}
        else:
            # Se for lista de objetos Alocacao
            alocacoes_dict = {aloc.posicao_id: aloc.valor_usd for aloc in alocacoes if aloc.valor_usd is not None}
        
        # Processar cada posição aberta
        for posicao_data in posicoes_abertas:
            if isinstance(posicao_data, tuple):
                # Se for tupla do banco: (id, ativo, side, preco_entrada)
                posicao_id, ativo, side, preco_entrada = posicao_data
            else:
                # Se for objeto Posicao
                posicao_id = posicao_data.id if hasattr(posicao_data, 'id') else None
                ativo = posicao_data.ativo
                side = posicao_data.side
                preco_entrada = posicao_data.preco_entrada
            
            if posicao_id in preco_atual_por_posicao:
                preco_atual = preco_atual_por_posicao[posicao_id]
                valor_investido = alocacoes_dict.get(posicao_id, 0)
                
                if valor_investido > 0:
                    # Calcular variação percentual
                    variacao_pct = (preco_atual - preco_entrada) / preco_entrada
                    if side == "short":
                        variacao_pct = -variacao_pct
                    
                    pnl_posicao = valor_investido * variacao_pct
                    pnl_total += pnl_posicao
        
        return pnl_total

    @staticmethod
    def calcular_valor_total_carteira(valor_disponivel, valor_investido, pnl_nao_realizado):
        """
        Calcula o valor total da carteira
        
        Args:
            valor_disponivel: Capital disponível
            valor_investido: Valor investido
            pnl_nao_realizado: PnL não realizado
        
        Returns:
            float: Valor total da carteira
        """
        return valor_disponivel + valor_investido + pnl_nao_realizado

    @staticmethod
    def calcular_valor_disponivel(capital_inicial, valor_investido):
        """
        Calcula o valor disponível baseado no capital inicial e valor investido
        
        Args:
            capital_inicial: Capital inicial total
            valor_investido: Valor já investido
        
        Returns:
            float: Valor disponível (nunca negativo)
        """
        return max(0.0, capital_inicial - valor_investido)
