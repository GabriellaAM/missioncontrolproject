from domain.carteira import Carteira
from domain.produto import Produto
from storage.parquet_repo import ParquetRepo
from services.calculadora import Calculadora
from datetime import datetime

# Serviço responsável por gerenciar carteiras de produtos
class CarteiraService:

    @staticmethod
    def criar_carteira(produto_id, valor_disponivel=0.0, valor_investido=0.0, 
                       pnl_nao_realizado=0.0, data_atualizacao=None):
        """
        Cria uma carteira manualmente
        
        Args:
            produto_id: ID do produto
            valor_disponivel: Capital disponível para investimento
            valor_investido: Valor investido em posições abertas
            pnl_nao_realizado: PnL não realizado das posições abertas
            data_atualizacao: Data da atualização (opcional, usa data atual se None)
        
        Returns:
            Carteira: Objeto Carteira criado
        """
        # Validações
        if not isinstance(produto_id, int) or produto_id <= 0:
            raise ValueError("produto_id deve ser um inteiro positivo")
        
        if not isinstance(valor_disponivel, (int, float)) or valor_disponivel < 0:
            raise ValueError("valor_disponivel deve ser um número positivo")
        
        if not isinstance(valor_investido, (int, float)) or valor_investido < 0:
            raise ValueError("valor_investido deve ser um número positivo")
        
        if not isinstance(pnl_nao_realizado, (int, float)):
            raise ValueError("pnl_nao_realizado deve ser um número")
        
        if data_atualizacao is None:
            data_atualizacao = datetime.now().strftime("%Y-%m-%d")
        
        return Carteira(
            produto_id=produto_id,
            valor_disponivel=valor_disponivel,
            valor_investido=valor_investido,
            pnl_nao_realizado=pnl_nao_realizado,
            data_atualizacao=data_atualizacao
        )

    @staticmethod
    def preencher_carteira(produto_id, preco_atual_por_posicao=None):
        """
        Preenche/atualiza a carteira automaticamente a partir das posições e alocações
        
        Args:
            produto_id: ID do produto
            preco_atual_por_posicao: Dicionário {posicao_id: preco_atual} para calcular PnL não realizado
        
        Returns:
            Carteira: Objeto Carteira preenchido
        """
        if not isinstance(produto_id, int) or produto_id <= 0:
            raise ValueError("produto_id deve ser um inteiro positivo")
        
        repo = ParquetRepo()
        
        # Buscar todas as alocações ativas do produto
        df_alocacoes = repo.carregar_alocacoes_ativas(produto_id)
        alocacoes = [(row['posicao_id'], row['valor_usd'], row['percentual']) 
                     for _, row in df_alocacoes.iterrows()] if not df_alocacoes.empty else []
        
        # Calcular valor investido usando a Calculadora
        valor_investido = Calculadora.calcular_valor_investido(alocacoes)
        
        # Buscar posições abertas para calcular PnL não realizado
        df_posicoes = repo.carregar_posicoes_abertas(produto_id)
        posicoes_abertas = [(row['id'], row['ativo'], row['side'], row['preco_entrada'])
                          for _, row in df_posicoes.iterrows()] if not df_posicoes.empty else []
        
        # Calcular PnL não realizado usando a Calculadora
        pnl_nao_realizado = Calculadora.calcular_pnl_nao_realizado_total(
            posicoes_abertas, 
            alocacoes, 
            preco_atual_por_posicao or {}
        )
        
        # Buscar valor total inicial (se houver carteira existente)
        carteira_existente = repo.carregar_carteira(produto_id)
        
        if carteira_existente:
            valor_total_inicial = carteira_existente.valor_total or 0
            valor_disponivel = carteira_existente.valor_disponivel or 0
        else:
            # Se não existe carteira, assumimos que o valor disponível é 0
            # e o valor total inicial é o valor investido
            valor_disponivel = 0.0
            valor_total_inicial = valor_investido
        
        return Carteira(
            produto_id=produto_id,
            valor_disponivel=valor_disponivel,
            valor_investido=valor_investido,
            pnl_nao_realizado=pnl_nao_realizado,
            data_atualizacao=datetime.now().strftime("%Y-%m-%d")
        )

    @staticmethod
    def atualizar_carteira_com_capital_inicial(produto_id, capital_inicial, preco_atual_por_posicao=None):
        """
        Cria ou atualiza carteira com capital inicial e recalcula valores
        
        Args:
            produto_id: ID do produto
            capital_inicial: Capital inicial total da carteira
            preco_atual_por_posicao: Dicionário {posicao_id: preco_atual} para calcular PnL
        
        Returns:
            Carteira: Objeto Carteira atualizado
        """
        if not isinstance(produto_id, int) or produto_id <= 0:
            raise ValueError("produto_id deve ser um inteiro positivo")
        
        if not isinstance(capital_inicial, (int, float)) or capital_inicial <= 0:
            raise ValueError("capital_inicial deve ser um número positivo")
        
        # Preencher carteira automaticamente
        carteira = CarteiraService.preencher_carteira(produto_id, preco_atual_por_posicao)
        
        # Ajustar valor disponível baseado no capital inicial usando a Calculadora
        carteira.valor_disponivel = Calculadora.calcular_valor_disponivel(
            capital_inicial, 
            carteira.valor_investido
        )
        
        # Recalcular valor total usando a Calculadora
        carteira.valor_total = Calculadora.calcular_valor_total_carteira(
            carteira.valor_disponivel,
            carteira.valor_investido,
            carteira.pnl_nao_realizado
        )
        
        return carteira

