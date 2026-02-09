"""
Serviço de Cálculo de Rentabilidade por Turma.

Este módulo implementa o cálculo correto de rentabilidade (TWR - Time-Weighted Return)
por turma, usando o preco_entrada_turma correto para trades replicados.

A correção principal é que:
- Trades NATIVOS: usam preco_entrada original da posição
- Trades REPLICADOS: usam preco_entrada_turma (preço na data de inserção na turma)

Fórmulas:
- Custo Inicial = preco_entrada_turma * quantidade
- Valor de Mercado = preco_atual * quantidade
- PnL Não Realizado = Valor de Mercado - Custo Inicial
- Capital em Caixa = Capital Base - Soma(Custos Ativos) + PnL Fechado Acumulado
- Valor Total Portfolio = Capital em Caixa + Valor de Mercado Ativos
- Rentabilidade Diária % = (Valor Hoje / Valor Ontem - 1) * 100
- Rentabilidade Acumulada % = (Valor Hoje / Capital Base - 1) * 100
"""

import psycopg2
import psycopg2.extras
import os
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from dotenv import load_dotenv

from .cotacoes_service import CotacoesService

_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / '.env')


# =============================================================================
# ESTRUTURAS DE VALIDAÇÃO
# =============================================================================

class TipoAviso(Enum):
    """Tipos de avisos que podem ocorrer durante o cálculo."""
    PRECO_FALLBACK = "preco_fallback"  # Usou preço de entrada como fallback
    PRECO_SAIDA_FALLBACK = "preco_saida_fallback"  # Trade fechado sem preço de saída
    SEM_COINGECKO_ID = "sem_coingecko_id"  # Não consegue atualizar preço
    DATA_FUTURA = "data_futura"  # Data solicitada é no futuro
    TRADE_DUPLICADO = "trade_duplicado"  # Trade aparece mais de uma vez
    VARIACAO_PRECO_EXTREMA = "variacao_preco_extrema"  # Preço variou muito
    CAPITAL_CAIXA_NEGATIVO = "capital_caixa_negativo"  # Capital em caixa < 0
    API_TIMEOUT = "api_timeout"  # CoinGecko não respondeu


class TipoErro(Enum):
    """Tipos de erros que impedem o cálculo."""
    SIDE_INVALIDO = "side_invalido"
    PRECO_ENTRADA_INVALIDO = "preco_entrada_invalido"
    CAPITAL_BASE_INVALIDO = "capital_base_invalido"
    TURMA_NAO_ENCONTRADA = "turma_nao_encontrada"
    DADOS_CORROMPIDOS = "dados_corrompidos"
    QUANTIDADE_INVALIDA = "quantidade_invalida"
    DATA_INVALIDA = "data_invalida"  # Formato de data inválido
    VALOR_TOTAL_NEGATIVO = "valor_total_negativo"  # Valor total < 0
    CONEXAO_BANCO = "conexao_banco"  # Erro de conexão SQLite
    DATAS_INCONSISTENTES = "datas_inconsistentes"  # data_insercao > data_remocao
    POSICAO_NAO_ENCONTRADA = "posicao_nao_encontrada"  # Posição referenciada não existe
    INTEGRIDADE_TURMA_PRODUTO = "integridade_turma_produto"  # Turma não pertence ao produto


@dataclass
class Aviso:
    """Representa um aviso durante o cálculo (não impede, mas sinaliza)."""
    tipo: TipoAviso
    trade_id: Optional[int]
    ativo: Optional[str]
    mensagem: str


@dataclass
class Erro:
    """Representa um erro que impede o cálculo correto de um trade."""
    tipo: TipoErro
    trade_id: Optional[int]
    ativo: Optional[str]
    mensagem: str
    valor_encontrado: Any = None


@dataclass
class TradeSnapshot:
    """Snapshot de um trade em um dia específico."""
    trade_id: int
    turma_id: int
    posicao_id: int
    ativo: str
    side: str
    origem: str
    data_insercao: str
    data_remocao: Optional[str]
    preco_entrada_turma: float
    quantidade: float
    preco_cotacao: float
    status: str  # 'aberto' ou 'vendido'


@dataclass
class DailyPortfolio:
    """Resumo diário do portfolio de uma turma."""
    turma_id: int
    dia: str
    capital_alocado: float  # Valor de mercado dos ativos
    custo_ativos: float  # Soma dos custos iniciais dos ativos ativos
    pnl_fechado_acumulado: float  # PnL dos trades já fechados
    capital_em_caixa: float  # Capital disponível
    valor_total: float  # Valor total do portfolio
    rentabilidade_diaria_pct: float
    rentabilidade_acumulada_pct: float
    # Campos de validação
    avisos: List[Aviso] = field(default_factory=list)
    erros: List[Erro] = field(default_factory=list)
    trades_calculados: int = 0
    trades_com_erro: int = 0

    @property
    def tem_avisos(self) -> bool:
        return len(self.avisos) > 0

    @property
    def tem_erros(self) -> bool:
        return len(self.erros) > 0

    @property
    def confiabilidade_pct(self) -> float:
        """Retorna % de confiabilidade baseado em trades com erro/aviso."""
        total = self.trades_calculados + self.trades_com_erro
        if total == 0:
            return 100.0
        return (self.trades_calculados / total) * 100


class RentabilidadeService:
    """Serviço para calcular rentabilidade por turma."""

    # Valores válidos para o campo 'side'
    SIDES_VALIDOS = ('long', 'short')

    # Valor mínimo para capital_base (evitar divisão por zero)
    CAPITAL_BASE_MINIMO = 0.01

    # Limiar para aviso de variação de preço extrema (50% = 0.5)
    VARIACAO_PRECO_EXTREMA_LIMIAR = 0.5

    # Formato de data esperado
    FORMATO_DATA = "%Y-%m-%d"

    def __init__(self, db_path: Optional[Path] = None):
        # db_path parameter kept for signature compatibility but ignored
        self.db_url = os.getenv('SUPABASE_DB_URL')
        if self.db_url is None:
            raise ValueError("SUPABASE_DB_URL environment variable is not set")
        self.cotacoes_service = CotacoesService(db_path)

    def _validar_side(self, side: Any) -> Tuple[bool, Optional[str]]:
        """
        Valida se o valor de side é válido.

        Returns:
            Tuple (é_válido, side_normalizado ou None se inválido)
        """
        if side is None:
            return False, None

        side_lower = str(side).lower().strip()
        if side_lower in self.SIDES_VALIDOS:
            return True, side_lower

        return False, None

    def _validar_preco(self, preco: Any, nome_campo: str) -> Tuple[bool, float]:
        """
        Valida se um preço é válido (número > 0).

        Returns:
            Tuple (é_válido, valor_float ou 0.0 se inválido)
        """
        if preco is None:
            return False, 0.0

        try:
            valor = float(preco)
            if valor > 0:
                return True, valor
            return False, 0.0
        except (ValueError, TypeError):
            return False, 0.0

    @staticmethod
    def _calcular_pnl(side: str, preco_entrada: float, preco_saida: float, quantidade: float) -> float:
        """
        Calcula o PnL (Profit and Loss) de um trade.

        Args:
            side: 'long' ou 'short'
            preco_entrada: Preço de entrada do trade
            preco_saida: Preço de saída do trade
            quantidade: Quantidade negociada

        Returns:
            PnL calculado (positivo = lucro, negativo = prejuízo)

        Formula:
            Long:  PnL = (preco_saida × quantidade) - (preco_entrada × quantidade)
            Short: PnL = (preco_entrada × quantidade) - (preco_saida × quantidade)
        """
        custo_inicial = preco_entrada * quantidade
        if side == 'long':
            return (preco_saida * quantidade) - custo_inicial
        else:  # short
            return custo_inicial - (preco_saida * quantidade)

    def _validar_quantidade(self, quantidade: Any, usa_quantidade: bool) -> Tuple[bool, Optional[float], bool]:
        """
        Valida quantidade baseado na configuração do produto.

        Args:
            quantidade: Valor da quantidade
            usa_quantidade: Se o produto usa quantidade (True) ou não (False)

        Returns:
            Tuple (é_válido, quantidade_float, deve_ignorar)
            - é_válido: True se quantidade é válida para cálculo
            - quantidade_float: Valor numérico da quantidade ou None
            - deve_ignorar: True se o trade deve ser ignorado (produto não usa quantidade)
        """
        # Produto não usa quantidade - trade deve ser ignorado no cálculo
        if not usa_quantidade:
            return True, None, True  # válido, sem quantidade, ignorar

        # Produto usa quantidade - validar
        if quantidade is None:
            return False, None, False  # inválido, erro

        try:
            valor = float(quantidade)
            if valor > 0:
                return True, valor, False  # válido, com quantidade, não ignorar
            return False, None, False  # quantidade <= 0 é inválida
        except (ValueError, TypeError):
            return False, None, False  # não é número válido

    def _validar_data(self, dia: str) -> Tuple[bool, Optional[Erro]]:
        """
        Valida se a data está no formato correto (YYYY-MM-DD).

        Returns:
            Tuple (é_válido, erro ou None)
        """
        if dia is None:
            return False, Erro(
                tipo=TipoErro.DATA_INVALIDA,
                trade_id=None,
                ativo=None,
                mensagem="Data não pode ser None",
                valor_encontrado=None
            )

        try:
            datetime.strptime(dia, self.FORMATO_DATA)
            return True, None
        except ValueError:
            return False, Erro(
                tipo=TipoErro.DATA_INVALIDA,
                trade_id=None,
                ativo=None,
                mensagem=f"Data '{dia}' não está no formato YYYY-MM-DD",
                valor_encontrado=dia
            )

    def _verificar_data_futura(self, dia: str) -> Optional[Aviso]:
        """
        Verifica se a data é no futuro.

        Returns:
            Aviso se data é futura, None caso contrário
        """
        hoje = datetime.now().strftime(self.FORMATO_DATA)
        if dia > hoje:
            return Aviso(
                tipo=TipoAviso.DATA_FUTURA,
                trade_id=None,
                ativo=None,
                mensagem=f"Data '{dia}' é no futuro (hoje: {hoje})"
            )
        return None

    def _validar_datas_trade(self, data_insercao: str, data_remocao: Optional[str],
                              trade_id: int, ativo: str) -> Optional[Erro]:
        """
        Valida se as datas do trade são consistentes (inserção <= remoção).

        Returns:
            Erro se datas inconsistentes, None caso contrário
        """
        if data_remocao is None:
            return None  # Trade ainda aberto, OK

        if data_insercao > data_remocao:
            return Erro(
                tipo=TipoErro.DATAS_INCONSISTENTES,
                trade_id=trade_id,
                ativo=ativo,
                mensagem=f"Data de inserção ({data_insercao}) > data de remoção ({data_remocao})",
                valor_encontrado={'data_insercao': data_insercao, 'data_remocao': data_remocao}
            )
        return None

    def _detectar_trades_duplicados(self, trades: List[Dict]) -> List[Aviso]:
        """
        Detecta trades duplicados na lista.

        Returns:
            Lista de avisos para trades duplicados
        """
        avisos = []
        trade_ids_vistos = {}

        for trade in trades:
            trade_id = trade.get('trade_id')
            ativo = trade.get('ativo', 'DESCONHECIDO')

            if trade_id in trade_ids_vistos:
                avisos.append(Aviso(
                    tipo=TipoAviso.TRADE_DUPLICADO,
                    trade_id=trade_id,
                    ativo=ativo,
                    mensagem=f"Trade {trade_id} ({ativo}) aparece duplicado"
                ))
            else:
                trade_ids_vistos[trade_id] = True

        return avisos

    def _verificar_variacao_preco_extrema(self, preco_entrada: float, preco_atual: float,
                                           trade_id: int, ativo: str) -> Optional[Aviso]:
        """
        Verifica se houve variação de preço extrema (> limiar configurado).

        Returns:
            Aviso se variação extrema, None caso contrário
        """
        if preco_entrada <= 0:
            return None

        variacao = abs(preco_atual - preco_entrada) / preco_entrada

        if variacao > self.VARIACAO_PRECO_EXTREMA_LIMIAR:
            variacao_pct = variacao * 100
            return Aviso(
                tipo=TipoAviso.VARIACAO_PRECO_EXTREMA,
                trade_id=trade_id,
                ativo=ativo,
                mensagem=f"{ativo} variou {variacao_pct:.1f}% (entrada: {preco_entrada}, atual: {preco_atual})"
            )
        return None

    def _get_connection(self):
        """
        Obtém conexão com o banco de dados.

        Raises:
            psycopg2.Error: Se não conseguir conectar
        """
        return psycopg2.connect(self.db_url)

    def _get_connection_safe(self) -> Tuple[Optional[Any], Optional[Erro]]:
        """
        Obtém conexão com o banco de dados de forma segura.

        Returns:
            Tuple (conexão ou None, erro ou None)
        """
        try:
            conn = psycopg2.connect(self.db_url)
            return conn, None
        except psycopg2.Error as e:
            return None, Erro(
                tipo=TipoErro.CONEXAO_BANCO,
                trade_id=None,
                ativo=None,
                mensagem=f"Erro ao conectar com banco de dados: {str(e)}",
                valor_encontrado=str(self.db_url)
            )

    def _verificar_posicao_existe(self, posicao_id: int, conn) -> Optional[Erro]:
        """
        Verifica se uma posição existe no banco.

        Returns:
            Erro se posição não existe, None caso contrário
        """
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("SELECT id FROM posicoes WHERE id = %s", (posicao_id,))
        if cursor.fetchone() is None:
            return Erro(
                tipo=TipoErro.POSICAO_NAO_ENCONTRADA,
                trade_id=None,
                ativo=None,
                mensagem=f"Posição {posicao_id} não encontrada no banco de dados",
                valor_encontrado=posicao_id
            )
        return None

    def obter_capital_base(self, turma_id: int) -> Tuple[float, Optional[Erro]]:
        """
        Obtém o capital base de uma turma com validação.

        Returns:
            Tuple (capital_base, erro ou None se válido)
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("SELECT capital_base FROM turmas WHERE id = %s", (turma_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return 0.0, Erro(
                tipo=TipoErro.TURMA_NAO_ENCONTRADA,
                trade_id=None,
                ativo=None,
                mensagem=f"Turma {turma_id} não encontrada no banco de dados",
                valor_encontrado=None
            )

        capital_base = row['capital_base']

        if capital_base is None or capital_base < self.CAPITAL_BASE_MINIMO:
            return 0.0, Erro(
                tipo=TipoErro.CAPITAL_BASE_INVALIDO,
                trade_id=None,
                ativo=None,
                mensagem=f"Capital base inválido para turma {turma_id}: {capital_base}",
                valor_encontrado=capital_base
            )

        return float(capital_base), None

    def obter_preco_cotacao(self, trade_id: int, dia: str,
                             ativo: Optional[str] = None,
                             exchange_symbol: Optional[str] = None,
                             coingecko_id: Optional[str] = None) -> Tuple[Optional[float], str]:
        """
        Obtém o preço de cotação de um trade para um dia específico.

        - Se dia == hoje: busca preço em tempo real (sem salvar no banco)
        - Se dia < hoje: usa o preço mais recente até aquela data do banco

        Args:
            trade_id: ID do trade
            dia: Data no formato YYYY-MM-DD
            ativo: Nome do ativo (para busca em tempo real)
            exchange_symbol: Símbolo na exchange
            coingecko_id: ID no CoinGecko

        Returns:
            Tuple (preco ou None, status: 'ok', 'timeout', 'erro', 'fallback_db')
        """
        hoje = datetime.now().strftime("%Y-%m-%d")

        # Se for hoje, buscar preço em tempo real (sem salvar)
        if dia == hoje and ativo:
            preco, status = self.cotacoes_service.obter_preco_atual_com_status(
                ativo,
                exchange_symbol=exchange_symbol,
                coingecko_id=coingecko_id
            )
            if preco is not None:
                return preco, status
            # Se teve timeout ou erro, ainda tenta o banco como fallback
            if status == 'timeout':
                # Buscar do banco mesmo para hoje como fallback
                conn = self._get_connection()
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute("""
                    SELECT preco FROM trade_valores_diarios
                    WHERE trade_id = %s AND data <= %s
                    ORDER BY data DESC
                    LIMIT 1
                """, (trade_id, dia))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return row['preco'], 'timeout'  # Retorna preço do banco + sinaliza timeout
                return None, 'timeout'

        # Buscar do banco de dados (para dias anteriores ou fallback)
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""
            SELECT preco FROM trade_valores_diarios
            WHERE trade_id = %s AND data <= %s
            ORDER BY data DESC
            LIMIT 1
        """, (trade_id, dia))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row['preco'], 'fallback_db'
        return None, 'erro'

    def obter_trades_ativos_no_dia(self, turma_id: int, dia: str) -> List[Dict]:
        """
        Obtém todos os trades que estavam ativos em uma turma em um dia específico.

        Um trade está ativo no dia se:
        - data_insercao <= dia
        - (data_remocao IS NULL OR data_remocao > dia)

        IMPORTANTE: Para o dia da saída, o trade ainda conta como ativo para
        calcular o PnL realizado.
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute("""
            SELECT
                ct.trade_id,
                ct.turma_id,
                tt.posicao_id,
                p.ativo,
                p.side,
                p.exchange_symbol,
                p.coingecko_id,
                ct.origem,
                ct.data_insercao,
                ct.data_remocao,
                ct.preco_entrada_turma,
                pap.quantidade,
                p.preco_saida,
                p.status as status_posicao,
                COALESCE(prod.usa_quantidade, 0) as usa_quantidade
            FROM carteira_turma ct
            JOIN trades_turma tt ON ct.trade_id = tt.id
            JOIN posicoes p ON tt.posicao_id = p.id
            JOIN turmas t ON ct.turma_id = t.id
            JOIN produtos prod ON t.produto_id = prod.id
            LEFT JOIN posicao_atributos_produto pap ON p.id = pap.posicao_id
            WHERE ct.turma_id = %s
            AND date(ct.data_insercao) <= date(%s)
            AND (ct.data_remocao IS NULL OR date(ct.data_remocao) >= date(%s))
        """, (turma_id, dia, dia))

        trades = []
        for row in cursor.fetchall():
            trade = dict(row)

            # Determinar se está 'aberto' ou 'vendido' neste dia
            if trade['data_remocao'] and trade['data_remocao'] == dia:
                trade['status'] = 'vendido'
            else:
                trade['status'] = 'aberto'

            trades.append(trade)

        conn.close()
        return trades

    def _filtrar_trades_para_dia(self, todos_trades: List[Dict], dia: str) -> List[Dict]:
        """
        Filtra trades pré-carregados para um dia específico (equivalente em memória
        de obter_trades_ativos_no_dia).

        Um trade está ativo no dia se:
        - data_insercao <= dia
        - (data_remocao IS NULL OR data_remocao >= dia)
        """
        resultado = []
        for trade in todos_trades:
            data_insercao = trade.get('data_insercao')
            data_remocao = trade.get('data_remocao')

            if data_insercao and data_insercao > dia:
                continue
            if data_remocao and data_remocao < dia:
                continue

            trade_copia = dict(trade)
            if data_remocao and data_remocao == dia:
                trade_copia['status'] = 'vendido'
            else:
                trade_copia['status'] = 'aberto'

            resultado.append(trade_copia)
        return resultado

    def calcular_portfolio_dia(self, turma_id: int, dia: str,
                                valor_anterior: Optional[float] = None,
                                precos_cache: Optional[Dict[str, Dict[str, float]]] = None,
                                trades_prefetched: Optional[List[Dict]] = None,
                                capital_base_cache: Optional[float] = None,
                                pnl_fechado_anterior_override: Optional[float] = None) -> DailyPortfolio:
        """
        Calcula o portfolio de uma turma para um dia específico.

        Inclui validações completas:
        - Verifica capital_base > 0
        - Valida side (long/short) de cada trade
        - Valida preco_entrada_turma > 0
        - Valida preco_saida para trades fechados
        - Sinaliza quando fallbacks são usados

        Args:
            turma_id: ID da turma
            dia: Data no formato YYYY-MM-DD
            valor_anterior: Valor total do portfolio no dia anterior
            precos_cache: Dict[coingecko_id, Dict[data, preco]] pré-carregado.
                         Se fornecido, usa em vez de queries individuais ao banco/API.
            trades_prefetched: Lista de todos os trades da turma (pré-carregados).
                              Se fornecido, filtra em memória em vez de query SQL por dia.
            capital_base_cache: Capital base já obtido (evita query extra).
            pnl_fechado_anterior_override: PnL fechado acumulado até o dia anterior
                                           já calculado pelo chamador. Evita query SQL.

        Returns:
            DailyPortfolio com métricas, avisos e erros
        """
        avisos: List[Aviso] = []
        erros: List[Erro] = []
        trades_calculados = 0
        trades_com_erro = 0

        # =================================================================
        # VALIDAÇÃO 0: Data válida (formato YYYY-MM-DD)
        # =================================================================
        data_valida, erro_data = self._validar_data(dia)
        if not data_valida:
            erros.append(erro_data)
            return DailyPortfolio(
                turma_id=turma_id,
                dia=dia if dia else "INVALID",
                capital_alocado=0.0,
                custo_ativos=0.0,
                pnl_fechado_acumulado=0.0,
                capital_em_caixa=0.0,
                valor_total=0.0,
                rentabilidade_diaria_pct=0.0,
                rentabilidade_acumulada_pct=0.0,
                avisos=avisos,
                erros=erros,
                trades_calculados=0,
                trades_com_erro=0
            )

        # Verificar se data é no futuro (aviso, não impede cálculo)
        aviso_futuro = self._verificar_data_futura(dia)
        if aviso_futuro:
            avisos.append(aviso_futuro)

        # =================================================================
        # VALIDAÇÃO 1: Capital Base
        # =================================================================
        if capital_base_cache is not None:
            capital_base = capital_base_cache
        else:
            capital_base, erro_capital = self.obter_capital_base(turma_id)
            if erro_capital:
                erros.append(erro_capital)
                return DailyPortfolio(
                    turma_id=turma_id,
                    dia=dia,
                    capital_alocado=0.0,
                    custo_ativos=0.0,
                    pnl_fechado_acumulado=0.0,
                    capital_em_caixa=0.0,
                    valor_total=0.0,
                    rentabilidade_diaria_pct=0.0,
                    rentabilidade_acumulada_pct=0.0,
                    avisos=avisos,
                    erros=erros,
                    trades_calculados=0,
                    trades_com_erro=0
                )

        # =================================================================
        # OBTER TRADES: prefetched (filtro em memória) ou query SQL
        # =================================================================
        if trades_prefetched is not None:
            trades = self._filtrar_trades_para_dia(trades_prefetched, dia)
        else:
            trades = self.obter_trades_ativos_no_dia(turma_id, dia)

        # =================================================================
        # VALIDAÇÃO: Detectar trades duplicados
        # =================================================================
        avisos_duplicados = self._detectar_trades_duplicados(trades)
        avisos.extend(avisos_duplicados)

        capital_alocado = 0.0
        custo_ativos = 0.0
        pnl_fechado_dia = 0.0

        for trade in trades:
            trade_id = trade['trade_id']
            ativo = trade.get('ativo', 'DESCONHECIDO')

            # =============================================================
            # VALIDAÇÃO: Datas do trade (inserção <= remoção)
            # =============================================================
            erro_datas = self._validar_datas_trade(
                trade.get('data_insercao'),
                trade.get('data_remocao'),
                trade_id,
                ativo
            )
            if erro_datas:
                erros.append(erro_datas)
                trades_com_erro += 1
                continue  # Pula este trade

            # =============================================================
            # VALIDAÇÃO 2: Side (long/short)
            # =============================================================
            side_valido, side_normalizado = self._validar_side(trade.get('side'))
            if not side_valido:
                erros.append(Erro(
                    tipo=TipoErro.SIDE_INVALIDO,
                    trade_id=trade_id,
                    ativo=ativo,
                    mensagem=f"Side inválido para {ativo}: esperado 'long' ou 'short'",
                    valor_encontrado=trade.get('side')
                ))
                trades_com_erro += 1
                continue  # Pula este trade

            # =============================================================
            # VALIDAÇÃO 3: Preço de entrada
            # =============================================================
            preco_entrada_valido, preco_entrada = self._validar_preco(
                trade.get('preco_entrada_turma'), 'preco_entrada_turma'
            )
            if not preco_entrada_valido:
                erros.append(Erro(
                    tipo=TipoErro.PRECO_ENTRADA_INVALIDO,
                    trade_id=trade_id,
                    ativo=ativo,
                    mensagem=f"Preço de entrada inválido para {ativo}",
                    valor_encontrado=trade.get('preco_entrada_turma')
                ))
                trades_com_erro += 1
                continue  # Pula este trade

            # =============================================================
            # VALIDAÇÃO 4: Quantidade (baseado em usa_quantidade do produto)
            # =============================================================
            usa_quantidade = bool(trade.get('usa_quantidade', 0))
            qtd_valida, quantidade, deve_ignorar = self._validar_quantidade(
                trade.get('quantidade'), usa_quantidade
            )

            if deve_ignorar:
                # Produto não usa quantidade - trade não participa do cálculo
                continue

            if not qtd_valida:
                erros.append(Erro(
                    tipo=TipoErro.QUANTIDADE_INVALIDA,
                    trade_id=trade_id,
                    ativo=ativo,
                    mensagem=f"Quantidade inválida para {ativo} (produto requer quantidade > 0)",
                    valor_encontrado=trade.get('quantidade')
                ))
                trades_com_erro += 1
                continue  # Pula este trade

            # =============================================================
            # OBTER PREÇO DE COTAÇÃO (cache > API > banco > fallback)
            # =============================================================
            coingecko_id = trade.get('coingecko_id')
            preco_cotacao = None
            status_cotacao = 'ok'
            usou_fallback_preco = False

            if precos_cache is not None and coingecko_id:
                # Buscar no cache pré-carregado
                precos_ativo = precos_cache.get(coingecko_id, {})
                preco_cotacao = precos_ativo.get(dia)
                # Se não tem preço exato para o dia, buscar o mais recente anterior
                if preco_cotacao is None and precos_ativo:
                    datas_anteriores = [d for d in precos_ativo if d <= dia]
                    if datas_anteriores:
                        preco_cotacao = precos_ativo[max(datas_anteriores)]
            else:
                # Caminho original: query individual
                preco_cotacao, status_cotacao = self.obter_preco_cotacao(
                    trade_id,
                    dia,
                    ativo=ativo,
                    exchange_symbol=trade.get('exchange_symbol'),
                    coingecko_id=coingecko_id
                )

                if status_cotacao == 'timeout':
                    avisos.append(Aviso(
                        tipo=TipoAviso.API_TIMEOUT,
                        trade_id=trade_id,
                        ativo=ativo,
                        mensagem=f"Timeout na API CoinGecko para {ativo} - usando preço do banco"
                    ))

            if preco_cotacao is None:
                preco_cotacao = preco_entrada
                usou_fallback_preco = True
                avisos.append(Aviso(
                    tipo=TipoAviso.PRECO_FALLBACK,
                    trade_id=trade_id,
                    ativo=ativo,
                    mensagem=f"Usando preço de entrada como fallback para {ativo}"
                ))

            # Verificar se tem coingecko_id (necessário para atualização)
            if not trade.get('coingecko_id'):
                avisos.append(Aviso(
                    tipo=TipoAviso.SEM_COINGECKO_ID,
                    trade_id=trade_id,
                    ativo=ativo,
                    mensagem=f"{ativo} sem coingecko_id - não consegue atualizar preço"
                ))

            # Verificar variação de preço extrema (aviso)
            aviso_variacao = self._verificar_variacao_preco_extrema(
                preco_entrada, preco_cotacao, trade_id, ativo
            )
            if aviso_variacao:
                avisos.append(aviso_variacao)

            custo_inicial = preco_entrada * quantidade

            if trade['status'] == 'aberto':
                # Trade ainda aberto: contribui para valor de mercado
                if side_normalizado == 'long':
                    valor_mercado = preco_cotacao * quantidade
                else:  # short
                    pnl_nao_realizado = (preco_entrada - preco_cotacao) * quantidade
                    valor_mercado = custo_inicial + pnl_nao_realizado

                capital_alocado += valor_mercado
                custo_ativos += custo_inicial
                trades_calculados += 1

            else:  # vendido
                # ==========================================================
                # VALIDAÇÃO 4: Preço de saída para trades fechados
                # ==========================================================
                preco_saida = trade.get('preco_saida')
                usou_fallback_saida = False

                if preco_saida is None or preco_saida <= 0:
                    # Fallback: usar cotação atual (com aviso)
                    preco_saida = preco_cotacao
                    usou_fallback_saida = True
                    avisos.append(Aviso(
                        tipo=TipoAviso.PRECO_SAIDA_FALLBACK,
                        trade_id=trade_id,
                        ativo=ativo,
                        mensagem=f"Trade fechado {ativo} sem preço de saída - usando cotação"
                    ))

                # Calcular PnL baseado no side
                pnl = self._calcular_pnl(side_normalizado, preco_entrada, preco_saida, quantidade)
                pnl_fechado_dia += pnl
                trades_calculados += 1

        # Obter PnL fechado acumulado até o dia anterior
        if pnl_fechado_anterior_override is not None:
            pnl_fechado_anterior = pnl_fechado_anterior_override
        else:
            pnl_fechado_anterior = self._obter_pnl_fechado_acumulado(turma_id, dia)
        pnl_fechado_acumulado = pnl_fechado_anterior + pnl_fechado_dia

        # Capital em caixa
        capital_em_caixa = capital_base - custo_ativos + pnl_fechado_acumulado

        # Valor total do portfolio
        valor_total = capital_em_caixa + capital_alocado

        # =================================================================
        # VALIDAÇÃO: Capital em caixa negativo (aviso)
        # =================================================================
        if capital_em_caixa < 0:
            avisos.append(Aviso(
                tipo=TipoAviso.CAPITAL_CAIXA_NEGATIVO,
                trade_id=None,
                ativo=None,
                mensagem=f"Capital em caixa negativo: {capital_em_caixa:.2f}"
            ))

        # =================================================================
        # VALIDAÇÃO: Valor total negativo (erro grave)
        # =================================================================
        if valor_total < 0:
            erros.append(Erro(
                tipo=TipoErro.VALOR_TOTAL_NEGATIVO,
                trade_id=None,
                ativo=None,
                mensagem=f"Valor total do portfolio negativo: {valor_total:.2f}",
                valor_encontrado=valor_total
            ))

        # Rentabilidade diária (com proteção contra divisão por zero)
        if valor_anterior is None or valor_anterior <= 0:
            valor_anterior = capital_base
        rentabilidade_diaria = ((valor_total / valor_anterior) - 1) * 100

        # Rentabilidade acumulada (capital_base já validado > 0)
        rentabilidade_acumulada = ((valor_total / capital_base) - 1) * 100

        return DailyPortfolio(
            turma_id=turma_id,
            dia=dia,
            capital_alocado=capital_alocado,
            custo_ativos=custo_ativos,
            pnl_fechado_acumulado=pnl_fechado_acumulado,
            capital_em_caixa=capital_em_caixa,
            valor_total=valor_total,
            rentabilidade_diaria_pct=rentabilidade_diaria,
            rentabilidade_acumulada_pct=rentabilidade_acumulada,
            avisos=avisos,
            erros=erros,
            trades_calculados=trades_calculados,
            trades_com_erro=trades_com_erro
        )

    def _obter_pnl_fechado_acumulado(self, turma_id: int, dia: str) -> float:
        """
        Calcula o PnL fechado acumulado de trades fechados ANTES do dia especificado.

        Inclui validações de side e preços.
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute("""
            SELECT
                ct.preco_entrada_turma,
                pap.quantidade,
                p.preco_saida,
                p.side,
                p.ativo,
                COALESCE(prod.usa_quantidade, 0) as usa_quantidade
            FROM carteira_turma ct
            JOIN trades_turma tt ON ct.trade_id = tt.id
            JOIN posicoes p ON tt.posicao_id = p.id
            JOIN turmas t ON ct.turma_id = t.id
            JOIN produtos prod ON t.produto_id = prod.id
            LEFT JOIN posicao_atributos_produto pap ON p.id = pap.posicao_id
            WHERE ct.turma_id = %s
            AND ct.data_remocao IS NOT NULL
            AND date(ct.data_remocao) < date(%s)
        """, (turma_id, dia))

        pnl_total = 0.0
        for row in cursor.fetchall():
            preco_entrada = row['preco_entrada_turma']
            quantidade = row['quantidade']
            preco_saida = row['preco_saida']
            side = row['side']
            usa_quantidade = bool(row['usa_quantidade'])

            # Produto não usa quantidade - não participa do cálculo
            if not usa_quantidade:
                continue

            # Validar quantidade
            qtd_valida, quantidade, _ = self._validar_quantidade(quantidade, usa_quantidade)
            if not qtd_valida:
                continue  # Pula trade com quantidade inválida

            # Validar side
            side_valido, side_normalizado = self._validar_side(side)
            if not side_valido:
                continue  # Pula trade com side inválido

            # Validar preços
            if preco_saida is None or preco_saida <= 0:
                continue
            if preco_entrada is None or preco_entrada <= 0:
                continue

            pnl = self._calcular_pnl(side_normalizado, preco_entrada, preco_saida, quantidade)
            pnl_total += pnl

        conn.close()
        return pnl_total

    def calcular_serie_rentabilidade(self, turma_id: int,
                                      data_inicio: Optional[str] = None,
                                      data_fim: Optional[str] = None,
                                      precos_cache: Optional[Dict[str, Dict[str, float]]] = None) -> List[DailyPortfolio]:
        """
        Calcula a série histórica de rentabilidade de uma turma.

        Otimizado: pré-carrega todos os trades e preços históricos antes do loop,
        evitando queries SQL e chamadas API repetidas por dia.

        Args:
            turma_id: ID da turma
            data_inicio: Data inicial (se None, usa data_inicio da turma)
            data_fim: Data final (se None, usa hoje)
            precos_cache: Dict[coingecko_id, Dict[data, preco]] pré-carregado.
                         Se fornecido, pula as chamadas à API CoinGecko.
                         Útil para processar múltiplas turmas sem chamadas redundantes.

        Returns:
            Lista de DailyPortfolio ordenada por data
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Obter data_inicio e capital_base da turma
        cursor.execute("SELECT data_inicio, capital_base FROM turmas WHERE id = %s", (turma_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return []

        if data_inicio is None:
            data_inicio = row['data_inicio']
        capital_base = float(row['capital_base']) if row['capital_base'] else 1500.0

        if data_fim is None:
            data_fim = datetime.now().strftime("%Y-%m-%d")

        conn.close()

        # =====================================================================
        # PRÉ-CARREGAR: todos os trades da turma (uma única query)
        # =====================================================================
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""
            SELECT
                ct.trade_id,
                ct.turma_id,
                tt.posicao_id,
                p.ativo,
                p.side,
                p.exchange_symbol,
                p.coingecko_id,
                ct.origem,
                ct.data_insercao,
                ct.data_remocao,
                ct.preco_entrada_turma,
                pap.quantidade,
                p.preco_saida,
                p.status as status_posicao,
                COALESCE(prod.usa_quantidade, 0) as usa_quantidade
            FROM carteira_turma ct
            JOIN trades_turma tt ON ct.trade_id = tt.id
            JOIN posicoes p ON tt.posicao_id = p.id
            JOIN turmas t ON ct.turma_id = t.id
            JOIN produtos prod ON t.produto_id = prod.id
            LEFT JOIN posicao_atributos_produto pap ON p.id = pap.posicao_id
            WHERE ct.turma_id = %s
            AND date(ct.data_insercao) <= date(%s)
        """, (turma_id, data_fim))
        todos_trades = [dict(r) for r in cursor.fetchall()]
        conn.close()

        # =====================================================================
        # PRÉ-CARREGAR: preços históricos via API (pula se cache externo)
        # =====================================================================
        if precos_cache is None:
            coingecko_ids = list(set(
                t['coingecko_id'] for t in todos_trades
                if t.get('coingecko_id')
            ))

            dias_necessarios = (datetime.strptime(data_fim, "%Y-%m-%d") -
                               datetime.strptime(data_inicio, "%Y-%m-%d")).days + 5
            dias_api = min(dias_necessarios, 365)

            precos_cache = self.cotacoes_service.obter_historicos_batch(
                coingecko_ids, dias=dias_api
            )

            hoje = datetime.now().strftime("%Y-%m-%d")
            precos_hoje = self.cotacoes_service.obter_precos_batch_coingecko(coingecko_ids)
            for cg_id, preco in precos_hoje.items():
                if cg_id not in precos_cache:
                    precos_cache[cg_id] = {}
                precos_cache[cg_id][hoje] = preco

        # =====================================================================
        # LOOP DIA-A-DIA: tudo em memória, sem queries
        # =====================================================================
        current = datetime.strptime(data_inicio, "%Y-%m-%d")
        end = datetime.strptime(data_fim, "%Y-%m-%d")

        serie = []
        valor_anterior = None
        pnl_fechado_acumulado = 0.0

        while current <= end:
            dia = current.strftime("%Y-%m-%d")

            portfolio = self.calcular_portfolio_dia(
                turma_id, dia,
                valor_anterior=valor_anterior,
                precos_cache=precos_cache,
                trades_prefetched=todos_trades,
                capital_base_cache=capital_base,
                pnl_fechado_anterior_override=pnl_fechado_acumulado
            )

            # Acumular PnL fechado para o próximo dia
            pnl_fechado_acumulado = portfolio.pnl_fechado_acumulado

            serie.append(portfolio)
            valor_anterior = portfolio.valor_total
            current += timedelta(days=1)

        return serie

    def construir_precos_cache(self, turma_ids: Optional[List[int]] = None,
                               dias: int = 365) -> Dict[str, Dict[str, float]]:
        """
        Constrói cache de preços para múltiplas turmas em poucas chamadas API.

        Coleta todos os coingecko_ids únicos de todas as turmas, busca histórico
        e preço atual, e retorna um cache compartilhável.

        Args:
            turma_ids: Lista de IDs de turmas. Se None, usa todas as turmas.
            dias: Dias de histórico a buscar (max 365).

        Returns:
            Dict[coingecko_id, Dict[data, preco]]
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if turma_ids:
            placeholders = ','.join(['%s'] * len(turma_ids))
            cursor.execute(f"""
                SELECT DISTINCT p.coingecko_id
                FROM carteira_turma ct
                JOIN trades_turma tt ON ct.trade_id = tt.id
                JOIN posicoes p ON tt.posicao_id = p.id
                WHERE ct.turma_id IN ({placeholders})
                AND p.coingecko_id IS NOT NULL
            """, turma_ids)
        else:
            cursor.execute("""
                SELECT DISTINCT p.coingecko_id
                FROM carteira_turma ct
                JOIN trades_turma tt ON ct.trade_id = tt.id
                JOIN posicoes p ON tt.posicao_id = p.id
                WHERE p.coingecko_id IS NOT NULL
            """)

        coingecko_ids = [row['coingecko_id'] for row in cursor.fetchall()]
        conn.close()

        if not coingecko_ids:
            return {}

        # Histórico (1 chamada por ativo)
        precos_cache = self.cotacoes_service.obter_historicos_batch(
            coingecko_ids, dias=min(dias, 365)
        )

        # Preço de hoje (1 chamada batch para todos)
        hoje = datetime.now().strftime("%Y-%m-%d")
        precos_hoje = self.cotacoes_service.obter_precos_batch_coingecko(coingecko_ids)
        for cg_id, preco in precos_hoje.items():
            if cg_id not in precos_cache:
                precos_cache[cg_id] = {}
            precos_cache[cg_id][hoje] = preco

        return precos_cache

    def comparar_turmas(self, turma_ids: List[int],
                        data_inicio: Optional[str] = None,
                        data_fim: Optional[str] = None) -> Dict[int, List[DailyPortfolio]]:
        """
        Calcula rentabilidade de múltiplas turmas para comparação.

        Returns:
            Dict mapeando turma_id -> lista de DailyPortfolio
        """
        # Pré-carregar preços de todos os ativos de todas as turmas
        precos_cache = self.construir_precos_cache(turma_ids)

        resultado = {}
        for turma_id in turma_ids:
            resultado[turma_id] = self.calcular_serie_rentabilidade(
                turma_id, data_inicio, data_fim, precos_cache=precos_cache
            )
        return resultado

    def resumo_turma(self, turma_id: int) -> Dict[str, Any]:
        """
        Obtém resumo atual de uma turma.

        Returns:
            Dict com métricas resumidas, incluindo informações de validação
        """
        hoje = datetime.now().strftime("%Y-%m-%d")
        portfolio = self.calcular_portfolio_dia(turma_id, hoje)

        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Obter nome da turma
        cursor.execute("""
            SELECT t.nome, t.data_inicio, t.capital_base, p.nome as produto_nome
            FROM turmas t
            JOIN produtos p ON t.produto_id = p.id
            WHERE t.id = %s
        """, (turma_id,))
        turma_info = cursor.fetchone()

        # Contar trades ativos
        cursor.execute("""
            SELECT COUNT(*) as count FROM carteira_turma
            WHERE turma_id = %s AND ativo_atual = 1
        """, (turma_id,))
        trades_ativos = cursor.fetchone()['count']

        # Contar trades fechados
        cursor.execute("""
            SELECT COUNT(*) as count FROM carteira_turma
            WHERE turma_id = %s AND ativo_atual = 0
        """, (turma_id,))
        trades_fechados = cursor.fetchone()['count']

        conn.close()

        return {
            'turma_id': turma_id,
            'nome': turma_info['nome'] if turma_info else None,
            'produto': turma_info['produto_nome'] if turma_info else None,
            'data_inicio': turma_info['data_inicio'] if turma_info else None,
            'capital_base': turma_info['capital_base'] if turma_info else 1500.0,
            'trades_ativos': trades_ativos,
            'trades_fechados': trades_fechados,
            'valor_total': portfolio.valor_total,
            'rentabilidade_acumulada_pct': portfolio.rentabilidade_acumulada_pct,
            'capital_alocado': portfolio.capital_alocado,
            'capital_em_caixa': portfolio.capital_em_caixa,
            'data_calculo': hoje,
            # Informações de validação
            'tem_erros': portfolio.tem_erros,
            'tem_avisos': portfolio.tem_avisos,
            'trades_calculados': portfolio.trades_calculados,
            'trades_com_erro': portfolio.trades_com_erro,
            'confiabilidade_pct': portfolio.confiabilidade_pct,
            'erros': [{'tipo': e.tipo.value, 'ativo': e.ativo, 'mensagem': e.mensagem}
                      for e in portfolio.erros],
            'avisos': [{'tipo': a.tipo.value, 'ativo': a.ativo, 'mensagem': a.mensagem}
                       for a in portfolio.avisos]
        }

    def obter_rentabilidade_historica(self, turma_id: int,
                                       data_inicio: Optional[str] = None,
                                       data_fim: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém a rentabilidade acumulada histórica de uma turma.

        Retorna estrutura simplificada focada apenas na rentabilidade.

        Args:
            turma_id: ID da turma
            data_inicio: Data inicial (se None, usa data_inicio da turma)
            data_fim: Data final (se None, usa hoje)

        Returns:
            Dict com informações da turma e série de rentabilidade
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Obter informações da turma
        cursor.execute("""
            SELECT t.nome, t.data_inicio, t.capital_base, p.nome as produto_nome
            FROM turmas t
            JOIN produtos p ON t.produto_id = p.id
            WHERE t.id = %s
        """, (turma_id,))
        turma_info = cursor.fetchone()
        conn.close()

        if not turma_info:
            return {'erro': f'Turma {turma_id} não encontrada'}

        # Calcular série de rentabilidade
        serie = self.calcular_serie_rentabilidade(turma_id, data_inicio, data_fim)

        # Extrair apenas os dados de rentabilidade
        historico = [
            {
                'data': p.dia,
                'rentabilidade_acumulada_pct': round(p.rentabilidade_acumulada_pct, 4),
                'valor_total': round(p.valor_total, 2)
            }
            for p in serie
        ]

        # Calcular métricas resumidas
        if historico:
            rentab_atual = historico[-1]['rentabilidade_acumulada_pct']
            valor_atual = historico[-1]['valor_total']
            rentab_max = max(h['rentabilidade_acumulada_pct'] for h in historico)
            rentab_min = min(h['rentabilidade_acumulada_pct'] for h in historico)
        else:
            rentab_atual = 0.0
            valor_atual = turma_info['capital_base']
            rentab_max = 0.0
            rentab_min = 0.0

        return {
            'turma_id': turma_id,
            'nome': turma_info['nome'],
            'produto': turma_info['produto_nome'],
            'data_inicio': turma_info['data_inicio'],
            'capital_base': turma_info['capital_base'],
            'rentabilidade_atual_pct': rentab_atual,
            'rentabilidade_max_pct': rentab_max,
            'rentabilidade_min_pct': rentab_min,
            'valor_atual': valor_atual,
            'total_dias': len(historico),
            'historico': historico
        }

    def obter_rentabilidade_todas_turmas(self,
                                          data_inicio: Optional[str] = None,
                                          data_fim: Optional[str] = None,
                                          produto_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Obtém a rentabilidade acumulada histórica de todas as turmas.

        Args:
            data_inicio: Data inicial (se None, usa data_inicio de cada turma)
            data_fim: Data final (se None, usa hoje)
            produto_id: Filtrar por produto específico (opcional)

        Returns:
            Lista de dicts com rentabilidade histórica de cada turma
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Buscar todas as turmas
        if produto_id:
            cursor.execute("""
                SELECT id FROM turmas WHERE produto_id = %s ORDER BY data_inicio DESC
            """, (produto_id,))
        else:
            cursor.execute("SELECT id FROM turmas ORDER BY data_inicio DESC")

        turma_ids = [row['id'] for row in cursor.fetchall()]
        conn.close()

        resultado = []
        for turma_id in turma_ids:
            rentab = self.obter_rentabilidade_historica(turma_id, data_inicio, data_fim)
            if 'erro' not in rentab:
                resultado.append(rentab)

        return resultado

    def obter_rentabilidade_resumida_todas_turmas(self,
                                                   produto_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Obtém resumo da rentabilidade de todas as turmas (sem histórico completo).

        Usa batch queries SQL + uma única chamada batch à API CoinGecko para
        preços em tempo real. Mesma lógica de cálculo que calcular_portfolio_dia/resumo_turma.

        Args:
            produto_id: Filtrar por produto específico (opcional)

        Returns:
            Lista de dicts com resumo de rentabilidade de cada turma
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        hoje = datetime.now().strftime("%Y-%m-%d")

        # Query otimizada: busca turmas com métricas pré-calculadas em uma única query
        if produto_id:
            cursor.execute("""
                SELECT
                    t.id as turma_id,
                    t.nome,
                    t.data_inicio,
                    t.capital_base,
                    p.nome as produto_nome,
                    COALESCE(p.usa_quantidade, 0) as usa_quantidade,
                    (SELECT COUNT(*) FROM carteira_turma WHERE turma_id = t.id AND ativo_atual = 1) as trades_ativos,
                    (SELECT COUNT(*) FROM carteira_turma WHERE turma_id = t.id AND ativo_atual = 0) as trades_fechados
                FROM turmas t
                JOIN produtos p ON t.produto_id = p.id
                WHERE t.produto_id = %s
                ORDER BY t.data_inicio DESC
            """, (produto_id,))
        else:
            cursor.execute("""
                SELECT
                    t.id as turma_id,
                    t.nome,
                    t.data_inicio,
                    t.capital_base,
                    p.nome as produto_nome,
                    COALESCE(p.usa_quantidade, 0) as usa_quantidade,
                    (SELECT COUNT(*) FROM carteira_turma WHERE turma_id = t.id AND ativo_atual = 1) as trades_ativos,
                    (SELECT COUNT(*) FROM carteira_turma WHERE turma_id = t.id AND ativo_atual = 0) as trades_fechados
                FROM turmas t
                JOIN produtos p ON t.produto_id = p.id
                ORDER BY t.data_inicio DESC
            """)

        turmas = [dict(row) for row in cursor.fetchall()]

        if not turmas:
            conn.close()
            return []

        turma_ids = [t['turma_id'] for t in turmas]

        # Buscar todos os trades de todas as turmas de uma vez (inclui coingecko_id)
        placeholders = ','.join(['%s'] * len(turma_ids))
        cursor.execute(f"""
            SELECT
                ct.turma_id,
                ct.trade_id,
                ct.preco_entrada_turma,
                ct.data_insercao,
                ct.data_remocao,
                p.side,
                p.preco_saida,
                p.coingecko_id,
                pap.quantidade,
                (SELECT preco FROM trade_valores_diarios
                 WHERE trade_id = ct.trade_id
                 ORDER BY data DESC LIMIT 1) as ultimo_preco_db
            FROM carteira_turma ct
            JOIN trades_turma tt ON ct.trade_id = tt.id
            JOIN posicoes p ON tt.posicao_id = p.id
            LEFT JOIN posicao_atributos_produto pap ON p.id = pap.posicao_id
            WHERE ct.turma_id IN ({placeholders})
            AND date(ct.data_insercao) <= date(%s)
        """, (*turma_ids, hoje))

        trades_por_turma = {}
        coingecko_ids_necessarios = set()
        for row in cursor.fetchall():
            turma_id = row['turma_id']
            if turma_id not in trades_por_turma:
                trades_por_turma[turma_id] = []
            trade = dict(row)
            trades_por_turma[turma_id].append(trade)

            # Coletar coingecko_ids de trades que precisam de preço em tempo real
            data_remocao = trade.get('data_remocao')
            precisa_preco_atual = (data_remocao is None or data_remocao >= hoje)
            if precisa_preco_atual and trade.get('coingecko_id'):
                coingecko_ids_necessarios.add(trade['coingecko_id'])

        conn.close()

        # Buscar preços em tempo real via batch (uma única chamada API)
        precos_realtime = {}
        if coingecko_ids_necessarios:
            precos_realtime = self.cotacoes_service.obter_precos_batch_coingecko(
                list(coingecko_ids_necessarios)
            )

        resultado = []
        for turma in turmas:
            turma_id = turma['turma_id']
            capital_base = turma['capital_base'] or 1500.0
            usa_quantidade = bool(turma['usa_quantidade'])

            trades = trades_por_turma.get(turma_id, [])

            capital_alocado = 0.0
            custo_ativos = 0.0
            pnl_fechado = 0.0

            for trade in trades:
                if not usa_quantidade:
                    continue

                quantidade = trade.get('quantidade')
                if quantidade is None or quantidade <= 0:
                    continue

                preco_entrada = trade.get('preco_entrada_turma')
                if preco_entrada is None or preco_entrada <= 0:
                    continue

                side = str(trade.get('side', '')).lower()
                if side not in ('long', 'short'):
                    continue

                data_remocao = trade.get('data_remocao')
                custo_inicial = preco_entrada * quantidade

                # Resolver preço atual: API real-time > DB > preco_entrada
                cg_id = trade.get('coingecko_id')
                preco_atual = (
                    precos_realtime.get(cg_id)
                    or trade.get('ultimo_preco_db')
                    or preco_entrada
                )

                if data_remocao is None or data_remocao >= hoje:
                    if data_remocao == hoje:
                        # Fechado hoje - usar preco_saida, fallback para cotação atual
                        preco_saida = trade.get('preco_saida')
                        if preco_saida is None or preco_saida <= 0:
                            preco_saida = preco_atual
                        pnl_fechado += self._calcular_pnl(side, preco_entrada, preco_saida, quantidade)
                    else:
                        # Ainda aberto
                        if side == 'long':
                            valor_mercado = preco_atual * quantidade
                        else:
                            pnl_nao_realizado = (preco_entrada - preco_atual) * quantidade
                            valor_mercado = custo_inicial + pnl_nao_realizado
                        capital_alocado += valor_mercado
                        custo_ativos += custo_inicial
                else:
                    # Trade fechado antes de hoje
                    preco_saida = trade.get('preco_saida')
                    if preco_saida is None or preco_saida <= 0:
                        preco_saida = preco_atual
                    pnl_fechado += self._calcular_pnl(side, preco_entrada, preco_saida, quantidade)

            capital_em_caixa = capital_base - custo_ativos + pnl_fechado
            valor_total = capital_em_caixa + capital_alocado
            rentabilidade = ((valor_total / capital_base) - 1) * 100 if capital_base > 0 else 0

            resultado.append({
                'turma_id': turma_id,
                'nome': turma['nome'],
                'produto': turma['produto_nome'],
                'data_inicio': turma['data_inicio'],
                'capital_base': capital_base,
                'trades_ativos': turma['trades_ativos'],
                'trades_fechados': turma['trades_fechados'],
                'rentabilidade_atual_pct': round(rentabilidade, 4),
                'rentabilidade_acumulada_pct': round(rentabilidade, 4),
                'rentabilidade_max_pct': round(rentabilidade, 4),  # Simplificado
                'rentabilidade_min_pct': round(min(0, rentabilidade), 4),  # Simplificado
                'valor_atual': round(valor_total, 2),
                'valor_total': round(valor_total, 2),
                'total_dias': (datetime.now() - datetime.strptime(turma['data_inicio'], "%Y-%m-%d")).days if turma['data_inicio'] else 0,
                'data_calculo': hoje
            })

        return resultado


def criar_rentabilidade_service() -> RentabilidadeService:
    """Factory function para criar uma instância do serviço."""
    return RentabilidadeService()


if __name__ == "__main__":
    # Teste básico
    service = criar_rentabilidade_service()
    print("RentabilidadeService criado com sucesso")
    print(f"Database URL: {service.db_url}")

    # Testar com primeira turma
    conn = service._get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("SELECT id, nome FROM turmas LIMIT 3")
    turmas = cursor.fetchall()
    conn.close()

    if turmas:
        print("\nResumo das primeiras turmas:")
        for turma in turmas:
            print(f"\n=== {turma['nome']} ===")
            resumo = service.resumo_turma(turma['id'])
            print(f"  Valor Total: R$ {resumo['valor_total']:.2f}")
            print(f"  Rentabilidade: {resumo['rentabilidade_acumulada_pct']:.2f}%")
            print(f"  Trades Ativos: {resumo['trades_ativos']}")
            print(f"  Trades Fechados: {resumo['trades_fechados']}")
