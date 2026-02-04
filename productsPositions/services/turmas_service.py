"""
Serviço para gerenciar Turmas e Rentabilidade.

Este módulo implementa a lógica de negócio para:
- Criar e gerenciar turmas
- Sincronizar posições com turmas
- Calcular rentabilidade por turma
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from services.cotacoes_service import CotacoesService


@dataclass
class Turma:
    id: Optional[int]
    produto_id: int
    nome: str
    data_inicio: str
    capital_base: float = 1500.0
    descricao: Optional[str] = None
    data_criacao: Optional[str] = None


@dataclass
class TradeTurma:
    id: Optional[int]
    turma_id: int
    posicao_id: int


@dataclass
class CarteiraTurma:
    id: Optional[int]
    turma_id: int
    trade_id: int
    origem: str  # 'nativo' ou 'replicado'
    data_insercao: str
    data_remocao: Optional[str]
    preco_entrada_turma: float
    ativo_atual: bool = True


class TurmasService:
    """Serviço para gerenciar turmas e rentabilidade."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            script_dir = Path(__file__).parent
            products_positions_dir = script_dir.parent
            db_path = products_positions_dir / "data" / "products_positions.db"
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ================================================================
    # CRUD TURMAS
    # ================================================================

    def criar_turma(self, produto_id: int, nome: str, data_inicio: str,
                    capital_base: float = 1500.0, descricao: str = None,
                    posicoes_config: List[Dict] = None,
                    auto_fetch_prices: bool = True) -> Dict:
        """
        Cria uma nova turma com busca automática de preços via CoinGecko.

        Cada posição deve ter sua própria data_insercao. O preco_entrada_turma
        pode ser fornecido manualmente ou buscado automaticamente via CoinGecko.

        Args:
            produto_id: ID do produto
            nome: Nome da turma
            data_inicio: Data de início da turma (YYYY-MM-DD)
            capital_base: Capital inicial da turma
            descricao: Descrição opcional
            posicoes_config: Lista de dicts com configuração de cada posição.
                            Cada dict DEVE ter:
                              - posicao_id: int (obrigatório)
                              - data_insercao: str YYYY-MM-DD (obrigatório)
                              - preco_entrada_turma: float (OPCIONAL se auto_fetch_prices=True)
            auto_fetch_prices: Se True, busca preços automaticamente via CoinGecko
                              quando preco_entrada_turma não for fornecido

        Returns:
            Dict com:
                - turma_id: int
                - precos_resolvidos: List[Dict] com detalhes de cada preço
                - avisos: List[str] com avisos sobre fallbacks

        Raises:
            ValueError: Se posicoes_config não for fornecido ou estiver vazio

        Example:
            resultado = service.criar_turma(
                produto_id=1,
                nome="Turma Janeiro",
                data_inicio="2026-01-15",
                posicoes_config=[
                    {"posicao_id": 123, "data_insercao": "2026-01-15"},  # Preço buscado auto
                    {"posicao_id": 456, "data_insercao": "2026-01-20", "preco_entrada_turma": 3200.0},
                ],
                auto_fetch_prices=True
            )
            # resultado = {'turma_id': 42, 'precos_resolvidos': [...], 'avisos': [...]}
        """
        # Validar posicoes_config é obrigatório
        if posicoes_config is None or len(posicoes_config) == 0:
            raise ValueError(
                "posicoes_config é obrigatório. Use /api/turma/posicoes-elegiveis para "
                "listar posições elegíveis."
            )

        # Validar cada configuração de posição
        for i, config in enumerate(posicoes_config):
            if 'posicao_id' not in config:
                raise ValueError(f"posicoes_config[{i}]: posicao_id é obrigatório")
            if 'data_insercao' not in config:
                raise ValueError(f"posicoes_config[{i}]: data_insercao é obrigatório")
            # preco_entrada_turma só é obrigatório se auto_fetch_prices=False
            if not auto_fetch_prices and 'preco_entrada_turma' not in config:
                raise ValueError(f"posicoes_config[{i}]: preco_entrada_turma é obrigatório quando auto_fetch_prices=False")

        avisos = []
        precos_resolvidos = []

        # Resolver preços para cada posição
        cotacoes_service = CotacoesService(db_path=self.db_path)

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            for config in posicoes_config:
                posicao_id = config['posicao_id']
                data_insercao = config['data_insercao']
                preco_manual = config.get('preco_entrada_turma')

                if preco_manual is not None:
                    # Usuário forneceu preço manualmente
                    preco_resolvido = {
                        'posicao_id': posicao_id,
                        'preco': float(preco_manual),
                        'fonte': 'manual',
                        'data_referencia': data_insercao,
                        'moeda': 'USD',
                        'status': 'ok'
                    }
                elif auto_fetch_prices:
                    # Buscar preço automaticamente
                    preco_resolvido = self._resolver_preco_automatico(
                        cursor, cotacoes_service, posicao_id, data_insercao
                    )

                    if preco_resolvido.get('status') == 'erro':
                        avisos.append(f"Posição {posicao_id}: {preco_resolvido.get('erro')}")
                    elif preco_resolvido.get('aviso'):
                        avisos.append(f"Posição {posicao_id}: {preco_resolvido.get('aviso')}")
                else:
                    raise ValueError(f"posicoes_config[{posicao_id}]: preco_entrada_turma é obrigatório")

                precos_resolvidos.append(preco_resolvido)
                config['_preco_resolvido'] = preco_resolvido

            # Inserir turma
            cursor.execute("""
                INSERT INTO turmas (produto_id, nome, data_inicio, capital_base, descricao, data_criacao)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (produto_id, nome, data_inicio, capital_base, descricao, datetime.now().strftime("%Y-%m-%d")))

            turma_id = cursor.lastrowid

            # Adicionar cada posição com sua configuração resolvida
            for config in posicoes_config:
                preco_info = config.get('_preco_resolvido', {})
                self._adicionar_trade_turma(
                    cursor=cursor,
                    turma_id=turma_id,
                    posicao_id=config['posicao_id'],
                    origem='replicado',
                    data_insercao=config['data_insercao'],
                    preco_entrada_turma=preco_info.get('preco'),
                    preco_fonte=preco_info.get('fonte', 'manual'),
                    preco_data_referencia=preco_info.get('data_referencia'),
                    preco_moeda=preco_info.get('moeda', 'USD')
                )

            conn.commit()

            return {
                'turma_id': turma_id,
                'precos_resolvidos': precos_resolvidos,
                'avisos': avisos
            }

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _resolver_preco_automatico(self, cursor, cotacoes_service: CotacoesService,
                                    posicao_id: int, data_insercao: str) -> Dict:
        """
        Resolve o preço de entrada automaticamente com fallbacks.

        Ordem de tentativa:
        1. CoinGecko (se posição tem coingecko_id)
        2. Preço original da posição (fallback)
        """
        # Buscar dados da posição
        cursor.execute("""
            SELECT coingecko_id, preco_entrada, ativo FROM posicoes WHERE id = ?
        """, (posicao_id,))
        posicao = cursor.fetchone()

        if not posicao:
            return {
                'posicao_id': posicao_id,
                'status': 'erro',
                'erro': f'Posição {posicao_id} não encontrada'
            }

        coingecko_id = posicao['coingecko_id']
        preco_original = posicao['preco_entrada']
        ativo = posicao['ativo']

        # Tentar buscar via CoinGecko
        if coingecko_id:
            resultado = cotacoes_service.obter_preco_historico_exato(coingecko_id, data_insercao)

            if resultado.get('status') in ('ok', 'fallback'):
                resultado['posicao_id'] = posicao_id
                return resultado

        # Fallback: usar preço original da posição
        return {
            'posicao_id': posicao_id,
            'preco': preco_original,
            'fonte': 'original',
            'data_referencia': data_insercao,
            'moeda': 'USD',
            'status': 'fallback',
            'aviso': f"Usando preço original da posição {ativo} (CoinGecko indisponível)"
        }

    def listar_posicoes_elegiveis(self, produto_id: int, data_inicio: str) -> List[Dict]:
        """
        Lista posições abertas de um produto anteriores a uma data.

        Usado para selecionar quais posições replicar ao criar uma turma.

        Args:
            produto_id: ID do produto
            data_inicio: Data de início da turma (YYYY-MM-DD)

        Returns:
            Lista de dicts com dados das posições elegíveis
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    p.id,
                    p.ativo,
                    p.coingecko_id,
                    p.exchange_symbol,
                    p.side,
                    p.data_entrada,
                    p.preco_entrada,
                    p.status,
                    pap.quantidade
                FROM posicoes p
                LEFT JOIN posicao_atributos_produto pap ON p.id = pap.posicao_id
                WHERE p.produto_id = ?
                AND p.status = 'open'
                AND date(p.data_entrada) < date(?)
                ORDER BY p.data_entrada DESC
            """, (produto_id, data_inicio))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def listar_turmas(self, produto_id: Optional[int] = None) -> List[Dict]:
        """Lista todas as turmas, opcionalmente filtradas por produto."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if produto_id:
            cursor.execute("""
                SELECT t.*, p.nome as produto_nome
                FROM turmas t
                JOIN produtos p ON t.produto_id = p.id
                WHERE t.produto_id = ?
                ORDER BY t.data_inicio DESC
            """, (produto_id,))
        else:
            cursor.execute("""
                SELECT t.*, p.nome as produto_nome
                FROM turmas t
                JOIN produtos p ON t.produto_id = p.id
                ORDER BY t.data_inicio DESC
            """)

        result = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return result

    def obter_turma(self, turma_id: int) -> Optional[Dict]:
        """Obtém detalhes de uma turma específica."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT t.*, p.nome as produto_nome
            FROM turmas t
            JOIN produtos p ON t.produto_id = p.id
            WHERE t.id = ?
        """, (turma_id,))

        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # ================================================================
    # TRADES E CARTEIRA
    # ================================================================

    def _adicionar_trade_turma(self, cursor, turma_id: int, posicao_id: int,
                               origem: str, data_insercao: str,
                               preco_entrada_turma: Optional[float] = None,
                               preco_fonte: str = 'manual',
                               preco_data_referencia: Optional[str] = None,
                               preco_moeda: str = 'USD') -> int:
        """
        Adiciona um trade à turma (interno, usa cursor existente).

        Args:
            cursor: Cursor do banco de dados
            turma_id: ID da turma
            posicao_id: ID da posição
            origem: 'nativo' ou 'replicado'
            data_insercao: Data de inserção (YYYY-MM-DD)
            preco_entrada_turma: Preço de entrada (se None, usa preco_entrada da posição)
            preco_fonte: Fonte do preço ('coingecko', 'manual', 'original')
            preco_data_referencia: Data efetiva do preço (pode diferir de data_insercao)
            preco_moeda: Moeda do preço (default: 'USD')

        Returns:
            ID do trade_turma criado
        """
        # Criar link em trades_turma
        cursor.execute("""
            INSERT OR IGNORE INTO trades_turma (turma_id, posicao_id)
            VALUES (?, ?)
        """, (turma_id, posicao_id))

        # Obter ID do trade_turma
        cursor.execute("""
            SELECT id FROM trades_turma
            WHERE turma_id = ? AND posicao_id = ?
        """, (turma_id, posicao_id))
        trade_id = cursor.fetchone()['id']

        # Se preço não informado, buscar da posição (para nativos)
        if preco_entrada_turma is None:
            cursor.execute("SELECT preco_entrada FROM posicoes WHERE id = ?", (posicao_id,))
            preco_entrada_turma = cursor.fetchone()['preco_entrada']
            preco_fonte = 'original'

        # Se data_referencia não informada, usar data_insercao
        if preco_data_referencia is None:
            preco_data_referencia = data_insercao

        # Adicionar à carteira da turma com metadados de preço
        cursor.execute("""
            INSERT INTO carteira_turma
            (turma_id, trade_id, origem, data_insercao, preco_entrada_turma,
             preco_fonte, preco_data_referencia, preco_moeda, ativo_atual)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (turma_id, trade_id, origem, data_insercao, preco_entrada_turma,
              preco_fonte, preco_data_referencia, preco_moeda))

        return trade_id

    def adicionar_posicao_turma(self, turma_id: int, posicao_id: int,
                                 data_insercao: Optional[str] = None,
                                 preco_entrada_turma: Optional[float] = None) -> int:
        """
        Adiciona uma posição existente a uma turma.

        Determina automaticamente se é 'nativo' ou 'replicado':
        - Nativo: posição criada após data_inicio da turma
        - Replicado: posição criada antes da data_inicio da turma
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Obter dados da turma e posição
            cursor.execute("SELECT data_inicio FROM turmas WHERE id = ?", (turma_id,))
            turma = cursor.fetchone()
            if not turma:
                raise ValueError(f"Turma {turma_id} não encontrada")

            cursor.execute("SELECT data_entrada, preco_entrada FROM posicoes WHERE id = ?", (posicao_id,))
            posicao = cursor.fetchone()
            if not posicao:
                raise ValueError(f"Posição {posicao_id} não encontrada")

            # Determinar origem e data_insercao
            data_inicio_turma = turma['data_inicio']
            data_entrada_posicao = posicao['data_entrada']

            if data_entrada_posicao >= data_inicio_turma:
                origem = 'nativo'
                data_insercao = data_insercao or data_entrada_posicao
                preco_entrada_turma = preco_entrada_turma or posicao['preco_entrada']
                preco_fonte = 'original'
            else:
                origem = 'replicado'
                data_insercao = data_insercao or data_inicio_turma
                # Para replicado, preco_entrada_turma deve ser a cotação na data_insercao
                # Se não fornecido, será None (precisa buscar via cotações)
                preco_fonte = 'manual' if preco_entrada_turma else 'original'

            trade_id = self._adicionar_trade_turma(
                cursor=cursor,
                turma_id=turma_id,
                posicao_id=posicao_id,
                origem=origem,
                data_insercao=data_insercao,
                preco_entrada_turma=preco_entrada_turma,
                preco_fonte=preco_fonte,
                preco_data_referencia=data_insercao,
                preco_moeda='USD'
            )

            conn.commit()
            return trade_id

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def fechar_trade_turma(self, turma_id: int, posicao_id: int,
                           data_remocao: Optional[str] = None) -> bool:
        """
        Marca um trade como fechado na carteira da turma.

        Chamado quando uma posição é fechada no MissionControl.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            data_remocao = data_remocao or datetime.now().strftime("%Y-%m-%d")

            # Encontrar o trade_turma
            cursor.execute("""
                SELECT tt.id
                FROM trades_turma tt
                WHERE tt.turma_id = ? AND tt.posicao_id = ?
            """, (turma_id, posicao_id))

            trade = cursor.fetchone()
            if not trade:
                return False

            # Atualizar carteira_turma
            cursor.execute("""
                UPDATE carteira_turma
                SET data_remocao = ?, ativo_atual = 0
                WHERE turma_id = ? AND trade_id = ? AND ativo_atual = 1
            """, (data_remocao, turma_id, trade['id']))

            conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def listar_carteira_turma(self, turma_id: int, apenas_ativos: bool = False) -> List[Dict]:
        """Lista trades na carteira de uma turma."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                ct.id,
                ct.turma_id,
                ct.trade_id,
                ct.origem,
                ct.data_insercao,
                ct.data_remocao,
                ct.preco_entrada_turma,
                ct.ativo_atual,
                p.ativo,
                p.coingecko_id,
                p.side,
                p.data_entrada as data_entrada_original,
                p.preco_entrada as preco_entrada_original,
                p.data_saida,
                p.preco_saida,
                p.status as status_posicao,
                pap.quantidade
            FROM carteira_turma ct
            JOIN trades_turma tt ON ct.trade_id = tt.id
            JOIN posicoes p ON tt.posicao_id = p.id
            LEFT JOIN posicao_atributos_produto pap ON p.id = pap.posicao_id
            WHERE ct.turma_id = ?
        """

        if apenas_ativos:
            query += " AND ct.ativo_atual = 1"

        query += " ORDER BY ct.data_insercao DESC"

        cursor.execute(query, (turma_id,))
        result = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return result

    # ================================================================
    # SYNC COM POSIÇÕES (HOOKS)
    # ================================================================

    def sync_nova_posicao(self, posicao_id: int, produto_id: int,
                          data_entrada: str, preco_entrada: float):
        """
        Hook: chamado quando uma nova posição é criada.
        Adiciona a posição a todas as turmas ativas do produto.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Buscar turmas ativas do produto
            cursor.execute("""
                SELECT id, data_inicio
                FROM turmas
                WHERE produto_id = ?
            """, (produto_id,))

            turmas = cursor.fetchall()

            for turma in turmas:
                # Determinar se é nativo (posição criada após turma)
                if data_entrada >= turma['data_inicio']:
                    self._adicionar_trade_turma(
                        cursor=cursor,
                        turma_id=turma['id'],
                        posicao_id=posicao_id,
                        origem='nativo',
                        data_insercao=data_entrada,
                        preco_entrada_turma=preco_entrada,
                        preco_fonte='original',  # Native positions use actual entry price
                        preco_data_referencia=data_entrada,
                        preco_moeda='USD'
                    )

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def sync_posicao_fechada(self, posicao_id: int, data_saida: str):
        """
        Hook: chamado quando uma posição é fechada.
        Fecha o trade em todas as turmas onde está ativo.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Encontrar todos os trades_turma desta posição
            cursor.execute("""
                SELECT tt.id, ct.turma_id
                FROM trades_turma tt
                JOIN carteira_turma ct ON tt.id = ct.trade_id
                WHERE tt.posicao_id = ? AND ct.ativo_atual = 1
            """, (posicao_id,))

            trades = cursor.fetchall()

            for trade in trades:
                cursor.execute("""
                    UPDATE carteira_turma
                    SET data_remocao = ?, ativo_atual = 0
                    WHERE trade_id = ? AND ativo_atual = 1
                """, (data_saida, trade['id']))

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    # ================================================================
    # VALORES DIÁRIOS
    # ================================================================

    def registrar_valor_diario(self, trade_id: int, data: str, preco: float,
                                fonte: str = 'manual'):
        """Registra o valor diário de um trade."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trade_valores_diarios (trade_id, data, preco, fonte)
                VALUES (?, ?, ?, ?)
            """, (trade_id, data, preco, fonte))
            conn.commit()
        finally:
            conn.close()

    def obter_preco_na_data(self, trade_id: int, data: str) -> Optional[float]:
        """Obtém o preço mais recente até uma data específica."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT preco FROM trade_valores_diarios
            WHERE trade_id = ? AND data <= ?
            ORDER BY data DESC
            LIMIT 1
        """, (trade_id, data))

        row = cursor.fetchone()
        conn.close()
        return row['preco'] if row else None

    def listar_valores_diarios(self, trade_id: int,
                                data_inicio: Optional[str] = None,
                                data_fim: Optional[str] = None) -> List[Dict]:
        """Lista valores diários de um trade."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM trade_valores_diarios WHERE trade_id = ?"
        params = [trade_id]

        if data_inicio:
            query += " AND data >= ?"
            params.append(data_inicio)

        if data_fim:
            query += " AND data <= ?"
            params.append(data_fim)

        query += " ORDER BY data DESC"

        cursor.execute(query, params)
        result = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return result


# ================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ================================================================

def criar_turmas_service() -> TurmasService:
    """Factory function para criar uma instância do serviço."""
    return TurmasService()


if __name__ == "__main__":
    # Teste básico
    service = criar_turmas_service()
    print("TurmasService criado com sucesso")
    print(f"Database: {service.db_path}")

    # Listar turmas existentes
    turmas = service.listar_turmas()
    print(f"\nTurmas existentes: {len(turmas)}")
    for t in turmas:
        print(f"  - {t.get('nome', 'N/A')}: {t.get('data_inicio', 'N/A')}")
