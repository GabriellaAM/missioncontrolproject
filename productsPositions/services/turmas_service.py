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
                    capital_base: float = 1500.0, descricao: str = None) -> int:
        """
        Cria uma nova turma e replica posições abertas do produto.

        Args:
            produto_id: ID do produto
            nome: Nome da turma
            data_inicio: Data de início (YYYY-MM-DD)
            capital_base: Capital inicial da turma
            descricao: Descrição opcional

        Returns:
            ID da turma criada
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Inserir turma
            cursor.execute("""
                INSERT INTO turmas (produto_id, nome, data_inicio, capital_base, descricao)
                VALUES (?, ?, ?, ?, ?)
            """, (produto_id, nome, data_inicio, capital_base, descricao))

            turma_id = cursor.lastrowid

            # Buscar posições abertas do produto anteriores à data_inicio
            cursor.execute("""
                SELECT id, data_entrada, preco_entrada
                FROM posicoes
                WHERE produto_id = ?
                AND status = 'open'
                AND date(data_entrada) < date(?)
            """, (produto_id, data_inicio))

            posicoes_abertas = cursor.fetchall()

            # Replicar posições abertas para a turma
            for posicao in posicoes_abertas:
                self._adicionar_trade_turma(
                    cursor=cursor,
                    turma_id=turma_id,
                    posicao_id=posicao['id'],
                    origem='replicado',
                    data_insercao=data_inicio,
                    preco_entrada_turma=None  # Será buscado via API/parquet
                )

            conn.commit()
            return turma_id

        except Exception as e:
            conn.rollback()
            raise e
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
                               preco_entrada_turma: Optional[float] = None) -> int:
        """
        Adiciona um trade à turma (interno, usa cursor existente).

        Se preco_entrada_turma for None:
        - Para trades 'nativo': usa o preco_entrada da posição
        - Para trades 'replicado': precisa ser preenchido posteriormente via cotação
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

        # Adicionar à carteira da turma
        cursor.execute("""
            INSERT INTO carteira_turma
            (turma_id, trade_id, origem, data_insercao, preco_entrada_turma, ativo_atual)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (turma_id, trade_id, origem, data_insercao, preco_entrada_turma))

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
            else:
                origem = 'replicado'
                data_insercao = data_insercao or data_inicio_turma
                # Para replicado, preco_entrada_turma deve ser a cotação na data_insercao
                # Se não fornecido, será None (precisa buscar via cotações)

            trade_id = self._adicionar_trade_turma(
                cursor=cursor,
                turma_id=turma_id,
                posicao_id=posicao_id,
                origem=origem,
                data_insercao=data_insercao,
                preco_entrada_turma=preco_entrada_turma
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
                p.status as status_posicao
            FROM carteira_turma ct
            JOIN trades_turma tt ON ct.trade_id = tt.id
            JOIN posicoes p ON tt.posicao_id = p.id
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
                        preco_entrada_turma=preco_entrada
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
