import psycopg2
import psycopg2.extras
import sqlite3
import warnings
import pandas as pd
import uuid
import os
import socket
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import contextmanager
from dotenv import load_dotenv
import threading

# Suprime aviso do pandas ao usar nosso wrapper SQLite (compatível com DBAPI2)
warnings.filterwarnings("ignore", message=".*Other DBAPI2.*", category=UserWarning)


def _resolver_ipv4_do_host(db_url: str) -> str:
    """
    Resolve o hostname da URL do PostgreSQL para IPv4 e retorna o IP.
    psycopg2 usa libpq (C), que ignora monkey-patches do Python no socket.
    Por isso resolvemos manualmente e passamos o IP via hostaddr.
    """
    match = re.search(r'@([^:/@]+)', db_url)
    if match:
        hostname = match.group(1)
        try:
            resultado = socket.getaddrinfo(hostname, None, socket.AF_INET)
            if resultado:
                return resultado[0][4][0]  # endereco IPv4
        except Exception:
            pass
    return None


def _sqlite_path_from_url(url: str) -> Path:
    """Extrai path do arquivo a partir de URL sqlite:///path."""
    path_str = url.replace("sqlite:///", "").replace("sqlite://", "")
    return Path(path_str)


class _SqliteCursorWrapper:
    """Cursor que traduz placeholders %s (PostgreSQL) para ? (SQLite).
    Expõe todos os atributos DBAPI2 necessários (description, rowcount, etc.)
    para compatibilidade com pd.read_sql_query e com 'with cursor:'."""
    def __init__(self, cursor):
        self._cur = cursor
    def execute(self, sql, params=None):
        sql = sql.replace("%s", "?")
        if params is not None:
            self._cur.execute(sql, params)
        else:
            self._cur.execute(sql)
        return self
    def executemany(self, sql, params_list):
        sql = sql.replace("%s", "?")
        self._cur.executemany(sql, params_list)
        return self
    def fetchone(self): return self._cur.fetchone()
    def fetchall(self): return self._cur.fetchall()
    def fetchmany(self, size=None):
        return self._cur.fetchmany(size) if size else self._cur.fetchmany()
    def close(self): self._cur.close()
    def __iter__(self): return iter(self._cur)
    def __enter__(self): return self
    def __exit__(self, *args): self.close(); return False
    @property
    def description(self): return self._cur.description
    @property
    def rowcount(self): return self._cur.rowcount
    @property
    def lastrowid(self): return self._cur.lastrowid


class _SqliteConnectionWrapper:
    """Wrapper de conexão sqlite3 compatível com DBAPI2 + pandas."""
    _is_sqlite = True
    def __init__(self, conn):
        self._conn = conn
    def cursor(self):
        return _SqliteCursorWrapper(self._conn.cursor())
    def execute(self, sql, params=None):
        """Permite conn.execute() direto (usado por pandas internamente)."""
        sql = sql.replace("%s", "?")
        if params is not None:
            return self._conn.execute(sql, params)
        return self._conn.execute(sql)
    def commit(self): self._conn.commit()
    def rollback(self): self._conn.rollback()
    def close(self): self._conn.close()
    def __enter__(self): return self
    def __exit__(self, *a): self.close(); return False


def connect_pg(db_url, **kwargs):
    """
    Conexão com o banco: PostgreSQL (Supabase) ou SQLite local (fallback).
    - URL postgresql://... -> psycopg2 (força IPv4 quando possível).
    - URL sqlite:///path -> sqlite3 (fallback quando Supabase inacessível).
    """
    if not db_url:
        raise RuntimeError("db_url nao informado")
    if db_url.strip().lower().startswith("sqlite://"):
        path = _sqlite_path_from_url(db_url)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / path
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        return _SqliteConnectionWrapper(conn)
    hostaddr = _resolver_ipv4_do_host(db_url)
    if hostaddr:
        return psycopg2.connect(db_url, hostaddr=hostaddr, **kwargs)
    return psycopg2.connect(db_url, **kwargs)

# Carregar .env do root do projeto
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / '.env')

# Path do SQLite local (fallback quando Supabase inacessível)
_DEFAULT_SQLITE_PATH = Path(__file__).parent.parent / "data" / "products_positions.db"
# Evita imprimir a mensagem de fallback várias vezes por processo
_FALLBACK_PRINTED = False


class SQLiteRepo:
    """
    Repositorio: PostgreSQL (Supabase) ou SQLite local (fallback).
    Tenta Supabase primeiro; se falhar (DNS, timeout, rede), usa
    productsPositions/data/products_positions.db.
    """
    
    def __init__(self, db_path=None):
        self._use_sqlite = False
        self.db_url = (os.getenv('SUPABASE_DB_URL') or '').strip()
        if db_path:
            self.db_url = "sqlite:///" + str(Path(db_path).resolve())
            self._use_sqlite = True
        elif self.db_url and not self.db_url.lower().startswith("sqlite://"):
            # Tentar resolver o host primeiro (evita psycopg2 quando DNS falha)
            use_supabase = True
            host_match = re.search(r'@([^:/@]+)', self.db_url)
            if host_match:
                hostname = host_match.group(1)
                try:
                    socket.getaddrinfo(hostname, None, socket.AF_INET)
                except (socket.gaierror, socket.error, OSError):
                    use_supabase = False
            if use_supabase:
                try:
                    conn = connect_pg(self.db_url, connect_timeout=3)
                    conn.close()
                except (Exception, OSError):
                    use_supabase = False
            if not use_supabase:
                self.db_url = "sqlite:///" + str(_DEFAULT_SQLITE_PATH.resolve())
                self._use_sqlite = True
                global _FALLBACK_PRINTED
                if not _FALLBACK_PRINTED:
                    _FALLBACK_PRINTED = True
                    print("[SQLiteRepo] Supabase inacessível; usando banco local:", _DEFAULT_SQLITE_PATH)
        if not self.db_url:
            self.db_url = "sqlite:///" + str(_DEFAULT_SQLITE_PATH.resolve())
            self._use_sqlite = True
            print("[SQLiteRepo] SUPABASE_DB_URL não configurado; usando banco local:", _DEFAULT_SQLITE_PATH)
        self._sqlite_conn = None  # conexão única reutilizada em modo SQLite (apenas na thread que a criou)
        self._sqlite_conn_thread_id = None
        if self._use_sqlite:
            print(f"[SQLiteRepo INIT] Modo: SQLite | Path: {self.db_url}")
        else:
            host_info = re.search(r'@([^/]+)', self.db_url)
            print(f"[SQLiteRepo INIT] Modo: PostgreSQL | Host: {host_info.group(1) if host_info else '?'}")
            self._inicializar_banco()
    
    @contextmanager
    def _get_connection(self):
        """Context manager: PostgreSQL abre/fecha a cada uso; SQLite reutiliza uma conexão só na mesma thread."""
        if self._use_sqlite:
            current_id = threading.current_thread().ident
            if self._sqlite_conn is not None and self._sqlite_conn_thread_id == current_id:
                conn = self._sqlite_conn
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                return
            if self._sqlite_conn is None:
                self._sqlite_conn = connect_pg(self.db_url)
                self._sqlite_conn_thread_id = current_id
                conn = self._sqlite_conn
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                return
            conn = connect_pg(self.db_url)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return
        conn = connect_pg(self.db_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def connection(self):
        """Context manager para uso externo (ex: analytics). Em SQLite reutiliza a mesma conexão."""
        return self._get_connection()

    def _get_pg_columns(self, conn, table_name):
        """Retorna lista de colunas da tabela (PostgreSQL ou SQLite)."""
        if getattr(conn, "_is_sqlite", False):
            with conn.cursor() as cur:
                cur.execute("PRAGMA table_info(" + table_name.replace("'", "''") + ")")
                return [row[1] for row in cur.fetchall()]
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))
            return [row[0] for row in cur.fetchall()]

    def _ensure_column(self, conn, table_name, col_name, col_type='TEXT', default=None):
        """Adiciona coluna se nao existir."""
        colunas = self._get_pg_columns(conn, table_name)
        if col_name not in colunas:
            default_clause = f" DEFAULT {default}" if default is not None else ""
            with conn.cursor() as cur:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}{default_clause}")

    def _inicializar_banco(self):
        """Garante que o schema existe e dados essenciais (apenas PostgreSQL)."""
        if self._use_sqlite:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Garantir tipos essenciais
            self._garantir_tipos_essenciais(conn)

            # Migrar usa_quantidade para produtos existentes
            self._migrar_usa_quantidade(conn)

            # Garantir colunas ATR na tabela posicoes
            self._ensure_column(conn, 'posicoes', 'atr_period', 'INTEGER')
            self._ensure_column(conn, 'posicoes', 'atr_multiplier', 'DOUBLE PRECISION')
    
    def _garantir_tipos_essenciais(self, conn=None):
        """Garante que os tipos essenciais (Perpetuos e Spot) sempre existam"""
        if conn is None:
            with self._get_connection() as conn:
                self._garantir_tipos_essenciais(conn)
                return
        
        cursor = conn.cursor()
        tipos_essenciais = [
            ('Perpétuos', 'Contratos perpétuos de criptomoedas'),
            ('Spot', 'Trading spot de criptomoedas')
        ]
        
        for nome, descricao in tipos_essenciais:
            cursor.execute("""
                INSERT INTO tipos (nome, descricao, data_criacao)
                VALUES (%s, %s, %s)
                ON CONFLICT (nome) DO NOTHING
            """, (nome, descricao, datetime.now().strftime("%Y-%m-%d")))

    def _migrar_usa_quantidade(self, conn=None):
        """
        Migra o campo usa_quantidade para produtos existentes.

        Produtos que usam quantidade (usa_quantidade = 1):
        - Soros Spot, Soros Perpetuos, Memebot Perpetuos

        Produtos que NAO usam quantidade (usa_quantidade = 0):
        - HB, EXC, LC, Alphacoins, Crypto Signals (produtos de sinais)
        """
        if conn is None:
            with self._get_connection() as conn:
                self._migrar_usa_quantidade(conn)
                return

        cursor = conn.cursor()

        # Produtos que usam quantidade (padroes de nome)
        produtos_com_quantidade = [
            '%Soros Spot%',
            '%Soros Perp%',
            '%Memebot Perp%'
        ]

        # Atualizar produtos que usam quantidade
        for pattern in produtos_com_quantidade:
            cursor.execute("""
                UPDATE produtos
                SET usa_quantidade = 1
                WHERE nome LIKE %s
                AND (usa_quantidade IS NULL OR usa_quantidade = 0)
            """, (pattern,))

    def _gerar_id(self):
        """Gera um ID único (compatível com formato anterior)"""
        return int(uuid.uuid4().int % (10 ** 10))  # ID numérico de 10 dígitos
    
    def _validar_produto_existe(self, produto_id):
        """Valida se um produto existe"""
        produto = self.carregar_produto(produto_id)
        if produto is None:
            raise ValueError(f"Produto com ID {produto_id} não existe")
        return True
    
    def _validar_posicao_existe(self, posicao_id):
        """Valida se uma posicao existe"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM posicoes WHERE id = %s", (posicao_id,))
            if cursor.fetchone() is None:
                raise ValueError(f"Posicao com ID {posicao_id} nao existe")
        return True
    
    def _validar_tipo_existe(self, nome_tipo):
        """Valida se um tipo existe"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM tipos WHERE nome = %s", (nome_tipo,))
            if cursor.fetchone() is None:
                cursor.execute("SELECT nome FROM tipos")
                tipos = [row[0] for row in cursor.fetchall()]
                raise ValueError(
                    f"Tipo '{nome_tipo}' nao existe. "
                    f"Tipos disponiveis: {', '.join(tipos) if tipos else 'nenhum'}"
                )
        return True
    
    def _validar_ativo_existe(self, nome_ativo):
        """Valida se um ativo existe"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM ativos WHERE nome = %s", (nome_ativo,))
            if cursor.fetchone() is None:
                cursor.execute("SELECT nome FROM ativos")
                ativos = [row[0] for row in cursor.fetchall()]
                raise ValueError(
                    f"Ativo '{nome_ativo}' nao existe. "
                    f"Ativos disponiveis: {', '.join(ativos) if ativos else 'nenhum'}"
                )
        return True
    
    # ========== TIPOS ==========
    
    def registrar_tipo(self, nome, descricao=None):
        """Registra um novo tipo (se nao existir)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tipos (nome, descricao, data_criacao)
                VALUES (%s, %s, %s)
                ON CONFLICT (nome) DO NOTHING
            """, (nome, descricao or '', datetime.now().strftime("%Y-%m-%d")))
        return nome
    
    def listar_tipos(self):
        """Lista todos os tipos disponíveis"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM tipos")
            return [row[0] for row in cursor.fetchall()]
    
    # ========== ATIVOS ==========
    
    def registrar_ativo(self, nome, coingecko_id):
        """Registra um novo ativo (se nao existir)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT coingecko_id FROM ativos WHERE nome = %s", (nome,))
            existing = cursor.fetchone()
            
            if existing:
                if coingecko_id and existing[0] != coingecko_id:
                    cursor.execute("""
                        UPDATE ativos SET coingecko_id = %s WHERE nome = %s
                    """, (coingecko_id, nome))
            else:
                cursor.execute("""
                    INSERT INTO ativos (nome, coingecko_id)
                    VALUES (%s, %s)
                """, (nome, coingecko_id))
        return nome
    
    def obter_ativo(self, nome):
        """Obtem informacoes de um ativo"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome, coingecko_id FROM ativos WHERE nome = %s", (nome,))
            row = cursor.fetchone()
            if row:
                return {
                    'nome': row[0],
                    'coingecko_id': row[1] if row[1] else None
                }
        return None
    
    def listar_ativos(self):
        """Lista todos os ativos disponiveis"""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM ativos", conn)
            return df.to_dict('records') if not df.empty else []
    
    # ========== PRODUTOS ==========
    
    def salvar_produto(self, produto):
        """Salva um produto (com validacao de tipo)"""
        self._validar_tipo_existe(produto.tipo.nome)
        
        produto_id = self._gerar_id()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO produtos (id, nome, data_inicio, tipo, capital_inicial)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                produto_id,
                produto.nome,
                produto.data_inicio,
                produto.tipo.nome,
                produto.capital_inicial
            ))
        
        return produto_id
    
    def carregar_produto(self, produto_id):
        """Carrega um produto por ID (retorna dict)"""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM produtos WHERE id = %s",
                conn,
                params=(produto_id,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
    
    def carregar_produto_objeto(self, produto_id):
        """Carrega um produto por ID e retorna objeto Produto"""
        from domain.produto import Produto
        from domain.tipo import Tipo
        
        produto_dict = self.carregar_produto(produto_id)
        if not produto_dict:
            return None
        
        tipo = Tipo(produto_dict['tipo'])
        capital_inicial = produto_dict.get('capital_inicial', 0.0)
        if pd.isna(capital_inicial):
            capital_inicial = 0.0
        
        produto = Produto(
            nome=produto_dict['nome'],
            data_inicio=produto_dict['data_inicio'],
            tipo=tipo,
            capital_inicial=float(capital_inicial)
        )
        return produto
    
    def listar_produtos(self):
        """Lista todos os produtos"""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM produtos", conn)
            return df.to_dict('records') if not df.empty else []

    def contar_posicoes_abertas_por_produto(self):
        """Conta posicoes abertas de todos os produtos em uma unica query"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT produto_id, COUNT(*) as count
                FROM posicoes
                WHERE status = 'open'
                GROUP BY produto_id
            """)
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    def obter_produto_por_nome(self, nome):
        """Obtem um produto por nome"""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM produtos WHERE nome = %s",
                conn,
                params=(nome,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None

    def atualizar_produto(self, produto_id, **kwargs):
        """
        Atualiza um produto existente

        Args:
            produto_id: ID do produto
            **kwargs: Campos a atualizar (nome, data_inicio, tipo, capital_inicial)

        Returns:
            int: produto_id
        """
        self._validar_produto_existe(produto_id)

        campos_permitidos = ['nome', 'data_inicio', 'tipo', 'capital_inicial', 'usa_quantidade']

        updates = []
        valores = []
        for campo, valor in kwargs.items():
            if campo in campos_permitidos:
                if campo == 'tipo':
                    self._validar_tipo_existe(valor)
                updates.append(f"{campo} = %s")
                valores.append(valor)
            else:
                raise ValueError(f"Campo '{campo}' nao e permitido para atualizacao")

        if not updates:
            return produto_id

        valores.append(produto_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE produtos SET {', '.join(updates)} WHERE id = %s
            """, valores)

        return produto_id

    def deletar_produto(self, produto_id, forcar=False):
        """
        Deleta um produto e todos os dados associados

        Args:
            produto_id: ID do produto a ser deletado
            forcar: Se True, deleta mesmo se houver posições/alocações

        Returns:
            dict: Resumo do que foi deletado

        Raises:
            ValueError: Se o produto não existe ou se há dados associados e forcar=False
        """
        # Validar que o produto existe
        produto = self.carregar_produto(produto_id)
        if not produto:
            raise ValueError(f"Produto com ID {produto_id} não existe")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = %s", (produto_id,))
            count_posicoes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM alocacoes WHERE produto_id = %s", (produto_id,))
            count_alocacoes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM carteiras WHERE produto_id = %s", (produto_id,))
            count_carteiras = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM stops s
                JOIN posicoes p ON s.posicao_id = p.id
                WHERE p.produto_id = %s
            """, (produto_id,))
            count_stops = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM posicao_atributos_produto
                WHERE produto_id = %s
            """, (produto_id,))
            count_atributos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM visualizacoes_config WHERE produto_id = %s", (produto_id,))
            count_visualizacoes = cursor.fetchone()[0]

            resumo = {
                'posicoes': count_posicoes,
                'alocacoes': count_alocacoes,
                'carteiras': count_carteiras,
                'stops': count_stops,
                'atributos': count_atributos,
                'visualizacoes': count_visualizacoes
            }

            if not forcar and (count_posicoes > 0 or count_alocacoes > 0):
                raise ValueError(
                    f"Produto {produto_id} possui dados associados: "
                    f"{count_posicoes} posicao(oes), {count_alocacoes} alocacao(oes). "
                    f"Use forcar=True para deletar mesmo assim."
                )

            cursor.execute("""
                DELETE FROM stops WHERE posicao_id IN (
                    SELECT id FROM posicoes WHERE produto_id = %s
                )
            """, (produto_id,))

            cursor.execute("DELETE FROM posicao_atributos_produto WHERE produto_id = %s", (produto_id,))
            cursor.execute("DELETE FROM alocacoes WHERE produto_id = %s", (produto_id,))
            cursor.execute("DELETE FROM posicoes WHERE produto_id = %s", (produto_id,))
            cursor.execute("DELETE FROM carteiras WHERE produto_id = %s", (produto_id,))
            cursor.execute("DELETE FROM visualizacoes_config WHERE produto_id = %s", (produto_id,))
            cursor.execute("DELETE FROM produtos WHERE id = %s", (produto_id,))

        # 7. Limpar colunas órfãs (fora da transação principal)
        colunas_removidas = self.limpar_colunas_orfas()
        if colunas_removidas:
            resumo['colunas_removidas'] = colunas_removidas

        return resumo

    # ========== ATRIBUTOS CONFIG ==========

    def carregar_atributos_config(self, produto_id):
        """
        Carrega configuracao de atributos de um produto.

        Args:
            produto_id: ID do produto

        Returns:
            list: Lista de dicionarios com configuracao de cada atributo
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT atributo_nome, atributo_tipo, atributo_label, obrigatorio
                FROM produto_atributos_config
                WHERE produto_id = %s
                ORDER BY atributo_nome
            """, conn, params=(produto_id,))
            return df.to_dict('records') if not df.empty else []

    def adicionar_atributo_config(self, produto_id, atributo_nome, atributo_tipo='text',
                                   atributo_label=None, obrigatorio=False):
        """
        Adiciona configuração de atributo para um produto.
        Se o atributo não existir na tabela posicao_atributos_produto, cria a coluna.

        Args:
            produto_id: ID do produto
            atributo_nome: Nome do atributo (será nome da coluna)
            atributo_tipo: Tipo do atributo ('text', 'float', 'int', 'date')
            atributo_label: Label para exibição (se None, usa atributo_nome)
            obrigatorio: Se o atributo é obrigatório

        Returns:
            bool: True se adicionado com sucesso
        """
        self._validar_produto_existe(produto_id)

        # Normalizar nome do atributo (lowercase, sem espacos)
        atributo_nome = atributo_nome.lower().strip().replace(' ', '_')

        # Mapear tipo para PostgreSQL
        tipo_pg = {
            'text': 'TEXT',
            'float': 'DOUBLE PRECISION',
            'int': 'INTEGER',
            'date': 'TEXT'
        }.get(atributo_tipo.lower(), 'TEXT')

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Verificar se coluna existe
            colunas_existentes = self._get_pg_columns(conn, 'posicao_atributos_produto')

            if atributo_nome not in colunas_existentes:
                cursor.execute(f"""
                    ALTER TABLE posicao_atributos_produto
                    ADD COLUMN {atributo_nome} {tipo_pg}
                """)

            # Adicionar config (upsert)
            cursor.execute("""
                INSERT INTO produto_atributos_config
                (produto_id, atributo_nome, atributo_tipo, atributo_label, obrigatorio)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (produto_id, atributo_nome) DO UPDATE SET
                    atributo_tipo = EXCLUDED.atributo_tipo,
                    atributo_label = EXCLUDED.atributo_label,
                    obrigatorio = EXCLUDED.obrigatorio
            """, (
                produto_id,
                atributo_nome,
                atributo_tipo.lower(),
                atributo_label or atributo_nome.replace('_', ' ').title(),
                1 if obrigatorio else 0
            ))

        return True

    def remover_atributo_config(self, produto_id, atributo_nome):
        """
        Remove configuracao de atributo de um produto.
        Nao remove a coluna da tabela (pode ser usada por outros produtos).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM produto_atributos_config
                WHERE produto_id = %s AND atributo_nome = %s
            """, (produto_id, atributo_nome.lower().strip()))
            return cursor.rowcount > 0

    def listar_colunas_atributos(self):
        """
        Lista todas as colunas disponiveis na tabela posicao_atributos_produto.

        Returns:
            list: Lista de nomes de colunas (exceto posicao_id e produto_id)
        """
        with self._get_connection() as conn:
            colunas = self._get_pg_columns(conn, 'posicao_atributos_produto')
            excluir = ['posicao_id', 'produto_id']
            return [c for c in colunas if c not in excluir]

    def editar_atributo_config(self, produto_id, atributo_nome, novo_label=None,
                               novo_tipo=None, novo_obrigatorio=None):
        """
        Edita configuração de um atributo para um produto.

        Args:
            produto_id: ID do produto
            atributo_nome: Nome do atributo a editar
            novo_label: Novo label (opcional)
            novo_tipo: Novo tipo (opcional) - não altera a coluna, apenas a config
            novo_obrigatorio: Novo valor de obrigatório (opcional)

        Returns:
            bool: True se editado com sucesso
        """
        updates = []
        valores = []

        if novo_label is not None:
            updates.append("atributo_label = %s")
            valores.append(novo_label)

        if novo_tipo is not None:
            updates.append("atributo_tipo = %s")
            valores.append(novo_tipo)

        if novo_obrigatorio is not None:
            updates.append("obrigatorio = %s")
            valores.append(1 if novo_obrigatorio else 0)

        if not updates:
            return False

        valores.extend([produto_id, atributo_nome.lower().strip()])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE produto_atributos_config
                SET {', '.join(updates)}
                WHERE produto_id = %s AND atributo_nome = %s
            """, valores)
            return cursor.rowcount > 0

    def deletar_coluna_atributo(self, nome_coluna):
        """
        Deleta uma coluna de atributo da tabela posicao_atributos_produto.
        So permite deletar se nenhum produto usa esse atributo.
        """
        nome_coluna = nome_coluna.lower().strip()

        colunas = self.listar_colunas_atributos()
        if nome_coluna not in colunas:
            raise ValueError(f"Coluna '{nome_coluna}' nao existe")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.id, p.nome
                FROM produto_atributos_config pac
                JOIN produtos p ON pac.produto_id = p.id
                WHERE pac.atributo_nome = %s
            """, (nome_coluna,))
            produtos_usando = cursor.fetchall()

            if produtos_usando:
                nomes = [f"{p[1]} (ID: {p[0]})" for p in produtos_usando]
                raise ValueError(
                    f"Coluna '{nome_coluna}' ainda e usada por: {', '.join(nomes)}"
                )

            cursor.execute(f"ALTER TABLE posicao_atributos_produto DROP COLUMN {nome_coluna}")
            return True

    def listar_colunas_orfas(self):
        """
        Lista colunas que existem na tabela mas não são usadas por nenhum produto.

        Returns:
            list: Lista de nomes de colunas órfãs
        """
        colunas = self.listar_colunas_atributos()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT atributo_nome FROM produto_atributos_config")
            colunas_em_uso = {row[0] for row in cursor.fetchall()}

        return [c for c in colunas if c not in colunas_em_uso]

    def limpar_colunas_orfas(self):
        """
        Remove todas as colunas órfãs (não usadas por nenhum produto).

        Returns:
            list: Lista de colunas removidas
        """
        orfas = self.listar_colunas_orfas()
        removidas = []

        for coluna in orfas:
            try:
                self.deletar_coluna_atributo(coluna)
                removidas.append(coluna)
            except Exception:
                pass  # Ignora erros silenciosamente

        return removidas

    # ========== VISUALIZAÇÕES ==========

    def obter_colunas_disponiveis(self, produto_id):
        """
        Retorna todas as colunas disponíveis para visualização de um produto.
        Inclui colunas da tabela posicoes + atributos configurados + campos calculados.

        Args:
            produto_id: ID do produto

        Returns:
            list: Lista de dicts com nome, label e tipo de cada coluna
        """
        # Colunas internas que não devem aparecer nas visualizações
        colunas_internas = ['atr_period', 'atr_multiplier', 'atr_data_inicio']

        colunas = []

        # Colunas básicas da posição
        colunas_posicao = [
            {'nome': 'id', 'label': 'ID', 'tipo': 'int', 'origem': 'posicao'},
            {'nome': 'ativo', 'label': 'Ativo', 'tipo': 'text', 'origem': 'posicao'},
            {'nome': 'side', 'label': 'Side', 'tipo': 'text', 'origem': 'posicao'},
            {'nome': 'data_entrada', 'label': 'Data Entrada', 'tipo': 'date', 'origem': 'posicao'},
            {'nome': 'preco_entrada', 'label': 'Preço Entrada', 'tipo': 'float', 'origem': 'posicao'},
            {'nome': 'data_saida', 'label': 'Data Saída', 'tipo': 'date', 'origem': 'posicao'},
            {'nome': 'preco_saida', 'label': 'Preço Saída', 'tipo': 'float', 'origem': 'posicao'},
            {'nome': 'status', 'label': 'Status', 'tipo': 'text', 'origem': 'posicao'},
            {'nome': 'coingecko_id', 'label': 'CoinGecko ID', 'tipo': 'text', 'origem': 'posicao'},
        ]
        colunas.extend(colunas_posicao)

        # Atributos configurados para este produto (excluindo colunas internas)
        configs = self.carregar_atributos_config(produto_id)
        for config in configs:
            if config['atributo_nome'] not in colunas_internas:
                colunas.append({
                    'nome': config['atributo_nome'],
                    'label': config['atributo_label'] or config['atributo_nome'].replace('_', ' ').title(),
                    'tipo': config['atributo_tipo'],
                    'origem': 'atributo'
                })

        # Campos calculados
        colunas_calculadas = [
            {'nome': 'pnl', 'label': 'PnL (%)', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'pnl_valor', 'label': 'PnL ($)', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'preco_atual', 'label': 'Preço Atual', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'preco_atual_total', 'label': 'Preço Atual Total', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'preco_saida_total', 'label': 'Preço Saída Total', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'preco_entrada_total', 'label': 'Preço Entrada Total', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'stop_atual', 'label': 'Stop Atual', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'risco_stop', 'label': 'Risco Stop (%)', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'rr', 'label': 'RR', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'alocacao', 'label': 'Alocação (%)', 'tipo': 'float', 'origem': 'calculado'},
            {'nome': 'dias_carteira', 'label': 'Dias em Carteira', 'tipo': 'int', 'origem': 'calculado'},
        ]
        colunas.extend(colunas_calculadas)

        return colunas

    def criar_visualizacao(self, produto_id, nome, colunas, ordenacao=None, filtros=None, colunas_labels=None):
        """
        Cria uma nova visualização customizada para um produto.

        Args:
            produto_id: ID do produto
            nome: Nome da visualização
            colunas: Lista de nomes de colunas na ordem desejada
            ordenacao: Dict com {'coluna': 'nome', 'direcao': 'asc'|'desc'} (opcional)
            filtros: Lista de dicts com {'coluna': 'nome', 'operador': '>', 'valor': 0} (opcional)
            colunas_labels: Dict com mapeamento nome_coluna -> label_customizado (opcional)

        Returns:
            int: ID da visualização criada
        """
        import json

        self._validar_produto_existe(produto_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO visualizacoes_config (produto_id, nome, colunas, colunas_labels, ordenacao, filtros, data_criacao)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                produto_id,
                nome,
                json.dumps(colunas),
                json.dumps(colunas_labels) if colunas_labels else None,
                json.dumps(ordenacao) if ordenacao else None,
                json.dumps(filtros) if filtros else None,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            return cursor.fetchone()[0]

    def listar_visualizacoes(self, produto_id):
        """
        Lista todas as visualizacoes de um produto.
        """
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, nome, colunas, colunas_labels, ordenacao, filtros, data_criacao
                FROM visualizacoes_config
                WHERE produto_id = %s
                ORDER BY nome
            """, (produto_id,))

            visualizacoes = []
            for row in cursor.fetchall():
                visualizacoes.append({
                    'id': row[0],
                    'nome': row[1],
                    'colunas': json.loads(row[2]),
                    'colunas_labels': json.loads(row[3]) if row[3] else None,
                    'ordenacao': json.loads(row[4]) if row[4] else None,
                    'filtros': json.loads(row[5]) if row[5] else None,
                    'data_criacao': row[6]
                })
            return visualizacoes

    def carregar_visualizacao(self, visualizacao_id):
        """
        Carrega uma visualização específica.

        Args:
            visualizacao_id: ID da visualização

        Returns:
            dict: Dados da visualização ou None
        """
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, produto_id, nome, colunas, colunas_labels, ordenacao, filtros, data_criacao
                FROM visualizacoes_config
                WHERE id = %s
            """, (visualizacao_id,))

            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'produto_id': row[1],
                    'nome': row[2],
                    'colunas': json.loads(row[3]),
                    'colunas_labels': json.loads(row[4]) if row[4] else None,
                    'ordenacao': json.loads(row[5]) if row[5] else None,
                    'filtros': json.loads(row[6]) if row[6] else None,
                    'data_criacao': row[7]
                }
            return None

    def atualizar_visualizacao(self, visualizacao_id, nome=None, colunas=None, colunas_labels=None, ordenacao=None, filtros=None):
        """
        Atualiza uma visualização existente.

        Args:
            visualizacao_id: ID da visualização
            nome: Novo nome (opcional)
            colunas: Nova lista de colunas (opcional)
            colunas_labels: Novo mapeamento de labels (opcional)
            ordenacao: Nova ordenação (opcional)
            filtros: Novos filtros (opcional)

        Returns:
            bool: True se atualizado com sucesso
        """
        import json

        updates = []
        valores = []

        if nome is not None:
            updates.append("nome = %s")
            valores.append(nome)

        if colunas is not None:
            updates.append("colunas = %s")
            valores.append(json.dumps(colunas))

        if colunas_labels is not None:
            updates.append("colunas_labels = %s")
            valores.append(json.dumps(colunas_labels) if colunas_labels else None)

        if ordenacao is not None:
            updates.append("ordenacao = %s")
            valores.append(json.dumps(ordenacao) if ordenacao else None)

        if filtros is not None:
            updates.append("filtros = %s")
            valores.append(json.dumps(filtros) if filtros else None)

        if not updates:
            return False

        valores.append(visualizacao_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE visualizacoes_config
                SET {', '.join(updates)}
                WHERE id = %s
            """, valores)
            return cursor.rowcount > 0

    def deletar_visualizacao(self, visualizacao_id):
        """Deleta uma visualizacao."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM visualizacoes_config WHERE id = %s", (visualizacao_id,))
            return cursor.rowcount > 0

    # ========== POSIÇÕES ==========
    
    def salvar_posicao(self, produto_id, posicao):
        """Salva uma posicao (com validacao de integridade referencial)"""
        self._validar_produto_existe(produto_id)
        
        ativo_existente = self.obter_ativo(posicao.ativo)
        if posicao.coingecko_id:
            if ativo_existente and ativo_existente.get('coingecko_id'):
                if ativo_existente['coingecko_id'] != posicao.coingecko_id:
                    self.registrar_ativo(posicao.ativo, posicao.coingecko_id)
            else:
                self.registrar_ativo(posicao.ativo, posicao.coingecko_id)
        else:
            if not ativo_existente:
                self.registrar_ativo(posicao.ativo, None)
        
        posicao_id = self._gerar_id()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO posicoes (
                    id, produto_id, ativo, coingecko_id, exchange_symbol, side, data_entrada,
                    preco_entrada, data_saida, preco_saida, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                posicao_id,
                produto_id,
                posicao.ativo,
                posicao.coingecko_id if posicao.coingecko_id else None,
                getattr(posicao, 'exchange_symbol', None),
                posicao.side,
                posicao.data_entrada,
                posicao.preco_entrada,
                posicao.data_saida,
                posicao.preco_saida,
                posicao.status
            ))
            
            if posicao.stops:
                for stop in posicao.stops:
                    cursor.execute("""
                        INSERT INTO stops (posicao_id, data, valor)
                        VALUES (%s, %s, %s)
                    """, (posicao_id, stop['data'], stop['valor']))
        
        return posicao_id
    
    def carregar_posicoes_abertas(self, produto_id=None):
        """Carrega posicoes abertas com atributos do produto e tipo do produto"""
        with self._get_connection() as conn:
            if produto_id:
                df = pd.read_sql_query("""
                    SELECT p.*,
                           a.motivo, a.perfil, a.alvo1, a.alvo2,
                           pr.tipo as produto_tipo
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    LEFT JOIN produtos pr ON p.produto_id = pr.id
                    WHERE p.status = 'open' AND p.produto_id = %s
                """, conn, params=(produto_id,))
            else:
                df = pd.read_sql_query("""
                    SELECT p.*,
                           a.motivo, a.perfil, a.alvo1, a.alvo2,
                           pr.tipo as produto_tipo
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    LEFT JOIN produtos pr ON p.produto_id = pr.id
                    WHERE p.status = 'open'
                """, conn)
            return df
    
    def carregar_posicoes_fechadas(self, produto_id=None):
        """Carrega posicoes fechadas com atributos do produto"""
        with self._get_connection() as conn:
            if produto_id:
                df = pd.read_sql_query("""
                    SELECT p.*, 
                           a.motivo, a.perfil, a.alvo1, a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'closed' AND p.produto_id = %s
                """, conn, params=(produto_id,))
            else:
                df = pd.read_sql_query("""
                    SELECT p.*, 
                           a.motivo, a.perfil, a.alvo1, a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'closed'
                """, conn)
            return df
    
    def carregar_posicao(self, posicao_id):
        """Carrega uma posicao por ID com atributos do produto"""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT 
                    p.*,
                    a.motivo,
                    a.perfil,
                    a.alvo1,
                    a.alvo2,
                    a.quantidade,
                    a.preco_entrada_total
                FROM posicoes p
                LEFT JOIN posicao_atributos_produto a 
                    ON p.id = a.posicao_id
                WHERE p.id = %s
                """,
                conn,
                params=(posicao_id,)
            )
            
            if df.empty:
                return None
            
            posicao_dict = df.iloc[0].to_dict()
            
            df_stops = pd.read_sql_query(
                "SELECT data, valor FROM stops WHERE posicao_id = %s ORDER BY data",
                conn,
                params=(posicao_id,)
            )
            
            if not df_stops.empty:
                posicao_dict['stops'] = df_stops.to_dict('records')
            else:
                posicao_dict['stops'] = []
            
            return posicao_dict
    
    def atualizar_posicao(self, posicao_id, **kwargs):
        """Atualiza uma posicao existente"""
        self._validar_posicao_existe(posicao_id)

        campos_permitidos = ['ativo', 'coingecko_id', 'exchange_symbol', 'side', 'data_entrada',
                            'preco_entrada', 'data_saida', 'preco_saida', 'status',
                            'atr_period', 'atr_multiplier', 'atr_data_inicio']

        updates = []
        valores = []
        for campo, valor in kwargs.items():
            if campo in campos_permitidos:
                updates.append(f"{campo} = %s")
                valores.append(valor)
            else:
                raise ValueError(f"Campo '{campo}' nao e permitido para atualizacao")

        if not updates:
            return posicao_id

        valores.append(posicao_id)

        if 'ativo' in kwargs:
            novo_ativo = kwargs['ativo']
            coingecko_id = kwargs.get('coingecko_id')
            self.registrar_ativo(novo_ativo, coingecko_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE posicoes SET {', '.join(updates)} WHERE id = %s
            """, valores)

            if 'coingecko_id' in kwargs and 'ativo' not in kwargs:
                cursor.execute("SELECT ativo FROM posicoes WHERE id = %s", (posicao_id,))
                row = cursor.fetchone()
                if row:
                    ativo = row[0]
                    self.registrar_ativo(ativo, kwargs['coingecko_id'])

        return posicao_id

    def obter_ultimo_stop(self, posicao_id):
        """Obtem o ultimo stop salvo para uma posicao."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT valor FROM stops
                WHERE posicao_id = %s
                ORDER BY data DESC
                LIMIT 1
            """, (posicao_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def adicionar_stop_posicao(self, posicao_id, data, valor):
        """Adiciona um novo stop a uma posicao existente."""
        self._validar_posicao_existe(posicao_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO stops (posicao_id, data, valor)
                VALUES (%s, %s, %s)
            """, (posicao_id, data, float(valor)))
            db_tipo = "SQLite" if self._use_sqlite else "PostgreSQL"
            print(f"[STOP SAVE] posicao_id={posicao_id}, data={data}, valor={valor} | DB: {db_tipo}")
        
        return posicao_id
    
    def deletar_posicao(self, posicao_id, forcar=False):
        """
        Deleta uma posição do banco de dados
        
        Args:
            posicao_id: ID da posição a ser deletada
            forcar: Se True, deleta mesmo se houver alocações ativas
        
        Returns:
            bool: True se deletado com sucesso, False caso contrário
        
        Raises:
            ValueError: Se a posição não existe ou se há alocações ativas e forcar=False
        """
        # Validar que a posição existe
        posicao = self.carregar_posicao(posicao_id)
        if not posicao:
            raise ValueError(f"Posição com ID {posicao_id} não existe")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM alocacoes 
                WHERE posicao_id = %s AND status = 'active'
            """, (posicao_id,))
            count_alocacoes = cursor.fetchone()[0]
            
            if count_alocacoes > 0 and not forcar:
                raise ValueError(
                    f"Posicao {posicao_id} possui {count_alocacoes} alocacao(oes) ativa(s). "
                    f"Use forcar=True para deletar mesmo assim."
                )

            cursor.execute("DELETE FROM stops WHERE posicao_id = %s", (posicao_id,))
            cursor.execute("DELETE FROM alocacoes WHERE posicao_id = %s", (posicao_id,))
            cursor.execute("DELETE FROM posicoes WHERE id = %s", (posicao_id,))
            
            cursor.execute("SELECT id FROM posicoes WHERE id = %s", (posicao_id,))
            if cursor.fetchone() is None:
                return True
        
        return False
    
    # ========== ATRIBUTOS POR PRODUTO ==========
    
    def salvar_atributos_posicao(self, posicao_id, produto_id, **kwargs):
        """
        Salva ou atualiza atributos específicos de uma posição por produto.
        Aceita atributos dinâmicos baseados nas colunas da tabela.

        Args:
            posicao_id: ID da posição
            produto_id: ID do produto
            **kwargs: Atributos a salvar (ex: quantidade=10, perfil='conservador')

        Returns:
            int: posicao_id

        Nota:
            - RR é calculado dinamicamente, não é armazenado
            - preco_saida_total é calculado dinamicamente como quantidade * preco_saida (não é armazenado)
        """
        self._validar_posicao_existe(posicao_id)
        self._validar_produto_existe(produto_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM posicoes WHERE id = %s AND produto_id = %s",
                         (posicao_id, produto_id))
            if cursor.fetchone() is None:
                raise ValueError(f"Posicao {posicao_id} nao pertence ao produto {produto_id}")

        if not kwargs:
            return posicao_id

        colunas_existentes = self.listar_colunas_atributos()
        atributos_validos = {k: v for k, v in kwargs.items() if k in colunas_existentes}

        if not atributos_validos:
            return posicao_id

        with self._get_connection() as conn:
            cursor = conn.cursor()

            colunas = ['posicao_id', 'produto_id'] + list(atributos_validos.keys())
            placeholders = ['%s'] * len(colunas)
            valores = [posicao_id, produto_id] + list(atributos_validos.values())

            updates = []
            for col in atributos_validos.keys():
                updates.append(f"{col} = COALESCE(EXCLUDED.{col}, posicao_atributos_produto.{col})")

            query = f"""
                INSERT INTO posicao_atributos_produto ({', '.join(colunas)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT(posicao_id) DO UPDATE SET
                    produto_id = EXCLUDED.produto_id,
                    {', '.join(updates)}
            """

            cursor.execute(query, valores)

        return posicao_id
    
    def carregar_atributos_posicao(self, posicao_id):
        """Carrega atributos especificos de uma posicao."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM posicao_atributos_produto WHERE posicao_id = %s",
                conn,
                params=(posicao_id,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
    
    def atualizar_atributos_posicao(self, posicao_id, **kwargs):
        """
        Atualiza atributos específicos de uma posição.
        Aceita atributos dinâmicos baseados nas colunas da tabela.

        Args:
            posicao_id: ID da posição
            **kwargs: Campos a atualizar (qualquer coluna existente na tabela)

        Nota:
            - RR é calculado dinamicamente, não pode ser atualizado
            - preco_saida_total é calculado dinamicamente, não pode ser atualizado

        Returns:
            int: posicao_id
        """
        # Obter colunas permitidas dinamicamente da tabela
        colunas_existentes = self.listar_colunas_atributos()

        updates = []
        valores = []
        for campo, valor in kwargs.items():
            if campo in colunas_existentes:
                updates.append(f"{campo} = %s")
                valores.append(valor)

        if not updates:
            return posicao_id

        valores.append(posicao_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE posicao_atributos_produto
                SET {', '.join(updates)}
                WHERE posicao_id = %s
            """, valores)

        return posicao_id
    
    # ========== ALOCAÇÕES ==========
    
    def salvar_alocacao(self, produto_id, alocacao):
        """Salva uma alocacao (com validacao de integridade referencial)"""
        self._validar_produto_existe(produto_id)
        self._validar_posicao_existe(alocacao.posicao_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM posicoes 
                WHERE id = %s AND produto_id = %s
            """, (alocacao.posicao_id, produto_id))
            if cursor.fetchone() is None:
                raise ValueError(
                    f"Posicao {alocacao.posicao_id} nao pertence ao produto {produto_id}"
                )
        
        alocacao_id = self._gerar_id()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alocacoes (
                    id, produto_id, posicao_id, percentual, valor_usd, data, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                alocacao_id,
                produto_id,
                alocacao.posicao_id,
                alocacao.percentual,
                alocacao.valor_usd,
                alocacao.data,
                alocacao.status
            ))
        
        return alocacao_id
    
    def salvar_alocacoes(self, produto_id, alocacoes):
        """Salva multiplas alocacoes (com validacao de integridade referencial)"""
        self._validar_produto_existe(produto_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for alocacao in alocacoes:
                self._validar_posicao_existe(alocacao.posicao_id)
                
                cursor.execute("""
                    SELECT id FROM posicoes 
                    WHERE id = %s AND produto_id = %s
                """, (alocacao.posicao_id, produto_id))
                if cursor.fetchone() is None:
                    raise ValueError(
                        f"Posicao {alocacao.posicao_id} nao pertence ao produto {produto_id}"
                    )
                
                alocacao_id = self._gerar_id()
                cursor.execute("""
                    INSERT INTO alocacoes (
                        id, produto_id, posicao_id, percentual, valor_usd, data, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    alocacao_id,
                    produto_id,
                    alocacao.posicao_id,
                    alocacao.percentual,
                    alocacao.valor_usd,
                    alocacao.data,
                    alocacao.status
                ))
        
        return [self._gerar_id() for _ in alocacoes]
    
    def carregar_alocacoes_ativas(self, produto_id):
        """Carrega alocacoes ativas de um produto"""
        with self._get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT * FROM alocacoes 
                WHERE produto_id = %s AND status = 'active'
            """, conn, params=(produto_id,))
            return df
    
    def carregar_alocacao(self, alocacao_id):
        """Carrega uma alocacao por ID"""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM alocacoes WHERE id = %s",
                conn,
                params=(alocacao_id,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
    
    def atualizar_alocacao(self, alocacao_id, **kwargs):
        """Atualiza uma alocacao existente"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM alocacoes WHERE id = %s", (alocacao_id,))
            if cursor.fetchone() is None:
                raise ValueError(f"Alocacao com ID {alocacao_id} nao existe")
            
            campos_permitidos = ['percentual', 'valor_usd', 'data', 'status']
            
            updates = []
            valores = []
            for campo, valor in kwargs.items():
                if campo in campos_permitidos:
                    updates.append(f"{campo} = %s")
                    valores.append(valor)
                else:
                    raise ValueError(f"Campo '{campo}' nao e permitido para atualizacao")
            
            if updates:
                valores.append(alocacao_id)
                cursor.execute(f"""
                    UPDATE alocacoes SET {', '.join(updates)} WHERE id = %s
                """, valores)
        
        return alocacao_id
    
    # ========== CARTEIRAS ==========
    
    def salvar_carteira(self, carteira):
        """Salva ou atualiza uma carteira (UPSERT com validacao de integridade)"""
        self._validar_produto_existe(carteira.produto_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO carteiras (
                    produto_id, valor_disponivel, valor_investido,
                    pnl_nao_realizado, valor_total, data_atualizacao
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(produto_id) DO UPDATE SET
                    valor_disponivel = EXCLUDED.valor_disponivel,
                    valor_investido = EXCLUDED.valor_investido,
                    pnl_nao_realizado = EXCLUDED.pnl_nao_realizado,
                    valor_total = EXCLUDED.valor_total,
                    data_atualizacao = EXCLUDED.data_atualizacao
            """, (
                carteira.produto_id,
                carteira.valor_disponivel,
                carteira.valor_investido,
                carteira.pnl_nao_realizado,
                carteira.valor_total,
                carteira.data_atualizacao
            ))
        
        return carteira.produto_id
    
    def carregar_carteira(self, produto_id):
        """Carrega carteira de um produto"""
        from domain.carteira import Carteira
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM carteiras WHERE produto_id = %s",
                conn,
                params=(produto_id,)
            )
            
            if not df.empty:
                row = df.iloc[0]
                return Carteira(
                    produto_id=row['produto_id'],
                    valor_disponivel=row['valor_disponivel'] or 0.0,
                    valor_investido=row['valor_investido'] or 0.0,
                    pnl_nao_realizado=row['pnl_nao_realizado'] or 0.0,
                    valor_total=row['valor_total'],
                    data_atualizacao=row['data_atualizacao']
                )
            return None
    
    # ========== VALORES DIÁRIOS ==========
    # (Mantido igual - lê do Parquet do CoinGecko)
    
    def verificar_ativo_rastreado(self, ativo):
        """Verifica se um ativo ja esta sendo rastreado"""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM ativos_rastreados WHERE ativo = %s",
                conn,
                params=(ativo,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
    
    def registrar_ativo_rastreado(self, ativo, data_historico_inicial=None, coingecko_id=None):
        """Registra um ativo como rastreado"""
        data_atual = datetime.now().strftime("%Y-%m-%d")
        
        # Obter coingecko_id do ativo se não fornecido
        if coingecko_id is None:
            ativo_info = self.obter_ativo(ativo)
            if ativo_info:
                coingecko_id = ativo_info.get('coingecko_id')
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ativo FROM ativos_rastreados WHERE ativo = %s", (ativo,))
            if cursor.fetchone():
                cursor.execute("""
                    UPDATE ativos_rastreados 
                    SET data_ultima_atualizacao = %s,
                        data_historico_inicial = COALESCE(%s, data_historico_inicial),
                        coingecko_id = COALESCE(%s, coingecko_id)
                    WHERE ativo = %s
                """, (data_atual, data_historico_inicial, coingecko_id, ativo))
            else:
                cursor.execute("""
                    INSERT INTO ativos_rastreados (
                        ativo, coingecko_id, data_primeira_insercao,
                        data_ultima_atualizacao, data_historico_inicial
                    )
                    VALUES (%s, %s, %s, %s, %s)
                """, (ativo, coingecko_id, data_atual, data_atual, data_historico_inicial))
        
        return ativo
    
    def obter_valores_diarios_ativo(self, ativo, data_inicio=None, data_fim=None):
        """
        Obtém valores diários de um ativo diretamente do CoinGecko
        (não duplica dados, lê da fonte original em Parquet)
        """
        # Buscar coingecko_id do ativo
        ativo_info = self.obter_ativo(ativo)
        if not ativo_info or not ativo_info.get('coingecko_id'):
            return []  # Ativo não tem coingecko_id
        
        coingecko_id = ativo_info['coingecko_id']
        
        # Usar ValorDiarioService para ler diretamente do CoinGecko (Parquet)
        from services.valor_diario_service import ValorDiarioService
        
        valores = ValorDiarioService.ler_valores_do_coingecko(
            coingecko_id=coingecko_id,
            data_inicio=data_inicio,
            data_fim=data_fim
        )
        
        # Converter para formato esperado (lista de tuplas)
        return [(v['data'], v['preco']) for v in valores]


_REPO_INSTANCE = None


def get_repo():
    """Singleton do repositório (evita múltiplas instâncias e checagens por request)."""
    global _REPO_INSTANCE
    if _REPO_INSTANCE is None:
        try:
            _REPO_INSTANCE = SQLiteRepo()
        except Exception:
            _REPO_INSTANCE = SQLiteRepo(db_path=str(_DEFAULT_SQLITE_PATH))
    return _REPO_INSTANCE
