import sqlite3
import pandas as pd
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import contextmanager

class SQLiteRepo:
    """
    Repositório usando SQLite para armazenar todos os dados.
    Estrutura:
    - data/products_positions.db (banco SQLite único)
    - (valores diários continuam sendo lidos de data_parquet/crypto_data/coingecko/{coingecko_id}/data.parquet)
    """
    
    def __init__(self, db_path=None):
        if db_path is None:
            # Assumir que estamos em productsPositions/storage/
            script_dir = Path(__file__).parent
            # productsPositions/ é o diretório pai de storage/
            products_positions_dir = script_dir.parent
            db_path = products_positions_dir / "data" / "products_positions.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Inicializar banco de dados
        self._inicializar_banco()
    
    @contextmanager
    def _get_connection(self):
        """Context manager para conexões SQLite com commit automático"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Retorna dict-like rows
        # Habilitar foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            # Tentar fazer commit apenas se necessário
            # DDL statements (CREATE TABLE) não precisam de commit explícito
            try:
                # Verificar se está em transação e se não há statements em progresso
                if conn.in_transaction:
                    # Tentar fazer commit, mas ignorar erro se houver statements em progresso
                    try:
                        conn.commit()
                    except sqlite3.OperationalError as e:
                        if "SQL statements in progress" not in str(e):
                            raise
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                # Ignorar erros de commit (pode ser DDL ou conexão já fechada)
                pass
        except Exception:
            try:
                if conn.in_transaction:
                    conn.rollback()
            except (sqlite3.OperationalError, sqlite3.ProgrammingError):
                pass
            raise
        finally:
            try:
                conn.close()
            except (sqlite3.OperationalError, sqlite3.ProgrammingError):
                pass
    
    def _inicializar_banco(self):
        """Cria todas as tabelas se não existirem"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de tipos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tipos (
                    nome TEXT PRIMARY KEY,
                    descricao TEXT,
                    data_criacao TEXT NOT NULL
                )
            """)
            
            # Tabela de ativos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ativos (
                    nome TEXT PRIMARY KEY,
                    coingecko_id TEXT
                )
            """)
            
            # Tabela de produtos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS produtos (
                    id INTEGER PRIMARY KEY,
                    nome TEXT NOT NULL,
                    data_inicio TEXT NOT NULL,
                    tipo TEXT NOT NULL,
                    capital_inicial REAL DEFAULT 0.0,
                    FOREIGN KEY (tipo) REFERENCES tipos(nome)
                )
            """)
            
            # Tabela de posições
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posicoes (
                    id INTEGER PRIMARY KEY,
                    produto_id INTEGER NOT NULL,
                    ativo TEXT NOT NULL,
                    coingecko_id TEXT,
                    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
                    data_entrada TEXT NOT NULL,
                    preco_entrada REAL NOT NULL,
                    data_saida TEXT,
                    preco_saida REAL,
                    status TEXT NOT NULL CHECK(status IN ('open', 'closed')),
                    FOREIGN KEY (produto_id) REFERENCES produtos(id),
                    FOREIGN KEY (ativo) REFERENCES ativos(nome)
                )
            """)
            
            # Tabela de stops (normalizada)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    posicao_id INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    valor REAL NOT NULL,
                    FOREIGN KEY (posicao_id) REFERENCES posicoes(id) ON DELETE CASCADE
                )
            """)
            
            # Tabela de alocações
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alocacoes (
                    id INTEGER PRIMARY KEY,
                    produto_id INTEGER NOT NULL,
                    posicao_id INTEGER NOT NULL,
                    percentual REAL NOT NULL CHECK(percentual >= 0 AND percentual <= 100),
                    valor_usd REAL,
                    data TEXT,
                    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
                    FOREIGN KEY (produto_id) REFERENCES produtos(id),
                    FOREIGN KEY (posicao_id) REFERENCES posicoes(id) ON DELETE CASCADE
                )
            """)
            
            # Tabela de carteiras
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS carteiras (
                    produto_id INTEGER PRIMARY KEY,
                    valor_disponivel REAL NOT NULL DEFAULT 0.0,
                    valor_investido REAL NOT NULL DEFAULT 0.0,
                    pnl_nao_realizado REAL NOT NULL DEFAULT 0.0,
                    valor_total REAL NOT NULL,
                    data_atualizacao TEXT NOT NULL,
                    FOREIGN KEY (produto_id) REFERENCES produtos(id)
                )
            """)
            
            # Tabela de ativos rastreados
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ativos_rastreados (
                    ativo TEXT PRIMARY KEY,
                    coingecko_id TEXT NOT NULL,
                    data_primeira_insercao TEXT NOT NULL,
                    data_ultima_atualizacao TEXT NOT NULL,
                    data_historico_inicial TEXT,
                    FOREIGN KEY (ativo) REFERENCES ativos(nome)
                )
            """)
            
            # Tabela de atributos específicos por produto
            # Nota: RR é calculado dinamicamente, não é armazenado
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posicao_atributos_produto (
                    posicao_id INTEGER PRIMARY KEY,
                    produto_id INTEGER NOT NULL,
                    motivo TEXT,
                    perfil TEXT,
                    alvo1 REAL,
                    alvo2 REAL,
                    FOREIGN KEY (posicao_id) REFERENCES posicoes(id) ON DELETE CASCADE,
                    FOREIGN KEY (produto_id) REFERENCES produtos(id)
                )
            """)
            
            # Criar índices para performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_posicoes_produto ON posicoes(produto_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_posicoes_status ON posicoes(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stops_posicao ON stops(posicao_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alocacoes_produto ON alocacoes(produto_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alocacoes_posicao ON alocacoes(posicao_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alocacoes_status ON alocacoes(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_atributos_produto ON posicao_atributos_produto(produto_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_atributos_posicao ON posicao_atributos_produto(posicao_id)")
            
            # Habilitar WAL mode para melhor concorrência
            cursor.execute("PRAGMA journal_mode=WAL")
            
            # Garantir tipos essenciais
            self._garantir_tipos_essenciais(conn)
    
    def _garantir_tipos_essenciais(self, conn=None):
        """Garante que os tipos essenciais (Perpétuos e Spot) sempre existam"""
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
                INSERT OR IGNORE INTO tipos (nome, descricao, data_criacao)
                VALUES (?, ?, ?)
            """, (nome, descricao, datetime.now().strftime("%Y-%m-%d")))
    
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
        """Valida se uma posição existe"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM posicoes WHERE id = ?", (posicao_id,))
            if cursor.fetchone() is None:
                raise ValueError(f"Posição com ID {posicao_id} não existe")
        return True
    
    def _validar_tipo_existe(self, nome_tipo):
        """Valida se um tipo existe"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM tipos WHERE nome = ?", (nome_tipo,))
            if cursor.fetchone() is None:
                # Listar tipos disponíveis
                cursor.execute("SELECT nome FROM tipos")
                tipos = [row[0] for row in cursor.fetchall()]
                raise ValueError(
                    f"Tipo '{nome_tipo}' não existe. "
                    f"Tipos disponíveis: {', '.join(tipos) if tipos else 'nenhum'}"
                )
        return True
    
    def _validar_ativo_existe(self, nome_ativo):
        """Valida se um ativo existe"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM ativos WHERE nome = ?", (nome_ativo,))
            if cursor.fetchone() is None:
                # Listar ativos disponíveis
                cursor.execute("SELECT nome FROM ativos")
                ativos = [row[0] for row in cursor.fetchall()]
                raise ValueError(
                    f"Ativo '{nome_ativo}' não existe. "
                    f"Ativos disponíveis: {', '.join(ativos) if ativos else 'nenhum'}"
                )
        return True
    
    # ========== TIPOS ==========
    
    def registrar_tipo(self, nome, descricao=None):
        """Registra um novo tipo (se não existir)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO tipos (nome, descricao, data_criacao)
                VALUES (?, ?, ?)
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
        """Registra um novo ativo (se não existir)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Verificar se já existe
            cursor.execute("SELECT coingecko_id FROM ativos WHERE nome = ?", (nome,))
            existing = cursor.fetchone()
            
            if existing:
                # Atualizar coingecko_id se fornecido e diferente
                if coingecko_id and existing[0] != coingecko_id:
                    cursor.execute("""
                        UPDATE ativos SET coingecko_id = ? WHERE nome = ?
                    """, (coingecko_id, nome))
            else:
                # Inserir novo ativo
                cursor.execute("""
                    INSERT INTO ativos (nome, coingecko_id)
                    VALUES (?, ?)
                """, (nome, coingecko_id))
        return nome
    
    def obter_ativo(self, nome):
        """Obtém informações de um ativo"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome, coingecko_id FROM ativos WHERE nome = ?", (nome,))
            row = cursor.fetchone()
            if row:
                return {
                    'nome': row[0],
                    'coingecko_id': row[1] if row[1] else None
                }
        return None
    
    def listar_ativos(self):
        """Lista todos os ativos disponíveis"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM ativos", conn)
            return df.to_dict('records') if not df.empty else []
        finally:
            conn.close()
    
    # ========== PRODUTOS ==========
    
    def salvar_produto(self, produto):
        """Salva um produto (com validação de tipo)"""
        # Validar que o tipo existe
        self._validar_tipo_existe(produto.tipo.nome)
        
        produto_id = self._gerar_id()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO produtos (id, nome, data_inicio, tipo, capital_inicial)
                VALUES (?, ?, ?, ?, ?)
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
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM produtos WHERE id = ?",
                conn,
                params=(produto_id,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        finally:
            conn.close()
    
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
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM produtos", conn)
            return df.to_dict('records') if not df.empty else []
        finally:
            conn.close()
    
    def obter_produto_por_nome(self, nome):
        """Obtém um produto por nome"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM produtos WHERE nome = ?",
                conn,
                params=(nome,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        finally:
            conn.close()
    
    # ========== POSIÇÕES ==========
    
    def salvar_posicao(self, produto_id, posicao):
        """Salva uma posição (com validação de integridade referencial)"""
        # Validar que o produto existe
        self._validar_produto_existe(produto_id)
        
        # Registrar ativo se não existir (com coingecko_id se fornecido)
        if posicao.coingecko_id:
            ativo_existente = self.obter_ativo(posicao.ativo)
            if ativo_existente and ativo_existente.get('coingecko_id'):
                # Se já existe com coingecko_id diferente, atualizar
                if ativo_existente['coingecko_id'] != posicao.coingecko_id:
                    self.registrar_ativo(posicao.ativo, posicao.coingecko_id)
            else:
                # Registrar novo ativo
                self.registrar_ativo(posicao.ativo, posicao.coingecko_id)
        
        posicao_id = self._gerar_id()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO posicoes (
                    id, produto_id, ativo, coingecko_id, side, data_entrada,
                    preco_entrada, data_saida, preco_saida, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                posicao_id,
                produto_id,
                posicao.ativo,
                posicao.coingecko_id if posicao.coingecko_id else None,
                posicao.side,
                posicao.data_entrada,
                posicao.preco_entrada,
                posicao.data_saida,
                posicao.preco_saida,
                posicao.status
            ))
            
            # Salvar stops na tabela separada
            if posicao.stops:
                for stop in posicao.stops:
                    cursor.execute("""
                        INSERT INTO stops (posicao_id, data, valor)
                        VALUES (?, ?, ?)
                    """, (posicao_id, stop['data'], stop['valor']))
        
        return posicao_id
    
    def carregar_posicoes_abertas(self, produto_id=None):
        """Carrega posições abertas com atributos do produto"""
        conn = sqlite3.connect(self.db_path)
        try:
            if produto_id:
                df = pd.read_sql_query("""
                    SELECT p.*, 
                           a.motivo, a.perfil, a.alvo1, a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'open' AND p.produto_id = ?
                """, conn, params=(produto_id,))
            else:
                df = pd.read_sql_query("""
                    SELECT p.*, 
                           a.motivo, a.perfil, a.alvo1, a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'open'
                """, conn)
            return df
        finally:
            conn.close()
    
    def carregar_posicoes_fechadas(self, produto_id=None):
        """Carrega posições fechadas com atributos do produto"""
        conn = sqlite3.connect(self.db_path)
        try:
            if produto_id:
                df = pd.read_sql_query("""
                    SELECT p.*, 
                           a.motivo, a.perfil, a.alvo1, a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'closed' AND p.produto_id = ?
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
        finally:
            conn.close()
    
    def carregar_posicao(self, posicao_id):
        """Carrega uma posição por ID com atributos do produto"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Carregar com JOIN para incluir atributos
            df = pd.read_sql_query("""
                SELECT p.*, 
                       a.motivo, a.perfil, a.alvo1, a.alvo2
                FROM posicoes p
                LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                WHERE p.id = ?
            """, conn, params=(posicao_id,))
            
            if df.empty:
                return None
            
            posicao_dict = df.iloc[0].to_dict()
            
            # Carregar stops da tabela separada
            df_stops = pd.read_sql_query(
                "SELECT data, valor FROM stops WHERE posicao_id = ? ORDER BY data",
                conn,
                params=(posicao_id,)
            )
            
            if not df_stops.empty:
                posicao_dict['stops'] = df_stops.to_dict('records')
            else:
                posicao_dict['stops'] = []
            
            return posicao_dict
        finally:
            conn.close()
    
    def atualizar_posicao(self, posicao_id, **kwargs):
        """Atualiza uma posição existente"""
        self._validar_posicao_existe(posicao_id)
        
        campos_permitidos = ['ativo', 'coingecko_id', 'side', 'data_entrada', 
                            'preco_entrada', 'data_saida', 'preco_saida', 'status']
        
        updates = []
        valores = []
        for campo, valor in kwargs.items():
            if campo in campos_permitidos:
                updates.append(f"{campo} = ?")
                valores.append(valor)
            else:
                raise ValueError(f"Campo '{campo}' não é permitido para atualização")
        
        if not updates:
            return posicao_id
        
        valores.append(posicao_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE posicoes SET {', '.join(updates)} WHERE id = ?
            """, valores)
            
            # Se atualizando coingecko_id, atualizar também na tabela de ativos
            if 'coingecko_id' in kwargs:
                # Obter ativo da posição
                cursor.execute("SELECT ativo FROM posicoes WHERE id = ?", (posicao_id,))
                row = cursor.fetchone()
                if row:
                    ativo = row[0]
                    self.registrar_ativo(ativo, kwargs['coingecko_id'])
        
        return posicao_id
    
    def adicionar_stop_posicao(self, posicao_id, data, valor):
        """
        Adiciona um novo stop a uma posição existente
        
        Args:
            posicao_id: ID da posição
            data: Data do stop (YYYY-MM-DD)
            valor: Valor do stop
        
        Returns:
            int: ID da posição atualizada
        """
        self._validar_posicao_existe(posicao_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO stops (posicao_id, data, valor)
                VALUES (?, ?, ?)
            """, (posicao_id, data, float(valor)))
        
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
        
        # Verificar se há alocações ativas associadas
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM alocacoes 
                WHERE posicao_id = ? AND status = 'active'
            """, (posicao_id,))
            count_alocacoes = cursor.fetchone()[0]
            
            if count_alocacoes > 0 and not forcar:
                raise ValueError(
                    f"Posição {posicao_id} possui {count_alocacoes} alocação(ões) ativa(s). "
                    f"Use forcar=True para deletar mesmo assim."
                )
            
            # Deletar a posição (stops e alocações serão deletados em cascata devido ao ON DELETE CASCADE)
            cursor.execute("DELETE FROM posicoes WHERE id = ?", (posicao_id,))
            
            # Verificar se foi deletado
            cursor.execute("SELECT id FROM posicoes WHERE id = ?", (posicao_id,))
            if cursor.fetchone() is None:
                return True
        
        return False
    
    # ========== ATRIBUTOS POR PRODUTO ==========
    
    def salvar_atributos_posicao(self, posicao_id, produto_id, motivo=None, perfil=None,
                                 alvo1=None, alvo2=None):
        """
        Salva ou atualiza atributos específicos de uma posição por produto
        
        Args:
            posicao_id: ID da posição
            produto_id: ID do produto
            motivo: Motivo do encerramento (Crypto Signals)
            perfil: Perfil de risco (Crypto Signals)
            alvo1: Primeiro alvo de preço (Crypto Signals)
            alvo2: Segundo alvo de preço (Crypto Signals)
        
        Returns:
            int: posicao_id
        
        Nota: RR é calculado dinamicamente, não é armazenado
        """
        self._validar_posicao_existe(posicao_id)
        self._validar_produto_existe(produto_id)
        
        # Verificar se posição pertence ao produto
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM posicoes WHERE id = ? AND produto_id = ?", 
                         (posicao_id, produto_id))
            if cursor.fetchone() is None:
                raise ValueError(f"Posição {posicao_id} não pertence ao produto {produto_id}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # UPSERT (INSERT OR REPLACE)
            cursor.execute("""
                INSERT INTO posicao_atributos_produto (
                    posicao_id, produto_id, motivo, perfil, alvo1, alvo2
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(posicao_id) DO UPDATE SET
                    produto_id = excluded.produto_id,
                    motivo = excluded.motivo,
                    perfil = excluded.perfil,
                    alvo1 = excluded.alvo1,
                    alvo2 = excluded.alvo2
            """, (
                posicao_id, produto_id, motivo, perfil, alvo1, alvo2
            ))
        
        return posicao_id
    
    def carregar_atributos_posicao(self, posicao_id):
        """
        Carrega atributos específicos de uma posição
        
        Args:
            posicao_id: ID da posição
        
        Returns:
            dict ou None: Dicionário com atributos ou None se não existir
        """
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM posicao_atributos_produto WHERE posicao_id = ?",
                conn,
                params=(posicao_id,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        finally:
            conn.close()
    
    def atualizar_atributos_posicao(self, posicao_id, **kwargs):
        """
        Atualiza atributos específicos de uma posição
        
        Args:
            posicao_id: ID da posição
            **kwargs: Campos a atualizar (motivo, perfil, alvo1, alvo2)
            
        Nota: RR é calculado dinamicamente, não pode ser atualizado
        
        Returns:
            int: posicao_id
        """
        campos_permitidos = ['motivo', 'perfil', 'alvo1', 'alvo2']
        
        updates = []
        valores = []
        for campo, valor in kwargs.items():
            if campo in campos_permitidos:
                updates.append(f"{campo} = ?")
                valores.append(valor)
            else:
                raise ValueError(f"Campo '{campo}' não é permitido para atualização")
        
        if not updates:
            return posicao_id
        
        valores.append(posicao_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE posicao_atributos_produto 
                SET {', '.join(updates)} 
                WHERE posicao_id = ?
            """, valores)
        
        return posicao_id
    
    # ========== ALOCAÇÕES ==========
    
    def salvar_alocacao(self, produto_id, alocacao):
        """Salva uma alocação (com validação de integridade referencial)"""
        # Validar que o produto existe
        self._validar_produto_existe(produto_id)
        
        # Validar que a posição existe
        self._validar_posicao_existe(alocacao.posicao_id)
        
        # Validar que a posição pertence ao produto
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM posicoes 
                WHERE id = ? AND produto_id = ?
            """, (alocacao.posicao_id, produto_id))
            if cursor.fetchone() is None:
                raise ValueError(
                    f"Posição {alocacao.posicao_id} não pertence ao produto {produto_id}"
                )
        
        alocacao_id = self._gerar_id()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alocacoes (
                    id, produto_id, posicao_id, percentual, valor_usd, data, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
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
        """Salva múltiplas alocações (com validação de integridade referencial)"""
        # Validar que o produto existe
        self._validar_produto_existe(produto_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for alocacao in alocacoes:
                # Validar que a posição existe
                self._validar_posicao_existe(alocacao.posicao_id)
                
                # Validar que a posição pertence ao produto
                cursor.execute("""
                    SELECT id FROM posicoes 
                    WHERE id = ? AND produto_id = ?
                """, (alocacao.posicao_id, produto_id))
                if cursor.fetchone() is None:
                    raise ValueError(
                        f"Posição {alocacao.posicao_id} não pertence ao produto {produto_id}"
                    )
                
                alocacao_id = self._gerar_id()
                cursor.execute("""
                    INSERT INTO alocacoes (
                        id, produto_id, posicao_id, percentual, valor_usd, data, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    alocacao_id,
                    produto_id,
                    alocacao.posicao_id,
                    alocacao.percentual,
                    alocacao.valor_usd,
                    alocacao.data,
                    alocacao.status
                ))
        
        return [self._gerar_id() for _ in alocacoes]  # Retornar IDs (simplificado)
    
    def carregar_alocacoes_ativas(self, produto_id):
        """Carrega alocações ativas de um produto"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("""
                SELECT * FROM alocacoes 
                WHERE produto_id = ? AND status = 'active'
            """, conn, params=(produto_id,))
            return df
        finally:
            conn.close()
    
    def carregar_alocacao(self, alocacao_id):
        """Carrega uma alocação por ID"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM alocacoes WHERE id = ?",
                conn,
                params=(alocacao_id,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        finally:
            conn.close()
    
    def atualizar_alocacao(self, alocacao_id, **kwargs):
        """Atualiza uma alocação existente"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM alocacoes WHERE id = ?", (alocacao_id,))
            if cursor.fetchone() is None:
                raise ValueError(f"Alocação com ID {alocacao_id} não existe")
            
            campos_permitidos = ['percentual', 'valor_usd', 'data', 'status']
            
            updates = []
            valores = []
            for campo, valor in kwargs.items():
                if campo in campos_permitidos:
                    updates.append(f"{campo} = ?")
                    valores.append(valor)
                else:
                    raise ValueError(f"Campo '{campo}' não é permitido para atualização")
            
            if updates:
                valores.append(alocacao_id)
                cursor.execute(f"""
                    UPDATE alocacoes SET {', '.join(updates)} WHERE id = ?
                """, valores)
        
        return alocacao_id
    
    # ========== CARTEIRAS ==========
    
    def salvar_carteira(self, carteira):
        """Salva ou atualiza uma carteira (UPSERT com validação de integridade)"""
        # Validar que o produto existe
        self._validar_produto_existe(carteira.produto_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO carteiras (
                    produto_id, valor_disponivel, valor_investido,
                    pnl_nao_realizado, valor_total, data_atualizacao
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(produto_id) DO UPDATE SET
                    valor_disponivel = excluded.valor_disponivel,
                    valor_investido = excluded.valor_investido,
                    pnl_nao_realizado = excluded.pnl_nao_realizado,
                    valor_total = excluded.valor_total,
                    data_atualizacao = excluded.data_atualizacao
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
        
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM carteiras WHERE produto_id = ?",
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
        finally:
            conn.close()
    
    # ========== VALORES DIÁRIOS ==========
    # (Mantido igual - lê do Parquet do CoinGecko)
    
    def verificar_ativo_rastreado(self, ativo):
        """Verifica se um ativo já está sendo rastreado"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM ativos_rastreados WHERE ativo = ?",
                conn,
                params=(ativo,)
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        finally:
            conn.close()
    
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
            # Verificar se já existe
            cursor.execute("SELECT ativo FROM ativos_rastreados WHERE ativo = ?", (ativo,))
            if cursor.fetchone():
                # Atualizar
                cursor.execute("""
                    UPDATE ativos_rastreados 
                    SET data_ultima_atualizacao = ?,
                        data_historico_inicial = COALESCE(?, data_historico_inicial),
                        coingecko_id = COALESCE(?, coingecko_id)
                    WHERE ativo = ?
                """, (data_atual, data_historico_inicial, coingecko_id, ativo))
            else:
                # Inserir
                cursor.execute("""
                    INSERT INTO ativos_rastreados (
                        ativo, coingecko_id, data_primeira_insercao,
                        data_ultima_atualizacao, data_historico_inicial
                    )
                    VALUES (?, ?, ?, ?, ?)
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

