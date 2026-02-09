"""
Script de migracao: SQLite -> Supabase (PostgreSQL)

Cria as tabelas no Supabase e copia todos os dados do SQLite.
Uso:
    python migrar_para_supabase.py
    python migrar_para_supabase.py --apenas-schema   # Só cria tabelas, sem dados
"""
import sys
import sqlite3
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv

# Carregar .env
load_dotenv(Path(__file__).parent.parent.parent / '.env')

DB_URL = os.getenv('SUPABASE_DB_URL')
SQLITE_PATH = Path(__file__).parent.parent / "data" / "products_positions.db"

# Ordem de criação (respeita foreign keys)
TABELAS_ORDEM = [
    'tipos', 'ativos', 'produtos', 'posicoes', 'stops', 'alocacoes',
    'carteiras', 'ativos_rastreados', 'posicao_atributos_produto',
    'produto_atributos_config', 'visualizacoes_config',
    'turmas', 'trades_turma', 'carteira_turma', 'trade_valores_diarios'
]

# Schema PostgreSQL
SCHEMA_SQL = """
-- Tipos
CREATE TABLE IF NOT EXISTS tipos (
    nome TEXT PRIMARY KEY,
    descricao TEXT,
    data_criacao TEXT NOT NULL
);

-- Ativos
CREATE TABLE IF NOT EXISTS ativos (
    nome TEXT PRIMARY KEY,
    coingecko_id TEXT
);

-- Produtos
CREATE TABLE IF NOT EXISTS produtos (
    id BIGINT PRIMARY KEY,
    nome TEXT NOT NULL,
    data_inicio TEXT NOT NULL,
    tipo TEXT NOT NULL REFERENCES tipos(nome),
    capital_inicial DOUBLE PRECISION DEFAULT 0.0,
    usa_quantidade INTEGER DEFAULT 0
);

-- Posições
CREATE TABLE IF NOT EXISTS posicoes (
    id BIGINT PRIMARY KEY,
    produto_id BIGINT NOT NULL REFERENCES produtos(id),
    ativo TEXT NOT NULL REFERENCES ativos(nome),
    coingecko_id TEXT,
    exchange_symbol TEXT,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    data_entrada TEXT NOT NULL,
    preco_entrada DOUBLE PRECISION NOT NULL,
    data_saida TEXT,
    preco_saida DOUBLE PRECISION,
    status TEXT NOT NULL CHECK(status IN ('open', 'closed')),
    atr_data_inicio TEXT
);

-- Stops
CREATE TABLE IF NOT EXISTS stops (
    id SERIAL PRIMARY KEY,
    posicao_id BIGINT NOT NULL REFERENCES posicoes(id) ON DELETE CASCADE,
    data TEXT NOT NULL,
    valor DOUBLE PRECISION NOT NULL
);

-- Alocações
CREATE TABLE IF NOT EXISTS alocacoes (
    id BIGINT PRIMARY KEY,
    produto_id BIGINT NOT NULL REFERENCES produtos(id),
    posicao_id BIGINT NOT NULL REFERENCES posicoes(id) ON DELETE CASCADE,
    percentual DOUBLE PRECISION NOT NULL CHECK(percentual >= 0 AND percentual <= 100),
    valor_usd DOUBLE PRECISION,
    data TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'inactive'))
);

-- Carteiras
CREATE TABLE IF NOT EXISTS carteiras (
    produto_id BIGINT PRIMARY KEY REFERENCES produtos(id),
    valor_disponivel DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    valor_investido DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    pnl_nao_realizado DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    valor_total DOUBLE PRECISION NOT NULL,
    data_atualizacao TEXT NOT NULL
);

-- Ativos rastreados
CREATE TABLE IF NOT EXISTS ativos_rastreados (
    ativo TEXT PRIMARY KEY REFERENCES ativos(nome),
    coingecko_id TEXT NOT NULL,
    data_primeira_insercao TEXT NOT NULL,
    data_ultima_atualizacao TEXT NOT NULL,
    data_historico_inicial TEXT
);

-- Atributos de posição por produto
CREATE TABLE IF NOT EXISTS posicao_atributos_produto (
    posicao_id BIGINT PRIMARY KEY REFERENCES posicoes(id) ON DELETE CASCADE,
    produto_id BIGINT NOT NULL REFERENCES produtos(id),
    motivo TEXT,
    perfil TEXT,
    alvo1 DOUBLE PRECISION,
    alvo2 DOUBLE PRECISION,
    quantidade DOUBLE PRECISION,
    preco_entrada_total DOUBLE PRECISION,
    preco_entrada_exchange DOUBLE PRECISION,
    pnl_exchange DOUBLE PRECISION,
    leverage INTEGER
);

-- Config de atributos por produto
CREATE TABLE IF NOT EXISTS produto_atributos_config (
    produto_id BIGINT NOT NULL REFERENCES produtos(id) ON DELETE CASCADE,
    atributo_nome TEXT NOT NULL,
    atributo_tipo TEXT NOT NULL DEFAULT 'text',
    atributo_label TEXT,
    obrigatorio INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (produto_id, atributo_nome)
);

-- Visualizações customizadas
CREATE TABLE IF NOT EXISTS visualizacoes_config (
    id SERIAL PRIMARY KEY,
    produto_id BIGINT NOT NULL REFERENCES produtos(id) ON DELETE CASCADE,
    nome TEXT NOT NULL,
    colunas TEXT NOT NULL,
    colunas_labels TEXT,
    ordenacao TEXT,
    filtros TEXT,
    data_criacao TEXT NOT NULL,
    UNIQUE(produto_id, nome)
);

-- Turmas
CREATE TABLE IF NOT EXISTS turmas (
    id SERIAL PRIMARY KEY,
    produto_id BIGINT NOT NULL REFERENCES produtos(id) ON DELETE CASCADE,
    nome TEXT NOT NULL,
    data_inicio TEXT NOT NULL,
    capital_base DOUBLE PRECISION DEFAULT 1500.0,
    descricao TEXT,
    data_criacao TEXT
);

-- Trades por turma
CREATE TABLE IF NOT EXISTS trades_turma (
    id SERIAL PRIMARY KEY,
    turma_id INTEGER NOT NULL REFERENCES turmas(id) ON DELETE CASCADE,
    posicao_id BIGINT NOT NULL REFERENCES posicoes(id) ON DELETE CASCADE,
    UNIQUE(turma_id, posicao_id)
);

-- Carteira por turma
CREATE TABLE IF NOT EXISTS carteira_turma (
    id SERIAL PRIMARY KEY,
    turma_id INTEGER NOT NULL REFERENCES turmas(id) ON DELETE CASCADE,
    trade_id INTEGER NOT NULL REFERENCES trades_turma(id) ON DELETE CASCADE,
    origem TEXT NOT NULL CHECK(origem IN ('nativo', 'replicado')),
    data_insercao TEXT NOT NULL,
    data_remocao TEXT,
    preco_entrada_turma DOUBLE PRECISION,
    preco_fonte TEXT DEFAULT 'manual',
    preco_data_referencia TEXT,
    preco_moeda TEXT DEFAULT 'USD',
    ativo_atual INTEGER DEFAULT 1
);

-- Valores diários por trade
CREATE TABLE IF NOT EXISTS trade_valores_diarios (
    trade_id INTEGER NOT NULL REFERENCES trades_turma(id) ON DELETE CASCADE,
    data TEXT NOT NULL,
    preco DOUBLE PRECISION NOT NULL,
    fonte TEXT,
    PRIMARY KEY (trade_id, data)
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_posicoes_produto ON posicoes(produto_id);
CREATE INDEX IF NOT EXISTS idx_posicoes_status ON posicoes(status);
CREATE INDEX IF NOT EXISTS idx_posicoes_produto_status ON posicoes(produto_id, status);
CREATE INDEX IF NOT EXISTS idx_stops_posicao ON stops(posicao_id);
CREATE INDEX IF NOT EXISTS idx_stops_posicao_data ON stops(posicao_id, data DESC);
CREATE INDEX IF NOT EXISTS idx_alocacoes_produto ON alocacoes(produto_id);
CREATE INDEX IF NOT EXISTS idx_alocacoes_posicao ON alocacoes(posicao_id);
CREATE INDEX IF NOT EXISTS idx_alocacoes_status ON alocacoes(status);
CREATE INDEX IF NOT EXISTS idx_alocacoes_data ON alocacoes(data DESC);
CREATE INDEX IF NOT EXISTS idx_atributos_produto ON posicao_atributos_produto(produto_id);
CREATE INDEX IF NOT EXISTS idx_atributos_posicao ON posicao_atributos_produto(posicao_id);
CREATE INDEX IF NOT EXISTS idx_atributos_config_produto ON produto_atributos_config(produto_id);
CREATE INDEX IF NOT EXISTS idx_visualizacoes_produto ON visualizacoes_config(produto_id);
CREATE INDEX IF NOT EXISTS idx_turmas_produto ON turmas(produto_id);
CREATE INDEX IF NOT EXISTS idx_turmas_data_inicio ON turmas(data_inicio);
CREATE INDEX IF NOT EXISTS idx_trades_turma_turma ON trades_turma(turma_id);
CREATE INDEX IF NOT EXISTS idx_trades_turma_posicao ON trades_turma(posicao_id);
CREATE INDEX IF NOT EXISTS idx_carteira_turma_turma ON carteira_turma(turma_id);
CREATE INDEX IF NOT EXISTS idx_carteira_turma_turma_ativo ON carteira_turma(turma_id, ativo_atual);
CREATE INDEX IF NOT EXISTS idx_carteira_turma_trade ON carteira_turma(trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_valores_trade_data ON trade_valores_diarios(trade_id, data DESC);
"""


def criar_schema(pg_conn):
    """Cria todas as tabelas e índices no PostgreSQL."""
    print("Criando schema no Supabase...", flush=True)
    with pg_conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    pg_conn.commit()
    print("  Schema criado com sucesso!", flush=True)


def detectar_colunas_extras_sqlite(sqlite_conn):
    """Detecta colunas extras em posicao_atributos_produto (adicionadas dinamicamente)."""
    cursor = sqlite_conn.cursor()
    cursor.execute("PRAGMA table_info(posicao_atributos_produto)")
    colunas_sqlite = [row[1] for row in cursor.fetchall()]

    colunas_base = [
        'posicao_id', 'produto_id', 'motivo', 'perfil', 'alvo1', 'alvo2',
        'quantidade', 'preco_entrada_total', 'preco_entrada_exchange',
        'pnl_exchange', 'leverage'
    ]
    extras = [c for c in colunas_sqlite if c not in colunas_base]
    return extras


def adicionar_colunas_extras(pg_conn, colunas_extras):
    """Adiciona colunas extras (dinâmicas) na tabela posicao_atributos_produto."""
    if not colunas_extras:
        return
    print(f"  Adicionando {len(colunas_extras)} coluna(s) extra(s): {colunas_extras}", flush=True)
    with pg_conn.cursor() as cur:
        for col in colunas_extras:
            try:
                cur.execute(f"ALTER TABLE posicao_atributos_produto ADD COLUMN {col} TEXT")
            except psycopg2.errors.DuplicateColumn:
                pg_conn.rollback()
                continue
    pg_conn.commit()


def corrigir_referencias_orfas(sqlite_conn, pg_conn):
    """Insere registros faltantes em tabelas referenciadas (ex: ativos usados em posicoes que nao existem)."""
    sqlite_cursor = sqlite_conn.cursor()

    # Ativos referenciados em posicoes mas que nao existem na tabela ativos
    sqlite_cursor.execute("""
        SELECT DISTINCT p.ativo FROM posicoes p
        LEFT JOIN ativos a ON a.nome = p.ativo
        WHERE a.nome IS NULL
    """)
    ativos_orfaos = [row[0] for row in sqlite_cursor.fetchall()]

    if ativos_orfaos:
        print(f"  Corrigindo {len(ativos_orfaos)} ativo(s) orfao(s): {ativos_orfaos}", flush=True)
        with pg_conn.cursor() as pg_cur:
            for ativo in ativos_orfaos:
                pg_cur.execute(
                    "INSERT INTO ativos (nome) VALUES (%s) ON CONFLICT DO NOTHING",
                    (ativo,)
                )
        pg_conn.commit()

    # Ativos referenciados em ativos_rastreados mas que nao existem na tabela ativos
    sqlite_cursor.execute("""
        SELECT DISTINCT ar.ativo FROM ativos_rastreados ar
        LEFT JOIN ativos a ON a.nome = ar.ativo
        WHERE a.nome IS NULL
    """)
    ativos_rastreados_orfaos = [row[0] for row in sqlite_cursor.fetchall()]

    if ativos_rastreados_orfaos:
        print(f"  Corrigindo {len(ativos_rastreados_orfaos)} ativo(s) rastreado(s) orfao(s): {ativos_rastreados_orfaos}", flush=True)
        with pg_conn.cursor() as pg_cur:
            for ativo in ativos_rastreados_orfaos:
                pg_cur.execute(
                    "INSERT INTO ativos (nome) VALUES (%s) ON CONFLICT DO NOTHING",
                    (ativo,)
                )
        pg_conn.commit()


def copiar_dados(sqlite_conn, pg_conn):
    """Copia dados de todas as tabelas do SQLite para PostgreSQL."""
    sqlite_cursor = sqlite_conn.cursor()

    # Corrigir referencias orfas antes de copiar
    corrigir_referencias_orfas(sqlite_conn, pg_conn)

    # Desabilitar verificacao de FK temporariamente para bulk insert
    with pg_conn.cursor() as cur:
        cur.execute("SET session_replication_role = 'replica'")
    pg_conn.commit()

    for tabela in TABELAS_ORDEM:
        # Verificar se tabela existe no SQLite
        sqlite_cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tabela,)
        )
        if not sqlite_cursor.fetchone():
            print(f"  [{tabela}] Tabela não existe no SQLite - pulando.", flush=True)
            continue

        # Obter colunas do SQLite
        sqlite_cursor.execute(f"PRAGMA table_info({tabela})")
        colunas_sqlite = [row[1] for row in sqlite_cursor.fetchall()]

        # Obter colunas do PostgreSQL
        with pg_conn.cursor() as pg_cur:
            pg_cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (tabela,))
            colunas_pg = [row[0] for row in pg_cur.fetchall()]

        # Usar apenas colunas que existem em ambos
        colunas_comuns = [c for c in colunas_sqlite if c in colunas_pg]

        if not colunas_comuns:
            print(f"  [{tabela}] Sem colunas em comum - pulando.", flush=True)
            continue

        # Ler dados do SQLite
        colunas_str = ', '.join(colunas_comuns)
        sqlite_cursor.execute(f"SELECT {colunas_str} FROM {tabela}")
        rows = sqlite_cursor.fetchall()

        if not rows:
            print(f"  [{tabela}] 0 registros.", flush=True)
            continue

        # Inserir no PostgreSQL
        placeholders = ', '.join(['%s'] * len(colunas_comuns))
        insert_sql = f"INSERT INTO {tabela} ({colunas_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

        with pg_conn.cursor() as pg_cur:
            psycopg2.extras.execute_batch(pg_cur, insert_sql, rows, page_size=500)

        pg_conn.commit()
        print(f"  [{tabela}] {len(rows)} registros migrados.", flush=True)

    # Reabilitar verificacao de FK
    with pg_conn.cursor() as cur:
        cur.execute("SET session_replication_role = 'origin'")
    pg_conn.commit()

    # Atualizar sequences para tabelas SERIAL (stops, visualizacoes_config, turmas, etc.)
    tabelas_serial = ['stops', 'visualizacoes_config', 'turmas', 'trades_turma', 'carteira_turma']
    with pg_conn.cursor() as pg_cur:
        for tabela in tabelas_serial:
            try:
                pg_cur.execute(f"""
                    SELECT setval(pg_get_serial_sequence('{tabela}', 'id'),
                                  COALESCE((SELECT MAX(id) FROM {tabela}), 0) + 1, false)
                """)
            except Exception:
                pg_conn.rollback()
    pg_conn.commit()


def main():
    parser = argparse.ArgumentParser(description='Migrar SQLite para Supabase (PostgreSQL)')
    parser.add_argument('--apenas-schema', action='store_true', help='Só cria tabelas, sem copiar dados')
    args = parser.parse_args()

    if not DB_URL:
        print("ERRO: SUPABASE_DB_URL não configurado no .env!", flush=True)
        sys.exit(1)

    if not SQLITE_PATH.exists():
        print(f"ERRO: SQLite não encontrado em {SQLITE_PATH}", flush=True)
        sys.exit(1)

    print("=" * 50, flush=True)
    print("MIGRACAO SQLite -> Supabase (PostgreSQL)", flush=True)
    print("=" * 50, flush=True)

    # Conectar
    print(f"\nConectando ao SQLite: {SQLITE_PATH}", flush=True)
    sqlite_conn = sqlite3.connect(SQLITE_PATH)

    print(f"Conectando ao Supabase...", flush=True)
    pg_conn = psycopg2.connect(DB_URL)
    print("  Conectado!\n", flush=True)

    # 0. Limpar tabelas existentes (na ordem inversa para respeitar FKs)
    print("\nLimpando tabelas existentes no Supabase...", flush=True)
    with pg_conn.cursor() as cur:
        for tabela in reversed(TABELAS_ORDEM):
            try:
                cur.execute(f"DROP TABLE IF EXISTS {tabela} CASCADE")
            except Exception:
                pg_conn.rollback()
    pg_conn.commit()
    print("  Tabelas limpas!", flush=True)

    # 1. Criar schema
    criar_schema(pg_conn)

    # 2. Detectar e adicionar colunas extras (dinâmicas)
    colunas_extras = detectar_colunas_extras_sqlite(sqlite_conn)
    if colunas_extras:
        adicionar_colunas_extras(pg_conn, colunas_extras)

    # 3. Copiar dados
    if not args.apenas_schema:
        print("\nCopiando dados...", flush=True)
        copiar_dados(sqlite_conn, pg_conn)

    # Fechar conexões
    sqlite_conn.close()
    pg_conn.close()

    print("\n" + "=" * 50, flush=True)
    print("MIGRACAO CONCLUIDA COM SUCESSO!", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
