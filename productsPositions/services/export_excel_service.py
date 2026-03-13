"""
Serviço para exportar rentabilidade e posições em Excel.

Gera um arquivo .xlsx com abas conforme a exibição de cada produto:
- Rentabilidade: série diária de rentabilidade acumulada
- Posições Abertas, Fechadas, Histórico (colunas conforme tabela do dashboard)
- Para ICOs: aba Possíveis ICOs
"""

import io
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

from services.rentabilidade_service import RentabilidadeService
from services.turmas_service import TurmasService
from services.portfolio_service import (
    get_portfolio_data,
    get_portfolio_rentabilidade_serie,
)
from analytics.queries import posicoes_abertas, posicoes_fechadas, historico_posicoes

# Mapeamento produto_id -> config (espelho de servidor_dashboard.PORTFOLIO_PRODUCTS)
_PORTFOLIO_PRODUCTS = {
    2150859854: {'type': 'group', 'keys': ['EXC', 'HB', 'LC']},
    2000449260: {'type': 'redirect', 'target_id': 2150859854},
    2394004756: {'type': 'redirect', 'target_id': 2150859854},
    3476245316: {'type': 'single', 'keys': ['AC']},
}

# Colunas por tipo de produto e tabela (campo_dados -> header_excel)
# Ordem = ordem de exibição na tabela do dashboard
_COLUNAS_TURMAS_ABERTAS = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('side', 'Side'), ('origem', 'Origem'),
    ('data_insercao', 'Data Entrada'), ('preco_entrada_turma', 'Preço Entrada'),
    ('quantidade', 'Qtd'), ('preco_entrada_total', 'Entrada Total'),
    ('preco_atual', 'Preço Atual'), ('preco_saida_total', 'Atual Total'),
    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]
_COLUNAS_TURMAS_FECHADAS = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('side', 'Side'), ('origem', 'Origem'),
    ('data_insercao', 'Data Entrada'), ('data_remocao', 'Data Saída'),
    ('dias', 'Dias'), ('preco_entrada_turma', 'Preço Entrada'),
    ('preco_entrada_total', 'Entrada Total'), ('preco_atual', 'Preço Saída'),
    ('preco_saida_total', 'Saída Total'), ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]
_COLUNAS_TURMAS_HISTORICO = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('side', 'Side'), ('origem', 'Origem'),
    ('_status', 'Status'), ('data_insercao', 'Data Entrada'), ('data_remocao', 'Data Saída'),
    ('dias', 'Dias'), ('preco_entrada_turma', 'Preço Entrada'),
    ('preco_entrada_total', 'Entrada Total'), ('preco_atual', 'Preço Saída/Atual'),
    ('preco_saida_total', 'Saída Total'), ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]

_COLUNAS_PORTFOLIO_ABERTAS = [
    ('sub_portfolio', 'Sub-Portfólio'), ('ativo', 'Ativo'), ('alocacao_pct', 'Alocação%'), ('data_entrada', 'Data Entrada'),
    ('dias', 'Dias'), ('preco_entrada', 'Preço Entrada'), ('preco_atual', 'Preço Atual'),
    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]
_COLUNAS_PORTFOLIO_FECHADAS = [
    ('sub_portfolio', 'Sub-Portfólio'), ('ativo', 'Ativo'), ('data_entrada', 'Data Entrada'), ('data_saida', 'Data Saída'),
    ('dias', 'Dias'), ('preco_entrada', 'Preço Entrada'), ('preco_saida', 'Preço Saída'),
    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]
_COLUNAS_PORTFOLIO_HISTORICO = [
    ('sub_portfolio', 'Sub-Portfólio'), ('ativo', 'Ativo'), ('_status', 'Status'), ('data_entrada', 'Data Entrada'),
    ('data_saida', 'Data Saída'), ('dias', 'Dias'),
    ('preco_entrada', 'Preço Entrada'), ('preco_atual', 'Preço Saída/Atual'),
    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]

_COLUNAS_CRYPTO_ABERTAS = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('side', 'Tipo'), ('perfil', 'Perfil'),
    ('data_insercao', 'Dt. Abertura'), ('dias', 'Duração'),
    ('preco_entrada_turma', 'P. Entrada'), ('preco_atual', 'P. Atual'),
    ('alvo1', 'Alvo 1'), ('alvo2', 'Alvo 2'), ('stop_atual', 'Stop'),
    ('rr', 'RR'), ('pnl_pct', 'PnL%'),
]
_COLUNAS_CRYPTO_FECHADAS = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('side', 'Tipo'), ('perfil', 'Perfil'),
    ('data_insercao', 'Dt. Abertura'), ('data_remocao', 'Dt. Encerram.'),
    ('dias', 'Duração'), ('preco_entrada_turma', 'P. Entrada'), ('preco_atual', 'P. Saída'),
    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]
_COLUNAS_CRYPTO_HISTORICO = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('_status', 'Status'), ('side', 'Tipo'), ('perfil', 'Perfil'),
    ('data_insercao', 'Dt. Abertura'), ('data_remocao', 'Dt. Saída'),
    ('dias', 'Duração'), ('preco_entrada_turma', 'P. Entrada'), ('preco_atual', 'P. Saída/Atual'),
    ('alvo1', 'Alvo 1'), ('alvo2', 'Alvo 2'), ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]

_COLUNAS_ICOS_POSSIVEIS = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('rank', 'Rank'), ('tipo_janela', 'Tipo Janela'),
    ('tese', 'Tese'), ('risco', 'Risco'), ('atencao', 'Atenção'),
    ('execucao', 'Execução'), ('por_que', 'Por quê'),
]
_COLUNAS_ICOS_ABERTAS = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('categoria', 'Categoria'), ('tipo_ico', 'Tipo'),
    ('data_insercao', 'Data de ICO'), ('dias', 'Duração'),
    ('preco_entrada_turma', 'P. Entrada'), ('preco_atual', 'P. Atual'),
    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
]
_COLUNAS_ICOS_FECHADAS = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('categoria', 'Categoria'), ('tipo_ico', 'Tipo'),
    ('data_insercao', 'Data de ICO'), ('data_remocao', 'Data Saída'),
    ('preco_entrada_turma', 'P. Entrada'), ('preco_atual', 'P. Saída'),
    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'), ('resultado', 'Resultado'),
]
_COLUNAS_ICOS_HISTORICO = [
    ('turma', 'Turma'), ('ativo', 'Ativo'), ('_status', 'Status'), ('categoria', 'Categoria'), ('tipo_ico', 'Tipo'),
    ('data_insercao', 'Data de ICO'), ('data_remocao', 'Data Saída'),
    ('dias', 'Duração'), ('preco_entrada_turma', 'P. Entrada'), ('preco_atual', 'P. Saída/Atual'),
    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'), ('resultado', 'Resultado'),
]


def _list_to_df(data: List[Dict], colunas: List[Tuple[str, str]], status_fn=None) -> pd.DataFrame:
    """Converte lista de dicts em DataFrame com colunas na ordem definida."""
    if not data:
        return pd.DataFrame()
    rows = []
    for t in data:
        row = {}
        for campo, header in colunas:
            if campo == '_status':
                val = status_fn(t) if status_fn else ''
            else:
                val = t.get(campo)
            if hasattr(val, 'isoformat'):
                val = val.isoformat()[:10] if val else None
            row[header] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _is_trade_open(t):
    """Determina se trade está aberto (carteira)."""
    if t.get('ativo_atual') in (0, False):
        return False
    if t.get('data_remocao') or t.get('data_saida'):
        return False
    if str(t.get('status_posicao') or '').lower() == 'closed':
        return False
    return True


def _is_possivel_ico(t):
    """Possíveis ICOs: abertas com preço de entrada zero."""
    if not _is_trade_open(t):
        return False
    pe = t.get('preco_entrada_turma')
    if pe is None:
        return True
    return (float(pe) if pe else 0) < 0.0001


def _is_aberta_ico(t):
    """Abertas ICOs: em carteira com preço definido."""
    if not _is_trade_open(t):
        return False
    pe = t.get('preco_entrada_turma')
    if pe is None:
        return False
    return (float(pe) if pe else 0) >= 0.0001


def _status_turmas(t):
    return 'Aberto' if _is_trade_open(t) else 'Fechado'


def _status_portfolio(t):
    return 'Aberto' if not t.get('data_saida') else 'Fechado'


def _status_crypto(t):
    return 'Aberto' if _is_trade_open(t) else 'Fechado'


def _status_icos(t):
    if _is_possivel_ico(t):
        return 'Possível'
    if _is_aberta_ico(t):
        return 'Aberto'
    return 'Fechado'


def _montar_resumo_pnl(dados_posicoes: Optional[Dict]) -> pd.DataFrame:
    """Resumo PnL por ativo para produtos sem rentabilidade acumulada (ex: ICOs, Crypto Signals)."""
    if not dados_posicoes:
        return pd.DataFrame()
    tipo = dados_posicoes.get('tipo', '')
    if tipo in ('turmas', 'crypto', 'icos'):
        carteira = dados_posicoes.get('historico', []) or []
        if not carteira:
            return pd.DataFrame()
        colunas = [
            ('turma', 'Turma'), ('ativo', 'Ativo'), ('_status', 'Status'),
            ('preco_entrada_turma', 'P. Entrada'), ('preco_atual', 'P. Atual/Saída'),
            ('pnl_pct', 'PnL%'), ('dias', 'Dias'),
        ]
        def status_fn(t):
            if tipo == 'icos' and _is_possivel_ico(t):
                return 'Possível'
            return 'Aberto' if _is_trade_open(t) else 'Fechado'
        return _list_to_df(carteira, colunas, status_fn=status_fn)
    if tipo == 'portfolio':
        hist = dados_posicoes.get('historico', []) or []
        if not hist:
            return pd.DataFrame()
        for t in hist:
            t['_preco_fim'] = t.get('preco_saida') or t.get('preco_atual')
        colunas = [
            ('sub_portfolio', 'Sub-Portfólio'), ('ativo', 'Ativo'), ('_status', 'Status'),
            ('preco_entrada', 'P. Entrada'), ('_preco_fim', 'P. Atual/Saída'),
            ('pnl_pct', 'PnL%'), ('dias', 'Dias'),
        ]
        return _list_to_df(hist, colunas, status_fn=_status_portfolio)
    return pd.DataFrame()


def _montar_rentabilidade_turmas(
    produto_id: int,
    rentabilidade_service: RentabilidadeService,
    turmas_service: TurmasService,
) -> pd.DataFrame:
    """Monta DataFrame de rentabilidade para produtos com turmas."""
    turmas = turmas_service.listar_turmas(produto_id)
    if not turmas:
        return pd.DataFrame()

    turma_ids = [t['id'] for t in turmas]
    precos_cache = rentabilidade_service.construir_precos_cache(turma_ids)

    all_data = []
    for turma in turmas:
        turma_id = turma['id']
        turma_nome = turma['nome']
        serie = rentabilidade_service.calcular_serie_rentabilidade(
            turma_id, precos_cache=precos_cache
        )
        for portfolio in serie:
            all_data.append({
                'Data': portfolio.dia,
                'Turma': turma_nome,
                'Rentabilidade Acumulada (%)': round(portfolio.rentabilidade_acumulada_pct, 4),
                'Valor Total (R$)': round(portfolio.valor_total, 2),
                'Capital Alocado (R$)': round(portfolio.capital_alocado, 2),
                'Capital em Caixa (R$)': round(portfolio.capital_em_caixa, 2),
            })

    return pd.DataFrame(all_data)


def _montar_rentabilidade_portfolio(
    portfolio_keys: List[str],
    repo=None,
) -> pd.DataFrame:
    """Monta DataFrame de rentabilidade para produtos portfolio."""
    all_data = []
    nomes = {'EXC': 'Principal', 'HB': 'High Beta', 'LC': 'Low Caps', 'AC': 'Alphacoins'}
    for key in portfolio_keys:
        try:
            serie = get_portfolio_rentabilidade_serie(key, repo=repo)
        except Exception:
            continue
        nome = nomes.get(key, key)
        for item in serie:
            dia = item.get('dia') if isinstance(item, dict) else getattr(item, 'name', str(item))
            rentab = item.get('rentabilidade_acumulada_pct', 0) if isinstance(item, dict) else float(item)
            all_data.append({
                'Data': dia,
                'Turma': nome,
                'Rentabilidade Acumulada (%)': round(float(rentab), 4),
                'Valor Total (R$)': None,
                'Capital Alocado (R$)': None,
                'Capital em Caixa (R$)': None,
            })
    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data)


def _montar_posicoes_portfolio(
    portfolio_keys: List[str],
    repo=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna (df_abertas, df_fechadas, df_historico) para produtos portfolio."""
    abertas_all, fechadas_all = [], []
    nomes = {'EXC': 'Principal', 'HB': 'High Beta', 'LC': 'Low Caps', 'AC': 'Alphacoins'}
    for key in portfolio_keys:
        try:
            data, _ = get_portfolio_data(key, repo=repo)
        except Exception:
            continue
        nome = nomes.get(key, key)
        for t in data.get('posicoes_abertas', []):
            row = dict(t)
            row['sub_portfolio'] = nome
            abertas_all.append(row)
        for t in data.get('posicoes_fechadas', []):
            row = dict(t)
            row['sub_portfolio'] = nome
            fechadas_all.append(row)

    df_abertas = pd.DataFrame(abertas_all) if abertas_all else pd.DataFrame()
    df_fechadas = pd.DataFrame(fechadas_all) if fechadas_all else pd.DataFrame()

    # Histórico = abertas + fechadas
    all_list = abertas_all + fechadas_all
    df_hist = pd.DataFrame(all_list) if all_list else pd.DataFrame()
    if not df_hist.empty and 'data_entrada' in df_hist.columns:
        df_hist = df_hist.sort_values('data_entrada', ascending=False)

    return df_abertas, df_fechadas, df_hist


def _df_carteira_para_excel(
    carteira: List[Dict],
    tipo: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Converte carteira enriquecida em DataFrames para Excel.
    tipo: 'turmas' | 'crypto' | 'icos'
    Retorna (abertas, fechadas, historico, possiveis ou None)
    """
    if tipo == 'turmas':
        open_fn = lambda t: _is_trade_open(t)
        closed_fn = lambda t: not _is_trade_open(t)
        col_ab, col_fech, col_hist = _COLUNAS_TURMAS_ABERTAS, _COLUNAS_TURMAS_FECHADAS, _COLUNAS_TURMAS_HISTORICO
        status_fn = _status_turmas
    elif tipo == 'crypto':
        open_fn = _is_trade_open
        closed_fn = lambda t: not _is_trade_open(t)
        col_ab, col_fech, col_hist = _COLUNAS_CRYPTO_ABERTAS, _COLUNAS_CRYPTO_FECHADAS, _COLUNAS_CRYPTO_HISTORICO
        status_fn = _status_crypto
    else:  # icos
        open_fn = _is_aberta_ico
        closed_fn = lambda t: not _is_trade_open(t)
        col_ab, col_fech, col_hist = _COLUNAS_ICOS_ABERTAS, _COLUNAS_ICOS_FECHADAS, _COLUNAS_ICOS_HISTORICO
        status_fn = _status_icos

    abertas = [t for t in carteira if open_fn(t)]
    fechadas = [t for t in carteira if closed_fn(t)]
    historico = list(carteira)

    df_ab = _list_to_df(abertas, col_ab)
    df_fech = _list_to_df(fechadas, col_fech)
    df_hist = _list_to_df(historico, col_hist, status_fn=status_fn)

    df_poss = None
    if tipo == 'icos':
        possiveis = [t for t in carteira if _is_possivel_ico(t)]
        df_poss = _list_to_df(possiveis, _COLUNAS_ICOS_POSSIVEIS)

    return df_ab, df_fech, df_hist, df_poss


def _df_portfolio_para_excel(
    df_abertas: pd.DataFrame,
    df_fechadas: pd.DataFrame,
    df_historico: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Converte dados portfolio em DataFrames para Excel com colunas corretas."""
    col_ab, col_fech, col_hist = _COLUNAS_PORTFOLIO_ABERTAS, _COLUNAS_PORTFOLIO_FECHADAS, _COLUNAS_PORTFOLIO_HISTORICO

    def to_records(df):
        return df.to_dict('records') if isinstance(df, pd.DataFrame) and not df.empty else []

    recs_ab = to_records(df_abertas)
    recs_fech = to_records(df_fechadas)
    recs_hist = to_records(df_historico)

    df_ab = _list_to_df(recs_ab, col_ab)
    df_fech = _list_to_df(recs_fech, col_fech)
    df_hist = _list_to_df(recs_hist, col_hist, status_fn=_status_portfolio)

    return df_ab, df_fech, df_hist


def gerar_excel_completo(
    produto_id: int,
    produto_nome: str,
    db_url: Optional[str] = None,
    repo=None,
    dados_posicoes: Optional[Dict[str, Any]] = None,
) -> io.BytesIO:
    """
    Gera arquivo Excel completo com rentabilidade e posições.

    Args:
        produto_id: ID do produto
        produto_nome: Nome do produto
        db_url: URL do banco
        repo: Repositório
        dados_posicoes: Opcional. Dict com:
            - tipo: 'turmas'|'portfolio'|'crypto'|'icos'
            - abertas: list[dict] ou DataFrame
            - fechadas: list[dict] ou DataFrame
            - historico: list[dict] ou DataFrame
            - possiveis: list[dict] (apenas para icos)
            Quando fornecido, usa essas colunas conforme a tabela do dashboard.
    """
    buffer = io.BytesIO()

    pf_cfg = _PORTFOLIO_PRODUCTS.get(produto_id)
    is_portfolio = pf_cfg is not None and pf_cfg.get('type') in ('group', 'single')
    portfolio_keys = []
    if is_portfolio:
        if pf_cfg.get('type') == 'redirect':
            target = _PORTFOLIO_PRODUCTS.get(pf_cfg['target_id'])
            portfolio_keys = (target or {}).get('keys', [])
        else:
            portfolio_keys = pf_cfg.get('keys', [])

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # --- Aba Rentabilidade ---
        df_rent = pd.DataFrame()
        if is_portfolio and portfolio_keys:
            df_rent = _montar_rentabilidade_portfolio(portfolio_keys, repo=repo)
        else:
            try:
                rent_svc = RentabilidadeService(db_url=db_url)
                turmas_svc = TurmasService(db_url=db_url)
                df_rent = _montar_rentabilidade_turmas(produto_id, rent_svc, turmas_svc)
            except Exception:
                pass

        if not df_rent.empty:
            df_rent.to_excel(writer, sheet_name='Rentabilidade', index=False)
        else:
            # Produtos sem rentabilidade acumulada (ex: ICOs): aba "PnL por Ativo" com resumo da carteira
            df_pnl = _montar_resumo_pnl(dados_posicoes)
            if not df_pnl.empty:
                df_pnl.to_excel(writer, sheet_name='PnL por Ativo', index=False)
            else:
                pd.DataFrame([{'mensagem': 'Sem dados de rentabilidade ou PnL para este produto.'}]).to_excel(
                    writer, sheet_name='Rentabilidade', index=False
                )

        # --- Abas Posições (conforme tabela do dashboard) ---
        if dados_posicoes:
            tipo = dados_posicoes.get('tipo', 'turmas')
            if tipo in ('turmas', 'crypto', 'icos'):
                carteira = dados_posicoes.get('historico', []) or []
                df_ab, df_fech, df_hist, df_poss = _df_carteira_para_excel(carteira, tipo)
            else:
                df_ab = pd.DataFrame(dados_posicoes.get('abertas', []))
                df_fech = pd.DataFrame(dados_posicoes.get('fechadas', []))
                df_hist = pd.DataFrame(dados_posicoes.get('historico', []))
                df_poss = None
                if not df_ab.empty or not df_fech.empty or not df_hist.empty:
                    df_ab, df_fech, df_hist = _df_portfolio_para_excel(df_ab, df_fech, df_hist)
        else:
            # Fallback: dados genéricos (posicoes ou portfolio)
            df_ab, df_fech, df_hist, df_poss = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
            if is_portfolio and portfolio_keys:
                df_ab, df_fech, df_hist = _montar_posicoes_portfolio(portfolio_keys, repo=repo)
                df_ab, df_fech, df_hist = _df_portfolio_para_excel(df_ab, df_fech, df_hist)
            else:
                # Analytics queries retornam data_entrada, preco_entrada (não data_insercao, preco_entrada_turma)
                col_ab_fb = [
                    ('ativo', 'Ativo'), ('side', 'Side'), ('data_entrada', 'Data Entrada'),
                    ('preco_entrada', 'Preço Entrada'), ('quantidade', 'Qtd'),
                    ('preco_atual', 'Preço Atual'), ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
                ]
                col_fech_fb = [
                    ('ativo', 'Ativo'), ('side', 'Side'), ('data_entrada', 'Data Entrada'),
                    ('data_saida', 'Data Saída'), ('dias', 'Dias'),
                    ('preco_entrada', 'Preço Entrada'), ('preco_saida', 'Preço Saída'),
                    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
                ]
                col_hist_fb = [
                    ('ativo', 'Ativo'), ('side', 'Side'), ('_status', 'Status'),
                    ('data_entrada', 'Data Entrada'), ('data_saida', 'Data Saída'), ('dias', 'Dias'),
                    ('preco_entrada', 'Preço Entrada'), ('preco_atual', 'Preço Saída/Atual'),
                    ('stop_atual', 'Stop'), ('pnl_pct', 'PnL%'),
                ]
                df_ab_raw = posicoes_abertas(produto_id)
                df_fech_raw = posicoes_fechadas(produto_id)
                df_hist_raw = historico_posicoes(produto_id)
                if df_ab_raw is not None and not df_ab_raw.empty:
                    df_ab = _list_to_df(df_ab_raw.to_dict('records'), col_ab_fb)
                if df_fech_raw is not None and not df_fech_raw.empty:
                    df_fech = _list_to_df(df_fech_raw.to_dict('records'), col_fech_fb)
                if df_hist_raw is not None and not df_hist_raw.empty:
                    df_hist = _list_to_df(df_hist_raw.to_dict('records'), col_hist_fb, status_fn=_status_portfolio)

        # Escrever abas
        for nome_aba, df in [
            ('Posições Abertas', df_ab),
            ('Posições Fechadas', df_fech),
            ('Histórico', df_hist),
        ]:
            if not df.empty:
                df.to_excel(writer, sheet_name=nome_aba, index=False)
            else:
                msg = 'Nenhuma posição aberta.' if 'Abertas' in nome_aba else 'Nenhuma posição fechada.' if 'Fechadas' in nome_aba else 'Nenhum histórico de posições.'
                pd.DataFrame([{'mensagem': msg}]).to_excel(writer, sheet_name=nome_aba, index=False)

        if df_poss is not None:
            if not df_poss.empty:
                df_poss.to_excel(writer, sheet_name='Possíveis ICOs', index=False)
            else:
                pd.DataFrame([{'mensagem': 'Nenhum possível ICO.'}]).to_excel(
                    writer, sheet_name='Possíveis ICOs', index=False
                )

    buffer.seek(0)
    return buffer
