"""
Dashboard Web para visualizacao de todos os produtos e posicoes
Servidor HTTP que fornece interface web completa para gerenciar o sistema
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
# scripts/ no path para import turmas_dashboard (mesmo diretório)
_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(1, str(_scripts))

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json
import os
import gzip
import urllib.parse
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

TZ_BRASILIA = ZoneInfo('America/Sao_Paulo')


def _now_brasilia():
    """Retorna datetime atual no fuso de Brasilia."""
    return datetime.now(TZ_BRASILIA)
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from storage.sqlite_repo import get_repo
from domain.posicao import Posicao
from domain.produto import Produto
from domain.tipo import Tipo
from analytics.notebook_utils import (
    display_posicoes_abertas,
    display_posicoes_fechadas,
    display_historico_posicoes,
    display_carteira,
    display_alocacoes,
)
from analytics.queries import invalidar_cache_atr
from services.atr_stop_service import atualizar_stops_posicoes_abertas, calcular_stop_para_posicao
from services.bitget_service import sync_positions_with_exchange, auto_sync_positions, fetch_bitget_tickers_perpetuals, fetch_bitget_tickers_spot, get_bitget_credentials
from services.notificacao_service import notificar_stop_atingido
from services.turmas_service import TurmasService
from services.rentabilidade_service import RentabilidadeService
from services.cotacoes_service import CotacoesService
from turmas_dashboard import (
    get_lista_turmas_html,
    get_form_nova_turma_html,
    get_turma_detalhes_html,
    get_rentabilidade_chart_html,
    get_comparar_turmas_html,
    get_produto_dashboard_html
)
from portfolio_dashboard import get_portfolio_dashboard_html
from cryptosignals_dashboard import get_cryptosignals_dashboard_html
from icos_dashboard import get_icos_dashboard_html
from services.portfolio_service import get_portfolio_data, get_portfolio_pnl, get_portfolio_rentabilidade_serie, PORTFOLIO_CONFIG
from services.btc_cache_service import BTCCacheService

logger = logging.getLogger(__name__)

# Mapeamento de produto_id para configuração de portfolio
# EXC, HB, LC são sub-portfolios de "Exponential Coins"
# AC (Alphacoins) é produto separado
PORTFOLIO_PRODUCTS = {
    2150859854: {'type': 'group', 'group_name': 'Exponential Coins', 'keys': ['EXC', 'HB', 'LC']},
    2000449260: {'type': 'redirect', 'target_id': 2150859854},  # HB -> Exponential Coins
    2394004756: {'type': 'redirect', 'target_id': 2150859854},  # LC -> Exponential Coins
    3476245316: {'type': 'single', 'group_name': 'Alphacoins', 'keys': ['AC']},
}

SOROS_GROUPS = {
    'Soros Spot': {
        'display_name': 'Soros Spot',
        'members': ['Soros Spot 1', 'Soros Spot 2'],
        'tab_names': ['Soros Spot Turma 1', 'Soros Spot Turma 2'],
    },
}

CUSTOM_DASHBOARD_PRODUCTS = {
    4970919917: 'cryptosignals',
}

ICOS_PRODUCT_NAMES = ['icos', 'ico']

# Nomes para identificar produtos nos formulários específicos
CRYPTO_SIGNALS_NAMES = ['crypto signals', 'cryptosignals', 'crypto_signals']
ICOS_FORM_NAMES = ['icos', 'ico']


def _is_produto_crypto_signals(produto):
    """Retorna True se o produto é Crypto Signals."""
    if not produto:
        return False
    nome = (produto.get('nome') or '').strip().lower()
    return nome in CRYPTO_SIGNALS_NAMES or 'crypto' in nome and 'signal' in nome


def _is_produto_icos(produto):
    """Retorna True se o produto é ICOs."""
    if not produto:
        return False
    nome = (produto.get('nome') or '').strip().lower()
    return nome in ICOS_FORM_NAMES


_soros_name_to_id_cache = {}

def _resolve_soros_group(repo, produto_nome):
    """Resolve group info for a Soros product by name.
    Returns (group_cfg, member_index) or (None, -1) if not in any group."""
    for _gkey, gcfg in SOROS_GROUPS.items():
        if produto_nome in gcfg['members']:
            idx = gcfg['members'].index(produto_nome)
            return gcfg, idx
    return None, -1

def _get_soros_id_by_name(repo, nome):
    """Look up product ID by name, with caching."""
    if nome in _soros_name_to_id_cache:
        return _soros_name_to_id_cache[nome]
    produtos = repo.listar_produtos()
    for p in produtos:
        p_nome = (p.get('nome') or '').strip()
        if p_nome in [m for g in SOROS_GROUPS.values() for m in g['members']]:
            _soros_name_to_id_cache[p_nome] = int(p.get('id') or 0)
    return _soros_name_to_id_cache.get(nome)


def _enrich_carteira_cs_attributes(carteira, repo):
    """Enrich Crypto Signals carteira with product-specific attributes (perfil, alvo1, alvo2, rr)."""
    posicao_ids = [int(t['posicao_id']) for t in carteira if t.get('posicao_id')]
    if not posicao_ids:
        return
    try:
        with repo.connection() as conn:
            placeholders = ','.join(['%s'] * len(posicao_ids))
            query = f"""
                SELECT posicao_id, motivo, perfil, alvo1, alvo2
                FROM posicao_atributos_produto
                WHERE posicao_id IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=posicao_ids)
        attr_map = {}
        for _, row in df.iterrows():
            attr_map[int(row['posicao_id'])] = {
                'motivo': row.get('motivo'),
                'perfil': row.get('perfil'),
                'alvo1': float(row['alvo1']) if pd.notna(row.get('alvo1')) else None,
                'alvo2': float(row['alvo2']) if pd.notna(row.get('alvo2')) else None,
            }
        for trade in carteira:
            pid = int(trade.get('posicao_id', 0))
            attrs = attr_map.get(pid, {})
            trade['perfil'] = attrs.get('perfil')
            trade['alvo1'] = attrs.get('alvo1')
            trade['alvo2'] = attrs.get('alvo2')
            preco_atual = trade.get('preco_atual')
            alvo2 = attrs.get('alvo2')
            stop = trade.get('stop_atual')
            if preco_atual and alvo2 and stop and stop not in (-1,) and preco_atual != 0:
                try:
                    denom = abs(stop / preco_atual - 1)
                    trade['rr'] = round(abs(alvo2 / preco_atual - 1) / denom, 2) if denom > 0 else None
                except Exception:
                    trade['rr'] = None
            else:
                trade['rr'] = None
    except Exception as e:
        print(f"[DASHBOARD] Erro ao enriquecer atributos CS: {e}", flush=True)


def _enrich_carteira_icos_attributes(carteira, repo):
    """Enrich ICOs carteira with product-specific attributes (categoria, tipo_ico, resultado)."""
    posicao_ids = [int(t['posicao_id']) for t in carteira if t.get('posicao_id')]
    if not posicao_ids:
        return
    try:
        with repo.connection() as conn:
            colunas = repo.listar_colunas_atributos()
            icos_cols = [c for c in colunas if c in (
                'categoria', 'tipo_ico', 'resultado', 'local', 'ficha_tecnica',
                'disponivel_pos_ico', 'tese', 'risco', 'atencao', 'execucao',
                'rank', 'tipo_janela', 'por_que', 'motivo', 'perfil',
            )]
            if not icos_cols:
                return
            cols_sql = ', '.join(icos_cols)
            placeholders = ','.join(['%s'] * len(posicao_ids))
            query = f"""
                SELECT posicao_id, {cols_sql}
                FROM posicao_atributos_produto
                WHERE posicao_id IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=posicao_ids)
        attr_map = {}
        for _, row in df.iterrows():
            attrs = {}
            for col in icos_cols:
                val = row.get(col)
                if pd.notna(val):
                    attrs[col] = val
            attr_map[int(row['posicao_id'])] = attrs
        for trade in carteira:
            pid = int(trade.get('posicao_id', 0))
            attrs = attr_map.get(pid, {})
            for col in icos_cols:
                trade[col] = attrs.get(col)
    except Exception as e:
        print(f"[DASHBOARD] Erro ao enriquecer atributos ICOs: {e}", flush=True)


def _enrich_carteira_trades(carteira, precos_atuais=None, repo=None):
    """Enriquece trades da carteira com campos computados (dias, PnL, preço atual, stop)."""
    hoje = date.today()
    hoje_str = hoje.isoformat()

    # Batch load stops para todas as posições da carteira
    stops_map = {}
    if repo:
        posicao_ids = [int(t['posicao_id']) for t in carteira if t.get('posicao_id')]
        if posicao_ids:
            from analytics.queries import _batch_load_stops, _batch_load_stops_por_ativo
            stops_map = _batch_load_stops(repo, posicao_ids)

    for trade in carteira:
        data_insercao = trade.get('data_insercao', '')
        data_remocao = trade.get('data_remocao')
        data_saida = trade.get('data_saida')
        # Normalizar datas (podem vir como date do DB)
        if hasattr(data_insercao, 'isoformat'):
            data_insercao = data_insercao.isoformat()[:10]
        if hasattr(data_remocao, 'isoformat'):
            data_remocao = data_remocao.isoformat()[:10]
        if hasattr(data_saida, 'isoformat'):
            data_saida = data_saida.isoformat()[:10]
        trade['data_insercao'] = data_insercao
        if data_remocao is not None:
            trade['data_remocao'] = data_remocao
        if data_saida is not None:
            trade['data_saida'] = data_saida
        status = (trade.get('status_posicao') or '').strip().lower()
        # Fechado se tiver data de saída/remocão OU status 'closed' (fonte da verdade: posicoes.status)
        is_closed = bool(data_remocao or data_saida or status == 'closed')
        data_fim = data_remocao or data_saida

        if not is_closed:
            try:
                d_ins = datetime.strptime(data_insercao, '%Y-%m-%d').date()
                trade['dias'] = (hoje - d_ins).days
            except Exception:
                trade['dias'] = 0
        else:
            try:
                d_ins = datetime.strptime(data_insercao, '%Y-%m-%d').date()
                d_end = datetime.strptime(data_fim or hoje_str, '%Y-%m-%d').date()
                trade['dias'] = (d_end - d_ins).days
            except Exception:
                trade['dias'] = 0

        # Preço de entrada: para turmas, SEMPRE usar preco_entrada_turma (preço na data de inserção na turma)
        # Desconsidera data/preço de entrada no produto - regra: dados da turma
        preco_entrada_orig = trade.get('preco_entrada_original')
        preco_entrada_turma = trade.get('preco_entrada_turma', 0) or 0
        if preco_entrada_turma and float(preco_entrada_turma) > 0:
            preco_entrada = float(preco_entrada_turma)
        else:
            preco_entrada = (float(preco_entrada_orig) if preco_entrada_orig is not None else None) or preco_entrada_turma
        side = (trade.get('side') or '').upper()
        quantidade = trade.get('quantidade')

        pos_id = trade.get('posicao_id')
        trade['stop_atual'] = stops_map.get(int(pos_id)) if pos_id else None

        # preco_entrada_total = quantidade * preco_entrada (mesmo critério do sistema antigo)
        trade['preco_entrada_total'] = (quantidade * preco_entrada) if quantidade and preco_entrada else None

        if not is_closed:
            key = _price_key(trade)
            preco_atual = precos_atuais.get(key) if key and precos_atuais else None
            trade['preco_atual'] = preco_atual if preco_atual else preco_entrada
        else:
            trade['preco_atual'] = trade.get('preco_saida') or preco_entrada

        # preco_saida_total = quantidade * preco_atual (ou preco_saida)
        trade['preco_saida_total'] = (quantidade * trade['preco_atual']) if quantidade and trade['preco_atual'] else None

        # PnL%: mesma mecânica do sistema antigo (queries.py perpétuos)
        # ((preco_saida_total / preco_entrada_total) - 1) * 100, invertido para SHORT
        pet = trade.get('preco_entrada_total')
        pst = trade.get('preco_saida_total')
        if pet and pst and pet != 0:
            pnl = ((pst / pet) - 1.0) * 100.0
            if side == 'SHORT':
                pnl = -pnl
            trade['pnl_pct'] = pnl
        elif preco_entrada and preco_entrada != 0 and trade['preco_atual']:
            pnl = ((trade['preco_atual'] / preco_entrada) - 1.0) * 100.0
            if side == 'SHORT':
                pnl = -pnl
            trade['pnl_pct'] = pnl
        else:
            trade['pnl_pct'] = 0.0

    return carteira


def _price_key(trade):
    """Chave de lookup: coingecko_id ou exchange_symbol (para posições sem coingecko_id)."""
    return trade.get('coingecko_id') or (trade.get('exchange_symbol') or '').strip().upper() or None


def _obter_preco_btc_bitget_coingecko_fallback(cotacoes_service):
    """
    Preço atual do BTC: Bitget primeiro (spot ou perp), CoinGecko como fallback.
    Usado no benchmark de rentabilidade acumulada.
    """
    try:
        tickers = fetch_bitget_tickers_spot()
        if tickers and 'BTCUSDT' in tickers and tickers['BTCUSDT'] and tickers['BTCUSDT'] > 0:
            return float(tickers['BTCUSDT'])
    except Exception:
        pass
    try:
        tickers = fetch_bitget_tickers_perpetuals()
        if tickers and 'BTCUSDT' in tickers and tickers['BTCUSDT'] and tickers['BTCUSDT'] > 0:
            return float(tickers['BTCUSDT'])
    except Exception:
        pass
    return cotacoes_service.obter_preco_coingecko('bitcoin') if cotacoes_service else None


def _exchange_symbol_to_cmc_symbol(ex_sym):
    """Extrai símbolo base para CoinMarketCap (ex: BTCUSDT -> BTC)."""
    if not ex_sym:
        return None
    s = (ex_sym or '').strip().upper()
    for suffix in ('USDT', 'USDC', 'BUSD', 'USD'):
        if s.endswith(suffix) and len(s) > len(suffix):
            return s[:-len(suffix)]
    return s if len(s) <= 10 else None


def _obter_precos_bitget_primeiro_coingecko_fallback(carteira, tipo_produto=None, db_url=None):
    """
    Busca preços atuais: Bitget primeiro, CoinGecko como fallback, CoinMarketCap como terceiro fallback.
    Para trades ativos da carteira. Indexa por _price_key (coingecko_id ou exchange_symbol).
    """
    trades_ativos = [t for t in carteira if t.get('ativo_atual')]
    if not trades_ativos:
        return {}

    precos = {}
    tipo_lower = (tipo_produto or '').lower()

    # 1) Bitget primeiro — nunca sobrescrever com CoinGecko/CMC depois
    use_perp = 'perp' in tipo_lower
    use_spot = 'spot' in tipo_lower
    if not use_perp and not use_spot:
        use_perp = use_spot = True  # produto indefinido: tenta os dois
    if use_perp:
        try:
            tickers = fetch_bitget_tickers_perpetuals()
            for t in trades_ativos:
                ex_sym = (t.get('exchange_symbol') or '').strip().upper()
                if ex_sym and ex_sym in tickers and tickers[ex_sym] > 0:
                    key = _price_key(t)
                    if key and key not in precos:
                        precos[key] = tickers[ex_sym]
        except Exception:
            pass
    if use_spot:
        try:
            tickers = fetch_bitget_tickers_spot()
            for t in trades_ativos:
                ex_sym = (t.get('exchange_symbol') or '').strip().upper()
                if ex_sym and ex_sym in tickers and tickers[ex_sym] > 0:
                    key = _price_key(t)
                    if key and key not in precos:
                        precos[key] = tickers[ex_sym]
        except Exception:
            pass

    # 2) CoinGecko como fallback — só preenche o que ainda não tem preço
    coingecko_ids_faltando = list(set(
        t['coingecko_id'] for t in trades_ativos
        if t.get('coingecko_id') and t['coingecko_id'] not in precos
    ))
    if coingecko_ids_faltando:
        try:
            cotacoes_service = CotacoesService(db_url=db_url)
            cg_precos = cotacoes_service.obter_precos_coingecko_fallback(coingecko_ids_faltando)
            for cg_id, preco in cg_precos.items():
                if cg_id not in precos:
                    precos[cg_id] = preco
        except Exception:
            pass

    # 3) CoinMarketCap como terceiro fallback — trades que ainda não têm preço
    trades_sem_preco = [t for t in trades_ativos if _price_key(t) and _price_key(t) not in precos]
    if trades_sem_preco and os.environ.get('COINMARKETCAP_API_KEY', '').strip():
        try:
            from services.portfolio_service import TICKER_TO_COINGECKO
            COINGECKO_TO_TICKER = {v: k for k, v in TICKER_TO_COINGECKO.items()}

            symbol_to_keys = {}  # symbol -> [key]
            for t in trades_sem_preco:
                key = _price_key(t)
                if not key:
                    continue
                sym = None
                ex_sym = (t.get('exchange_symbol') or '').strip().upper()
                cg_id = t.get('coingecko_id')
                ativo = (t.get('ativo') or '').strip().upper()
                if ex_sym:
                    sym = _exchange_symbol_to_cmc_symbol(ex_sym)
                if not sym and cg_id:
                    sym = COINGECKO_TO_TICKER.get(cg_id)
                if not sym and ativo:
                    sym = ativo if 1 <= len(ativo) <= 10 else None
                if sym:
                    symbol_to_keys.setdefault(sym, []).append(key)

            if symbol_to_keys:
                cotacoes_service = CotacoesService(db_url=db_url)
                cmc_precos = cotacoes_service.obter_precos_coinmarketcap_fallback(list(symbol_to_keys.keys()))
                for sym, preco in cmc_precos.items():
                    for key in symbol_to_keys.get(sym, []):
                        if key not in precos:
                            precos[key] = preco
        except Exception:
            pass

    return precos


def _obter_precos_com_fonte(carteira, tipo_produto=None, db_url=None):
    """
    Igual a _obter_precos_bitget_primeiro_coingecko_fallback, mas retorna
    dict[key] = {"preco": float, "fonte": "bitget_spot"|"bitget_perp"|"coingecko"|"coinmarketcap"|"db"}
    para diagnóstico de origem dos preços.
    """
    trades_ativos = [t for t in carteira if t.get('ativo_atual')]
    if not trades_ativos:
        return {}

    precos = {}
    tipo_lower = (tipo_produto or '').lower()
    use_perp = 'perp' in tipo_lower
    use_spot = 'spot' in tipo_lower
    if not use_perp and not use_spot:
        use_perp = use_spot = True

    if use_perp:
        try:
            tickers = fetch_bitget_tickers_perpetuals()
            for t in trades_ativos:
                ex_sym = (t.get('exchange_symbol') or '').strip().upper()
                if ex_sym and ex_sym in tickers and tickers[ex_sym] > 0:
                    key = _price_key(t)
                    if key and key not in precos:
                        precos[key] = {"preco": tickers[ex_sym], "fonte": "bitget_perp"}
        except Exception:
            pass
    if use_spot:
        try:
            tickers = fetch_bitget_tickers_spot()
            for t in trades_ativos:
                ex_sym = (t.get('exchange_symbol') or '').strip().upper()
                if ex_sym and ex_sym in tickers and tickers[ex_sym] > 0:
                    key = _price_key(t)
                    if key and key not in precos:
                        precos[key] = {"preco": tickers[ex_sym], "fonte": "bitget_spot"}
        except Exception:
            pass

    coingecko_ids_faltando = list(set(
        t['coingecko_id'] for t in trades_ativos
        if t.get('coingecko_id') and t['coingecko_id'] not in precos
    ))
    if coingecko_ids_faltando:
        try:
            cotacoes_service = CotacoesService(db_url=db_url)
            cg_precos = cotacoes_service.obter_precos_coingecko_fallback(coingecko_ids_faltando)
            for cg_id, preco in cg_precos.items():
                if cg_id not in precos:
                    precos[cg_id] = {"preco": preco, "fonte": "coingecko"}
        except Exception:
            pass

    trades_sem_preco = [t for t in trades_ativos if _price_key(t) and _price_key(t) not in precos]
    if trades_sem_preco and os.environ.get('COINMARKETCAP_API_KEY', '').strip():
        try:
            from services.portfolio_service import TICKER_TO_COINGECKO
            COINGECKO_TO_TICKER = {v: k for k, v in TICKER_TO_COINGECKO.items()}
            symbol_to_keys = {}
            for t in trades_sem_preco:
                key = _price_key(t)
                if not key:
                    continue
                sym = None
                ex_sym = (t.get('exchange_symbol') or '').strip().upper()
                cg_id = t.get('coingecko_id')
                ativo = (t.get('ativo') or '').strip().upper()
                if ex_sym:
                    sym = _exchange_symbol_to_cmc_symbol(ex_sym)
                if not sym and cg_id:
                    sym = COINGECKO_TO_TICKER.get(cg_id)
                if not sym and ativo:
                    sym = ativo if 1 <= len(ativo) <= 10 else None
                if sym:
                    symbol_to_keys.setdefault(sym, []).append(key)
            if symbol_to_keys:
                cotacoes_service = CotacoesService(db_url=db_url)
                cmc_precos = cotacoes_service.obter_precos_coinmarketcap_fallback(list(symbol_to_keys.keys()))
                for sym, preco in cmc_precos.items():
                    for key in symbol_to_keys.get(sym, []):
                        if key not in precos:
                            precos[key] = {"preco": preco, "fonte": "coinmarketcap"}
        except Exception:
            pass

    return precos


def _normalizar_data_entrada(val, fallback_today=True):
    """Converte data_entrada (date, datetime, Timestamp, NaT, str) para YYYY-MM-DD. Usa hoje só se fallback_today e valor vazio."""
    if val is None or val == '':
        return date.today().isoformat() if fallback_today else ''
    try:
        if pd.isna(val):  # pandas NaT
            return date.today().isoformat() if fallback_today else ''
        if hasattr(val, 'strftime'):
            return val.strftime('%Y-%m-%d')
        s = str(val).strip()[:10]
        if s and len(s) == 10 and s[4] == '-' and s[7] == '-':
            return s
    except Exception:
        pass
    return date.today().isoformat() if fallback_today else ''


def atualizar_dados_produto(repo, produto_id: int) -> dict:
    """
    Atualiza dados do produto em paralelo (Bitget sync + ATR stops).
    Executado automaticamente ao carregar a pagina do produto.

    Returns:
        dict com resultados: {bitget: {...}, atr: {...}}
    """
    resultado = {'bitget': None, 'atr': None}

    def sync_bitget():
        try:
            return auto_sync_positions(repo, produto_id, verbose=True)
        except Exception as e:
            print(f"[BITGET] Erro ao sincronizar produto {produto_id}: {e}")
            return {'opened': 0, 'closed': 0, 'synced': 0, 'errors': [str(e)]}

    def update_atr():
        try:
            return atualizar_stops_posicoes_abertas(repo=repo, produto_id=produto_id, verbose=False)
        except Exception as e:
            print(f"[ATR] Erro ao atualizar stops produto {produto_id}: {e}")
            return {'updated': 0, 'errors': [str(e)]}

    # Executa Bitget sync e ATR update em paralelo
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bitget = executor.submit(sync_bitget)
        future_atr = executor.submit(update_atr)

        resultado['bitget'] = future_bitget.result()
        resultado['atr'] = future_atr.result()

    # Log resumido
    bg = resultado.get('bitget') or {}
    opened = bg.get('opened', 0)
    closed = bg.get('closed', 0)
    synced = bg.get('synced', 0)
    if opened or closed or synced:
        print(f"[BITGET] Produto {produto_id}: {opened} abertas, {closed} fechadas, {synced} atualizadas")
    if bg.get('errors'):
        for err in bg['errors']:
            print(f"[BITGET] Produto {produto_id} ERRO: {err}")
    if resultado['atr'] and resultado['atr'].get('updated', 0) > 0:
        print(f"[ATR] Produto {produto_id}: {resultado['atr']['updated']} stops atualizados")

    return resultado


# ============================================================
# FUNCOES DE VISUALIZACAO (copiadas de visualizar_dados.py)
# ============================================================

def aplicar_visualizacao(df, visualizacao):
    """Aplica uma visualizacao salva a um DataFrame."""
    if df is None or df.empty:
        return df

    # Selecionar apenas colunas que existem no DataFrame
    colunas_config = visualizacao.get('colunas', [])
    if colunas_config:
        colunas_disponiveis = [c for c in colunas_config if c in df.columns]
        if colunas_disponiveis:
            # Garantir que stop_atual apareça mesmo se não estiver na config da visualização
            if 'stop_atual' in df.columns and 'stop_atual' not in colunas_disponiveis:
                colunas_disponiveis.append('stop_atual')
            df = df[colunas_disponiveis]

    # Aplicar filtros (exceto status que ja foi aplicado)
    if visualizacao.get('filtros'):
        for filtro in visualizacao['filtros']:
            coluna = filtro['coluna']
            operador = filtro['operador']
            valor = filtro['valor']

            if coluna not in df.columns or coluna == 'status':
                continue

            try:
                if operador == '=':
                    df = df[df[coluna] == valor]
                elif operador == '!=':
                    df = df[df[coluna] != valor]
                elif operador == '>':
                    df = df[pd.to_numeric(df[coluna], errors='coerce') > float(valor)]
                elif operador == '<':
                    df = df[pd.to_numeric(df[coluna], errors='coerce') < float(valor)]
                elif operador == '>=':
                    df = df[pd.to_numeric(df[coluna], errors='coerce') >= float(valor)]
                elif operador == '<=':
                    df = df[pd.to_numeric(df[coluna], errors='coerce') <= float(valor)]
                elif operador == 'contem':
                    df = df[df[coluna].astype(str).str.contains(str(valor), case=False, na=False)]
            except Exception:
                pass

    # Aplicar ordenacao
    if visualizacao.get('ordenacao') and visualizacao['ordenacao'].get('coluna'):
        coluna = visualizacao['ordenacao']['coluna']
        direcao = visualizacao['ordenacao'].get('direcao', 'asc')
        if coluna in df.columns:
            df = df.sort_values(by=coluna, ascending=(direcao == 'asc'))

    # Renomear colunas com labels customizados
    if visualizacao.get('colunas_labels'):
        rename_map = {}
        for col_nome, col_label in visualizacao['colunas_labels'].items():
            if col_nome in df.columns and col_label:
                rename_map[col_nome] = col_label
        if rename_map:
            df = df.rename(columns=rename_map)

    # Substituir valores None por travessão
    df = df.fillna("—")

    return df


def obter_dados_para_visualizacao(produto_id, visualizacao):
    """Obtem os dados apropriados para uma visualizacao baseado nos filtros."""
    filtros = visualizacao.get('filtros', []) or []

    # Procurar filtro de status
    status_filtro = None
    for f in filtros:
        if f.get('coluna') == 'status':
            status_filtro = f.get('valor')
            break

    if status_filtro == 'open':
        df = display_posicoes_abertas(produto_id, formatar=True, filtrar_colunas=False)
    elif status_filtro == 'closed':
        df = display_posicoes_fechadas(produto_id, formatar=True, filtrar_colunas=False)
    else:
        df = display_historico_posicoes(produto_id, formatar=True, filtrar_colunas=False)

    return df


# ============================================================
# HTML TEMPLATES
# ============================================================

def _get_preencher_precos_btn(produto_id):
    """Gera o botão dropdown + JS para preencher preços de um produto."""
    return f'''
        <div class="dropdown-precos" style="display:inline-block; position:relative;">
            <button type="button" class="btn btn-sm btn-secondary" onclick="this.nextElementSibling.classList.toggle('show')">
                Preencher Precos &#9662;
            </button>
            <div class="dropdown-precos-menu" style="display:none; position:absolute; right:0; top:100%; margin-top:4px; background:#16213e; border:1px solid #39fda3; border-radius:8px; min-width:220px; z-index:100; box-shadow:0 4px 12px rgba(0,0,0,.4);">
                <a href="#" onclick="return preencherPrecos({produto_id},'entrada')" style="display:block;padding:10px 16px;color:#eee;text-decoration:none;font-size:.85em;border-bottom:1px solid #2a2a4a;">
                    Precos de <b style="color:#39fda3;">Entrada</b>
                </a>
                <a href="#" onclick="return preencherPrecos({produto_id},'saida')" style="display:block;padding:10px 16px;color:#eee;text-decoration:none;font-size:.85em;border-bottom:1px solid #2a2a4a;">
                    Precos de <b style="color:#39fda3;">Saida</b>
                </a>
                <a href="#" onclick="return preencherPrecos({produto_id},'ambos')" style="display:block;padding:10px 16px;color:#eee;text-decoration:none;font-size:.85em;">
                    <b style="color:#39fda3;">Todos</b> (Entrada + Saida)
                </a>
            </div>
        </div>
        <style>
            .dropdown-precos-menu.show {{ display:block!important; }}
        </style>
    '''


def _get_preencher_precos_js():
    """Retorna o JavaScript para o botão de preencher preços (incluir uma vez por página)."""
    return '''
    <script>
    // Fechar dropdown ao clicar fora
    document.addEventListener('click', function(e) {
        document.querySelectorAll('.dropdown-precos-menu.show').forEach(function(m) {
            if (!m.parentElement.contains(e.target)) m.classList.remove('show');
        });
    });
    function preencherPrecos(produtoId, tipo) {
        document.querySelectorAll('.dropdown-precos-menu.show').forEach(function(m) { m.classList.remove('show'); });
        var nomes = {entrada: 'preços de entrada', saida: 'preços de saída', ambos: 'todos os preços'};
        if (!confirm('Preencher ' + nomes[tipo] + ' faltantes? O processo roda em background.')) return false;
        var toast = document.createElement('div');
        toast.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#16213e;border:1px solid #39fda3;color:#39fda3;padding:14px 24px;border-radius:10px;z-index:9999;font-size:.9em;box-shadow:0 4px 16px rgba(0,0,0,.5);';
        toast.textContent = 'Iniciando preenchimento de ' + nomes[tipo] + '...';
        document.body.appendChild(toast);
        fetch('/api/posicoes/preencher-precos?produto_id=' + produtoId + '&tipo=' + tipo)
            .then(function(r) { return r.json(); })
            .then(function(res) {
                if (res.sucesso) {
                    toast.innerHTML = '<b style="color:#ffd93d;">OK!</b> ' + res.mensagem;
                } else {
                    toast.innerHTML = '<b style="color:#ff6b6b;">Erro:</b> ' + (res.erro || 'Falha desconhecida');
                }
                setTimeout(function() { toast.remove(); }, 8000);
            })
            .catch(function(e) {
                toast.innerHTML = '<b style="color:#ff6b6b;">Erro:</b> ' + e.message;
                setTimeout(function() { toast.remove(); }, 8000);
            });
        return false;
    }
    </script>
    '''


def _get_form_modal_overlay_script(produto_id):
    """Script e HTML para abrir formularios em modal na mesma pagina (sem recarregar o fundo)."""
    return f'''
    <div id="formModalOverlay" style="display:none; position:fixed; inset:0; z-index:9999;">
        <div class="form-page-overlay" style="position:fixed;inset:0;background:rgba(0,0,0,0.7);cursor:pointer;z-index:1;" onclick="formModalClose(event)"></div>
        <div class="form-page-modal" style="position:fixed;inset:0;display:flex;align-items:flex-start;justify-content:center;padding:24px;overflow:auto;pointer-events:none;z-index:2;">
            <div class="form-modal-card" style="pointer-events:auto;max-width:95vw;width:fit-content;margin:auto;overflow:visible;display:flex;flex-direction:column;align-items:stretch;" onclick="event.stopPropagation()">
                <div id="formModalContent" style="min-width:0;"></div>
            </div>
        </div>
    </div>
    <style>
        #formModalContent .form-modal-loader {{
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 40px; color: #5a6a7a; font-size: 0.9em;
        }}
        #formModalContent .form-modal-loader .loader-spinner {{
            width: 36px; height: 36px;
            border: 3px solid rgba(78,204,163,0.15);
            border-top-color: #39fda3;
            border-radius: 50%;
            animation: formModalSpin 0.8s linear infinite;
        }}
        #formModalContent .form-modal-loader .loader-text {{ margin-top: 14px; }}
        @keyframes formModalSpin {{ to {{ transform: rotate(360deg); }} }}
    </style>
    <script>
    (function() {{
        const produtoId = {produto_id};
        const loaderHtml = '<div class="form-modal-loader"><div class="loader-spinner"></div><div class="loader-text">Carregando...</div></div>';
        function isFormLink(a) {{
            if (!a || a.tagName !== 'A' || !a.href) return false;
            try {{
                const url = new URL(a.href);
                if (url.origin !== location.origin) return false;
                if (url.pathname.indexOf('/alocacao/') >= 0) return false;
                const path = url.pathname;
                if (path === '/' || path === '') return false;
                if (/^\\/produto\\/\\d+\\/?$/.test(path)) return false;
                const formPaths = ['/posicao/', '/posicoes/', '/stop/', '/atr/', '/produto/', '/turmas/nova'];
                return formPaths.some(p => path.indexOf(p) === 0);
            }} catch (e) {{ return false; }}
        }}
        function formModalOpen(url) {{
            const overlay = document.getElementById('formModalOverlay');
            const content = document.getElementById('formModalContent');
            content.innerHTML = loaderHtml;
            overlay.style.display = 'block';
            document.body.style.overflow = 'hidden';
            const sep = url.indexOf('?') >= 0 ? '&' : '?';
            const returnPath = encodeURIComponent(location.pathname || '/');
            fetch(url + sep + '_modal=1&return=' + returnPath).then(r => r.text()).then(html => {{
                const wrap = document.createElement('div');
                wrap.innerHTML = html;
                const scripts = wrap.querySelectorAll('script');
                content.innerHTML = '';
                wrap.childNodes.forEach(n => {{
                    if (n.tagName === 'SCRIPT') return;
                    content.appendChild(n.cloneNode(true));
                }});
                scripts.forEach(s => {{
                    const ns = document.createElement('script');
                    if (s.src) ns.src = s.src;
                    else ns.textContent = s.textContent;
                    content.appendChild(ns);
                }});
            }}).catch(e => {{
                content.innerHTML = '<p style="color:#e74c3c;padding:20px;">Erro ao carregar: ' + (e.message || 'Erro desconhecido') + '</p>';
                alert('Erro ao carregar: ' + e.message);
            }});
        }}
        window.formModalClose = function(e) {{
            if (e && e.target !== e.currentTarget) return;
            document.getElementById('formModalOverlay').style.display = 'none';
            document.body.style.overflow = '';
        }};
        function formModalCloseNow() {{
            document.getElementById('formModalOverlay').style.display = 'none';
            document.body.style.overflow = '';
        }};
        document.addEventListener('click', function(e) {{
            const a = e.target.closest('a');
            if (!a || !a.href) return;
            try {{
                const url = new URL(a.href);
                if (url.origin !== location.origin) return;
                const path = url.pathname;
                const prodPath = '/produto/' + produtoId;
                if (isFormLink(a)) {{
                    e.preventDefault();
                    formModalOpen(a.href);
                    return;
                }}
                if (path === prodPath || path === prodPath + '/' || path.indexOf(prodPath + '/') === 0) {{
                    if (document.getElementById('formModalOverlay').style.display === 'block') {{
                        e.preventDefault();
                        formModalCloseNow();
                        if (path !== location.pathname) location.href = a.href;
                    }}
                }}
            }} catch (err) {{}}
        }}, true);
    }})();
    </script>
    '''


def get_favicon_tag():
    """Tag link do favicon para incluir no head"""
    return '<link rel="icon" type="image/png" href="/assets/favicon.png?v=1">'


def get_base_styles():
    """Estilos CSS compartilhados"""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .navbar {
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar .navbar-brand {
            display: flex; align-items: center; gap: 10px;
            text-decoration: none;
            padding: 0; margin: 0; border-radius: 0;
            background: none !important;
        }
        .navbar .navbar-brand:hover { background: none !important; }
        .navbar .navbar-logo { height: 40px; width: auto; display: block; }
        .navbar .navbar-brand h1 { font-size: 1.5em; color: #39fda3; margin: 0; transition: color 0.2s; }
        .navbar .navbar-brand:hover h1 { color: #2edb8d; }
; }
        .navbar-links a {
            color: #39fda3;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 5px;
            transition: all 0.3s;
            margin-left: 5px;
        }
        .navbar-links a:hover {
            background-color: #39fda3;
            color: #1a1a2e;
        }
        .navbar-links { display: flex; gap: 5px; flex-wrap: wrap; }
        .navbar-links a.active { background: rgba(78, 204, 163, 0.25); }
        .turmas-subnav {
            display: flex;
            gap: 0;
            padding: 0 30px;
            background: rgba(0,0,0,0.15);
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .turmas-subnav a {
            color: #888;
            text-decoration: none;
            padding: 10px 20px;
            font-size: 0.9em;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .turmas-subnav a:hover { color: #ccc; }
        .turmas-subnav a.active {
            color: #39fda3;
            border-bottom-color: #39fda3;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 30px; }
        .container.form-page { padding: 16px 24px; }
        /* Form modal overlay: fundo = produto, escurecido, form centralizado */
        .form-page-bg { position: fixed; inset: 0; width: 100%; height: 100%; border: none; z-index: 0; }
        .form-page-bg-placeholder { position: fixed; inset: 0; z-index: 0.5; background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%); pointer-events: none; transition: opacity 0.3s ease; }
        .form-page-bg-placeholder.hidden { opacity: 0; }
        .form-page-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 1; }
        .form-page-modal { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; z-index: 2; padding: 24px; overflow-y: auto; }
        .form-page-modal .form-modal-card { margin: auto; }
        .card {
            background: linear-gradient(135deg, #16213e 0%, #1f2833 100%);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .card:hover {
            box-shadow: 0 8px 25px rgba(78, 204, 163, 0.15);
        }
        .card h2 { color: #39fda3; margin-bottom: 15px; font-size: 1.3em; }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }
        /* Product cards */
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 16px;
        }
        .product-card {
            background: #16213e;
            border-radius: 10px;
            padding: 18px 20px;
            border-left: 3px solid #39fda3;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
        }
        .product-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        }
        .product-card-new {
            background: transparent;
            border: 2px dashed #2a3a5a;
            border-left: 2px dashed #2a3a5a;
            box-shadow: none;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            gap: 8px;
            min-height: 120px;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s, transform 0.2s;
        }
        .product-card-new:hover {
            border-color: #39fda3;
            background: rgba(57, 253, 163, 0.05);
            transform: translateY(-2px);
            box-shadow: none;
        }
        .product-card-new-icon {
            font-size: 2em;
            color: #3a4a6a;
            font-weight: 300;
            line-height: 1;
            transition: color 0.2s;
        }
        .product-card-new:hover .product-card-new-icon { color: #39fda3; }
        .product-card-new-label {
            font-size: 0.85em;
            color: #4a5a6a;
            font-weight: 500;
            transition: color 0.2s;
        }
        .product-card-new:hover .product-card-new-label { color: #39fda3; }
        .product-card-name {
            color: #39fda3;
            font-size: 1.05em;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .product-rentab {
            display: flex;
            align-items: baseline;
            gap: 8px;
            margin-bottom: 10px;
        }
        .product-rentab-value {
            font-size: 1.15em;
            font-weight: 700;
        }
        .product-rentab-value.positive { color: #39fda3; }
        .product-rentab-value.negative { color: #e74c3c; }
        .product-rentab-label {
            font-size: 0.78em;
            color: #5a6a7a;
        }
        .product-meta {
            color: #6b7b8d;
            font-size: 0.88em;
            display: flex;
            gap: 6px;
            align-items: center;
        }
        .product-meta span { white-space: nowrap; }
        .product-meta .dot { color: #3a4a5a; }
        .product-actions {
            margin-top: auto;
            padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.06);
        }
        .product-actions a {
            display: block;
            padding: 9px 16px;
            border-radius: 6px;
            text-align: center;
            text-decoration: none;
            font-size: 0.85em;
            font-weight: 500;
            color: #39fda3;
            background: rgba(78, 204, 163, 0.08);
            border: 1px solid rgba(78, 204, 163, 0.2);
            transition: all 0.2s;
        }
        .product-actions a:hover {
            background: rgba(78, 204, 163, 0.15);
            border-color: rgba(78, 204, 163, 0.4);
        }
        .product-nav-loader {
            position: fixed; inset: 0; z-index: 99999;
            background: #1a1a2e;
            display: none; flex-direction: column;
            align-items: center; justify-content: center;
            transition: opacity 0.2s;
        }
        .product-nav-loader.show {
            display: flex;
            opacity: 1;
        }
        .product-nav-loader .loader-spinner {
            width: 40px; height: 40px;
            border: 3px solid rgba(78,204,163,0.15);
            border-top-color: #39fda3;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        .product-nav-loader .loader-text {
            margin-top: 16px;
            color: #5a6a7a;
            font-family: 'Segoe UI', sans-serif;
            font-size: 0.9em;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .badge-spot { background-color: #39fda3; color: #1a1a2e; }
        .badge-perpetuos { background-color: #ff6b6b; color: white; }
        .badge-outro { background-color: #ffd93d; color: #1a1a2e; }
        .badge-success { background-color: #39fda3; color: #1a1a2e; }
        .badge-danger { background-color: #ff6b6b; color: white; }
        .bdg-stop {
            display: inline-block; padding: 3px 10px; border-radius: 12px;
            font-size: 0.8em; font-weight: 600;
            background: rgba(231, 76, 60, 0.15); color: #e74c3c;
            border: 1px solid rgba(231, 76, 60, 0.3);
        }
        .stats { display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap; }
        .stat {
            background: rgba(78, 204, 163, 0.1);
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 3px solid #39fda3;
            flex: 1;
            min-width: 100px;
        }
        .stat-label { font-size: 0.75em; color: #888; }
        .stat-value { font-size: 1.1em; font-weight: bold; color: #39fda3; }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
            cursor: pointer;
            border: none;
            font-size: 0.9em;
        }
        .btn-primary { background-color: #39fda3; color: #1a1a2e; }
        .btn-primary:hover { background-color: #3db892; }
        .btn-secondary {
            background-color: transparent;
            color: #39fda3;
            border: 2px solid #39fda3;
        }
        .btn-secondary:hover { background-color: #39fda3; color: #1a1a2e; }
        .btn-danger { background-color: #ff6b6b; color: white; }
        .btn-danger:hover { background-color: #ff5252; }
        .btn-warning { background-color: #ffa726; color: #1a1a2e; }
        .btn-warning:hover { background-color: #ff9800; }
        .btn-sm { padding: 6px 12px; font-size: 0.8em; }
        .actions { margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
            font-size: 0.9em;
        }
        th {
            background: linear-gradient(135deg, #39fda3 0%, #3db892 100%);
            color: #1a1a2e;
            padding: 12px 10px;
            text-align: left;
            font-weight: bold;
            white-space: nowrap;
        }
        td { padding: 10px; border-bottom: 1px solid #2a2a4a; }
        tr:hover { background-color: rgba(78, 204, 163, 0.1); }
        .loading {
            text-align: center;
            padding: 40px;
            color: #39fda3;
            display: none;
        }
        .loading.show { display: block; }
        .loading-spinner {
            border: 4px solid #2a2a4a;
            border-top: 4px solid #39fda3;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .tab {
            padding: 10px 20px;
            background: #16213e;
            border: 2px solid #2a2a4a;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            color: #888;
            text-decoration: none;
        }
        .tab:hover, .tab.active { border-color: #39fda3; color: #39fda3; }
        .tab.active { background: rgba(78, 204, 163, 0.2); }
        .timestamp { color: #666; font-size: 0.9em; margin: 0; }
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .btn-turmas {
            padding: 8px 16px;
            background: rgba(78, 204, 163, 0.15);
            color: #39fda3;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.9em;
            border: 1px solid rgba(78, 204, 163, 0.4);
            transition: all 0.2s;
        }
        .btn-turmas:hover {
            background: rgba(78, 204, 163, 0.25);
            border-color: #39fda3;
        }
        .empty-state { text-align: center; padding: 40px; color: #666; }
        .empty-state h3 { color: #888; margin-bottom: 10px; }
        .alert {
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        .alert.show { display: block; }
        .alert-success { background: rgba(78, 204, 163, 0.2); border: 1px solid #39fda3; color: #39fda3; }
        .alert-error { background: rgba(255, 107, 107, 0.2); border: 1px solid #ff6b6b; color: #ff6b6b; }
        /* Forms */
        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #39fda3;
            font-weight: bold;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #2a2a4a;
            border-radius: 8px;
            background: #16213e;
            color: #eee;
            font-size: 1em;
        }
        .form-group input:focus, .form-group select:focus {
            border-color: #39fda3;
            outline: none;
        }
        /* Icone de calendario nos inputs de data - cor #39fda3 (Chrome/Safari/Edge) */
        input[type="date"]::-webkit-calendar-picker-indicator {
            filter: invert(78%) sepia(62%) saturate(512%) hue-rotate(85deg) brightness(102%) contrast(98%);
            cursor: pointer;
            opacity: 0.9;
        }
        .form-row { display: flex; gap: 20px; }
        .form-row .form-group { flex: 1; }
        /* Formulários compactos (cabem na tela sem scroll) */
        .form-compact .form-group { margin-bottom: 8px; }
        .form-compact .form-group label { margin-bottom: 4px; font-size: 0.85em; }
        .form-compact .form-group input,
        .form-compact .form-group select,
        .form-compact .form-group textarea { padding: 6px 10px; font-size: 0.9em; }
        .form-compact .form-row { gap: 12px; margin-bottom: 8px; }
        .form-compact .form-row .form-group { margin-bottom: 0; }
        .form-compact .form-row-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 8px; }
        .form-compact .form-row-3 .form-group { margin-bottom: 0; flex: none; }
        .form-compact .form-row-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 8px; }
        .form-compact .form-row-4 .form-group { margin-bottom: 0; flex: none; }
        .form-compact .form-card { padding: 16px; max-width: 1200px; }
        .form-compact .form-card h2 { margin-bottom: 10px; font-size: 1.1em; }
        .form-compact .form-card > p { margin-bottom: 12px; font-size: 0.9em; }
        .form-compact hr { margin: 12px 0; }
        .form-compact .actions { margin-top: 12px; }
        @media (max-width: 900px) {
            .form-compact .form-row-3, .form-compact .form-row-4 {
                grid-template-columns: 1fr 1fr;
            }
        }
        @media (max-width: 600px) {
            .form-compact .form-row-3, .form-compact .form-row-4 {
                grid-template-columns: 1fr;
            }
        }
        .viz-list { list-style: none; }
        .viz-item {
            background: rgba(78, 204, 163, 0.1);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .viz-item:hover { background: rgba(78, 204, 163, 0.2); }
        .menu-section { margin-bottom: 30px; }
        .menu-section h3 {
            color: #39fda3;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #2a2a4a;
        }
        .menu-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        .menu-item {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            text-decoration: none;
            color: #eee;
            border: 2px solid #2a2a4a;
            transition: all 0.3s;
        }
        .menu-item:hover {
            border-color: #39fda3;
            transform: translateY(-3px);
        }
        .menu-item-icon { font-size: 2em; margin-bottom: 10px; }
        .menu-item-label { font-weight: bold; }
        .table-container { overflow-x: auto; }
        .table-container::-webkit-scrollbar { height: 6px; }
        .table-container::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
        .table-container::-webkit-scrollbar-thumb { background: rgba(78, 204, 163, 0.35); border-radius: 3px; }
    """


def get_form_page_with_background(inner_content, produto_id, title="Form"):
    """Envolve o conteúdo do formulário com fundo (página do produto escurecida) e centraliza na tela."""
    produto_url = f"/produto/{produto_id}"
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>{title}</title>
        <style>html, body {{ background: #1a1a2e; }}</style>
        <style>{get_base_styles()}</style>
    </head>
    <body style="background:#1a1a2e;">
        <iframe class="form-page-bg" src="{produto_url}" title="Fundo" onload="document.getElementById('form-bg-placeholder').classList.add('hidden')"></iframe>
        <div id="form-bg-placeholder" class="form-page-bg-placeholder"></div>
        <div class="form-page-overlay"></div>
        <div class="form-page-modal">
            <div class="form-modal-card">
                {inner_content}
            </div>
        </div>
    </body>
    </html>
    """


def get_navbar(current_page=""):
    """Gera a barra de navegacao principal (logo + titulo Empiricus Crypto, sem links)."""
    return """
    <nav class="navbar">
        <a href="/" class="navbar-brand">
            <img src="/assets/empiricus_crypto_logo.png?v=2" alt="Empiricus Crypto" class="navbar-logo">
            <h1>Empiricus Crypto</h1>
        </a>
    </nav>
    """


def get_turmas_subnav(current_sub=""):
    """Gera sub-navegacao interna das paginas de turmas"""
    return f"""
    <div class="turmas-subnav">
        <a href="/turmas" class="{'active' if current_sub == 'turmas' else ''}">Turmas</a>
        <a href="/turmas/comparar" class="{'active' if current_sub == 'comparar' else ''}">Comparar</a>
    </div>
    """


def get_loader_only_html():
    """Pagina minima com loader; o JS busca o conteudo completo e substitui o documento."""
    return """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """ + get_favicon_tag() + """
        <title>Carregando...</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                min-height: 100vh;
                background: #1a1a2e;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .loader-spinner {
                width: 40px; height: 40px;
                border: 3px solid rgba(78,204,163,0.2);
                border-top-color: #39fda3;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }
            .loader-text {
                margin-top: 16px;
                color: #5a6a7a;
                font-size: 0.9em;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="loader-spinner"></div>
        <div class="loader-text">Carregando...</div>
        <script>
            fetch(window.location.pathname + '?__content=1')
                .then(function(r) { return r.text(); })
                .then(function(html) {
                    document.open();
                    document.write(html);
                    document.close();
                })
                .catch(function() {
                    document.body.innerHTML = '<p style="color:#e74c3c;">Falha ao carregar. <a href="/" style="color:#39fda3;">Tentar de novo</a></p>';
                });
        </script>
    </body>
    </html>
    """


def _produto_sort_key(p):
    """Retorna chave de ordenacao para cards: soros, memebot, exponential coins, alphacoins, crypto signals, icos."""
    nome = (p.get('nome') or '').strip().lower()
    if 'soros' in nome:
        return 0
    if 'memebot' in nome:
        return 1
    if 'exponential' in nome:
        return 2
    if 'alphacoins' in nome or 'alpha coins' in nome:
        return 3
    if 'crypto signals' in nome:
        return 4
    if 'icos' in nome or (nome == 'ico'):
        return 5
    return 6


def _get_dashboard_data(repo):
    """Retorna (produtos_visiveis, stats) para a home do dashboard. Reutilizado pela rota / e /api/dashboard/home."""
    produtos = repo.listar_produtos()
    _hidden_ids = {int(pid) for pid, cfg in PORTFOLIO_PRODUCTS.items() if cfg.get('type') == 'redirect'}
    _hidden_nomes_exatos = {'HB', 'LC', 'High Beta', 'Low Caps'}
    _soros_secondary_names = set()
    _soros_primary_rename = {}
    for _gkey, gcfg in SOROS_GROUPS.items():
        for m in gcfg['members'][1:]:
            _soros_secondary_names.add(m)
        _soros_primary_rename[gcfg['members'][0]] = gcfg['display_name']
    produtos_visiveis = []
    for p in produtos:
        try:
            pid = int(p.get('id') or 0)
        except (TypeError, ValueError):
            pid = 0
        nome = (p.get('nome') or '').strip()
        if pid in _hidden_ids:
            continue
        if nome in _hidden_nomes_exatos or nome.upper() in ('HB', 'LC'):
            continue
        if nome in _soros_secondary_names:
            continue
        pf_cfg = PORTFOLIO_PRODUCTS.get(pid)
        if pf_cfg and 'group_name' in pf_cfg:
            p = dict(p)
            p['nome'] = pf_cfg['group_name']
        if nome in _soros_primary_rename:
            p = dict(p)
            p['nome'] = _soros_primary_rename[nome]
        produtos_visiveis.append(p)
    contagem_abertas = repo.contar_posicoes_abertas_por_produto()
    contagem_fechadas = repo.contar_posicoes_fechadas_por_produto()
    rentabilidade_por_produto = {}
    try:
        rentabilidade_service = RentabilidadeService(db_url=repo.db_url)
        for p in produtos_visiveis:
            pid = p.get('id')
            if _is_produto_crypto_signals(p) or _is_produto_icos(p):
                continue
            pf_cfg = PORTFOLIO_PRODUCTS.get(pid)
            if pf_cfg and pf_cfg.get('type') in ('group', 'single') and pf_cfg.get('keys'):
                try:
                    data, _ = get_portfolio_data(pf_cfg['keys'][0], repo=repo)
                    rentab = data['resumo'].get('rentabilidade_acumulada_pct', 0)
                    if rentab is not None:
                        rentabilidade_por_produto[pid] = rentab
                except Exception:
                    pass
            else:
                resumos = rentabilidade_service.obter_rentabilidade_resumida_todas_turmas(produto_id=pid)
                if resumos:
                    primeira_turma = resumos[-1]
                    rentab = primeira_turma.get('rentabilidade_atual_pct') or primeira_turma.get('rentabilidade_acumulada_pct')
                    if rentab is not None:
                        rentabilidade_por_produto[pid] = rentab
    except Exception:
        pass
    _redirect_to_group = {}
    for _pid, _cfg in PORTFOLIO_PRODUCTS.items():
        if _cfg.get('type') == 'redirect':
            _redirect_to_group[_pid] = _cfg['target_id']
    stats = {}
    for p in produtos_visiveis:
        pid = p['id']
        abertas = contagem_abertas.get(pid, 0)
        fechadas = contagem_fechadas.get(pid, 0)
        for sub_pid, group_pid in _redirect_to_group.items():
            if group_pid == pid:
                abertas += contagem_abertas.get(sub_pid, 0)
                fechadas += contagem_fechadas.get(sub_pid, 0)
        stats[pid] = {
            'posicoes_abertas': abertas,
            'posicoes_fechadas': fechadas,
            'rentabilidade_acumulada_pct': rentabilidade_por_produto.get(pid),
        }
    return produtos_visiveis, stats


def _build_dashboard_cards_html(produtos, stats):
    """Gera o HTML dos cards do dashboard. Usado por get_dashboard_html e /api/dashboard/home."""
    produtos_ordenados = sorted(produtos, key=_produto_sort_key)
    cards_html = ""
    for p in produtos_ordenados:
        prod_stats = stats.get(p['id'], {})
        posicoes_abertas = prod_stats.get('posicoes_abertas', 0)
        posicoes_fechadas = prod_stats.get('posicoes_fechadas', 0)
        rentabilidade_pct = prod_stats.get('rentabilidade_acumulada_pct')
        rentab_html = ""
        if rentabilidade_pct is not None:
            rentab_class = 'positive' if rentabilidade_pct >= 0 else 'negative'
            rentab_str = f"+{rentabilidade_pct:.2f}%" if rentabilidade_pct >= 0 else f"{rentabilidade_pct:.2f}%"
            rentab_html = f"""<div class="product-rentab">
                    <div class="product-rentab-value {rentab_class}">{rentab_str}</div>
                </div>"""
        cards_html += f"""
        <div class="product-card">
            <div class="product-card-name">{p['nome']}</div>
            {rentab_html}
            <div class="product-meta">
                <span>{posicoes_abertas} ativos</span>
                <span class="dot">&middot;</span>
                <span>{posicoes_fechadas} fechados</span>
            </div>
            <div class="product-actions">
                <a href="/produto/{p['id']}">Ver Detalhes</a>
            </div>
        </div>
        """
    cards_html += """
        <a href="/produto/novo" class="product-card product-card-new">
            <div class="product-card-new-icon">+</div>
            <div class="product-card-new-label">Novo Produto</div>
        </a>
        """
    if not cards_html.strip():
        cards_html = """
        <div class="empty-state">
            <h3>Nenhum produto encontrado</h3>
            <p>Crie um produto para comecar.</p>
            <a href="/produto/novo" class="btn btn-primary" style="margin-top: 20px;">Criar Produto</a>
        </div>
        """
    return cards_html


def get_dashboard_html(produtos, stats, repo, skip_loader=False):
    """Gera HTML da pagina principal do dashboard. skip_loader=True quando a pagina e carregada via fetch (__content=1)."""
    timestamp = _now_brasilia().strftime("%d/%m/%Y %H:%M:%S")
    cards_html = _build_dashboard_cards_html(produtos, stats)

    product_nav_loader_html = """
        <div class="product-nav-loader" id="productNavLoader">
            <div class="loader-spinner"></div>
            <div class="loader-text">Carregando produto...</div>
        </div>
        <script>
        (function() {
            document.addEventListener('click', function(e) {
                var a = e.target.closest('a[href^="/produto/"]');
                if (a && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
                    var href = a.getAttribute('href');
                    if (href === '/produto/novo') return;
                    e.preventDefault();
                    var loader = document.getElementById('productNavLoader');
                    if (loader) loader.classList.add('show');
                    window.location.href = href;
                }
            });
        })();
        </script>
    """
    if skip_loader:
        return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Dashboard - Empiricus Crypto</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {product_nav_loader_html}
        {get_navbar('home')}
        <div class="container">
            <div class="dashboard-header">
                <p class="timestamp" id="dashboardTimestamp">Ultima atualizacao: {timestamp}</p>
                <a href="/turmas" class="btn-turmas">Ver Turmas</a>
            </div>
            <div class="product-grid" id="dashboardProductGrid">{cards_html}</div>
        </div>
        {_get_form_modal_overlay_script(0)}
        <script>
        (function() {{
            var REFRESH_MS = 300000;
            function refreshDashboard() {{
                fetch('/api/dashboard/home')
                    .then(function(r) {{ return r.json(); }})
                    .then(function(data) {{
                        var ts = document.getElementById('dashboardTimestamp');
                        var grid = document.getElementById('dashboardProductGrid');
                        if (ts) ts.textContent = 'Ultima atualizacao: ' + data.timestamp;
                        if (grid) grid.innerHTML = data.cards_html;
                    }})
                    .catch(function() {{}});
            }}
            setInterval(refreshDashboard, REFRESH_MS);
        }})();
        </script>
    </body>
    </html>
    """
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Dashboard - Empiricus Crypto</title>
        <style>
            .page-loader {{
                position: fixed; inset: 0; z-index: 9999;
                background: #1a1a2e;
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                transition: opacity 0.3s;
            }}
            .page-loader.hide {{ opacity: 0; pointer-events: none; }}
            .loader-spinner {{
                width: 36px; height: 36px;
                border: 3px solid rgba(78,204,163,0.15);
                border-top-color: #39fda3;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }}
            .loader-text {{
                margin-top: 14px;
                color: #5a6a7a;
                font-family: 'Segoe UI', sans-serif;
                font-size: 0.85em;
            }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
            .page-content {{ opacity: 0; transition: opacity 0.3s; }}
            .page-content.show {{ opacity: 1; }}
        </style>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        <div class="page-loader" id="pageLoader">
            <div class="loader-spinner"></div>
            <div class="loader-text">Carregando...</div>
        </div>
        <div class="page-content" id="pageContent">
            {get_navbar('home')}
            <div class="container">
                <div class="dashboard-header">
                    <p class="timestamp" id="dashboardTimestamp">Ultima atualizacao: {timestamp}</p>
                    <a href="/turmas" class="btn-turmas">Ver Turmas</a>
                </div>
                <div class="product-grid" id="dashboardProductGrid">{cards_html}</div>
            </div>
        </div>
        {product_nav_loader_html}
        {_get_form_modal_overlay_script(0)}
        <script>
            document.getElementById('pageLoader').classList.add('hide');
            document.getElementById('pageContent').classList.add('show');
            setTimeout(function() {{ document.getElementById('pageLoader').remove(); }}, 400);
            (function() {{
                var REFRESH_MS = 300000;
                function refreshDashboard() {{
                    fetch('/api/dashboard/home')
                        .then(function(r) {{ return r.json(); }})
                        .then(function(data) {{
                            var ts = document.getElementById('dashboardTimestamp');
                            var grid = document.getElementById('dashboardProductGrid');
                            if (ts) ts.textContent = 'Ultima atualizacao: ' + data.timestamp;
                            if (grid) grid.innerHTML = data.cards_html;
                        }})
                        .catch(function() {{}});
                }}
                setInterval(refreshDashboard, REFRESH_MS);
            }})();
        </script>
    </body>
    </html>
    """


def get_menu_html():
    """Gera HTML da pagina de menu completo"""
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Menu - Products & Positions</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar('menu')}
        <div class="container">
            <h2 style="color: #39fda3; margin-bottom: 30px;">Menu de Operacoes</h2>

            <div class="menu-section">
                <h3>Produtos</h3>
                <div class="menu-grid">
                    <a href="/produto/novo" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Criar Produto</div>
                    </a>
                    <a href="/produtos/editar" class="menu-item">
                        <div class="menu-item-icon">E</div>
                        <div class="menu-item-label">Editar Produto</div>
                    </a>
                    <a href="/produtos/deletar" class="menu-item">
                        <div class="menu-item-icon">X</div>
                        <div class="menu-item-label">Deletar Produto</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Posicoes</h3>
                <div class="menu-grid">
                    <a href="/posicao/nova" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Criar Posicao</div>
                    </a>
                    <a href="/posicoes/editar" class="menu-item">
                        <div class="menu-item-icon">E</div>
                        <div class="menu-item-label">Editar Posicao</div>
                    </a>
                    <a href="/posicoes/deletar" class="menu-item">
                        <div class="menu-item-icon">X</div>
                        <div class="menu-item-label">Deletar Posicao</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Stops</h3>
                <div class="menu-grid">
                    <a href="/stop/novo" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Adicionar Stop</div>
                    </a>
                    <a href="/stops/atualizar-atr" class="menu-item">
                        <div class="menu-item-icon">A</div>
                        <div class="menu-item-label">Atualizar ATR</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Alocações</h3>
                <div class="menu-grid">
                    <a href="/alocacao/nova" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Criar Alocacao</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Turmas & Rentabilidade</h3>
                <div class="menu-grid">
                    <a href="/turmas" class="menu-item">
                        <div class="menu-item-icon">T</div>
                        <div class="menu-item-label">Ver Turmas</div>
                    </a>
                    <a href="/turmas/nova" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Nova Turma</div>
                    </a>
                    <a href="/turmas/comparar" class="menu-item">
                        <div class="menu-item-icon">C</div>
                        <div class="menu-item-label">Comparar</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Visualizacao</h3>
                <div class="menu-grid">
                    <a href="/" class="menu-item">
                        <div class="menu-item-icon">D</div>
                        <div class="menu-item-label">Dashboard</div>
                    </a>
                </div>
            </div>
        </div>
        {_get_form_modal_overlay_script(0)}
    </body>
    </html>
    """


def get_produto_html(produto, visualizacoes, repo):
    """Gera HTML da pagina de detalhes de um produto com visualizacoes salvas"""
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    produto_id = produto['id']
    nome = produto['nome']
    tipo = produto.get('tipo', 'Outro')

    # Se tiver visualizacoes, mostrar a primeira por padrao
    tabs_html = ""
    content_html = ""
    n_posicoes = 0
    viz_nome = ""

    if visualizacoes:
        for i, viz in enumerate(visualizacoes):
            active_class = 'active' if i == 0 else ''
            tabs_html += f'<a href="/produto/{produto_id}/viz/{viz["id"]}" class="tab {active_class}">{viz["nome"]}</a>'

        # Mostrar primeira visualizacao
        first_viz = visualizacoes[0]
        viz_nome = first_viz['nome']
        df = obter_dados_para_visualizacao(produto_id, first_viz)
        df_viz = aplicar_visualizacao(df, first_viz)

        if df_viz is not None and not df_viz.empty:
            df_viz = df_viz.fillna("—")
            content_html = f'<div class="table-container">{df_viz.to_html(index=False, classes="dataframe", escape=False)}</div>'
            n_posicoes = len(df_viz)
        else:
            content_html = '<div class="empty-state"><p>Nenhum dado encontrado</p></div>'
    else:
        # Sem visualizacoes - mostrar posicoes abertas padrao
        tabs_html = f'''
            <a href="/produto/{produto_id}/abertas" class="tab active">Posicoes Abertas</a>
            <a href="/produto/{produto_id}/fechadas" class="tab">Posicoes Fechadas</a>
        '''
        viz_nome = "Posições Abertas"
        df = display_posicoes_abertas(produto_id, formatar=True, filtrar_colunas=False)
        if df is not None and not df.empty:
            df = df.fillna("—")
            content_html = f'<div class="table-container">{df.to_html(index=False, classes="dataframe", escape=False)}</div>'
            n_posicoes = len(df)
        else:
            content_html = '<div class="empty-state"><p>Nenhuma posicao aberta</p></div>'

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>{nome} - Dashboard</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <p class="timestamp">Ultima atualizacao: {timestamp}</p>

            <div class="card" style="margin-bottom: 30px;">
                <h2>{nome}</h2>
                <span class="badge badge-{'spot' if 'spot' in tipo.lower() else 'perpetuos' if 'perp' in tipo.lower() else 'outro'}">{tipo}</span>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">Data Inicio</div>
                        <div class="stat-value">{produto.get('data_inicio', 'N/A')}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Posicoes</div>
                        <div class="stat-value">{n_posicoes}</div>
                    </div>
                </div>
                <div class="actions">
                    <a href="/produto/{produto_id}/editar" class="btn btn-secondary">Editar Produto</a>
                    <a href="/produto/{produto_id}/visualizacoes" class="btn btn-secondary">Gerenciar Visualizacoes</a>
                    <a href="/produto/{produto_id}/atributos" class="btn btn-secondary">Gerenciar Atributos</a>
                </div>
            </div>

            <div class="tabs">{tabs_html}</div>

            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h2 style="margin: 0;">{viz_nome}</h2>
                    <div class="actions" style="margin: 0;">
                        <a href="/posicao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Posicao</a>
                        <a href="/alocacao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Alocacao</a>
                        <a href="/posicoes/editar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Editar Posicao</a>
                        <a href="/stop/novo?produto_id={produto_id}" class="btn btn-sm btn-secondary">+ Stop</a>
                        <a href="/atr/config?produto_id={produto_id}" class="btn btn-sm btn-secondary">ATR Stop</a>
                        <a href="/posicao/fechar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Fechar Posicao</a>
                        <a href="/posicoes/deletar?produto_id={produto_id}" class="btn btn-sm btn-danger">Deletar Posicao</a>
                        {_get_preencher_precos_btn(produto_id)}
                    </div>
                </div>
                {content_html}
        </div>
    </div>

    {_get_form_modal_overlay_script(produto_id)}
    {_get_preencher_precos_js()}
    </body>
    </html>
    """


def get_visualizacao_html(produto, visualizacao, df_viz):
    """Gera HTML para uma visualizacao especifica"""
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    produto_id = produto['id']
    nome_produto = produto['nome']
    nome_viz = visualizacao['nome']
    tipo = produto.get('tipo', 'Outro')

    repo = get_repo()
    visualizacoes = repo.listar_visualizacoes(produto_id)

    tabs_html = ""
    for viz in visualizacoes:
        active_class = 'active' if viz['id'] == visualizacao['id'] else ''
        tabs_html += f'<a href="/produto/{produto_id}/viz/{viz["id"]}" class="tab {active_class}">{viz["nome"]}</a>'

    if df_viz is not None and not df_viz.empty:
        df_viz = df_viz.fillna("—")
        content_html = f'<div class="table-container">{df_viz.to_html(index=False, classes="dataframe", escape=False)}</div>'
        n_posicoes = len(df_viz)
    else:
        content_html = '<div class="empty-state"><p>Nenhum dado encontrado para esta visualizacao</p></div>'
        n_posicoes = 0

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>{nome_viz} - {nome_produto}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <p class="timestamp">Ultima atualizacao: {timestamp}</p>

            <div class="card" style="margin-bottom: 20px;">
                <h2>{nome_produto}</h2>
                <span class="badge badge-{'spot' if 'spot' in tipo.lower() else 'perpetuos' if 'perp' in tipo.lower() else 'outro'}">{tipo}</span>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">Data Inicio</div>
                        <div class="stat-value">{produto.get('data_inicio', 'N/A')}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Posicoes</div>
                        <div class="stat-value">{n_posicoes}</div>
                    </div>
                </div>
                <div class="actions">
                    <a href="/produto/{produto_id}/editar" class="btn btn-secondary">Editar Produto</a>
                    <a href="/produto/{produto_id}/visualizacoes" class="btn btn-secondary">Gerenciar Visualizacoes</a>
                    <a href="/produto/{produto_id}/atributos" class="btn btn-secondary">Gerenciar Atributos</a>
                </div>
            </div>

            <div class="tabs">{tabs_html}</div>

            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h2 style="margin: 0;">{nome_viz}</h2>
                    <div class="actions" style="margin: 0;">
                        <a href="/posicao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Posicao</a>
                        <a href="/alocacao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Alocacao</a>
                        <a href="/posicoes/editar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Editar Posicao</a>
                        <a href="/stop/novo?produto_id={produto_id}" class="btn btn-sm btn-secondary">+ Stop</a>
                        <a href="/atr/config?produto_id={produto_id}" class="btn btn-sm btn-secondary">ATR Stop</a>
                        <a href="/posicao/fechar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Fechar Posicao</a>
                        <a href="/posicoes/deletar?produto_id={produto_id}" class="btn btn-sm btn-danger">Deletar Posicao</a>
                        {_get_preencher_precos_btn(produto_id)}
                    </div>
                </div>
                {content_html}
        </div>
    </div>

    {_get_form_modal_overlay_script(produto_id)}
    {_get_preencher_precos_js()}
    </body>
    </html>
    """


def get_form_produto_html(produto=None, as_inner=False):
    """Gera formulario para criar/editar produto. Se as_inner=True (apenas em edicao), retorna só o card (para overlay)."""
    is_edit = produto is not None
    titulo = "Editar Produto" if is_edit else "Novo Produto"
    action = f"/api/produto/{produto['id']}/editar" if is_edit else "/api/produto/criar"
    cancel_url = f"/produto/{produto['id']}" if is_edit else "/"

    card_html = f"""
            <div class="card" style="max-width: 600px;">
                <h2>{titulo}</h2>
                <div id="alert" class="alert"></div>
                <form id="produtoForm">
                    <div class="form-group">
                        <label>Nome do Produto</label>
                        <input type="text" name="nome" required value="{produto.get('nome', '') if is_edit else ''}">
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Tipo</label>
                            <select name="tipo">
                                <option value="Spot" {'selected' if is_edit and produto.get('tipo') == 'Spot' else ''}>Spot</option>
                                <option value="Perpetuos" {'selected' if is_edit and produto.get('tipo') == 'Perpetuos' else ''}>Perpetuos</option>
                                <option value="Outro" {'selected' if is_edit and produto.get('tipo') not in ['Spot', 'Perpetuos'] else ''}>Outro</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Capital Inicial (USD)</label>
                            <input type="number" name="capital_inicial" step="0.01" value="{produto.get('capital_inicial', '') if is_edit else ''}">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Data de Inicio</label>
                        <input type="date" name="data_inicio" value="{produto.get('data_inicio', '') if is_edit else date.today().isoformat()}">
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">{'Salvar' if is_edit else 'Criar'}</button>
                        <a href="{cancel_url}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        """
    script = f"""
        <script>
            document.getElementById('produtoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('{action}', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Produto salvo com sucesso!';
                        setTimeout(() => window.location.href = '/', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    """
    if as_inner:
        return card_html + script
    if is_edit:
        return get_form_page_with_background(card_html + script, produto['id'], 'Editar Produto')
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>{titulo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            {card_html}
        </div>
        {script}
    </body>
    </html>
    """


def _get_campos_extras_nova_posicao(produto):
    """Retorna HTML dos campos extras para Nova Posição conforme o produto."""
    if not produto:
        return ""
    if _is_produto_crypto_signals(produto):
        return """
                    <hr style="margin: 12px 0; border-color: rgba(78,204,163,0.2);">
                    <p style="color: #39fda3; font-size: 0.85em; margin-bottom: 8px;">Crypto Signals</p>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Perfil</label>
                            <select name="perfil">
                                <option value="">Selecione...</option>
                                <option value="Arrojado">Arrojado</option>
                                <option value="Moderado">Moderado</option>
                                <option value="Conservador">Conservador</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Alvo 1 (USD)</label>
                            <input type="number" name="alvo1" step="0.00000001" placeholder="Preço alvo 1">
                        </div>
                        <div class="form-group">
                            <label>Alvo 2 (USD)</label>
                            <input type="number" name="alvo2" step="0.00000001" placeholder="Preço alvo 2">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Relatório (URL)</label>
                            <input type="url" name="relatorio" placeholder="https://...">
                        </div>
                        <div class="form-group">
                            <label>Motivo (opcional)</label>
                            <input type="text" name="motivo" placeholder="Ex: Breakout, reversão">
                        </div>
                    </div>"""
    if _is_produto_icos(produto):
        return """
                    <hr style="margin: 12px 0; border-color: rgba(78,204,163,0.2);">
                    <p style="color: #39fda3; font-size: 0.85em; margin-bottom: 8px;">ICOs</p>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Categoria</label>
                            <input type="text" name="categoria" placeholder="DeFi, Gaming">
                        </div>
                        <div class="form-group">
                            <label>Tipo ICO</label>
                            <input type="text" name="tipo_ico" placeholder="TGE, IDO">
                        </div>
                        <div class="form-group">
                            <label>Rank</label>
                            <input type="text" name="rank" placeholder="1, 2, 3">
                        </div>
                    </div>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Tipo Janela</label>
                            <input type="text" name="tipo_janela" placeholder="Janela">
                        </div>
                        <div class="form-group">
                            <label>Tese</label>
                            <input type="text" name="tese" placeholder="Tese">
                        </div>
                        <div class="form-group">
                            <label>Risco</label>
                            <input type="text" name="risco" placeholder="Risco">
                        </div>
                    </div>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Atenção</label>
                            <input type="text" name="atencao" placeholder="Atenção">
                        </div>
                        <div class="form-group">
                            <label>Execução</label>
                            <input type="text" name="execucao" placeholder="Execução">
                        </div>
                        <div class="form-group">
                            <label>Por quê</label>
                            <input type="text" name="por_que" placeholder="Justificativa">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Local (URL)</label>
                            <input type="url" name="local" placeholder="https://...">
                        </div>
                        <div class="form-group">
                            <label>Ficha Técnica (URL)</label>
                            <input type="url" name="ficha_tecnica" placeholder="https://...">
                        </div>
                    </div>"""
    return ""


def get_form_posicao_html(produto_id=None, produtos=None, produto=None, as_inner=False):
    """Gera formulario para criar posicao. Se as_inner=True, retorna só o card (para overlay)."""
    produtos_options = ""
    if produtos:
        for p in produtos:
            selected = 'selected' if produto_id and p['id'] == produto_id else ''
            produtos_options += f'<option value="{p["id"]}" {selected}>{p["nome"]}</option>'

    # Resolver produto para campos extras
    if produto is None and produto_id and produtos:
        produto = next((p for p in produtos if p.get('id') == produto_id), None)
    campos_extras = _get_campos_extras_nova_posicao(produto)

    cancel_url = f"/produto/{produto_id}" if produto_id else "/"
    card_html = f"""
            <div class="card form-card form-compact" style="max-width: 1200px;">
                <h2>Nova Posicao</h2>
                <div id="alert" class="alert"></div>
                <form id="posicaoForm" class="form-compact">
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Produto</label>
                            <select name="produto_id" required>{produtos_options}</select>
                        </div>
                        <div class="form-group">
                            <label>Ativo</label>
                            <input type="text" name="ativo" required placeholder="BTC">
                        </div>
                        <div class="form-group">
                            <label>CoinGecko ID</label>
                            <input type="text" name="coingecko_id" placeholder="bitcoin">
                        </div>
                    </div>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Exchange Symbol</label>
                            <input type="text" name="exchange_symbol" placeholder="BTCUSDT">
                        </div>
                        <div class="form-group">
                            <label>Side</label>
                            <select name="side">
                                <option value="long">Long</option>
                                <option value="short">Short</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Data de Entrada</label>
                            <input type="date" name="data_entrada" value="{date.today().isoformat()}">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Preco de Entrada (USD)</label>
                            <input type="number" name="preco_entrada" step="0.00000001" required>
                        </div>
                        <div class="form-group">
                            <label>Quantidade</label>
                            <input type="number" name="quantidade" step="0.00000001" placeholder="Opcional">
                        </div>
                    </div>
                    {campos_extras}
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Criar Posicao</button>
                        <a href="{cancel_url}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>"""
    script = f"""
        <script>
            document.getElementById('posicaoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/posicao/criar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Posicao criada!';
                        setTimeout(() => window.location.href = '/produto/' + obj.produto_id, 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>"""
    if as_inner:
        return card_html + script
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Nova Posicao</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container form-page">
            {card_html}
        </div>
        {script}
    </body>
    </html>
    """


def get_lista_produtos_html(produtos, acao="editar"):
    """Lista produtos para selecao (editar/deletar)"""
    titulo = "Editar Produto" if acao == "editar" else "Deletar Produto"

    items_html = ""
    for p in produtos:
        if acao == "editar":
            link = f"/produto/{p['id']}/editar"
            btn_class = "btn-primary"
            btn_text = "Editar"
        else:
            link = f"/produto/{p['id']}/deletar"
            btn_class = "btn-danger"
            btn_text = "Deletar"

        items_html += f"""
        <div class="viz-item">
            <div>
                <strong>{p['nome']}</strong>
                <span class="badge badge-outro" style="margin-left: 10px;">{p.get('tipo', 'Outro')}</span>
            </div>
            <a href="{link}" class="btn btn-sm {btn_class}">{btn_text}</a>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>{titulo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>{titulo}</h2>
                <p style="color: #888; margin-bottom: 20px;">Selecione um produto:</p>
                <div class="viz-list">{items_html}</div>
                <div class="actions" style="margin-top: 20px;">
                    <a href="/" class="btn btn-secondary">Voltar</a>
                </div>
            </div>
        </div>
        {_get_form_modal_overlay_script(0)}
    </body>
    </html>
    """


def get_confirmar_delete_html(produto, as_inner=False):
    """Pagina de confirmacao de exclusao. Se as_inner=True, retorna só o card (para overlay)."""
    card_html = f"""
            <div class="card" style="max-width: 500px; text-align: center;">
                <h2 style="color: #ff6b6b;">Confirmar Exclusao</h2>
                <p style="margin: 20px 0;">Tem certeza que deseja deletar o produto:</p>
                <p style="font-size: 1.3em; color: #39fda3; font-weight: bold;">{produto['nome']}</p>
                <p style="color: #ff6b6b; margin: 20px 0;">Esta acao ira deletar todas as posicoes, stops e visualizacoes associadas!</p>
                <div id="alert" class="alert"></div>
                <div class="actions" style="justify-content: center;">
                    <button class="btn btn-danger" onclick="deletarProduto()">Sim, Deletar</button>
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                </div>
            </div>
        """
    script = f"""
        <script>
            async function deletarProduto() {{
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/produto/{produto["id"]}/deletar', {{
                        method: 'POST'
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Produto deletado!';
                        setTimeout(() => window.location.href = '/', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }}
        </script>
    """
    if as_inner:
        return card_html + script
    return get_form_page_with_background(card_html + script, produto['id'], 'Confirmar Exclusao')


def get_lista_posicoes_html(produto, posicoes, acao="editar", as_inner=False):
    """Lista posicoes de um produto para selecao. Se as_inner=True, retorna só o card (para overlay)."""
    titulo_map = {
        "editar": "Editar Posicao",
        "stop": "Adicionar Stop",
        "fechar": "Fechar Posicao",
        "atr": "Configurar ATR Stop",
        "deletar": "Deletar Posicao"
    }
    titulo = titulo_map.get(acao, "Selecionar Posicao")

    items_html = ""
    is_spot = 'spot' in produto.get('tipo', '').lower()

    if posicoes is not None and not posicoes.empty:
        for _, pos in posicoes.iterrows():
            pos_id = pos.get('id') or pos.get('ID')
            ativo = pos.get('ativo') or pos.get('Ativo', 'N/A')
            side = pos.get('side') or pos.get('tipo') or pos.get('Tipo', 'N/A')
            preco = pos.get('preco_entrada') or pos.get('Preço Entrada', 0)

            if acao == "editar":
                link = f"/posicao/{pos_id}/editar?produto_id={produto['id']}"
                btn_class = "btn-primary"
                btn_text = "Editar"
            elif acao == "stop":
                link = f"/posicao/{pos_id}/stop?produto_id={produto['id']}"
                btn_class = "btn-warning"
                btn_text = "+ Stop"
            elif acao == "atr":
                link = f"/posicao/{pos_id}/atr?produto_id={produto['id']}"
                btn_class = "btn-secondary"
                btn_text = "Configurar"
            elif acao == "deletar":
                link = f"/posicao/{pos_id}/deletar?produto_id={produto['id']}"
                btn_class = "btn-danger"
                btn_text = "Deletar"
            else:  # fechar
                link = f"/posicao/{pos_id}/fechar?produto_id={produto['id']}"
                btn_class = "btn-danger"
                btn_text = "Fechar"

            # Só mostra badge de side se não for spot
            if is_spot:
                side_badge_html = ""
            else:
                tipo_badge = 'success' if side.lower() == 'long' else 'danger'
                side_badge_html = f'<span class="badge badge-{tipo_badge}" style="margin-left: 10px;">{side}</span>'

            items_html += f"""
            <div class="viz-item">
                <div>
                    <strong>{ativo}</strong>
                    {side_badge_html}
                    <span style="margin-left: 10px; color: #888;">${preco:,.4f}</span>
                </div>
                <a href="{link}" class="btn btn-sm {btn_class}">{btn_text}</a>
            </div>
            """
    else:
        items_html = '<p style="text-align: center; color: #888;">Nenhuma posicao aberta</p>'

    card_html = f"""
            <div class="card" style="max-width: 600px;">
                <h2>{titulo}</h2>
                <p style="color: #39fda3; margin-bottom: 20px;">{produto['nome']}</p>
                <div class="viz-list">{items_html}</div>
                <div class="actions" style="margin-top: 20px;">
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Voltar</a>
                </div>
            </div>"""
    if as_inner:
        return card_html
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>{titulo} - {produto['nome']}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            {card_html}
        </div>
    </body>
    </html>
    """


def get_form_adicionar_stop_html(produto, posicao, as_inner=False):
    """Formulario para adicionar stop. Se as_inner=True, retorna só o card (para overlay)."""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    side = posicao.get('side') or posicao.get('tipo') or posicao.get('Tipo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    is_spot = 'spot' in produto.get('tipo', '').lower()
    side_info = "" if is_spot else f" ({side})"

    card_html = f"""
            <div class="card form-card form-compact" style="max-width: 600px;">
                <h2>Adicionar Stop</h2>
                <p style="color: #39fda3; margin-bottom: 12px; font-size: 0.9em;">
                    {ativo}{side_info} - Entrada: ${preco_entrada:,.4f}
                </p>
                <div id="alert" class="alert"></div>
                <form id="stopForm" class="form-compact">
                    <input type="hidden" name="posicao_id" value="{pos_id}">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Preco do Stop</label>
                            <input type="number" name="preco" step="0.00000001" required>
                        </div>
                        <div class="form-group">
                            <label>Data do Stop</label>
                            <input type="date" name="data_stop" value="{date.today().isoformat()}">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Motivo (opcional)</label>
                        <input type="text" name="motivo" placeholder="Ex: Stop loss tecnico">
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Adicionar Stop</button>
                        <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>"""
    script = f"""
        <script>
            document.getElementById('stopForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/stop/criar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Stop adicionado!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>"""
    if as_inner:
        return card_html + script
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Adicionar Stop - {ativo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container form-page">
            {card_html}
        </div>
        {script}
    </body>
    </html>
    """


def get_form_fechar_posicao_html(produto, posicao, as_inner=False):
    """Formulario para fechar uma posicao. Se as_inner=True, retorna só o card (para overlay)."""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    side = posicao.get('side') or posicao.get('tipo') or posicao.get('Tipo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    is_spot = 'spot' in produto.get('tipo', '').lower()
    side_info = "" if is_spot else f" ({side})"

    # Campos extras para ICOs (Resultado)
    resultado_field = ""
    if _is_produto_icos(produto):
        resultado_field = """
                    <div class="form-group">
                        <label>Resultado</label>
                        <select name="resultado">
                            <option value="">Selecione...</option>
                            <option value="Sucesso">Sucesso</option>
                            <option value="Não ocorreu">Não ocorreu</option>
                        </select>
                    </div>"""

    card_html = f"""
            <div class="card form-card form-compact" style="max-width: 600px;">
                <h2>Fechar Posicao</h2>
                <p style="color: #39fda3; margin-bottom: 12px; font-size: 0.9em;">
                    {ativo}{side_info} - Entrada: ${preco_entrada:,.4f}
                </p>
                <div id="alert" class="alert"></div>
                <form id="fecharForm" class="form-compact">
                    <input type="hidden" name="posicao_id" value="{pos_id}">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Preco de Saida</label>
                            <input type="number" name="preco_saida" step="0.00000001" required>
                        </div>
                        <div class="form-group">
                            <label>Data de Saida</label>
                            <input type="date" name="data_saida" value="{date.today().isoformat()}">
                        </div>
                    </div>
                    {resultado_field}
                    <div class="form-group">
                        <label>Motivo (opcional)</label>
                        <input type="text" name="motivo" placeholder="Ex: Stop atingido, Take profit">
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-danger">Fechar Posicao</button>
                        <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        """
    script = f"""
        <script>
            document.getElementById('fecharForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/posicao/fechar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Posicao fechada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    """
    if as_inner:
        return card_html + script
    return get_form_page_with_background(card_html + script, produto['id'], 'Fechar Posicao')


def _get_campos_extras_editar_posicao(produto, posicao):
    """Retorna HTML dos campos extras para Editar Posição conforme o produto."""
    def _v(k, default=''):
        v = posicao.get(k) or posicao.get(k.replace('_', ' ').title(),
                         posicao.get(k.replace('_', ' ').title().replace(' ', '_')))
        return (str(v) if v is not None else '').replace('"', '&quot;') or default

    if not produto:
        return ""
    if _is_produto_crypto_signals(produto):
        perfil = _v('perfil')
        alvo1 = _v('alvo1')
        alvo2 = _v('alvo2')
        motivo = _v('motivo')
        relatorio = _v('relatorio')
        return f"""
                    <hr style="margin: 12px 0; border-color: rgba(78,204,163,0.2);">
                    <p style="color: #39fda3; font-size: 0.85em; margin-bottom: 8px;">Crypto Signals</p>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Perfil</label>
                            <select name="perfil">
                                <option value="">Selecione...</option>
                                <option value="Arrojado" {'selected' if perfil == 'Arrojado' else ''}>Arrojado</option>
                                <option value="Moderado" {'selected' if perfil == 'Moderado' else ''}>Moderado</option>
                                <option value="Conservador" {'selected' if perfil == 'Conservador' else ''}>Conservador</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Alvo 1 (USD)</label>
                            <input type="number" name="alvo1" step="0.00000001" value="{alvo1}" placeholder="Alvo 1">
                        </div>
                        <div class="form-group">
                            <label>Alvo 2 (USD)</label>
                            <input type="number" name="alvo2" step="0.00000001" value="{alvo2}" placeholder="Alvo 2">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Relatório (URL)</label>
                            <input type="url" name="relatorio" value="{relatorio}" placeholder="https://...">
                        </div>
                        <div class="form-group">
                            <label>Motivo</label>
                            <input type="text" name="motivo" value="{motivo}" placeholder="Breakout, reversão">
                        </div>
                    </div>"""
    if _is_produto_icos(produto):
        return f"""
                    <hr style="margin: 12px 0; border-color: rgba(78,204,163,0.2);">
                    <p style="color: #39fda3; font-size: 0.85em; margin-bottom: 8px;">ICOs</p>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Categoria</label>
                            <input type="text" name="categoria" value="{_v('categoria')}" placeholder="DeFi, Gaming">
                        </div>
                        <div class="form-group">
                            <label>Tipo ICO</label>
                            <input type="text" name="tipo_ico" value="{_v('tipo_ico')}" placeholder="TGE, IDO">
                        </div>
                        <div class="form-group">
                            <label>Rank</label>
                            <input type="text" name="rank" value="{_v('rank')}" placeholder="1, 2, 3">
                        </div>
                    </div>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Tipo Janela</label>
                            <input type="text" name="tipo_janela" value="{_v('tipo_janela')}" placeholder="Janela">
                        </div>
                        <div class="form-group">
                            <label>Tese</label>
                            <input type="text" name="tese" value="{_v('tese')}" placeholder="Tese">
                        </div>
                        <div class="form-group">
                            <label>Risco</label>
                            <input type="text" name="risco" value="{_v('risco')}" placeholder="Risco">
                        </div>
                    </div>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Atenção</label>
                            <input type="text" name="atencao" value="{_v('atencao')}" placeholder="Atenção">
                        </div>
                        <div class="form-group">
                            <label>Execução</label>
                            <input type="text" name="execucao" value="{_v('execucao')}" placeholder="Execução">
                        </div>
                        <div class="form-group">
                            <label>Por quê</label>
                            <input type="text" name="por_que" value="{_v('por_que')}" placeholder="Justificativa">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Local (URL)</label>
                            <input type="url" name="local" value="{_v('local')}" placeholder="https://...">
                        </div>
                        <div class="form-group">
                            <label>Ficha Técnica (URL)</label>
                            <input type="url" name="ficha_tecnica" value="{_v('ficha_tecnica')}" placeholder="https://...">
                        </div>
                    </div>"""
    return ""


def get_form_editar_posicao_html(produto, posicao, as_inner=False):
    """Formulario para editar uma posicao existente. Se as_inner=True, retorna só o card (para overlay)."""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', '')
    coingecko_id = posicao.get('coingecko_id') or posicao.get('CoinGecko ID', '')
    exchange_symbol = posicao.get('exchange_symbol') or posicao.get('Exchange Symbol', '')
    side = posicao.get('side') or posicao.get('tipo') or posicao.get('Tipo', 'long')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    quantidade = posicao.get('quantidade') or posicao.get('Quantidade', '')
    data_entrada = _normalizar_data_entrada(
        posicao.get('data_entrada') or posicao.get('Data Entrada'),
        fallback_today=True
    )
    is_spot = 'spot' in produto.get('tipo', '').lower()
    campos_extras = _get_campos_extras_editar_posicao(produto, posicao)

    # Para spot, não mostra seletor de tipo
    if is_spot:
        tipo_field_html = f"""
                        <div class="form-group">
                            <label>Data de Entrada</label>
                            <input type="date" name="data_entrada" value="{data_entrada}">
                        </div>
                        <input type="hidden" name="tipo" value="long">"""
    else:
        tipo_field_html = f"""
                        <div class="form-group">
                            <label>Tipo</label>
                            <select name="tipo" required>
                                <option value="long" {'selected' if side.lower() == 'long' else ''}>Long</option>
                                <option value="short" {'selected' if side.lower() == 'short' else ''}>Short</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Data de Entrada</label>
                            <input type="date" name="data_entrada" value="{data_entrada}">
                        </div>"""
    # Editar: linha com tipo/data + preco + quantidade
    base_row_class = "form-row-4" if not is_spot else "form-row-3"

    card_html = f"""
            <div class="card form-card form-compact" style="max-width: 1200px;">
                <h2>Editar Posicao</h2>
                <p style="color: #39fda3; margin-bottom: 12px; font-size: 0.9em;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <form id="editarPosicaoForm" class="form-compact">
                    <input type="hidden" name="posicao_id" value="{pos_id}">
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Ativo</label>
                            <input type="text" name="ativo" value="{ativo}" required>
                        </div>
                        <div class="form-group">
                            <label>CoinGecko ID</label>
                            <input type="text" name="coingecko_id" value="{coingecko_id or ''}">
                        </div>
                        <div class="form-group">
                            <label>Exchange Symbol</label>
                            <input type="text" name="exchange_symbol" value="{exchange_symbol or ''}" placeholder="BTCUSDT">
                        </div>
                    </div>
                    <div class="{base_row_class}">
                        {tipo_field_html}
                        <div class="form-group">
                            <label>Preco de Entrada</label>
                            <input type="number" name="preco_entrada" step="0.00000001" value="{preco_entrada}" required>
                        </div>
                        <div class="form-group">
                            <label>Quantidade</label>
                            <input type="number" name="quantidade" step="0.00000001" value="{quantidade or ''}">
                        </div>
                    </div>
                    {campos_extras}
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Salvar</button>
                        <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>"""
    script_editar = f"""
        <script>
            document.getElementById('editarPosicaoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/posicao/editar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Posicao atualizada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>"""
    if as_inner:
        return card_html + script_editar
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Editar Posicao - {ativo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container form-page">
            {card_html}
        </div>
        {script_editar}
    </body>
    </html>
    """


def get_form_atr_stop_html(produto, posicao, as_inner=False):
    """Formulario para configurar ATR Trailing Stop. Se as_inner=True, retorna só o card (para overlay)."""
    from datetime import date as dt_date
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    data_entrada = _normalizar_data_entrada(posicao.get('data_entrada') or posicao.get('Data Entrada'), fallback_today=False) or '—'
    current_period = posicao.get('atr_period') or 14
    current_mult = posicao.get('atr_multiplier') or 3.0
    current_data_inicio = posicao.get('atr_data_inicio') or ''
    # Se não tiver data de início, usar ontem como padrão
    if not current_data_inicio:
        from datetime import timedelta
        current_data_inicio = (dt_date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    card_html = f"""
            <div class="card form-card form-compact" style="max-width: 600px;">
                <h2>Configurar ATR Trailing Stop</h2>
                <p style="color: #39fda3; margin-bottom: 8px; font-size: 0.9em;">
                    {ativo} - Entrada: ${preco_entrada:,.4f} (em {data_entrada})
                </p>
                <p style="color: #888; font-size: 0.8em; margin-bottom: 12px;">
                    O ATR calcula o stop pela volatilidade. Use "Data de Início" para posições antigas.
                </p>
                <div id="alert" class="alert"></div>
                <form id="atrForm" class="form-compact">
                    <input type="hidden" name="posicao_id" value="{pos_id}">
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>ATR Period (dias)</label>
                            <input type="number" name="atr_period" value="{current_period}" min="1" max="100" required>
                        </div>
                        <div class="form-group">
                            <label>ATR Multiplier</label>
                            <input type="number" name="atr_multiplier" value="{current_mult}" step="0.1" min="0.5" max="10" required>
                        </div>
                        <div class="form-group">
                            <label>Data de Início</label>
                            <input type="date" name="atr_data_inicio" value="{current_data_inicio}" required>
                        </div>
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Salvar Configuracao</button>
                        <button type="button" class="btn btn-danger" onclick="removerATR()">Remover ATR</button>
                        <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        """
    script = f"""
        <script>
            document.getElementById('atrForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/posicao/atr', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = result.mensagem || 'ATR configurado!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});

            async function removerATR() {{
                const alert = document.getElementById('alert');
                if (!confirm('Remover configuracao de ATR desta posicao?')) return;

                try {{
                    const response = await fetch('/api/posicao/atr', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            posicao_id: {pos_id},
                            atr_period: null,
                            atr_multiplier: null
                        }})
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'ATR removido!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }}
        </script>
    """
    if as_inner:
        return card_html + script
    return get_form_page_with_background(card_html + script, produto['id'], 'ATR Stop')


# ============================================================
# VISUALIZACOES
# ============================================================

def get_lista_visualizacoes_html(produto, visualizacoes, as_inner=False):
    """Lista visualizacoes de um produto para gerenciamento. Se as_inner=True, retorna só o card (para overlay)."""
    items_html = ""

    if visualizacoes:
        for viz in visualizacoes:
            colunas_preview = ', '.join(viz.get('colunas', [])[:4])
            if len(viz.get('colunas', [])) > 4:
                colunas_preview += '...'

            ordenacao_info = ""
            if viz.get('ordenacao') and viz['ordenacao'].get('coluna'):
                ordenacao_info = f" | Ordenado por: {viz['ordenacao']['coluna']}"

            items_html += f"""
            <div class="viz-item">
                <div style="flex: 1;">
                    <strong>{viz['nome']}</strong>
                    <p style="color: #888; font-size: 0.85em; margin-top: 5px;">
                        Colunas: {colunas_preview}{ordenacao_info}
                    </p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <a href="/produto/{produto['id']}/viz/{viz['id']}/editar" class="btn btn-sm btn-primary">Editar</a>
                    <a href="/produto/{produto['id']}/viz/{viz['id']}/deletar" class="btn btn-sm btn-danger">Deletar</a>
                </div>
            </div>
            """
    else:
        items_html = '<p style="text-align: center; color: #888; padding: 20px;">Nenhuma visualizacao cadastrada</p>'

    card_html = f"""
            <div class="card" style="max-width: 700px;">
                <h2>Gerenciar Visualizacoes</h2>
                <p style="color: #39fda3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <div class="viz-list">{items_html}</div>
                <div class="actions" style="margin-top: 20px;">
                    <a href="/produto/{produto['id']}/viz/nova" class="btn btn-primary">+ Nova Visualizacao</a>
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Voltar</a>
                </div>
            </div>
        """
    if as_inner:
        return card_html
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Visualizacoes - {produto['nome']}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            {card_html}
        </div>
    </body>
    </html>
    """


def get_form_nova_visualizacao_html(produto, colunas_disponiveis, as_inner=False):
    """Formulario para criar nova visualizacao. Se as_inner=True, retorna só o card (para overlay)."""
    colunas_html = ""
    for i, col in enumerate(colunas_disponiveis):
        origem_badge = f'<span class="badge badge-outro" style="font-size: 0.7em;">{col["origem"]}</span>'
        colunas_html += f"""
        <label style="display: flex; align-items: center; gap: 10px; padding: 8px; background: #16213e; border-radius: 5px; cursor: pointer;">
            <input type="checkbox" name="colunas" value="{col['nome']}" style="width: 18px; height: 18px;">
            <span>{col['label']}</span>
            <span style="color: #666; font-size: 0.85em;">({col['nome']})</span>
            {origem_badge}
        </label>
        """

    colunas_ordenacao = ""
    for col in colunas_disponiveis:
        colunas_ordenacao += f'<option value="{col["nome"]}">{col["label"]}</option>'

    extra_styles = """
            .colunas-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 8px; max-height: 400px; overflow-y: auto; padding: 10px; background: #1a1a2e; border-radius: 8px; border: 1px solid #2a2a4a; }
            .filtros-container { margin-top: 15px; }
            .filtro-item { display: flex; gap: 10px; margin-bottom: 10px; padding: 10px; background: #16213e; border-radius: 8px; }
            .filtro-item select, .filtro-item input { padding: 8px; border: 1px solid #2a2a4a; border-radius: 5px; background: #1a1a2e; color: #eee; }
    """
    card_html = f"""
            <div class="card" style="max-width: 800px;">
                <h2>Nova Visualizacao</h2>
                <p style="color: #39fda3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <form id="vizForm">
                    <div class="form-group">
                        <label>Nome da Visualizacao</label>
                        <input type="text" name="nome" required placeholder="Ex: Posicoes Abertas Long">
                    </div>

                    <div class="form-group">
                        <label>Colunas (selecione na ordem desejada)</label>
                        <div style="margin-bottom: 10px;">
                            <button type="button" class="btn btn-sm btn-secondary" onclick="selecionarTodas()">Selecionar Todas</button>
                            <button type="button" class="btn btn-sm btn-secondary" onclick="limparSelecao()">Limpar</button>
                        </div>
                        <div class="colunas-grid">
                            {colunas_html}
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Ordenar por (opcional)</label>
                            <select name="ordenacao_coluna">
                                <option value="">Sem ordenacao</option>
                                {colunas_ordenacao}
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Direcao</label>
                            <select name="ordenacao_direcao">
                                <option value="asc">Crescente</option>
                                <option value="desc">Decrescente</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Filtro de Status (opcional)</label>
                        <select name="filtro_status">
                            <option value="">Todas as posicoes</option>
                            <option value="open">Apenas abertas</option>
                            <option value="closed">Apenas fechadas</option>
                        </select>
                    </div>

                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Criar Visualizacao</button>
                        <a href="/produto/{produto['id']}/visualizacoes" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        """
    script = f"""
        <script>
            function selecionarTodas() {{
                document.querySelectorAll('input[name="colunas"]').forEach(cb => cb.checked = true);
            }}
            function limparSelecao() {{
                document.querySelectorAll('input[name="colunas"]').forEach(cb => cb.checked = false);
            }}

            document.getElementById('vizForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const alert = document.getElementById('alert');

                // Coletar colunas selecionadas na ordem
                const colunas = Array.from(form.querySelectorAll('input[name="colunas"]:checked'))
                    .map(cb => cb.value);

                if (colunas.length === 0) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Selecione pelo menos uma coluna';
                    return;
                }}

                const dados = {{
                    produto_id: {produto['id']},
                    nome: form.nome.value,
                    colunas: colunas
                }};

                // Ordenacao
                if (form.ordenacao_coluna.value) {{
                    dados.ordenacao = {{
                        coluna: form.ordenacao_coluna.value,
                        direcao: form.ordenacao_direcao.value
                    }};
                }}

                // Filtro de status
                if (form.filtro_status.value) {{
                    dados.filtros = [{{
                        coluna: 'status',
                        operador: '=',
                        valor: form.filtro_status.value
                    }}];
                }}

                try {{
                    const response = await fetch('/api/visualizacao/criar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(dados)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Visualizacao criada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/visualizacoes', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    """
    if as_inner:
        return f"<style>{extra_styles}</style>" + card_html + script
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Nova Visualizacao - {produto['nome']}</title>
        <style>{get_base_styles()}{extra_styles}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            {card_html}
        </div>
        {script}
    </body>
    </html>
    """


def get_form_editar_visualizacao_html(produto, visualizacao, colunas_disponiveis, as_inner=False):
    """Formulario para editar visualizacao existente. Se as_inner=True, retorna só o card (para overlay)."""
    colunas_selecionadas = visualizacao.get('colunas', [])

    colunas_html = ""
    for col in colunas_disponiveis:
        checked = 'checked' if col['nome'] in colunas_selecionadas else ''
        origem_badge = f'<span class="badge badge-outro" style="font-size: 0.7em;">{col["origem"]}</span>'
        colunas_html += f"""
        <label style="display: flex; align-items: center; gap: 10px; padding: 8px; background: #16213e; border-radius: 5px; cursor: pointer;">
            <input type="checkbox" name="colunas" value="{col['nome']}" {checked} style="width: 18px; height: 18px;">
            <span>{col['label']}</span>
            <span style="color: #666; font-size: 0.85em;">({col['nome']})</span>
            {origem_badge}
        </label>
        """

    ordenacao = visualizacao.get('ordenacao', {}) or {}
    ordenacao_coluna = ordenacao.get('coluna', '')
    ordenacao_direcao = ordenacao.get('direcao', 'asc')

    colunas_ordenacao = '<option value="">Sem ordenacao</option>'
    for col in colunas_disponiveis:
        selected = 'selected' if col['nome'] == ordenacao_coluna else ''
        colunas_ordenacao += f'<option value="{col["nome"]}" {selected}>{col["label"]}</option>'

    # Determinar filtro de status atual
    filtros = visualizacao.get('filtros', []) or []
    filtro_status = ''
    for f in filtros:
        if f.get('coluna') == 'status':
            filtro_status = f.get('valor', '')
            break

    extra_styles = """
            .colunas-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 8px; max-height: 400px; overflow-y: auto; padding: 10px; background: #1a1a2e; border-radius: 8px; border: 1px solid #2a2a4a; }
    """
    card_html = f"""
            <div class="card" style="max-width: 800px;">
                <h2>Editar Visualizacao</h2>
                <p style="color: #39fda3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <form id="vizForm">
                    <div class="form-group">
                        <label>Nome da Visualizacao</label>
                        <input type="text" name="nome" required value="{visualizacao['nome']}">
                    </div>

                    <div class="form-group">
                        <label>Colunas</label>
                        <div style="margin-bottom: 10px;">
                            <button type="button" class="btn btn-sm btn-secondary" onclick="selecionarTodas()">Selecionar Todas</button>
                            <button type="button" class="btn btn-sm btn-secondary" onclick="limparSelecao()">Limpar</button>
                        </div>
                        <div class="colunas-grid">
                            {colunas_html}
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Ordenar por</label>
                            <select name="ordenacao_coluna">
                                {colunas_ordenacao}
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Direcao</label>
                            <select name="ordenacao_direcao">
                                <option value="asc" {'selected' if ordenacao_direcao == 'asc' else ''}>Crescente</option>
                                <option value="desc" {'selected' if ordenacao_direcao == 'desc' else ''}>Decrescente</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Filtro de Status</label>
                        <select name="filtro_status">
                            <option value="" {'selected' if filtro_status == '' else ''}>Todas as posicoes</option>
                            <option value="open" {'selected' if filtro_status == 'open' else ''}>Apenas abertas</option>
                            <option value="closed" {'selected' if filtro_status == 'closed' else ''}>Apenas fechadas</option>
                        </select>
                    </div>

                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Salvar</button>
                        <a href="/produto/{produto['id']}/visualizacoes" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        """
    script = f"""
        <script>
            function selecionarTodas() {{
                document.querySelectorAll('input[name="colunas"]').forEach(cb => cb.checked = true);
            }}
            function limparSelecao() {{
                document.querySelectorAll('input[name="colunas"]').forEach(cb => cb.checked = false);
            }}

            document.getElementById('vizForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const alert = document.getElementById('alert');

                const colunas = Array.from(form.querySelectorAll('input[name="colunas"]:checked'))
                    .map(cb => cb.value);

                if (colunas.length === 0) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Selecione pelo menos uma coluna';
                    return;
                }}

                const dados = {{
                    visualizacao_id: {visualizacao['id']},
                    nome: form.nome.value,
                    colunas: colunas
                }};

                if (form.ordenacao_coluna.value) {{
                    dados.ordenacao = {{
                        coluna: form.ordenacao_coluna.value,
                        direcao: form.ordenacao_direcao.value
                    }};
                }} else {{
                    dados.ordenacao = null;
                }}

                if (form.filtro_status.value) {{
                    dados.filtros = [{{
                        coluna: 'status',
                        operador: '=',
                        valor: form.filtro_status.value
                    }}];
                }} else {{
                    dados.filtros = null;
                }}

                try {{
                    const response = await fetch('/api/visualizacao/editar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(dados)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Visualizacao atualizada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/visualizacoes', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    """
    if as_inner:
        return f"<style>{extra_styles}</style>" + card_html + script
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Editar Visualizacao - {visualizacao['nome']}</title>
        <style>{get_base_styles()}{extra_styles}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            {card_html}
        </div>
        {script}
    </body>
    </html>
    """


def get_confirmar_delete_posicao_html(produto, posicao, as_inner=False):
    """Pagina de confirmacao para deletar posicao. Se as_inner=True, retorna só o card (para overlay)."""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    side = posicao.get('side') or posicao.get('tipo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    status = posicao.get('status', 'open')
    is_spot = 'spot' in produto.get('tipo', '').lower()
    side_info = "" if is_spot else f" ({side})"

    card_html = f"""
            <div class="card" style="max-width: 500px; text-align: center;">
                <h2 style="color: #ff6b6b;">Deletar Posicao</h2>
                <p style="margin: 20px 0;">Tem certeza que deseja deletar:</p>
                <p style="font-size: 1.3em; color: #39fda3; font-weight: bold;">{ativo}{side_info}</p>
                <p style="color: #888;">Entrada: ${preco_entrada:,.4f} | Status: {status}</p>
                <p style="color: #ff6b6b; margin: 20px 0; font-size: 0.9em;">
                    Esta acao ira deletar a posicao e todos os stops e alocacoes associados!
                </p>
                <div id="alert" class="alert"></div>
                <div class="actions" style="justify-content: center;">
                    <button class="btn btn-danger" onclick="deletarPosicao()">Sim, Deletar</button>
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                </div>
            </div>
        """
    script = f"""
        <script>
            async function deletarPosicao() {{
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/posicao/deletar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{posicao_id: {pos_id}}})
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Posicao deletada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }}
        </script>
    """
    if as_inner:
        return card_html + script
    return get_form_page_with_background(card_html + script, produto['id'], 'Deletar Posicao')


def get_lista_atributos_html(produto, configs, colunas_orfas, as_inner=False):
    """Lista atributos de um produto para gerenciamento. Se as_inner=True, retorna só o card (para overlay)."""
    items_html = ""

    if configs:
        for config in configs:
            obrig_badge = '<span class="badge badge-danger" style="margin-left: 8px;">obrigatorio</span>' if config['obrigatorio'] else ''
            items_html += f"""
            <div class="viz-item">
                <div style="flex: 1;">
                    <strong>{config['atributo_label'] or config['atributo_nome']}</strong>
                    <span style="color: #888; margin-left: 8px;">({config['atributo_nome']})</span>
                    <span class="badge badge-outro" style="margin-left: 8px;">{config['atributo_tipo']}</span>
                    {obrig_badge}
                </div>
                <div style="display: flex; gap: 8px;">
                    <a href="/produto/{produto['id']}/atributos/{config['atributo_nome']}/editar" class="btn btn-sm btn-primary">Editar</a>
                    <a href="/produto/{produto['id']}/atributos/{config['atributo_nome']}/remover" class="btn btn-sm btn-danger">Remover</a>
                </div>
            </div>
            """
    else:
        items_html = '<p style="text-align: center; color: #888; padding: 20px;">Nenhum atributo configurado para este produto</p>'

    # Mostrar colunas orfas se existirem
    orfas_html = ""
    if colunas_orfas:
        orfas_html = f"""
        <div style="margin-top: 30px; padding: 15px; background: rgba(255, 107, 107, 0.1); border-radius: 8px; border: 1px solid #ff6b6b;">
            <h3 style="color: #ff6b6b; margin-bottom: 10px;">Colunas Orfas ({len(colunas_orfas)})</h3>
            <p style="color: #888; font-size: 0.9em; margin-bottom: 10px;">Colunas que nao estao sendo usadas por nenhum produto:</p>
            <p style="color: #888;">{', '.join(colunas_orfas)}</p>
            <button class="btn btn-sm btn-danger" style="margin-top: 10px;" onclick="limparOrfas()">Limpar Colunas Orfas</button>
        </div>
        """

    card_html = f"""
            <div class="card" style="max-width: 700px;">
                <h2>Gerenciar Atributos</h2>
                <p style="color: #39fda3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <div class="viz-list">{items_html}</div>
                {orfas_html}
                <div class="actions" style="margin-top: 20px;">
                    <a href="/produto/{produto['id']}/atributos/novo" class="btn btn-primary">+ Novo Atributo</a>
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Voltar</a>
                </div>
            </div>
        """
    script = f"""
        <script>
            async function limparOrfas() {{
                if (!confirm('Remover todas as colunas orfas?')) return;
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/atributos/limpar-orfas', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}}
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Colunas orfas removidas: ' + result.removidas;
                        setTimeout(() => location.reload(), 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }}
        </script>
    """
    if as_inner:
        return card_html + script
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Atributos - {produto['nome']}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            {card_html}
        </div>
        {script}
    </body>
    </html>
    """


def get_form_atributo_html(produto, config=None, colunas_existentes=None, as_inner=False):
    """Formulario para criar/editar atributo. Se as_inner=True, retorna só o card (para overlay)."""
    is_edit = config is not None
    titulo = "Editar Atributo" if is_edit else "Novo Atributo"

    nome_value = config['atributo_nome'] if is_edit else ''
    label_value = config.get('atributo_label', '') or '' if is_edit else ''
    tipo_value = config['atributo_tipo'] if is_edit else 'text'
    obrig_checked = 'checked' if is_edit and config.get('obrigatorio') else ''

    # Lista de colunas existentes para sugestao
    colunas_datalist = ""
    if colunas_existentes and not is_edit:
        for col in colunas_existentes:
            colunas_datalist += f'<option value="{col}">'

    nome_field = f"""
        <input type="text" name="nome" value="{nome_value}" list="colunas_existentes" required
               placeholder="Ex: quantidade, risco, setor" {'readonly' if is_edit else ''}>
        <datalist id="colunas_existentes">{colunas_datalist}</datalist>
        <small style="color: #888;">Use colunas existentes ou crie uma nova</small>
    """ if not is_edit else f'<input type="text" name="nome" value="{nome_value}" readonly>'

    card_html = f"""
            <div class="card" style="max-width: 600px;">
                <h2>{titulo}</h2>
                <p style="color: #39fda3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <form id="atributoForm">
                    <div class="form-group">
                        <label>Nome do Atributo (coluna)</label>
                        {nome_field}
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Label (exibicao)</label>
                            <input type="text" name="label" value="{label_value}" placeholder="Ex: Quantidade, Nivel de Risco">
                        </div>
                        <div class="form-group">
                            <label>Tipo</label>
                            <select name="tipo">
                                <option value="text" {'selected' if tipo_value == 'text' else ''}>Texto</option>
                                <option value="float" {'selected' if tipo_value == 'float' else ''}>Numero decimal</option>
                                <option value="int" {'selected' if tipo_value == 'int' else ''}>Numero inteiro</option>
                                <option value="date" {'selected' if tipo_value == 'date' else ''}>Data</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-group">
                        <label style="display: flex; align-items: center; gap: 10px; cursor: pointer;">
                            <input type="checkbox" name="obrigatorio" {obrig_checked} style="width: 20px; height: 20px;">
                            <span>Campo obrigatorio</span>
                        </label>
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">{'Salvar' if is_edit else 'Criar'}</button>
                        <a href="/produto/{produto['id']}/atributos" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        """
    script = f"""
        <script>
            document.getElementById('atributoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const alert = document.getElementById('alert');

                const dados = {{
                    produto_id: {produto['id']},
                    nome: form.nome.value.toLowerCase().trim().replace(/ /g, '_'),
                    label: form.label.value || null,
                    tipo: form.tipo.value,
                    obrigatorio: form.obrigatorio.checked
                }};

                const endpoint = '{"editar" if is_edit else "criar"}';

                try {{
                    const response = await fetch('/api/atributo/' + endpoint, {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(dados)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Atributo {"atualizado" if is_edit else "criado"}!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/atributos', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    """
    if as_inner:
        return card_html + script
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>{titulo} - {produto['nome']}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            {card_html}
        </div>
        {script}
    </body>
    </html>
    """


def get_confirmar_remover_atributo_html(produto, config, as_inner=False):
    """Pagina de confirmacao para remover atributo do produto. Se as_inner=True, retorna só o card (para overlay)."""
    card_html = f"""
            <div class="card" style="max-width: 500px; text-align: center;">
                <h2 style="color: #ff6b6b;">Remover Atributo</h2>
                <p style="margin: 20px 0;">Remover atributo do produto {produto['nome']}:</p>
                <p style="font-size: 1.3em; color: #39fda3; font-weight: bold;">{config['atributo_label'] or config['atributo_nome']}</p>
                <p style="color: #888; margin: 20px 0; font-size: 0.9em;">
                    A coluna permanecera no banco de dados e podera ser usada por outros produtos.
                </p>
                <div id="alert" class="alert"></div>
                <div class="actions" style="justify-content: center;">
                    <button class="btn btn-danger" onclick="removerAtributo()">Sim, Remover</button>
                    <a href="/produto/{produto['id']}/atributos" class="btn btn-secondary">Cancelar</a>
                </div>
            </div>
        """
    script = f"""
        <script>
            async function removerAtributo() {{
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/atributo/remover', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            produto_id: {produto['id']},
                            nome: '{config["atributo_nome"]}'
                        }})
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Atributo removido!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/atributos', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }}
        </script>
    """
    if as_inner:
        return card_html + script
    return get_form_page_with_background(card_html + script, produto['id'], 'Remover Atributo')


def get_form_alocacao_html(produto_id=None, produtos=None, posicoes=None):
    """Pagina de alocacoes em formato planilha (tabela Data x Ativos com %) — tema escuro"""
    produtos_options = ""
    if produtos:
        for p in produtos:
            selected = 'selected' if produto_id and p['id'] == produto_id else ''
            produtos_options += f'<option value="{p["id"]}" {selected}>{p["nome"]}</option>'
    hoje = date.today().isoformat()
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {get_favicon_tag()}
        <title>Alocações - Tabela</title>
        <style>
            {get_base_styles()}

            /* ===== Planilha de alocacoes ===== */
            .aloc-header {{
                display: flex; align-items: center; gap: 1.2rem;
                margin-bottom: 1.5rem; flex-wrap: wrap;
            }}
            .aloc-header select {{
                padding: 10px 14px; border-radius: 8px;
                border: 2px solid #2a2a4a; background: #16213e;
                color: #eee; font-size: 1em; min-width: 200px;
            }}
            .aloc-header select:focus {{ border-color: #39fda3; outline: none; }}
            .aloc-header label {{ color: #39fda3; font-weight: bold; }}
            .aloc-meta {{
                display: flex; gap: 1rem; flex-wrap: wrap;
                margin-bottom: 1rem; align-items: center;
            }}
            .aloc-meta .pill {{
                background: rgba(78,204,163,.12); border: 1px solid rgba(78,204,163,.25);
                color: #39fda3; padding: 5px 14px; border-radius: 20px; font-size: .85em;
            }}
            .aloc-meta .pill b {{ color: #fff; }}

            /* Scroll wrapper */
            .sheet-wrap {{
                overflow-x: auto; border-radius: 12px;
                border: 1px solid #2a2a4a; max-height: 65vh; overflow-y: auto;
            }}
            .sheet-wrap::-webkit-scrollbar {{ width: 6px; height: 6px; }}
            .sheet-wrap::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.2); }}
            .sheet-wrap::-webkit-scrollbar-thumb {{ background: rgba(78, 204, 163, 0.35); border-radius: 3px; }}
            .sheet {{
                border-collapse: separate; border-spacing: 0;
                width: max-content; min-width: 100%; font-size: .85rem;
            }}
            .sheet th, .sheet td {{
                padding: 7px 10px; white-space: nowrap;
                border-right: 1px solid #2a2a4a; border-bottom: 1px solid #2a2a4a;
            }}
            .sheet thead {{ position: sticky; top: 0; z-index: 3; }}
            .sheet thead th {{
                background: linear-gradient(135deg, #39fda3 0%, #3db892 100%);
                color: #1a1a2e; font-weight: 700; text-align: center;
            }}
            .sheet thead th:first-child {{ text-align: left; }}
            .sheet tbody td {{ text-align: right; color: #ccc; }}
            .sheet tbody td:first-child {{ text-align: left; color: #eee; font-weight: 600; }}
            .sheet tbody tr:nth-child(odd) td {{ background: #16213e; }}
            .sheet tbody tr:nth-child(even) td {{ background: #1a1f33; }}
            .sheet tbody tr:hover td {{ background: rgba(78,204,163,.08); }}
            .sheet tbody td.cell-edit {{ cursor: cell; }}
            .sheet tbody td.cell-edit:hover {{ background: rgba(78,204,163,.12) !important; }}
            .sheet .cell-edit input,
            .sheet .cell-editing input {{
                width: 100%; min-width: 50px; text-align: right;
                padding: 4px 6px; font-size: .85rem;
                background: #16213e; color: #39fda3;
                border: 1px solid #39fda3; border-radius: 4px;
                outline: none; box-shadow: 0 0 0 2px rgba(78,204,163,.25);
            }}

            /* Celulas com valor > 0 ganham destaque */
            .sheet .val-pos {{ color: #39fda3; }}
            .sheet .val-zero {{ color: #444; }}

            /* Coluna Data fixa */
            .sheet .col-data {{
                position: sticky; left: 0; z-index: 2;
                border-right: 2px solid #39fda3;
            }}
            .sheet thead .col-data {{ z-index: 4; }}

            /* Coluna Total */
            .sheet .col-total {{
                font-weight: 700; border-left: 2px solid #39fda3;
            }}
            .sheet thead .col-total {{ color: #1a1a2e !important; }}
            .sheet tbody .col-total {{ color: #39fda3 !important; }}

            /* Nova linha (inputs) */
            .sheet tr.nova-linha td {{
                background: rgba(78,204,163,.08) !important;
                border-top: 2px solid #39fda3;
            }}
            .sheet .nova-linha input[type="date"],
            .sheet .nova-linha input[type="number"] {{
                background: #16213e; color: #eee;
                border: 1px solid #2a2a4a; border-radius: 5px;
                padding: 5px 7px; font-size: .85rem;
            }}
            .sheet .nova-linha input:focus {{
                border-color: #39fda3; outline: none;
                box-shadow: 0 0 0 2px rgba(78,204,163,.25);
            }}
            .sheet .nova-linha input[type="date"] {{ width: 135px; }}
            .sheet .nova-linha input[type="number"] {{ width: 68px; text-align: right; }}

            /* Paginacao */
            .pag-bar {{
                display: flex; align-items: center; gap: .7rem;
                margin-top: 1rem; flex-wrap: wrap;
            }}
            .pag-bar button {{
                background: #16213e; color: #39fda3; border: 1px solid #2a2a4a;
                padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: .85em;
            }}
            .pag-bar button:hover {{ border-color: #39fda3; }}
            .pag-bar button:disabled {{ opacity: .35; cursor: default; }}
            .pag-bar span {{ color: #888; font-size: .85em; }}

            /* Botao salvar */
            .save-bar {{
                display: flex; gap: 1rem; align-items: center;
                margin-top: 1.2rem; flex-wrap: wrap;
            }}

            /* Loading */
            .sheet-loading {{
                text-align: center; padding: 3rem; color: #39fda3;
            }}
            .sheet-loading .spinner {{
                width: 36px; height: 36px; margin: 0 auto 1rem;
                border: 3px solid #2a2a4a; border-top-color: #39fda3;
                border-radius: 50%; animation: spin .8s linear infinite;
            }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        </style>
    </head>
    <body>
        {get_navbar('alocacao')}
        <div class="container">
            <div class="card">
                <h2 style="margin:0 0 8px 0;">Alocações</h2>
                <p style="color:#718096; margin-bottom:1.2rem; font-size:.9em;">
                    Tabela estilo planilha: uma linha por data, colunas por ativo (%).
                    Para <b>adicionar um novo ativo (nova coluna)</b>, use o campo abaixo. Depois preencha o % na nova coluna e salve a linha.
                </p>

                <div class="aloc-header">
                    <label for="produtoSelect">Produto</label>
                    <select id="produtoSelect" onchange="carregarTabela()">
                        <option value="">Selecione um produto...</option>
                        {produtos_options}
                    </select>
                </div>

                <div id="addAtivoBar" style="display:none; margin-bottom:1rem; padding:14px; background:rgba(78,204,163,.08); border:1px solid rgba(78,204,163,.25); border-radius:8px;">
                    <div style="color:#39fda3; font-weight:bold; margin-bottom:10px;">Adicionar ativo (nova coluna)</div>
                    <div style="display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:8px;">
                        <input type="text" id="novoAtivoInput" placeholder="Ativo (ex: BTC, ETH)" style="width:120px; padding:8px 10px; border-radius:6px; border:1px solid #2a2a4a; background:#16213e; color:#eee;">
                        <input type="text" id="novoAtivoCoingecko" placeholder="CoinGecko ID (opcional, ex: bitcoin)" style="width:180px; padding:8px 10px; border-radius:6px; border:1px solid #2a2a4a; background:#16213e; color:#eee;">
                        <input type="text" id="novoAtivoSymbol" placeholder="Símbolo exchange (opcional, ex: BTCUSDT)" style="width:200px; padding:8px 10px; border-radius:6px; border:1px solid #2a2a4a; background:#16213e; color:#eee;">
                        <button type="button" class="btn btn-sm btn-primary" id="btnAdicionarAtivo">+ Adicionar</button>
                    </div>
                    <span id="addAtivoMsg" style="font-size:.85em; color:#888;"></span>
                </div>

                <div id="alert" class="alert"></div>
                <div id="metaInfo"></div>
                <div id="tabelaContainer">
                    <div class="empty-state">
                        <h3>Nenhum produto selecionado</h3>
                        <p>Escolha um produto acima para visualizar e adicionar alocacoes.</p>
                    </div>
                </div>
            </div>
        </div>
        <script>
        (function() {{
            const PAGE_SIZE = 30;
            let allData = null;
            let currentPage = 0;

            function esc(s) {{
                if (s == null) return '';
                const d = document.createElement('div');
                d.textContent = s;
                return d.innerHTML;
            }}

            function showAlert(msg, isError) {{
                const el = document.getElementById('alert');
                el.textContent = msg;
                el.className = 'alert show ' + (isError ? 'alert-error' : 'alert-success');
                setTimeout(() => el.className = 'alert', 5000);
            }}

            window.carregarTabela = async function() {{
                const pid = document.getElementById('produtoSelect').value;
                const ct = document.getElementById('tabelaContainer');
                const mi = document.getElementById('metaInfo');
                const addBar = document.getElementById('addAtivoBar');
                if (addBar) addBar.style.display = pid ? 'block' : 'none';
                if (!pid) {{
                    ct.innerHTML = '<div class="empty-state"><h3>Nenhum produto selecionado</h3><p>Escolha um produto acima.</p></div>';
                    mi.innerHTML = '';
                    return;
                }}
                ct.innerHTML = '<div class="sheet-loading"><div class="spinner"></div>Carregando...</div>';
                mi.innerHTML = '';
                try {{
                    const r = await fetch('/api/alocacoes/tabela?produto_id=' + pid);
                    const data = await r.json();
                    if (data.erro) throw new Error(data.erro);
                    allData = data;
                    currentPage = Math.max(0, Math.ceil(data.linhas.length / PAGE_SIZE) - 1);
                    renderMeta();
                    renderPage();
                }} catch (e) {{
                    ct.innerHTML = '<div class="empty-state"><h3>Erro</h3><p>' + esc(e.message) + '</p></div>';
                }}
            }};

            function renderMeta() {{
                const d = allData;
                const total = d.linhas.length;
                const mi = document.getElementById('metaInfo');
                mi.innerHTML = '<div class="aloc-meta">'
                    + '<div class="pill"><b>' + esc(d.produto_nome) + '</b></div>'
                    + '<div class="pill">' + d.colunas.length + ' ativos</div>'
                    + '<div class="pill">' + total + ' datas</div>'
                    + '</div>';
            }}

            function renderPage() {{
                const d = allData;
                const colunas = d.colunas;
                const linhas = d.linhas;
                const totalPages = Math.max(1, Math.ceil(linhas.length / PAGE_SIZE));
                const start = currentPage * PAGE_SIZE;
                const end = Math.min(start + PAGE_SIZE, linhas.length);
                const slice = linhas.slice(start, end);
                const hoje = '{hoje}';

                function escAttr(s) {{ return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;'); }}
                // Texto para clipboard: Data + valores separados por tab (para Ctrl+V colar nas colunas certas)
                function rowToTsv(linha) {{
                    const data = linha.data || '';
                    const vals = colunas.map(col => ((linha.ativos && linha.ativos[col]) != null ? Number(linha.ativos[col]) : 0).toFixed(2));
                    return data + '\\t' + vals.join('\\t');
                }}

                let h = '<div class="sheet-wrap" id="sheetWrap"><table class="sheet" id="sheetAloc"><thead><tr>';
                h += '<th class="col-data">Data</th>';
                colunas.forEach(c => h += '<th>' + esc(c) + '</th>');
                h += '<th class="col-total">Total</th>';
                h += '</tr></thead><tbody>';

                slice.forEach(linha => {{
                    let soma = 0;
                    const tsv = escAttr(rowToTsv(linha));
                    h += '<tr class="row-dados" data-row-copy="' + tsv + '"><td class="col-data">' + esc(linha.data) + '</td>';
                    colunas.forEach(col => {{
                        const v = (linha.ativos && linha.ativos[col]) || 0;
                        soma += v;
                        const cls = (v > 0 ? 'val-pos' : 'val-zero') + ' cell-edit';
                        h += '<td class="' + cls + '" data-data="' + esc(linha.data) + '" data-ativo="' + esc(col) + '" data-val="' + v.toFixed(2) + '">' + v.toFixed(2) + '%</td>';
                    }});
                    h += '<td class="col-total">' + soma.toFixed(1) + '%</td></tr>';
                }});

                // Nova linha
                h += '<tr class="nova-linha"><td class="col-data"><input type="date" id="novaData" value="' + hoje + '"></td>';
                colunas.forEach(col => {{
                    h += '<td><input type="number" step="0.01" min="0" max="100" placeholder="-" data-ativo="' + esc(col) + '"></td>';
                }});
                h += '<td class="col-total" id="novaTotal">0%</td></tr>';
                h += '</tbody></table></div>';

                // Paginacao
                h += '<div class="pag-bar">';
                h += '<button onclick="pagAloc(0)" ' + (currentPage <= 0 ? 'disabled' : '') + '>&#171; Inicio</button>';
                h += '<button onclick="pagAloc(' + (currentPage - 1) + ')" ' + (currentPage <= 0 ? 'disabled' : '') + '>&#8249; Anterior</button>';
                h += '<span>Pagina ' + (currentPage + 1) + ' de ' + totalPages + '</span>';
                h += '<button onclick="pagAloc(' + (currentPage + 1) + ')" ' + (currentPage >= totalPages - 1 ? 'disabled' : '') + '>Proxima &#8250;</button>';
                h += '<button onclick="pagAloc(' + (totalPages - 1) + ')" ' + (currentPage >= totalPages - 1 ? 'disabled' : '') + '>Fim &#187;</button>';
                h += '</div>';

                // Botao salvar e dica
                h += '<div class="save-bar">';
                h += '<button type="button" class="btn btn-primary" onclick="salvarLinha()">Salvar nova linha</button>';
                h += '<span style="color:#666; font-size:.85em;">Selecione uma linha, Ctrl+C para copiar; clique na nova linha e Ctrl+V para colar nas colunas.</span>';
                h += '</div>';

                document.getElementById('tabelaContainer').innerHTML = h;

                // Ctrl+C: ao copiar uma linha de dados, colocar no clipboard em formato tab-separado
                const wrap = document.getElementById('sheetWrap');
                if (wrap) {{
                    wrap.addEventListener('copy', function(e) {{
                        const sel = document.getSelection();
                        if (!sel || sel.rangeCount === 0) return;
                        let node = sel.anchorNode;
                        while (node && node !== wrap) {{
                            if (node.nodeType === 1 && node.classList && node.classList.contains('row-dados')) {{
                                const tsv = node.getAttribute('data-row-copy');
                                if (tsv) {{
                                    e.preventDefault();
                                    e.clipboardData.setData('text/plain', tsv);
                                }}
                                return;
                            }}
                            node = node.parentNode;
                        }}
                    }});
                }}

                // Ctrl+V na nova linha: distribuir valores por coluna (tab ou newline)
                // Enter na nova linha: salvar a linha
                document.querySelectorAll('#tabelaContainer .nova-linha input').forEach(inp => {{
                    inp.addEventListener('input', atualizarNovaTotal);
                    inp.addEventListener('keydown', function(e) {{
                        if (e.key === 'Enter') {{ e.preventDefault(); window.salvarLinha(); }}
                    }});
                    inp.addEventListener('paste', function(e) {{
                        e.preventDefault();
                        const text = (e.clipboardData || window.clipboardData).getData('text/plain');
                        const parts = text.trim().split(/[\\t\\n\\r]+/).map(s => s.trim()).filter(Boolean);
                        if (parts.length === 0) return;
                        const dataInp = document.getElementById('novaData');
                        const inputs = [dataInp, ...document.querySelectorAll('#tabelaContainer .nova-linha input[data-ativo]')];
                        parts.forEach((part, i) => {{
                            if (inputs[i]) {{
                                if (i === 0) inputs[i].value = part.replace(/\\s/g, '').substring(0, 10);
                                else inputs[i].value = part.replace(',', '.').replace(/[^0-9.-]/g, '') || '';
                            }}
                        }});
                        atualizarNovaTotal();
                    }});
                }});

                // Duplo clique na célula para editar; Enter salva
                document.querySelectorAll('#tabelaContainer .sheet tbody td.cell-edit').forEach(td => {{
                    td.addEventListener('dblclick', function() {{
                        const cell = this;
                        const dataVal = cell.getAttribute('data-data');
                        const ativoVal = cell.getAttribute('data-ativo');
                        const val = cell.getAttribute('data-val') || '0';
                        const produtoId = allData.produto_id;
                        const origHtml = cell.innerHTML;
                        cell.innerHTML = '<input type="number" step="0.01" min="0" max="100" value="' + esc(val) + '">';
                        const inp = cell.querySelector('input');
                        inp.focus();
                        inp.select();
                        let done = false;
                        function finish() {{
                            if (done) return;
                            done = true;
                            const newVal = inp.value.trim() === '' ? '0' : inp.value.replace(',', '.');
                            const num = parseFloat(newVal);
                            if (isNaN(num) || num < 0) {{ cell.innerHTML = origHtml; return; }}
                            fetch('/api/alocacao/celula', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{ produto_id: produtoId, data: dataVal, ativo: ativoVal, percentual: num }})
                            }}).then(r => r.json()).then(res => {{
                                if (res.sucesso) carregarTabela();
                                else {{ cell.innerHTML = origHtml; showAlert(res.erro || 'Erro ao salvar', true); }}
                            }}).catch(() => {{ cell.innerHTML = origHtml; }});
                        }}
                        inp.addEventListener('keydown', function(e) {{
                            if (e.key === 'Enter') {{ e.preventDefault(); finish(); }}
                            if (e.key === 'Escape') {{ done = true; cell.innerHTML = origHtml; }}
                        }});
                        inp.addEventListener('blur', function() {{ finish(); }});
                    }});
                }});
            }}

            function atualizarNovaTotal() {{
                let soma = 0;
                document.querySelectorAll('#tabelaContainer .nova-linha input[data-ativo]').forEach(inp => {{
                    const v = parseFloat(inp.value);
                    if (!isNaN(v)) soma += v;
                }});
                const el = document.getElementById('novaTotal');
                if (el) el.textContent = soma.toFixed(1) + '%';
            }}

            window.pagAloc = function(page) {{
                if (!allData) return;
                const totalPages = Math.max(1, Math.ceil(allData.linhas.length / PAGE_SIZE));
                currentPage = Math.max(0, Math.min(page, totalPages - 1));
                renderPage();
            }};

            window.salvarLinha = async function() {{
                if (!allData) return;
                const produtoId = allData.produto_id;
                const dataLinha = document.getElementById('novaData').value;
                if (!dataLinha) {{ showAlert('Informe a data.', true); return; }}
                const ativos = {{}};
                document.querySelectorAll('#tabelaContainer .nova-linha input[data-ativo]').forEach(inp => {{
                    const val = parseFloat(inp.value);
                    if (!isNaN(val) && val > 0) ativos[inp.getAttribute('data-ativo')] = val;
                }});
                if (Object.keys(ativos).length === 0) {{
                    showAlert('Informe ao menos um percentual > 0.', true);
                    return;
                }}
                try {{
                    const r = await fetch('/api/alocacao/linha', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ produto_id: produtoId, data: dataLinha, ativos: ativos }})
                    }});
                    const result = await r.json();
                    if (result.sucesso) {{
                        let msg = result.alocacoes_criadas + ' alocacao(oes) salvas.';
                        if (result.avisos && result.avisos.length) msg += ' Avisos: ' + result.avisos.join('; ');
                        showAlert(msg, false);
                        carregarTabela();
                    }} else {{
                        showAlert(result.erro || 'Erro desconhecido', true);
                    }}
                }} catch (e) {{
                    showAlert('Erro: ' + e.message, true);
                }}
            }};

            // Adicionar ativo (nova coluna)
            document.getElementById('btnAdicionarAtivo').addEventListener('click', async function() {{
                const pid = document.getElementById('produtoSelect').value;
                const inp = document.getElementById('novoAtivoInput');
                const inpCg = document.getElementById('novoAtivoCoingecko');
                const inpSym = document.getElementById('novoAtivoSymbol');
                const msgEl = document.getElementById('addAtivoMsg');
                const ativo = (inp && inp.value || '').trim().toUpperCase();
                const coingecko_id = (inpCg && inpCg.value || '').trim() || null;
                const exchange_symbol = (inpSym && inpSym.value || '').trim() || null;
                if (!pid) {{ showAlert('Selecione um produto antes.', true); return; }}
                if (!ativo) {{ showAlert('Digite o nome do ativo (ex: BTC, ETH).', true); return; }}
                if (msgEl) msgEl.textContent = 'Adicionando...';
                try {{
                    const body = {{ produto_id: parseInt(pid), ativo: ativo }};
                    if (coingecko_id) body.coingecko_id = coingecko_id;
                    if (exchange_symbol) body.exchange_symbol = exchange_symbol;
                    const r = await fetch('/api/alocacao/adicionar-ativo', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(body)
                    }});
                    const res = await r.json();
                    if (res.sucesso) {{
                        if (msgEl) {{ msgEl.textContent = res.mensagem || 'Ok'; msgEl.style.color = '#39fda3'; }}
                        if (inp) inp.value = '';
                        if (inpCg) inpCg.value = '';
                        if (inpSym) inpSym.value = '';
                        carregarTabela();
                    }} else {{
                        showAlert(res.erro || 'Erro ao adicionar ativo', true);
                        if (msgEl) msgEl.textContent = '';
                    }}
                }} catch (e) {{
                    showAlert('Erro: ' + e.message, true);
                    if (msgEl) msgEl.textContent = '';
                }}
            }});

            // Auto-load se produto ja selecionado
            if (document.getElementById('produtoSelect').value) carregarTabela();
        }})();
        </script>
    </body>
    </html>
    """


def get_confirmar_delete_visualizacao_html(produto, visualizacao, as_inner=False):
    """Pagina de confirmacao para deletar visualizacao. Se as_inner=True, retorna só o card (para overlay)."""
    card_html = f"""
            <div class="card" style="max-width: 500px; text-align: center;">
                <h2 style="color: #ff6b6b;">Deletar Visualizacao</h2>
                <p style="margin: 20px 0;">Tem certeza que deseja deletar:</p>
                <p style="font-size: 1.3em; color: #39fda3; font-weight: bold;">{visualizacao['nome']}</p>
                <p style="color: #888; margin: 20px 0;">Esta acao nao pode ser desfeita.</p>
                <div id="alert" class="alert"></div>
                <div class="actions" style="justify-content: center;">
                    <button class="btn btn-danger" onclick="deletarVisualizacao()">Sim, Deletar</button>
                    <a href="/produto/{produto['id']}/visualizacoes" class="btn btn-secondary">Cancelar</a>
                </div>
            </div>
        """
    script = f"""
        <script>
            async function deletarVisualizacao() {{
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/visualizacao/deletar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{visualizacao_id: {visualizacao['id']}}})
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Visualizacao deletada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/visualizacoes', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }}
        </script>
    """
    if as_inner:
        return card_html + script
    return get_form_page_with_background(card_html + script, produto['id'], 'Deletar Visualizacao')


# ============================================================
# Produtos cujas alocações definem as posições (sem filtro de datas)
# Alphacoins=3476245316, EXC=2150859854, HB=2000449260, LC=2394004756
#
# Ciclo de vida:
#   1. Alocação > 0 sem posição aberta → cria posição (data_entrada = data)
#   2. Alocações > 0 continuam → mesma posição
#   3. Alocação = 0 com posição aberta → fecha posição (data_saida = data)
#   4. Alocação > 0 novamente → cria NOVA posição
# ============================================================
_PRODUTOS_ALOCACAO_LIVRE = {3476245316, 2150859854, 2000449260, 2394004756}
_NOMES_ALOCACAO_LIVRE = {'Alphacoins', 'EXC', 'HB', 'LC'}

def _is_produto_livre(repo, produto_id):
    """Verifica se o produto é de alocação livre (AC/EXC/HB/LC)."""
    if produto_id in _PRODUTOS_ALOCACAO_LIVRE:
        return True
    produto = repo.carregar_produto(produto_id)
    return produto and produto.get('nome') in _NOMES_ALOCACAO_LIVRE

def _encontrar_ou_criar_posicao(repo, produto_id, ativo, data_linha):
    """
    Encontra uma posição para o ativo no produto.
    - Para produtos de alocação livre (AC/EXC/HB/LC): busca posição ABERTA
      (status='open') do ativo. Se não existir, cria uma nova.
    - Para outros produtos: exige posição aberta na data.
    Retorna posicao_id ou None.
    """
    livre = _is_produto_livre(repo, produto_id)

    with repo._get_connection() as conn:
        cur = conn.cursor()
        if livre:
            # Buscar posição ABERTA (status='open') para esse ativo
            cur.execute("""
                SELECT id FROM posicoes
                WHERE produto_id = %s AND ativo = %s AND status = 'open'
                ORDER BY id DESC LIMIT 1
            """, (produto_id, ativo))
        else:
            # Buscar posição aberta na data
            cur.execute("""
                SELECT id FROM posicoes
                WHERE produto_id = %s AND ativo = %s
                AND (data_entrada IS NULL OR data_entrada <= %s)
                AND (data_saida IS NULL OR data_saida >= %s)
                ORDER BY id DESC LIMIT 1
            """, (produto_id, ativo, data_linha, data_linha))
        row = cur.fetchone()
        if row:
            return row[0]

    # Não encontrou posição aberta; criar automaticamente se for produto livre
    if livre:
        try:
            posicao = Posicao(
                ativo=ativo,
                side='long',
                preco_entrada=0,
                data_entrada=data_linha,
            )
            posicao_id = repo.salvar_posicao(produto_id, posicao)
            print(f"[ALOCAÇÃO] Posição ABERTA automaticamente: {ativo} (data_entrada={data_linha}) no produto {produto_id}", flush=True)
            return posicao_id
        except Exception as e:
            print(f"[ALOCAÇÃO] Erro ao criar posição para {ativo}: {e}", flush=True)

    return None

def _fechar_posicao_se_zerado(repo, produto_id, ativo, data_linha):
    """
    Para produtos de alocação livre: quando uma alocação vai a 0%,
    fecha a posição aberta do ativo (data_saida = data, status = 'closed').
    Retorna True se fechou, False caso contrário.
    """
    if not _is_produto_livre(repo, produto_id):
        return False

    with repo._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM posicoes
            WHERE produto_id = %s AND ativo = %s AND status = 'open'
            ORDER BY id DESC LIMIT 1
        """, (produto_id, ativo))
        row = cur.fetchone()
        if not row:
            return False

    posicao_id = row[0]
    try:
        repo.atualizar_posicao(posicao_id, data_saida=data_linha, status='closed')
        print(f"[ALOCAÇÃO] Posição FECHADA automaticamente: {ativo} (data_saida={data_linha}) no produto {produto_id}", flush=True)
        return True
    except Exception as e:
        print(f"[ALOCAÇÃO] Erro ao fechar posição para {ativo}: {e}", flush=True)
        return False


def reconciliar_posicoes_com_alocacoes(repo, produto_id):
    """
    Para produtos de alocação livre (AC/EXC/HB/LC), reconstrói as posições
    a partir dos dados de alocação.

    Lógica central:
      - A timeline do produto = todas as datas onde QUALQUER ativo tem alocação.
      - Se um ativo está AUSENTE em uma data da timeline, significa 0% nessa data.
      - Abertas: ativos com alocação > 0 na DATA MAIS RECENTE da timeline.
      - Fechadas: ativos que tiveram alocação > 0 em algum período, mas na data
        mais recente estão ausentes (ou já saíram antes).
      - data_entrada: primeira data do bloco contínuo de presença.
      - data_saida: última data de presença antes do gap (posição fechada).

    Funciona tanto em SQLite quanto PostgreSQL.
    """
    if not _is_produto_livre(repo, produto_id):
        return

    from collections import defaultdict

    with repo._get_connection() as conn:
        cur = conn.cursor()

        # 1. Timeline: todas as datas distintas com alocação neste produto
        cur.execute("""
            SELECT DISTINCT CAST(a.data AS TEXT)
            FROM alocacoes a
            WHERE a.produto_id = %s AND a.status = 'active'
            ORDER BY 1
        """, (produto_id,))
        todas_datas = [str(r[0])[:10] for r in cur.fetchall()]

        if not todas_datas:
            return

        # 2. Todas as alocações > 0 (alocações = 0 nunca são armazenadas)
        cur.execute("""
            SELECT p.ativo, CAST(a.data AS TEXT), p.id
            FROM alocacoes a
            INNER JOIN posicoes p ON a.posicao_id = p.id
            WHERE a.produto_id = %s AND a.status = 'active' AND a.percentual > 0
        """, (produto_id,))
        allocs = cur.fetchall()

        # 3. Todas as posições do produto
        cur.execute("""
            SELECT id, ativo, data_entrada, data_saida, status
            FROM posicoes WHERE produto_id = %s
        """, (produto_id,))
        todas_posicoes = cur.fetchall()

    if not allocs:
        # Nenhuma alocação > 0 → fechar todas as posições abertas
        for pid, ativo, de, ds, st in todas_posicoes:
            if st == 'open':
                try:
                    repo.atualizar_posicao(pid, status='closed')
                except Exception:
                    pass
        return

    data_mais_recente = max(todas_datas)

    # Mapear: ativo → set de datas onde tem alocação > 0
    ativo_datas = defaultdict(set)
    for ativo, data, pos_id in allocs:
        ativo_datas[str(ativo).strip()].add(str(data)[:10])

    # ---------------------------------------------------------------
    # Detectar períodos por ativo (gap-and-island na timeline)
    # Um ativo AUSENTE em uma data da timeline = 0% = gap
    # ---------------------------------------------------------------
    ativo_periodos = {}
    for ativo, datas_presentes in ativo_datas.items():
        periodos = []
        inicio = fim = None
        for d in todas_datas:
            if d in datas_presentes:
                if inicio is None:
                    inicio = d
                fim = d
            else:
                # Gap: ativo ausente nesta data → fechar período atual
                if inicio is not None:
                    periodos.append({'inicio': inicio, 'fim': fim, 'aberto': False})
                    inicio = fim = None
        # Período final
        if inicio is not None:
            aberto = data_mais_recente in datas_presentes
            periodos.append({
                'inicio': inicio,
                'fim': None if aberto else fim,
                'aberto': aberto
            })
        ativo_periodos[ativo] = periodos

    # ---------------------------------------------------------------
    # Reconciliar: mapear cada período a uma posição existente
    # ---------------------------------------------------------------
    posicoes_por_ativo = defaultdict(list)
    for pid, ativo, de, ds, st in todas_posicoes:
        posicoes_por_ativo[str(ativo).strip()].append({
            'id': pid,
            'data_entrada': str(de)[:10] if de else None,
            'data_saida': str(ds)[:10] if ds else None,
            'status': st
        })

    n_atualizadas = 0

    for ativo, periodos in ativo_periodos.items():
        pos_list = sorted(posicoes_por_ativo.get(ativo, []), key=lambda x: x['id'])

        for i, per in enumerate(periodos):
            if i < len(pos_list):
                pos = pos_list[i]
            else:
                # Criar nova posição para este período
                nova = Posicao(ativo=ativo, side='long', preco_entrada=0, data_entrada=per['inicio'])
                novo_id = repo.salvar_posicao(produto_id, nova)
                pos = {'id': novo_id, 'data_entrada': per['inicio'], 'data_saida': None, 'status': 'open'}
                pos_list.append(pos)
                print(f"[RECONCILIAR] Posição criada: {ativo} ({per['inicio']})", flush=True)

            # Calcular updates necessários
            updates = {}
            if pos['data_entrada'] != per['inicio']:
                updates['data_entrada'] = per['inicio']

            if per['aberto']:
                if pos['status'] != 'open':
                    updates['status'] = 'open'
                if pos['data_saida'] is not None:
                    updates['data_saida'] = None
            else:
                if pos['status'] != 'closed':
                    updates['status'] = 'closed'
                if pos['data_saida'] != per['fim']:
                    updates['data_saida'] = per['fim']

            if updates:
                try:
                    repo.atualizar_posicao(pos['id'], **updates)
                    n_atualizadas += 1
                except Exception as e:
                    print(f"[RECONCILIAR] Erro ao atualizar posição {pos['id']} ({ativo}): {e}", flush=True)

        # Fechar posições excedentes para este ativo (sem período correspondente)
        for j in range(len(periodos), len(pos_list)):
            pos = pos_list[j]
            if pos['status'] == 'open':
                try:
                    repo.atualizar_posicao(pos['id'], status='closed')
                    n_atualizadas += 1
                except Exception as e:
                    print(f"[RECONCILIAR] Erro ao fechar excedente {pos['id']}: {e}", flush=True)

    # Fechar posições de ativos que NÃO têm nenhuma alocação > 0
    for ativo, pos_list in posicoes_por_ativo.items():
        if ativo not in ativo_periodos:
            for pos in pos_list:
                if pos['status'] == 'open':
                    try:
                        repo.atualizar_posicao(pos['id'], status='closed')
                        n_atualizadas += 1
                    except Exception as e:
                        print(f"[RECONCILIAR] Erro ao fechar órfã {pos['id']}: {e}", flush=True)

    if n_atualizadas:
        print(f"[RECONCILIAR] Produto {produto_id}: {n_atualizadas} posições atualizadas", flush=True)


def preencher_preco_saida_posicoes_fechadas(repo, produto_id=None):
    """
    Busca no CoinGecko (e fallback Bitget) o preço de fechamento na data_saida
    das posições fechadas dos produtos AC/EXC/HB/LC que estão sem preco_saida.

    produto_id: se informado, processa só esse produto; senão, os 4 produtos.
    Retorna dict: preenchidas, preenchidas_bitget, sem_identificador, erros, total.
    """
    ids_produtos = list(_PRODUTOS_ALOCACAO_LIVRE) if produto_id is None else [int(produto_id)]
    with repo._get_connection() as conn:
        cur = conn.cursor()
        if produto_id is not None:
            cur.execute("""
                SELECT p.id, p.ativo, CAST(p.data_saida AS TEXT), p.coingecko_id, p.exchange_symbol
                FROM posicoes p
                WHERE p.produto_id = %s AND p.status = 'closed'
                AND p.data_saida IS NOT NULL AND (p.preco_saida IS NULL OR p.preco_saida = 0)
                ORDER BY p.data_saida
            """, (produto_id,))
        else:
            placeholders = ','.join(['%s'] * len(ids_produtos))
            cur.execute(f"""
                SELECT p.id, p.ativo, CAST(p.data_saida AS TEXT), p.coingecko_id, p.exchange_symbol
                FROM posicoes p
                WHERE p.produto_id IN ({placeholders}) AND p.status = 'closed'
                AND p.data_saida IS NOT NULL AND (p.preco_saida IS NULL OR p.preco_saida = 0)
                ORDER BY p.data_saida
            """, ids_produtos)
        rows = cur.fetchall()

    if not rows:
        return {'preenchidas': 0, 'preenchidas_bitget': 0, 'sem_identificador': 0, 'erros': 0, 'total': 0}

    try:
        cotacoes = CotacoesService(db_url=repo.db_url)
    except Exception as e:
        print(f"[PREENCHER-PRECO-SAIDA] CotacoesService indisponível: {e}", flush=True)
        return {'preenchidas': 0, 'preenchidas_bitget': 0, 'sem_identificador': 0, 'erros': len(rows), 'total': len(rows), 'erro_servico': str(e)}

    preenchidas = 0
    preenchidas_bitget = 0
    sem_identificador = 0
    erros = 0

    import time as _time

    # Stablecoins: preço fixo = 1.0 (não faz sentido buscar API)
    _STABLECOINS = {'USDT', 'USDC', 'TUSD', 'BUSD', 'DAI', 'UST', 'GUSD', 'USDP', 'FDUSD'}

    print(f"[PREENCHER-PRECO-SAIDA] {len(rows)} posições fechadas sem preco_saida", flush=True)

    for (pos_id, ativo, data_saida, coingecko_id, exchange_symbol) in rows:
        data_str = str(data_saida)[:10] if data_saida else ''
        if not data_str:
            erros += 1
            continue
        ativo_nome = str(ativo).strip() if ativo else ''

        # Stablecoins: pular (deixar sem preco_saida — exibido como "—")
        if ativo_nome.upper() in _STABLECOINS:
            print(f"[PREENCHER-PRECO-SAIDA] SKIP stablecoin: pos_id={pos_id} {ativo_nome} {data_str}", flush=True)
            continue

        cg_id = (coingecko_id or '').strip() or None
        if not cg_id:
            info = repo.obter_ativo(ativo_nome)
            cg_id = (info.get('coingecko_id') or '').strip() or None if info else None
        symbol = (exchange_symbol or '').strip() or None
        if not symbol and ativo_nome:
            symbol = ativo_nome.upper().replace(' ', '') + 'USDT'

        preco = None
        fonte = None

        # 1. Tentar Bitget primeiro (API pública, rápida, sem key)
        if symbol:
            try:
                preco = cotacoes.obter_preco_fechamento_bitget_data(symbol, data_str)
                if preco is not None:
                    fonte = 'bitget'
            except Exception as e:
                print(f"[PREENCHER-PRECO-SAIDA] Bitget falhou para {symbol} em {data_str}: {e}", flush=True)

        # 2. Fallback CoinGecko
        if preco is None and cg_id:
            try:
                res = cotacoes.obter_preco_historico_exato(cg_id, data_str)
                if res.get('status') in ('ok', 'fallback') and res.get('preco') is not None:
                    preco = float(res['preco'])
                    fonte = 'coingecko'
            except Exception as e:
                print(f"[PREENCHER-PRECO-SAIDA] CoinGecko falhou para {cg_id} em {data_str}: {e}", flush=True)

        if preco is not None and fonte:
            try:
                repo.atualizar_posicao(pos_id, preco_saida=preco)
                if fonte == 'bitget':
                    preenchidas_bitget += 1
                else:
                    preenchidas += 1
                print(f"[PREENCHER-PRECO-SAIDA] OK pos_id={pos_id} {ativo_nome} {data_str} -> {preco:.6f} ({fonte})", flush=True)
            except Exception as e:
                erros += 1
                print(f"[PREENCHER-PRECO-SAIDA] Erro ao atualizar posição {pos_id}: {e}", flush=True)
        elif not cg_id and not symbol:
            sem_identificador += 1
            print(f"[PREENCHER-PRECO-SAIDA] SEM ID: pos_id={pos_id} {ativo_nome} {data_str}", flush=True)
        else:
            erros += 1
            print(f"[PREENCHER-PRECO-SAIDA] NÃO ENCONTRADO: pos_id={pos_id} {ativo_nome} {data_str} (cg={cg_id}, sym={symbol})", flush=True)

        _time.sleep(0.2)  # rate limit gentil

    resultado = {
        'preenchidas': preenchidas,
        'preenchidas_bitget': preenchidas_bitget,
        'sem_identificador': sem_identificador,
        'erros': erros,
        'total': len(rows)
    }
    print(f"[PREENCHER-PRECO-SAIDA] Resultado: {resultado}", flush=True)
    return resultado


def preencher_precos_posicoes(repo, produto_id, tipo='ambos'):
    """
    Preenche preços faltantes nas posições de um produto.

    tipo:
      - 'entrada': preenche preco_entrada na data_entrada (posições com preco_entrada NULL ou 0)
      - 'saida':   preenche preco_saida na data_saida (posições fechadas com preco_saida NULL ou 0)
      - 'ambos':   faz os dois

    Usa Bitget primeiro, depois CoinGecko como fallback.
    Pula stablecoins (USDT, USDC, TUSD, etc.).
    Retorna dict com contagem de preenchidos/erros.
    """
    import time as _time

    _STABLECOINS = {'USDT', 'USDC', 'TUSD', 'BUSD', 'DAI', 'UST', 'GUSD', 'USDP', 'FDUSD'}
    resultado = {
        'entrada_preenchidas': 0, 'entrada_total': 0,
        'saida_preenchidas': 0, 'saida_total': 0,
        'erros': 0, 'skipped_stable': 0
    }

    try:
        cotacoes = CotacoesService(db_url=repo.db_url)
    except Exception as e:
        print(f"[PREENCHER-PRECOS] CotacoesService indisponível: {e}", flush=True)
        resultado['erro_servico'] = str(e)
        return resultado

    def _buscar_preco(repo, ativo_nome, data_str, coingecko_id, exchange_symbol):
        """Busca preço histórico: Bitget → CoinGecko fallback. Retorna (preco, fonte) ou (None, None)."""
        cg_id = (coingecko_id or '').strip() or None
        if not cg_id:
            info = repo.obter_ativo(ativo_nome)
            cg_id = (info.get('coingecko_id') or '').strip() or None if info else None
        symbol = (exchange_symbol or '').strip() or None
        if not symbol and ativo_nome:
            symbol = ativo_nome.upper().replace(' ', '') + 'USDT'

        preco = None
        fonte = None

        # 1. Bitget (API pública, rápida)
        if symbol:
            try:
                preco = cotacoes.obter_preco_fechamento_bitget_data(symbol, data_str)
                if preco is not None:
                    fonte = 'bitget'
            except Exception:
                pass

        # 2. CoinGecko fallback
        if preco is None and cg_id:
            try:
                res = cotacoes.obter_preco_historico_exato(cg_id, data_str)
                if res.get('status') in ('ok', 'fallback') and res.get('preco') is not None:
                    preco = float(res['preco'])
                    fonte = 'coingecko'
            except Exception:
                pass

        return preco, fonte

    # --- Preços de ENTRADA ---
    if tipo in ('entrada', 'ambos'):
        with repo._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, ativo, CAST(data_entrada AS TEXT), coingecko_id, exchange_symbol
                FROM posicoes
                WHERE produto_id = %s
                  AND data_entrada IS NOT NULL
                  AND (preco_entrada IS NULL OR preco_entrada = 0)
            """, (produto_id,))
            rows_entrada = cur.fetchall()

        resultado['entrada_total'] = len(rows_entrada)
        print(f"[PREENCHER-PRECOS] Produto {produto_id}: {len(rows_entrada)} posições sem preco_entrada", flush=True)

        for (pos_id, ativo, data_entrada, cg_id, ex_sym) in rows_entrada:
            data_str = str(data_entrada)[:10] if data_entrada else ''
            if not data_str:
                resultado['erros'] += 1
                continue
            ativo_nome = str(ativo).strip() if ativo else ''
            if ativo_nome.upper() in _STABLECOINS:
                resultado['skipped_stable'] += 1
                print(f"[PREENCHER-PRECOS] SKIP stablecoin entrada: {ativo_nome}", flush=True)
                continue

            preco, fonte = _buscar_preco(repo, ativo_nome, data_str, cg_id, ex_sym)
            if preco is not None:
                try:
                    repo.atualizar_posicao(pos_id, preco_entrada=preco)
                    resultado['entrada_preenchidas'] += 1
                    print(f"[PREENCHER-PRECOS] OK ENTRADA pos_id={pos_id} {ativo_nome} {data_str} -> {preco:.6f} ({fonte})", flush=True)
                except Exception as e:
                    resultado['erros'] += 1
                    print(f"[PREENCHER-PRECOS] Erro ao salvar entrada {pos_id}: {e}", flush=True)
            else:
                resultado['erros'] += 1
                print(f"[PREENCHER-PRECOS] NAO ENCONTRADO ENTRADA: {ativo_nome} {data_str}", flush=True)
            _time.sleep(0.2)

    # --- Preços de SAÍDA ---
    if tipo in ('saida', 'ambos'):
        with repo._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, ativo, CAST(data_saida AS TEXT), coingecko_id, exchange_symbol
                FROM posicoes
                WHERE produto_id = %s AND status = 'closed'
                  AND data_saida IS NOT NULL
                  AND (preco_saida IS NULL OR preco_saida = 0)
            """, (produto_id,))
            rows_saida = cur.fetchall()

        resultado['saida_total'] = len(rows_saida)
        print(f"[PREENCHER-PRECOS] Produto {produto_id}: {len(rows_saida)} posições fechadas sem preco_saida", flush=True)

        for (pos_id, ativo, data_saida, cg_id, ex_sym) in rows_saida:
            data_str = str(data_saida)[:10] if data_saida else ''
            if not data_str:
                resultado['erros'] += 1
                continue
            ativo_nome = str(ativo).strip() if ativo else ''
            if ativo_nome.upper() in _STABLECOINS:
                resultado['skipped_stable'] += 1
                print(f"[PREENCHER-PRECOS] SKIP stablecoin saída: {ativo_nome}", flush=True)
                continue

            preco, fonte = _buscar_preco(repo, ativo_nome, data_str, cg_id, ex_sym)
            if preco is not None:
                try:
                    repo.atualizar_posicao(pos_id, preco_saida=preco)
                    resultado['saida_preenchidas'] += 1
                    print(f"[PREENCHER-PRECOS] OK SAIDA pos_id={pos_id} {ativo_nome} {data_str} -> {preco:.6f} ({fonte})", flush=True)
                except Exception as e:
                    resultado['erros'] += 1
                    print(f"[PREENCHER-PRECOS] Erro ao salvar saída {pos_id}: {e}", flush=True)
            else:
                resultado['erros'] += 1
                print(f"[PREENCHER-PRECOS] NAO ENCONTRADO SAIDA: {ativo_nome} {data_str}", flush=True)
            _time.sleep(0.2)

    print(f"[PREENCHER-PRECOS] Finalizado produto {produto_id}: {resultado}", flush=True)
    return resultado


# ============================================================
# HTTP SERVER
# ============================================================

class DashboardHandler(BaseHTTPRequestHandler):
    """Handler para requisicoes HTTP do dashboard"""

    # Tamanho minimo para usar GZIP (respostas muito pequenas ficam maiores comprimidas)
    _GZIP_MIN_SIZE = 256

    def _set_headers(self, status=200, content_type='text/html'):
        self.send_response(status)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_body(self, content_type, data_bytes, status=200, cache_control=None):
        """Envia resposta com body, opcionalmente comprimida (GZIP) e com Cache-Control."""
        accept_encoding = (self.headers.get('Accept-Encoding') or '').lower()
        use_gzip = 'gzip' in accept_encoding and len(data_bytes) >= self._GZIP_MIN_SIZE
        if use_gzip:
            body = gzip.compress(data_bytes, compresslevel=6)
        else:
            body = data_bytes

        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        if cache_control:
            self.send_header('Cache-Control', cache_control)
        if use_gzip:
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if not getattr(self, '_suppress_body', False):
            self.wfile.write(body)

    def _send_json(self, data, status=200, cache_control='private, max-age=0'):
        body = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
        self._send_body('application/json', body, status=status, cache_control=cache_control)

    def _send_html(self, html, status=200):
        body = html.encode('utf-8')
        self._send_body('text/html; charset=utf-8', body, status=status, cache_control='private, max-age=0')

    def _send_excel(self, buffer, filename):
        """Envia arquivo Excel para download"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        if not getattr(self, '_suppress_body', False):
            self.wfile.write(buffer.getvalue())

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_HEAD(self):
        """HEAD = mesma resposta que GET, sem body (evita 501 para Render e crons que usam HEAD)."""
        self._suppress_body = True
        try:
            self.do_GET()
        finally:
            self._suppress_body = False

    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError:
            data = {}

        try:
            repo = get_repo()
        except Exception as e:
            self._send_json({'erro': f'Falha na conexao com banco: {str(e)}'}, 500)
            return

        path = self.path

        # Criar produto
        if path == '/api/produto/criar':
            try:
                tipo_nome = data.get('tipo', 'Outro')
                produto = Produto(
                    nome=data.get('nome'),
                    data_inicio=data.get('data_inicio'),
                    tipo=Tipo(tipo_nome),
                    capital_inicial=float(data.get('capital_inicial', 0)) if data.get('capital_inicial') else 0.0
                )
                produto_id = repo.salvar_produto(produto)
                self._send_json({'sucesso': True, 'produto_id': produto_id})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Editar produto
        if path.startswith('/api/produto/') and path.endswith('/editar'):
            try:
                produto_id = int(path.split('/')[3])
                repo.atualizar_produto(
                    produto_id=produto_id,
                    nome=data.get('nome'),
                    tipo=data.get('tipo'),
                    data_inicio=data.get('data_inicio'),
                    capital_inicial=float(data.get('capital_inicial')) if data.get('capital_inicial') else None
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Deletar produto
        if path.startswith('/api/produto/') and path.endswith('/deletar'):
            try:
                produto_id = int(path.split('/')[3])
                repo.deletar_produto(produto_id, forcar=True)
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar posicao
        if path == '/api/posicao/criar':
            try:
                produto_id = int(data.get('produto_id'))
                posicao = Posicao(
                    ativo=data.get('ativo'),
                    side=data.get('side', 'long'),
                    data_entrada=data.get('data_entrada'),
                    preco_entrada=float(data.get('preco_entrada')),
                    coingecko_id=data.get('coingecko_id') or None,
                    exchange_symbol=data.get('exchange_symbol') or None
                )
                posicao_id = repo.salvar_posicao(produto_id, posicao)
                # Atributos: quantidade + campos específicos por produto
                attrs = {}
                if data.get('quantidade'):
                    attrs['quantidade'] = float(data.get('quantidade'))
                for k in ('perfil', 'alvo1', 'alvo2', 'motivo', 'relatorio',
                         'categoria', 'tipo_ico', 'rank', 'tipo_janela', 'tese',
                         'risco', 'atencao', 'execucao', 'por_que', 'local',
                         'ficha_tecnica', 'disponivel_pos_ico'):
                    v = data.get(k)
                    if v is not None and str(v).strip():
                        if k in ('alvo1', 'alvo2'):
                            try:
                                attrs[k] = float(v)
                            except (ValueError, TypeError):
                                pass
                        else:
                            attrs[k] = str(v).strip()
                if attrs:
                    repo.salvar_atributos_posicao(
                        posicao_id=posicao_id,
                        produto_id=produto_id,
                        **attrs
                    )

                # Sincronizar com turmas existentes
                try:
                    turmas_service = TurmasService(db_url=repo.db_url)
                    turmas_service.sync_nova_posicao(
                        posicao_id=posicao_id,
                        produto_id=produto_id,
                        data_entrada=data.get('data_entrada'),
                        preco_entrada=float(data.get('preco_entrada'))
                    )
                except Exception as sync_err:
                    print(f"Aviso: erro ao sincronizar posição com turmas: {sync_err}")

                self._send_json({'sucesso': True, 'posicao_id': posicao_id})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Editar posicao
        if path == '/api/posicao/editar':
            try:
                posicao_id = int(data.get('posicao_id'))
                repo.atualizar_posicao(
                    posicao_id=posicao_id,
                    ativo=data.get('ativo'),
                    side=data.get('tipo'),
                    data_entrada=data.get('data_entrada'),
                    preco_entrada=float(data.get('preco_entrada')),
                    coingecko_id=data.get('coingecko_id') or None,
                    exchange_symbol=data.get('exchange_symbol') or None
                )
                posicao = repo.carregar_posicao(posicao_id)
                if posicao:
                    attrs = {}
                    qty = data.get('quantidade')
                    if qty is not None and str(qty).strip() != '':
                        try:
                            attrs['quantidade'] = float(qty)
                        except (ValueError, TypeError):
                            pass
                    for k in ('perfil', 'alvo1', 'alvo2', 'motivo', 'relatorio',
                             'categoria', 'tipo_ico', 'rank', 'tipo_janela', 'tese',
                             'risco', 'atencao', 'execucao', 'por_que', 'local',
                             'ficha_tecnica', 'disponivel_pos_ico'):
                        v = data.get(k)
                        if v is not None and str(v).strip():
                            if k in ('alvo1', 'alvo2'):
                                try:
                                    attrs[k] = float(v)
                                except (ValueError, TypeError):
                                    pass
                            else:
                                attrs[k] = str(v).strip()
                    if attrs:
                        repo.salvar_atributos_posicao(
                            posicao_id=posicao_id,
                            produto_id=posicao['produto_id'],
                            **attrs
                        )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Fechar posicao
        if path == '/api/posicao/fechar':
            try:
                posicao_id = int(data.get('posicao_id'))
                data_saida = data.get('data_saida')
                repo.atualizar_posicao(
                    posicao_id=posicao_id,
                    data_saida=data_saida,
                    preco_saida=float(data.get('preco_saida')),
                    status='closed'
                )
                # Salvar atributos (resultado para ICOs, motivo)
                posicao = repo.carregar_posicao(posicao_id)
                if posicao:
                    attrs = {}
                    for k in ('resultado', 'motivo'):
                        v = data.get(k)
                        if v is not None and str(v).strip():
                            attrs[k] = str(v).strip()
                    if attrs:
                        repo.salvar_atributos_posicao(
                            posicao_id=posicao_id,
                            produto_id=posicao['produto_id'],
                            **attrs
                        )

                # Sincronizar fechamento com turmas
                try:
                    turmas_service = TurmasService(db_url=repo.db_url)
                    turmas_service.sync_posicao_fechada(
                        posicao_id=posicao_id,
                        data_saida=data_saida
                    )
                except Exception as sync_err:
                    print(f"Aviso: erro ao sincronizar fechamento com turmas: {sync_err}")

                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar stop
        if path == '/api/stop/criar':
            try:
                repo.adicionar_stop_posicao(
                    posicao_id=int(data.get('posicao_id')),
                    data=data.get('data_stop'),
                    valor=float(data.get('preco'))
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Configurar ATR Stop
        if path == '/api/posicao/atr':
            try:
                from datetime import date as dt_date
                posicao_id = int(data.get('posicao_id'))
                atr_period = data.get('atr_period')
                atr_multiplier = data.get('atr_multiplier')
                atr_data_inicio = data.get('atr_data_inicio')

                # Se valores são None ou null string, remove a configuração
                if atr_period in [None, 'null', '']:
                    atr_period = None
                else:
                    atr_period = int(atr_period)

                if atr_multiplier in [None, 'null', '']:
                    atr_multiplier = None
                else:
                    atr_multiplier = float(atr_multiplier)

                if atr_data_inicio in [None, 'null', '']:
                    atr_data_inicio = None

                repo.atualizar_posicao(
                    posicao_id=posicao_id,
                    atr_period=atr_period,
                    atr_multiplier=atr_multiplier,
                    atr_data_inicio=atr_data_inicio
                )

                # Invalidar cache ATR para que a próxima carga recalcule stops (incl. esta posição)
                posicao = repo.carregar_posicao(posicao_id)
                produto_id_atr = posicao.get('produto_id') if posicao else None
                if produto_id_atr is not None:
                    invalidar_cache_atr(produto_id_atr)
                    print(f"[ATR] Cache invalidado para produto {produto_id_atr} após config ATR posição {posicao_id}", flush=True)

                # Se ATR foi configurado, calcular e salvar o stop imediatamente
                mensagem = 'ATR configurado!'
                if atr_multiplier is not None and posicao and posicao.get('coingecko_id'):
                    # Usar atr_data_inicio se disponível, senão data_entrada
                    data_calculo = atr_data_inicio or posicao.get('data_entrada')
                    if hasattr(data_calculo, 'strftime'):
                        data_calculo = data_calculo.strftime('%Y-%m-%d')
                    else:
                        data_calculo = str(data_calculo)[:10] if data_calculo else None

                    exchange_symbol = posicao.get('exchange_symbol')
                    if exchange_symbol and str(exchange_symbol).lower() in ('none', 'nan', ''):
                        exchange_symbol = None

                    product_type = 'perpetuos'
                    produto_info = repo.carregar_produto(posicao.get('produto_id'))
                    if produto_info and produto_info.get('tipo'):
                        tipo_str = str(produto_info['tipo']).lower()
                        if 'spot' in tipo_str:
                            product_type = 'spot'

                    stop, breached, erro = calcular_stop_para_posicao(
                        coingecko_id=posicao['coingecko_id'],
                        side=posicao['side'],
                        data_entrada=data_calculo,
                        atr_period=atr_period,
                        atr_multiplier=atr_multiplier,
                        exchange_symbol=exchange_symbol,
                        product_type=product_type
                    )
                    if erro:
                        print(f"[ATR] Posição {posicao_id} ({posicao.get('ativo')}): {erro}", flush=True)
                        mensagem = 'ATR configurado. O stop será calculado na próxima atualização automática.'
                    elif breached:
                        ultimo_stop = repo.obter_ultimo_stop(posicao_id)
                        ja_notificado = (ultimo_stop is not None and float(ultimo_stop) == -1.0)
                        if not ja_notificado:
                            hoje = dt_date.today().strftime('%Y-%m-%d')
                            repo.adicionar_stop_posicao(posicao_id, hoje, -1)
                            try:
                                nome_produto = produto_info['nome'] if produto_info else "Desconhecido"
                                notificar_stop_atingido(
                                    ativo=posicao['ativo'],
                                    side=posicao['side'],
                                    preco_entrada=posicao.get('preco_entrada'),
                                    produto_nome=nome_produto,
                                    data_entrada=posicao.get('data_entrada')
                                )
                                print(f"[Dashboard] Notificação de stop enviada para {posicao['ativo']}", flush=True)
                            except Exception as e:
                                print(f"[Dashboard] Erro ao enviar notificação de stop: {e}", flush=True)
                        else:
                            print(f"[Dashboard] Stop já notificado para {posicao['ativo']} — pulando notificação", flush=True)
                        mensagem = 'ATR configurado - STOP ATINGIDO!'
                    elif stop is not None:
                        hoje = dt_date.today().strftime('%Y-%m-%d')
                        repo.adicionar_stop_posicao(posicao_id, hoje, stop)
                        mensagem = 'ATR configurado e stop calculado!'
                        print(f"[ATR] Stop salvo para posição {posicao_id} ({posicao.get('ativo')}): {stop}", flush=True)

                self._send_json({'sucesso': True, 'mensagem': mensagem})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar visualizacao
        if path == '/api/visualizacao/criar':
            try:
                viz_id = repo.criar_visualizacao(
                    produto_id=int(data.get('produto_id')),
                    nome=data.get('nome'),
                    colunas=data.get('colunas', []),
                    ordenacao=data.get('ordenacao'),
                    filtros=data.get('filtros')
                )
                self._send_json({'sucesso': True, 'visualizacao_id': viz_id})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Editar visualizacao
        if path == '/api/visualizacao/editar':
            try:
                viz_id = int(data.get('visualizacao_id'))
                repo.atualizar_visualizacao(
                    visualizacao_id=viz_id,
                    nome=data.get('nome'),
                    colunas=data.get('colunas'),
                    ordenacao=data.get('ordenacao'),
                    filtros=data.get('filtros')
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Deletar visualizacao
        if path == '/api/visualizacao/deletar':
            try:
                viz_id = int(data.get('visualizacao_id'))
                repo.deletar_visualizacao(viz_id)
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Deletar posicao
        if path == '/api/posicao/deletar':
            try:
                posicao_id = int(data.get('posicao_id'))
                repo.deletar_posicao(posicao_id, forcar=True)
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar atributo
        if path == '/api/atributo/criar':
            try:
                repo.adicionar_atributo_config(
                    produto_id=int(data.get('produto_id')),
                    atributo_nome=data.get('nome'),
                    atributo_tipo=data.get('tipo', 'text'),
                    atributo_label=data.get('label'),
                    obrigatorio=data.get('obrigatorio', False)
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Editar atributo
        if path == '/api/atributo/editar':
            try:
                repo.editar_atributo_config(
                    produto_id=int(data.get('produto_id')),
                    atributo_nome=data.get('nome'),
                    novo_label=data.get('label'),
                    novo_tipo=data.get('tipo'),
                    novo_obrigatorio=data.get('obrigatorio')
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Remover atributo do produto
        if path == '/api/atributo/remover':
            try:
                repo.remover_atributo_config(
                    produto_id=int(data.get('produto_id')),
                    atributo_nome=data.get('nome')
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Limpar colunas orfas
        if path == '/api/atributos/limpar-orfas':
            try:
                removidas = repo.limpar_colunas_orfas()
                self._send_json({'sucesso': True, 'removidas': len(removidas)})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar alocacao
        if path == '/api/alocacao/criar':
            try:
                from services.alocacao_service import AlocacaoService
                alocacao = AlocacaoService.criar_alocacao(
                    produto_id=int(data.get('produto_id')),
                    posicao_id=int(data.get('posicao_id')),
                    percentual=float(data.get('percentual')),
                    valor_usd=float(data.get('valor_usd')) if data.get('valor_usd') else None,
                    data=data.get('data_alocacao')
                )
                alocacao_id = repo.salvar_alocacao(int(data.get('produto_id')), alocacao)
                self._send_json({'sucesso': True, 'alocacao_id': alocacao_id})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar uma linha de alocações (uma data com % por ativo) — estilo planilha
        if path == '/api/alocacao/linha':
            try:
                produto_id = int(data.get('produto_id'))
                data_linha = (data.get('data') or '').strip()[:10]
                ativos = data.get('ativos') or {}
                if not data_linha:
                    self._send_json({'sucesso': False, 'erro': 'Campo "data" obrigatório'}, 400)
                    return
                from services.alocacao_service import AlocacaoService
                livre = _is_produto_livre(repo, produto_id)
                criadas = 0
                fechadas = 0
                erros = []
                for ativo, pct in ativos.items():
                    if not ativo:
                        continue
                    try:
                        pct_f = float(pct)
                    except (TypeError, ValueError):
                        continue

                    ativo_nome = ativo.strip()

                    if pct_f <= 0:
                        # Para produtos livres: alocação zerada → fechar posição
                        if livre:
                            if _fechar_posicao_se_zerado(repo, produto_id, ativo_nome, data_linha):
                                fechadas += 1
                        continue

                    # Alocação > 0: encontrar ou criar posição
                    posicao_id = _encontrar_ou_criar_posicao(repo, produto_id, ativo_nome, data_linha)
                    if not posicao_id:
                        erros.append(f"{ativo}: nenhuma posição encontrada/criada")
                        continue
                    alocacao = AlocacaoService.criar_alocacao(
                        produto_id=produto_id,
                        posicao_id=posicao_id,
                        percentual=pct_f,
                        valor_usd=None,
                        data=data_linha
                    )
                    repo.salvar_alocacao(produto_id, alocacao)
                    criadas += 1
                self._send_json({'sucesso': True, 'alocacoes_criadas': criadas, 'posicoes_fechadas': fechadas, 'avisos': erros})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Atualizar uma célula de alocação (produto_id, data, ativo, percentual)
        if path == '/api/alocacao/celula':
            try:
                produto_id = int(data.get('produto_id'))
                data_linha = (data.get('data') or '').strip()[:10]
                ativo = (data.get('ativo') or '').strip()
                try:
                    percentual = float(data.get('percentual'))
                except (TypeError, ValueError):
                    self._send_json({'sucesso': False, 'erro': 'percentual inválido'}, 400)
                    return
                if not data_linha or not ativo:
                    self._send_json({'sucesso': False, 'erro': 'data e ativo obrigatórios'}, 400)
                    return
                with repo._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT a.id, CAST(a.data AS TEXT) FROM alocacoes a
                        INNER JOIN posicoes p ON a.posicao_id = p.id
                        WHERE a.produto_id = %s AND p.ativo = %s AND a.status = 'active'
                    """, (produto_id, ativo))
                    rows = cur.fetchall()
                found = None
                for r in rows:
                    r_data = str(r[1])[:10] if r[1] else ''
                    if r_data == data_linha:
                        found = r[0]
                        break
                if found:
                    repo.atualizar_alocacao(found, percentual=percentual)
                    # Se editou para 0, verificar se deve fechar a posição
                    if percentual <= 0:
                        _fechar_posicao_se_zerado(repo, produto_id, ativo, data_linha)
                    self._send_json({'sucesso': True})
                else:
                    if percentual <= 0:
                        # Alocação 0 e não existe → nada a fazer (mas fechar posição se produto livre)
                        _fechar_posicao_se_zerado(repo, produto_id, ativo, data_linha)
                        self._send_json({'sucesso': True})
                        return
                    posicao_id = _encontrar_ou_criar_posicao(repo, produto_id, ativo, data_linha)
                    if not posicao_id:
                        self._send_json({'sucesso': False, 'erro': f'Não foi possível encontrar/criar posição para {ativo}'}, 404)
                        return
                    from services.alocacao_service import AlocacaoService
                    alocacao = AlocacaoService.criar_alocacao(
                        produto_id=produto_id,
                        posicao_id=posicao_id,
                        percentual=percentual,
                        valor_usd=None,
                        data=data_linha
                    )
                    repo.salvar_alocacao(produto_id, alocacao)
                    self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Adicionar novo ativo (nova coluna) na tabela de alocação: cria posição mínima + opcional coingecko_id/exchange_symbol
        if path == '/api/alocacao/adicionar-ativo':
            try:
                produto_id = int(data.get('produto_id'))
                ativo = (data.get('ativo') or '').strip()
                if not ativo:
                    self._send_json({'sucesso': False, 'erro': 'Informe o nome do ativo (ex: BTC, ETH).'}, 400)
                    return
                coingecko_id = (data.get('coingecko_id') or '').strip() or None
                exchange_symbol = (data.get('exchange_symbol') or '').strip() or None
                produto = repo.carregar_produto(produto_id)
                if not produto:
                    self._send_json({'sucesso': False, 'erro': 'Produto não encontrado.'}, 404)
                    return
                from datetime import date
                hoje = date.today().isoformat()
                posicao_id = _encontrar_ou_criar_posicao(repo, produto_id, ativo, hoje)
                if not posicao_id:
                    self._send_json({
                        'sucesso': False,
                        'erro': 'Não foi possível criar a posição. Para produtos que não são de alocação livre (AC/EXC/HB/LC), use "Nova Posição" na página do produto.'
                    }, 400)
                    return
                updates = {}
                if coingecko_id:
                    updates['coingecko_id'] = coingecko_id
                if exchange_symbol:
                    updates['exchange_symbol'] = exchange_symbol
                if updates:
                    try:
                        repo.atualizar_posicao(posicao_id, **updates)
                        if coingecko_id:
                            repo.registrar_ativo(ativo, coingecko_id)
                    except Exception as e:
                        print(f"[ADICIONAR-ATIVO] Aviso ao salvar coingecko_id/symbol: {e}", flush=True)
                self._send_json({'sucesso': True, 'mensagem': f'Ativo "{ativo}" adicionado. A tabela será atualizada.'})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # ============================================================
        # ROTAS DE TURMAS (POST)
        # ============================================================

        # Criar turma com busca automática de preços
        # posicoes_config: [{"posicao_id": 1, "data_insercao": "2026-01-15"}, ...]
        # preco_entrada_turma é OPCIONAL se auto_fetch_prices=true (default)
        # Use /api/turma/posicoes-elegiveis para listar posições elegíveis
        if path == '/api/turma/criar':
            try:
                turmas_service = TurmasService(db_url=repo.db_url)
                produto_id = int(data.get('produto_id'))
                data_inicio = data.get('data_inicio')

                # posicoes_config: se ausente, preencher com elegíveis (retrocompat)
                # se enviado explicitamente (mesmo vazio []), respeitar - permite criar turma sem posições
                posicoes_config = data.get('posicoes_config')
                if posicoes_config is None:
                    posicoes = turmas_service.listar_posicoes_elegiveis(produto_id, data_inicio)
                    posicoes_config = [
                        {'posicao_id': p['id'], 'data_insercao': data_inicio}
                        for p in posicoes
                    ]

                # auto_fetch_prices: se True (default), busca preços via CoinGecko
                auto_fetch_prices = data.get('auto_fetch_prices', True)

                resultado = turmas_service.criar_turma(
                    produto_id=produto_id,
                    nome=data.get('nome'),
                    data_inicio=data_inicio,
                    capital_base=float(data.get('capital_base', 1500)),
                    descricao=data.get('descricao'),
                    posicoes_config=posicoes_config,
                    auto_fetch_prices=auto_fetch_prices
                )

                self._send_json({
                    'sucesso': True,
                    'turma_id': resultado['turma_id'],
                    'precos_resolvidos': resultado['precos_resolvidos'],
                    'avisos': resultado['avisos']
                })
            except ValueError as e:
                # Validation error (missing posicoes_config, missing fields, etc.)
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        self._send_json({'erro': 'Rota nao encontrada'}, 404)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # Favicon na raiz (navegadores solicitam /favicon.ico automaticamente)
        if path == '/favicon.ico' or path == '/favicon.png':
            from pathlib import Path as _Path
            _assets_dir = _Path(__file__).resolve().parent.parent / 'assets'
            _file_path = _assets_dir / 'favicon.png'
            if _file_path.is_file():
                with open(_file_path, 'rb') as f:
                    self._send_body('image/png', f.read(), cache_control='public, max-age=86400')
                return
            self._send_html('', 404)
            return

        # Health check (sem conexao ao banco - para Render/cloud)
        if path == '/health':
            self._send_json(
                {'status': 'ok', 'timestamp': datetime.now().isoformat()},
                cache_control='public, max-age=30'
            )
            return

        try:
            repo = get_repo()
        except Exception as e:
            self._send_json({'erro': f'Falha na conexao com banco: {str(e)}'}, 500)
            return

        # API: Debug stops (temporário para diagnóstico)
        if path == '/api/debug/stops':
            try:
                produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
                with repo.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM stops")
                    total = cursor.fetchone()[0]
                    info = {
                        'db_tipo': 'SQLite' if repo._use_sqlite else 'PostgreSQL',
                        'total_stops': total,
                    }
                    if produto_id:
                        # Posições abertas deste produto
                        cursor.execute("""
                            SELECT id, ativo, side, status FROM posicoes 
                            WHERE produto_id = %s AND status = 'open' ORDER BY ativo
                        """, (produto_id,))
                        pos_abertas = cursor.fetchall()
                        pos_ids = [r[0] for r in pos_abertas]
                        info['posicoes_abertas'] = [{'id': r[0], 'ativo': r[1], 'side': r[2]} for r in pos_abertas]

                        # Stops para essas posições (mesma lógica do _batch_load_stops)
                        if pos_ids:
                            placeholders = ','.join(['%s'] * len(pos_ids))
                            cursor.execute(f"""
                                SELECT posicao_id, valor FROM (
                                    SELECT posicao_id, valor,
                                           ROW_NUMBER() OVER (PARTITION BY posicao_id ORDER BY data DESC, id DESC) as rn
                                    FROM stops
                                    WHERE posicao_id IN ({placeholders})
                                ) sub WHERE rn = 1
                            """, pos_ids)
                            batch_result = cursor.fetchall()
                            info['batch_load_result'] = [{'posicao_id': r[0], 'valor': r[1], 'tipo_id': str(type(r[0])), 'tipo_val': str(type(r[1]))} for r in batch_result]

                            # Verificar quais posições NÃO têm stop
                            ids_com_stop = {r[0] for r in batch_result}
                            info['posicoes_sem_stop'] = [pid for pid in pos_ids if pid not in ids_com_stop]

                            # Todos os stops dessas posições (para ver se existem)
                            cursor.execute(f"""
                                SELECT posicao_id, data, valor FROM stops 
                                WHERE posicao_id IN ({placeholders}) ORDER BY posicao_id, data DESC
                            """, pos_ids)
                            all_stops = cursor.fetchall()
                            info['todos_stops_posicoes'] = [{'posicao_id': r[0], 'data': r[1], 'valor': r[2]} for r in all_stops]
                            # Último stop por ativo no produto (fallback: qualquer posição)
                            ativos_abertas = [r[1] for r in pos_abertas]
                            if ativos_abertas:
                                ph = ','.join(['%s'] * len(ativos_abertas))
                                cursor.execute(f"""
                                    SELECT ativo, posicao_id, data, valor FROM (
                                        SELECT p.ativo, p.id as posicao_id, s.data, s.valor,
                                               ROW_NUMBER() OVER (PARTITION BY p.ativo ORDER BY s.data DESC) as rn
                                        FROM stops s JOIN posicoes p ON p.id = s.posicao_id
                                        WHERE p.produto_id = %s AND p.ativo IN ({ph})
                                    ) sub WHERE rn = 1
                                """, [produto_id] + ativos_abertas)
                                por_ativo = cursor.fetchall()
                                info['ultimo_stop_por_ativo'] = [{'ativo': r[0], 'posicao_id': r[1], 'data': r[2], 'valor': r[3]} for r in por_ativo]
                        else:
                            info['batch_load_result'] = []
                            info['posicoes_sem_stop'] = []
                    else:
                        cursor.execute("SELECT s.posicao_id, s.data, s.valor, p.ativo FROM stops s LEFT JOIN posicoes p ON s.posicao_id = p.id ORDER BY s.data DESC LIMIT 20")
                        rows = cursor.fetchall()
                        info['ultimos_20'] = [{'posicao_id': r[0], 'data': r[1], 'valor': r[2], 'ativo': r[3]} for r in rows]
                self._send_json(info)
            except Exception as e:
                import traceback
                self._send_json({'erro': str(e), 'traceback': traceback.format_exc(), 'db_tipo': 'SQLite' if repo._use_sqlite else 'PostgreSQL'}, 500)
            return

        # API: Tabela de alocações (pivot: datas x ativos) para tela estilo planilha
        if path == '/api/alocacoes/tabela':
            try:
                produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
                if not produto_id:
                    self._send_json({'erro': 'produto_id obrigatório'}, 400)
                    return
                produto = repo.carregar_produto(produto_id)
                if not produto:
                    self._send_json({'erro': 'Produto não encontrado'}, 404)
                    return
                with repo._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT DISTINCT ativo FROM posicoes WHERE produto_id = %s ORDER BY ativo
                    """, (produto_id,))
                    colunas = [r[0] for r in cur.fetchall()]
                    cur.execute("""
                        SELECT CAST(a.data AS TEXT), p.ativo, a.percentual
                        FROM alocacoes a
                        INNER JOIN posicoes p ON a.posicao_id = p.id
                        WHERE a.produto_id = %s AND a.status = 'active'
                        ORDER BY a.data, p.ativo
                    """, (produto_id,))
                    rows = cur.fetchall()
                from collections import defaultdict
                by_date = defaultdict(dict)
                for data, ativo, pct in rows:
                    d = str(data)[:10] if data else ''
                    by_date[d][ativo] = round(float(pct), 2)
                linhas = [{'data': d, 'ativos': by_date[d]} for d in sorted(by_date.keys())]
                self._send_json({
                    'produto_id': produto_id,
                    'produto_nome': produto.get('nome', ''),
                    'colunas': colunas,
                    'linhas': linhas
                })
            except Exception as e:
                self._send_json({'erro': str(e)}, 500)
            return

        # Servir arquivos estáticos da pasta assets
        if path.startswith('/assets/'):
            from pathlib import Path as _Path
            _assets_dir = _Path(__file__).resolve().parent.parent / 'assets'
            _file_path = _assets_dir / path[len('/assets/'):]
            if _file_path.resolve().is_file() and str(_file_path.resolve()).startswith(str(_assets_dir)):
                _ext = _file_path.suffix.lower()
                _mime = {
                    '.js': 'application/javascript',
                    '.css': 'text/css',
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.svg': 'image/svg+xml',
                    '.ico': 'image/x-icon',
                }.get(_ext, 'application/octet-stream')
                with open(_file_path, 'rb') as f:
                    data = f.read()
                self._send_body(_mime, data, cache_control='public, max-age=86400')
            else:
                self._send_html('<h1>404 Not Found</h1>', 404)
            return

        # Dashboard principal
        if path == '/' or path == '':
            if query.get('__content', [''])[0] != '1':
                self._send_html(get_loader_only_html())
                return
            produtos_visiveis, stats = _get_dashboard_data(repo)
            html = get_dashboard_html(produtos_visiveis, stats, repo, skip_loader=True)
            self._send_html(html)
            return

        # API: dados da home para atualizacao parcial (fetch sem reload)
        if path == '/api/dashboard/home':
            try:
                produtos_visiveis, stats = _get_dashboard_data(repo)
                cards_html = _build_dashboard_cards_html(produtos_visiveis, stats)
                timestamp = _now_brasilia().strftime("%d/%m/%Y %H:%M:%S")
                self._send_json({'timestamp': timestamp, 'cards_html': cards_html})
            except Exception as e:
                self._send_json({'erro': str(e)}, 500)
            return

        # Menu completo
        if path == '/menu':
            self._send_html(get_menu_html())
            return

        # Formulario novo produto
        if path == '/produto/novo':
            is_modal = query.get('_modal', [''])[0] == '1'
            inner = get_form_produto_html(as_inner=True)
            self._send_html(inner if is_modal else get_form_produto_html())
            return

        # Lista para editar produtos
        if path == '/produtos/editar':
            produtos = repo.listar_produtos()
            self._send_html(get_lista_produtos_html(produtos, "editar"))
            return

        # Lista para deletar produtos
        if path == '/produtos/deletar':
            produtos = repo.listar_produtos()
            self._send_html(get_lista_produtos_html(produtos, "deletar"))
            return

        # Formulario nova posicao
        if path == '/posicao/nova':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            produtos = repo.listar_produtos()
            is_modal = query.get('_modal', [''])[0] == '1'
            if produto_id:
                inner = get_form_posicao_html(produto_id, produtos, as_inner=True)
                self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Nova Posicao"))
            else:
                self._send_html(get_form_posicao_html(produto_id, produtos))
            return

        # Lista posicoes para editar
        if path == '/posicoes/editar':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            is_modal = query.get('_modal', [''])[0] == '1'
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    inner = get_lista_posicoes_html(produto, posicoes, "editar", as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Editar Posicao"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Lista posicoes para adicionar stop
        if path == '/stop/novo':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            is_modal = query.get('_modal', [''])[0] == '1'
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    inner = get_lista_posicoes_html(produto, posicoes, "stop", as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Adicionar Stop"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Lista posicoes para fechar
        if path == '/posicao/fechar':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            is_modal = query.get('_modal', [''])[0] == '1'
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    inner = get_lista_posicoes_html(produto, posicoes, "fechar", as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Fechar Posicao"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Lista posicoes para configurar ATR
        if path == '/atr/config':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            is_modal = query.get('_modal', [''])[0] == '1'
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    inner = get_lista_posicoes_html(produto, posicoes, "atr", as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "ATR Stop"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Lista posicoes para deletar
        if path == '/posicoes/deletar':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            is_modal = query.get('_modal', [''])[0] == '1'
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    inner = get_lista_posicoes_html(produto, posicoes, "deletar", as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Deletar Posicao"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Tela de alocações em tabela (estilo planilha: Data x Ativos)
        if path == '/alocacao/nova':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            produtos = repo.listar_produtos()
            self._send_html(get_form_alocacao_html(produto_id=produto_id, produtos=produtos))
            return

        # API: Listar posicoes abertas de um produto (para AJAX)
        if path.startswith('/api/posicoes/abertas/'):
            try:
                produto_id = int(path.split('/')[-1])
                posicoes_df = repo.carregar_posicoes_abertas(produto_id)
                posicoes_list = []
                if posicoes_df is not None and not posicoes_df.empty:
                    for _, row in posicoes_df.iterrows():
                        posicoes_list.append({
                            'id': int(row.get('id') or row.get('ID')),
                            'ativo': row.get('ativo') or row.get('Ativo', 'N/A'),
                            'side': row.get('side') or row.get('tipo', 'N/A'),
                            'preco_entrada': float(row.get('preco_entrada') or row.get('Preço Entrada', 0))
                        })
                self._send_json({'posicoes': posicoes_list})
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # Rotas de posicao especifica
        if path.startswith('/posicao/') and not path.startswith('/posicao/nova'):
            parts = path.split('/')
            if len(parts) >= 4:
                try:
                    posicao_id = int(parts[2])
                    acao = parts[3]
                    produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
                    is_modal = query.get('_modal', [''])[0] == '1'

                    if produto_id:
                        produto = repo.carregar_produto(produto_id)
                        posicao = repo.carregar_posicao(posicao_id)

                        if produto and posicao:
                            pid = produto['id']
                            if acao == 'editar':
                                attrs = repo.carregar_atributos_posicao(posicao_id)
                                if attrs:
                                    for k, v in attrs.items():
                                        if k not in ('posicao_id', 'produto_id') and posicao.get(k) is None:
                                            posicao[k] = v
                                inner = get_form_editar_posicao_html(produto, posicao, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, pid, "Editar Posicao"))
                                return
                            elif acao == 'stop':
                                inner = get_form_adicionar_stop_html(produto, posicao, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, pid, "Adicionar Stop"))
                                return
                            elif acao == 'atr':
                                inner = get_form_atr_stop_html(produto, posicao, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, pid, "ATR Stop"))
                                return
                            elif acao == 'fechar':
                                inner = get_form_fechar_posicao_html(produto, posicao, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, pid, "Fechar Posicao"))
                                return
                            elif acao == 'deletar':
                                inner = get_confirmar_delete_posicao_html(produto, posicao, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, pid, "Deletar Posicao"))
                                return
                except ValueError:
                    pass
            self._send_html("<h1>Posicao nao encontrada</h1>", 404)
            return

        # Rotas de produto
        if path.startswith('/produto/'):
            parts = path.split('/')

            if len(parts) >= 3:
                try:
                    produto_id = int(parts[2])
                except ValueError:
                    # Pode ser /produto/novo que ja foi tratado acima
                    self._send_html("<h1>Rota invalida</h1>", 404)
                    return

                produto = repo.carregar_produto(produto_id)
                if not produto:
                    self._send_html("<h1>Produto nao encontrado</h1>", 404)
                    return

                is_modal = query.get('_modal', [''])[0] == '1'

                # Editar produto
                if len(parts) >= 4 and parts[3] == 'editar':
                    inner = get_form_produto_html(produto, as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Editar Produto"))
                    return

                # Confirmar delete
                if len(parts) >= 4 and parts[3] == 'deletar':
                    inner = get_confirmar_delete_html(produto, as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Confirmar Exclusao"))
                    return

                # Gerenciar visualizacoes - lista
                if len(parts) >= 4 and parts[3] == 'visualizacoes':
                    visualizacoes = repo.listar_visualizacoes(produto_id)
                    inner = get_lista_visualizacoes_html(produto, visualizacoes, as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Visualizacoes"))
                    return

                # Gerenciar atributos
                if len(parts) >= 4 and parts[3] == 'atributos':
                    # Novo atributo
                    if len(parts) >= 5 and parts[4] == 'novo':
                        colunas_existentes = repo.listar_colunas_atributos()
                        inner = get_form_atributo_html(produto, None, colunas_existentes, as_inner=True)
                        self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Novo Atributo"))
                        return

                    # Atributo especifico
                    if len(parts) >= 6:
                        atributo_nome = parts[4]
                        acao = parts[5]
                        configs = repo.carregar_atributos_config(produto_id)
                        config = next((c for c in configs if c['atributo_nome'] == atributo_nome), None)

                        if config:
                            if acao == 'editar':
                                inner = get_form_atributo_html(produto, config, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Editar Atributo"))
                                return
                            elif acao == 'remover':
                                inner = get_confirmar_remover_atributo_html(produto, config, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Remover Atributo"))
                                return

                    # Lista de atributos (default)
                    configs = repo.carregar_atributos_config(produto_id)
                    colunas_orfas = repo.listar_colunas_orfas()
                    inner = get_lista_atributos_html(produto, configs, colunas_orfas, as_inner=True)
                    self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Atributos"))
                    return

                # Visualizacoes - criar/editar/deletar/ver
                if len(parts) >= 5 and parts[3] == 'viz':
                    # Nova visualizacao
                    if parts[4] == 'nova':
                        colunas_disponiveis = repo.obter_colunas_disponiveis(produto_id)
                        inner = get_form_nova_visualizacao_html(produto, colunas_disponiveis, as_inner=True)
                        self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Nova Visualizacao"))
                        return

                    # Visualizacao especifica
                    try:
                        viz_id = int(parts[4])
                        viz = repo.carregar_visualizacao(viz_id)
                        if viz and viz['produto_id'] == produto_id:
                            # Editar visualizacao
                            if len(parts) >= 6 and parts[5] == 'editar':
                                colunas_disponiveis = repo.obter_colunas_disponiveis(produto_id)
                                inner = get_form_editar_visualizacao_html(produto, viz, colunas_disponiveis, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Editar Visualizacao"))
                                return

                            # Deletar visualizacao
                            if len(parts) >= 6 and parts[5] == 'deletar':
                                inner = get_confirmar_delete_visualizacao_html(produto, viz, as_inner=True)
                                self._send_html(inner if is_modal else get_form_page_with_background(inner, produto_id, "Deletar Visualizacao"))
                                return

                            # Ver visualizacao
                            # Para produtos de alocação livre, reconciliar posições com alocações
                            reconciliar_posicoes_com_alocacoes(repo, produto_id)

                            # posicoes_abertas() já faz Bitget sync + ATR stops internamente
                            df = obter_dados_para_visualizacao(produto_id, viz)
                            df_viz = aplicar_visualizacao(df, viz)
                            self._send_html(get_visualizacao_html(produto, viz, df_viz))
                            return
                    except ValueError:
                        pass
                    self._send_html("<h1>Visualizacao nao encontrada</h1>", 404)
                    return

                # Posicoes abertas/fechadas (fallback sem visualizacoes)
                if len(parts) >= 4 and parts[3] in ['abertas', 'fechadas']:
                    tipo = parts[3]

                    # Para produtos de alocação livre, reconciliar posições com alocações
                    reconciliar_posicoes_com_alocacoes(repo, produto_id)

                    # posicoes_abertas() já faz Bitget sync + ATR stops internamente
                    visualizacoes = repo.listar_visualizacoes(produto_id)

                    if tipo == 'abertas':
                        df = display_posicoes_abertas(produto_id, formatar=True, filtrar_colunas=False)
                    else:
                        df = display_posicoes_fechadas(produto_id, formatar=True, filtrar_colunas=False)

                    # Criar visualizacao fake para reutilizar template
                    fake_viz = {'id': 0, 'nome': f'Posicoes {"Abertas" if tipo == "abertas" else "Fechadas"}', 'produto_id': produto_id}
                    self._send_html(get_visualizacao_html(produto, fake_viz, df))
                    return

                # Dashboard de portfolio (EXC/HB/LC/Alphacoins)
                pf_cfg = PORTFOLIO_PRODUCTS.get(produto_id)
                if pf_cfg:
                    if pf_cfg['type'] == 'redirect':
                        self.send_response(302)
                        self.send_header('Location', f"/produto/{pf_cfg['target_id']}")
                        self.end_headers()
                        return
                    try:
                        keys = pf_cfg['keys']
                        group_name = pf_cfg['group_name']
                        data_inicio = str(produto.get('data_inicio', ''))[:10]
                        # POSSÍVEL CAUSA: forçar data_inicio para EXC faz o benchmark BTC ser pedido desde 2017-10-02;
                        # a API Bitget usa limit=200, então o histórico real pode truncar em ~200 dias
                        if produto_id == 2150859854 and data_inicio != '2017-10-02':
                            data_inicio = '2017-10-02'
                        sub_portfolios = []
                        for k in keys:
                            cfg = PORTFOLIO_CONFIG[k]
                            csv_path = _root / cfg['csv']
                            if not csv_path.exists():
                                csv_path = _root / 'data' / 'allocations' / cfg['csv']
                            if not csv_path.exists():
                                print(f"[DASHBOARD] ERRO: CSV não encontrado para portfolio {k}: {csv_path}", flush=True)
                                self._send_html(f"<h1>Erro</h1><p>Arquivo de alocação não encontrado: {cfg['csv']}</p>", 500)
                                return
                            sub_portfolios.append({'key': k, 'nome': cfg['nome'], 'data_inicio': data_inicio})

                        first_key = keys[0]
                        print(f"[DASHBOARD] Carregando dados do portfolio {first_key}...", flush=True)
                        active_data, _ = get_portfolio_data(first_key, repo=repo)
                        print(f"[DASHBOARD] Dados carregados: resumo={active_data.get('resumo')}, "
                              f"serie_len={len(active_data.get('rentabilidade_serie', []))}, "
                              f"abertas={len(active_data.get('posicoes_abertas', []))}, "
                              f"fechadas={len(active_data.get('posicoes_fechadas', []))}", flush=True)
                        self._send_html(get_portfolio_dashboard_html(
                            group_name, sub_portfolios, active_data, first_key, produto_id))
                        return
                    except Exception as e:
                        print(f"[DASHBOARD] ERRO ao montar portfolio dashboard: {e}", flush=True)
                        import traceback; traceback.print_exc()
                        self._send_html(f"<h1>Erro</h1><p>Erro ao carregar portfolio: {str(e)}</p><pre>{traceback.format_exc()}</pre>", 500)
                        return

                # Soros groups: Soros Spot 2 aparece como Turma 2 dentro da página de Soros Spot 1
                produto_nome = (produto.get('nome') or '').strip()
                soros_gcfg, soros_member_idx = _resolve_soros_group(repo, produto_nome)
                if soros_gcfg:
                    primary_name = soros_gcfg['members'][0]
                    primary_id = _get_soros_id_by_name(repo, primary_name) or produto_id
                    if soros_member_idx > 0:
                        # Redireciona Soros Spot 2 para Soros Spot 1 com turma pré-selecionada
                        turmas_service_soros = TurmasService(db_url=repo.db_url)
                        spot2_id = _get_soros_id_by_name(repo, soros_gcfg['members'][1])
                        if primary_id and spot2_id:
                            turmas_spot2 = turmas_service_soros.listar_turmas(spot2_id)
                            turma_id_redirect = turmas_spot2[0]['id'] if turmas_spot2 else None
                            loc = f"/produto/{primary_id}"
                            if turma_id_redirect:
                                loc += f"?turma={turma_id_redirect}"
                            self.send_response(302)
                            self.send_header('Location', loc)
                            self.end_headers()
                            return

                    # Usar produto primário (Soros Spot 1) e agregar turmas de todos os members
                    produto_id = primary_id
                    produto = repo.carregar_produto(produto_id) or produto
                    produto_nome = primary_name
                product_tabs = None

                # Dashboard customizado: Crypto Signals
                if produto_id in CUSTOM_DASHBOARD_PRODUCTS and CUSTOM_DASHBOARD_PRODUCTS[produto_id] == 'cryptosignals':
                    try:
                        turmas_service = TurmasService(db_url=repo.db_url)
                        turmas_produto = turmas_service.listar_turmas(produto_id)
                        if turmas_produto:
                            reconciliar_posicoes_com_alocacoes(repo, produto_id)
                            try:
                                display_posicoes_abertas(produto_id, formatar=False, filtrar_colunas=False)
                            except Exception:
                                pass
                            try:
                                turmas_service.reconciliar_posicoes_com_turmas(produto_id, verbose=True)
                            except Exception:
                                pass
                            primeira_turma_id = turmas_produto[0]['id']
                            carteira = turmas_service.listar_carteira_turma(primeira_turma_id)
                            precos_atuais = _obter_precos_bitget_primeiro_coingecko_fallback(carteira, tipo_produto=produto.get('tipo', ''), db_url=repo.db_url)
                            _enrich_carteira_trades(carteira, precos_atuais, repo=repo)
                            _enrich_carteira_cs_attributes(carteira, repo)
                            self._send_html(get_cryptosignals_dashboard_html(
                                produto.get('nome', 'Crypto Signals'), carteira,
                                produto_id=produto_id))
                            return
                    except Exception as e:
                        print(f"[DASHBOARD] Erro ao montar dashboard Crypto Signals: {e}", flush=True)
                        import traceback; traceback.print_exc()

                # Dashboard customizado: ICOs
                produto_nome_lower = (produto.get('nome') or '').strip().lower()
                if produto_nome_lower in ICOS_PRODUCT_NAMES:
                    try:
                        turmas_service = TurmasService(db_url=repo.db_url)
                        turmas_icos = turmas_service.listar_turmas(produto_id)
                        if turmas_icos:
                            reconciliar_posicoes_com_alocacoes(repo, produto_id)
                            try:
                                display_posicoes_abertas(produto_id, formatar=False, filtrar_colunas=False)
                            except Exception:
                                pass
                            try:
                                turmas_service.reconciliar_posicoes_com_turmas(produto_id, verbose=True)
                            except Exception:
                                pass
                            primeira_turma_id = turmas_icos[0]['id']
                            carteira = turmas_service.listar_carteira_turma(primeira_turma_id)
                            precos_atuais = _obter_precos_bitget_primeiro_coingecko_fallback(carteira, tipo_produto=produto.get('tipo', ''), db_url=repo.db_url)
                            _enrich_carteira_trades(carteira, precos_atuais, repo=repo)
                            _enrich_carteira_icos_attributes(carteira, repo)
                            self._send_html(get_icos_dashboard_html(
                                produto.get('nome', 'ICOs'), carteira,
                                produto_id=produto_id))
                            return
                    except Exception as e:
                        print(f"[DASHBOARD] Erro ao montar dashboard ICOs: {e}", flush=True)
                        import traceback; traceback.print_exc()

                # Pagina do produto (default) - Dashboard rico se tiver turmas
                turmas_produto = []
                try:
                    turmas_service = TurmasService(db_url=repo.db_url)
                    if soros_gcfg:
                        # Agregar turmas de Soros Spot 1 + Soros Spot 2 como Turma 1 e Turma 2
                        for i, member_name in enumerate(soros_gcfg['members']):
                            mid = _get_soros_id_by_name(repo, member_name) if i > 0 else produto_id
                            if not mid:
                                continue
                            turmas_member = turmas_service.listar_turmas(mid)
                            for j, t in enumerate(turmas_member):
                                t_copy = dict(t)
                                if i < len(soros_gcfg['tab_names']) and j == 0:
                                    t_copy['nome'] = soros_gcfg['tab_names'][i]
                                turmas_produto.append(t_copy)
                    else:
                        turmas_produto = turmas_service.listar_turmas(produto_id)
                except Exception as e:
                    print(f"[DASHBOARD] Erro ao buscar turmas para produto {produto_id}: {e}", flush=True)

                # Pré-selecionar turma via ?turma=ID (ex: redirect de Soros Spot 2)
                turma_query = query.get('turma', [''])[0]
                if turma_query and turmas_produto:
                    try:
                        tid = int(turma_query)
                        idx = next((i for i, t in enumerate(turmas_produto) if t['id'] == tid), None)
                        if idx is not None and idx > 0:
                            t = turmas_produto.pop(idx)
                            turmas_produto.insert(0, t)
                    except (ValueError, TypeError):
                        pass

                # Auto-criar turma para produtos do grupo Soros sem turmas
                if not turmas_produto and soros_gcfg:
                    try:
                        print(f"[DASHBOARD] Auto-criando turma para produto {produto_id} ({produto_nome})...", flush=True)
                        data_inicio_prod = str(produto.get('data_inicio', ''))[:10] or date.today().isoformat()
                        capital_prod = produto.get('capital_inicial', 1500) or 1500
                        if not turmas_service:
                            turmas_service = TurmasService(db_url=repo.db_url)
                        conn = turmas_service._get_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO turmas (produto_id, nome, data_inicio, capital_base, data_criacao)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (produto_id, produto_nome, data_inicio_prod, capital_prod, date.today().isoformat()))
                        conn.commit()
                        conn.close()
                        turmas_produto = turmas_service.listar_turmas(produto_id)
                        print(f"[DASHBOARD] Turma auto-criada. turmas={len(turmas_produto)}", flush=True)
                    except Exception as e:
                        print(f"[DASHBOARD] Erro ao auto-criar turma: {e}", flush=True)

                if turmas_produto:
                    try:
                        # Para grupo Soros: sync e reconciliação em TODOS os members (Spot 1 e Spot 2)
                        ids_para_sync = [produto_id]
                        if soros_gcfg:
                            for member_name in soros_gcfg['members'][1:]:
                                mid = _get_soros_id_by_name(repo, member_name)
                                if mid and mid != produto_id:
                                    ids_para_sync.append(mid)
                        for pid in ids_para_sync:
                            reconciliar_posicoes_com_alocacoes(repo, pid)
                            try:
                                display_posicoes_abertas(pid, formatar=False, filtrar_colunas=False)
                            except Exception:
                                pass
                            try:
                                turmas_service.reconciliar_posicoes_com_turmas(pid, verbose=True)
                            except Exception as e_reconciliar:
                                print(f"[DASHBOARD] Aviso: reconciliação turmas produto {pid} falhou: {e_reconciliar}", flush=True)

                        rentabilidade_service = RentabilidadeService(db_url=repo.db_url)
                        primeira_turma_id = turmas_produto[0]['id']

                        carteira = turmas_service.listar_carteira_turma(primeira_turma_id)
                        precos_atuais = _obter_precos_bitget_primeiro_coingecko_fallback(carteira, tipo_produto=produto.get('tipo', ''), db_url=repo.db_url)
                        _enrich_carteira_trades(carteira, precos_atuais, repo=repo)
                        resumo = rentabilidade_service.resumo_turma(primeira_turma_id, precos_atuais=precos_atuais)

                        mostrar_caixa_alocacao = get_bitget_credentials(produto.get('nome') or '') is not None
                        self._send_html(get_produto_dashboard_html(
                            produto, turmas_produto, resumo, carteira,
                            mostrar_caixa_alocacao=mostrar_caixa_alocacao,
                            product_tabs=product_tabs))
                        return
                    except Exception as e:
                        print(f"[DASHBOARD] Erro ao montar dashboard rico: {e}", flush=True)

                # Fallback: produto sem turmas (ou erro) - página clássica
                reconciliar_posicoes_com_alocacoes(repo, produto_id)
                visualizacoes = repo.listar_visualizacoes(produto_id)
                self._send_html(get_produto_html(produto, visualizacoes, repo))
                return

        # API: Lista produtos
        if path == '/api/produtos':
            produtos = repo.listar_produtos()
            self._send_json({'produtos': produtos})
            return

        # ============================================================
        # ROTAS DE TURMAS
        # ============================================================

        # Lista de turmas
        if path == '/turmas':
            turmas_service = TurmasService(db_url=repo.db_url)
            rentabilidade_service = RentabilidadeService(db_url=repo.db_url)

            # OTIMIZAÇÃO: Não chama preencher_historico_faltante() no carregamento
            # Use /api/cotacoes/atualizar para atualizar preços quando necessário

            turmas = turmas_service.listar_turmas()

            # Agrupar Soros Spot 1 e Soros Spot 2 na mesma aba
            for gcfg in SOROS_GROUPS.values():
                soros_members = set(gcfg['members'])
                soros_display = gcfg.get('display_name', 'Soros Spot')
                for i, t in enumerate(turmas):
                    pnome = (t.get('produto_nome') or '').strip()
                    if pnome in soros_members:
                        t = dict(t)
                        t['produto_nome'] = soros_display
                        idx = gcfg['members'].index(pnome)
                        if idx < len(gcfg.get('tab_names', [])):
                            t['nome'] = gcfg['tab_names'][idx]
                        turmas[i] = t
                break

            # Batch otimizado: uma query SQL + uma chamada API para preços em tempo real
            resumos_lista = rentabilidade_service.obter_rentabilidade_resumida_todas_turmas()
            resumos = {r['turma_id']: r for r in resumos_lista}

            self._send_html(get_lista_turmas_html(turmas, resumos))
            return

        # Formulario nova turma
        if path == '/turmas/nova':
            produtos = repo.listar_produtos()
            is_modal = query.get('_modal', [''])[0] == '1'
            return_url = query.get('return', [''])[0]
            produto_id = query.get('produto_id', [''])[0]
            if not return_url and produto_id:
                return_url = f'/produto/{produto_id}'
            if not return_url:
                return_url = '/'
            inner = get_form_nova_turma_html(produtos, as_inner=True, return_url=return_url)
            self._send_html(inner if is_modal else get_form_nova_turma_html(produtos, return_url=return_url))
            return

        # Comparar turmas
        if path == '/turmas/comparar':
            turmas_service = TurmasService(db_url=repo.db_url)
            turmas = turmas_service.listar_turmas()
            self._send_html(get_comparar_turmas_html(turmas, {}))
            return

        # Rotas de turma especifica: /turmas/{id}, /turmas/{id}/abertas, /turmas/{id}/fechadas, /turmas/{id}/historico, /turmas/{id}/rentabilidade
        if path.startswith('/turmas/') and path != '/turmas/nova' and path != '/turmas/comparar' and path != '/turmas/historico':
            parts = path.split('/')
            if len(parts) >= 3:
                try:
                    turma_id = int(parts[2])
                    turmas_service = TurmasService(db_url=repo.db_url)
                    rentabilidade_service = RentabilidadeService(db_url=repo.db_url)

                    # OTIMIZAÇÃO: Não chama preencher_historico_faltante() no carregamento
                    # Use /api/cotacoes/atualizar para atualizar preços quando necessário

                    turma = turmas_service.obter_turma(turma_id)
                    if not turma:
                        self._send_html("<h1>Turma nao encontrada</h1>", 404)
                        return

                    # Grafico de rentabilidade
                    if len(parts) >= 4 and parts[3] == 'rentabilidade':
                        serie = rentabilidade_service.calcular_serie_rentabilidade(turma_id)
                        self._send_html(get_rentabilidade_chart_html(turma, serie))
                        return

                    # Determinar aba ativa
                    tab_ativa = 'abertas'
                    if len(parts) >= 4 and parts[3] in ('abertas', 'fechadas', 'historico'):
                        tab_ativa = parts[3]

                    # Detalhes da turma com abas (usa helpers compartilhados)
                    produto_turma = repo.carregar_produto(turma.get('produto_id'))
                    tipo_produto_turma = (produto_turma or {}).get('tipo', '')
                    produto_id_turma = turma.get('produto_id')
                    if produto_id_turma:
                        try:
                            turmas_service.reconciliar_posicoes_com_turmas(produto_id_turma)
                        except Exception:
                            pass
                    carteira = turmas_service.listar_carteira_turma(turma_id)
                    precos_atuais = _obter_precos_bitget_primeiro_coingecko_fallback(carteira, tipo_produto=tipo_produto_turma, db_url=repo.db_url)
                    _enrich_carteira_trades(carteira, precos_atuais, repo=repo)
                    resumo = rentabilidade_service.resumo_turma(turma_id, precos_atuais=precos_atuais)

                    self._send_html(get_turma_detalhes_html(turma, resumo, carteira, tab_ativa))
                    return
                except ValueError:
                    pass
            self._send_html("<h1>Turma nao encontrada</h1>", 404)
            return

        # API: Dashboard data de uma turma (resumo + carteira enriquecida)
        if path.startswith('/api/turma/') and path.endswith('/dashboard-data'):
            try:
                turma_id = int(path.split('/')[3])
                turmas_service = TurmasService(db_url=repo.db_url)
                rentabilidade_service = RentabilidadeService(db_url=repo.db_url)

                turma = turmas_service.obter_turma(turma_id)
                if not turma:
                    self._send_json({'erro': 'Turma não encontrada'}, 404)
                    return

                produto_turma = repo.carregar_produto(turma.get('produto_id'))
                tipo_produto_turma = (produto_turma or {}).get('tipo', '')
                produto_id_turma = turma.get('produto_id')
                if produto_id_turma:
                    try:
                        turmas_service.reconciliar_posicoes_com_turmas(produto_id_turma)
                    except Exception:
                        pass
                carteira = turmas_service.listar_carteira_turma(turma_id)
                precos_atuais = _obter_precos_bitget_primeiro_coingecko_fallback(carteira, tipo_produto=tipo_produto_turma, db_url=repo.db_url)
                _enrich_carteira_trades(carteira, precos_atuais, repo=repo)
                resumo = rentabilidade_service.resumo_turma(turma_id, precos_atuais=precos_atuais)

                # Serializar carteira (converter Decimal/date)
                import decimal as _dec
                def _clean(obj):
                    if isinstance(obj, _dec.Decimal):
                        return float(obj)
                    if hasattr(obj, 'isoformat'):
                        return obj.isoformat()
                    return obj

                carteira_clean = []
                for t in carteira:
                    carteira_clean.append({k: _clean(v) for k, v in t.items()})

                resumo_clean = {
                    'rentabilidade_acumulada_pct': resumo.get('rentabilidade_acumulada_pct', 0),
                    'valor_total': resumo.get('valor_total', 0),
                    'capital_alocado': resumo.get('capital_alocado', 0),
                    'capital_em_caixa': resumo.get('capital_em_caixa', 0),
                    'capital_base': resumo.get('capital_base', 1500),
                    'trades_ativos': resumo.get('trades_ativos', 0),
                    'trades_fechados': resumo.get('trades_fechados', 0),
                }

                self._send_json({'resumo': resumo_clean, 'carteira': carteira_clean})
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Rentabilidade de uma turma (para grafico comparativo)
        # GET /api/turma/{id}/rentabilidade ou ?inicio=YYYY-MM-DD&fim=YYYY-MM-DD
        if path.startswith('/api/turma/') and '/rentabilidade' in path:
            import time as _time_mod
            _t0 = _time_mod.time()
            _turma_id_log = '?'
            try:
                parts = path.split('/')
                turma_id = int(parts[3]) if len(parts) > 3 else 0
                _turma_id_log = turma_id
                if not turma_id:
                    self._send_json({'erro': 'ID da turma inválido'}, 400)
                    return
                print(f"[API] /api/turma/{turma_id}/rentabilidade - iniciando...", flush=True)
                data_inicio = query.get('inicio', [None])[0]
                data_fim = query.get('fim', [None])[0]
                rentabilidade_service = RentabilidadeService(db_url=repo.db_url)
                serie = rentabilidade_service.calcular_serie_rentabilidade(
                    turma_id, data_inicio=data_inicio, data_fim=data_fim
                )
                data = [
                    {
                        'dia': p.dia if not hasattr(p.dia, 'isoformat') else p.dia.isoformat()[:10],
                        'valor_total': p.valor_total,
                        'rentabilidade_acumulada_pct': p.rentabilidade_acumulada_pct,
                        'capital_alocado': getattr(p, 'capital_alocado', 0),
                        'capital_em_caixa': getattr(p, 'capital_em_caixa', 0),
                    }
                    for p in serie
                ]
                _elapsed = _time_mod.time() - _t0
                print(f"[API] /api/turma/{turma_id}/rentabilidade - OK, {len(data)} pontos em {_elapsed:.1f}s", flush=True)
                self._send_json(data)
            except Exception as e:
                _elapsed = _time_mod.time() - _t0
                print(f"[API] /api/turma/{_turma_id_log}/rentabilidade - ERRO em {_elapsed:.1f}s: {e}", flush=True)
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Dados de dashboard de portfolio
        # GET /api/portfolio/{key}/dashboard-data
        if path.startswith('/api/portfolio/') and path.endswith('/dashboard-data'):
            try:
                pf_key = path.split('/')[3]
                print(f"[PORTFOLIO API] Requisição para portfolio: {pf_key}", flush=True)
                if pf_key not in PORTFOLIO_CONFIG:
                    self._send_json({'erro': f'Portfolio desconhecido: {pf_key}'}, 404)
                    return
                data, _ = get_portfolio_data(pf_key, repo=repo)
                print(f"[PORTFOLIO API] Dados carregados com sucesso para {pf_key}: "
                      f"serie_len={len(data.get('rentabilidade_serie', []))}, "
                      f"abertas={len(data.get('posicoes_abertas', []))}", flush=True)
                self._send_json(data)
            except Exception as e:
                print(f"[PORTFOLIO API] ERRO ao carregar {pf_key}: {e}", flush=True)
                import traceback; traceback.print_exc()
                self._send_json({'erro': str(e)}, 500)
            return

        # API: PnL por ativo em um período (lógica do notebook)
        # GET /api/portfolio/{key}/pnl?inicio=2025-01-01&fim=2026-02-20
        if path.startswith('/api/portfolio/') and path.endswith('/pnl'):
            try:
                pf_key = path.split('/')[3]
                if pf_key not in PORTFOLIO_CONFIG:
                    self._send_json({'erro': f'Portfolio desconhecido: {pf_key}'}, 404)
                    return
                inicio = query.get('inicio', [None])[0]
                fim = query.get('fim', [None])[0]
                result = get_portfolio_pnl(pf_key, inicio=inicio, fim=fim, repo=repo)
                self._send_json({'ativos': result, 'inicio': inicio, 'fim': fim})
            except Exception as e:
                print(f"[PORTFOLIO PNL API] ERRO: {e}", flush=True)
                import traceback; traceback.print_exc()
                self._send_json({'erro': str(e)}, 500)
            return

        # API: Série de rentabilidade dia a dia para um período (sem downsampling)
        # GET /api/portfolio/{key}/rentabilidade_serie?inicio=2025-01-01&fim=2026-02-20
        if path.startswith('/api/portfolio/') and '/rentabilidade_serie' in path:
            try:
                parts = path.split('/')
                pf_key = parts[3] if len(parts) > 3 else None
                if not pf_key or pf_key not in PORTFOLIO_CONFIG:
                    self._send_json({'erro': 'Portfolio desconhecido'}, 404)
                    return
                inicio = query.get('inicio', [None])[0]
                fim = query.get('fim', [None])[0]
                serie = get_portfolio_rentabilidade_serie(pf_key, inicio=inicio, fim=fim, repo=repo)
                self._send_json({'rentabilidade_serie': serie})
            except Exception as e:
                print(f"[PORTFOLIO RENT SERIE API] ERRO: {e}", flush=True)
                import traceback; traceback.print_exc()
                self._send_json({'erro': str(e)}, 500)
            return

        # API: Série de rentabilidade do BTC (benchmark) - cache + Bitget paginado ou CoinGecko range
        if path == '/api/benchmark/btc':
            try:
                data_inicio = query.get('data_inicio', [None])[0]
                if not data_inicio:
                    self._send_json({'erro': 'data_inicio é obrigatório'}, 400)
                    return
                data_inicio_str = str(data_inicio)[:10]
                try:
                    data_inicio_obj = date.fromisoformat(data_inicio_str)
                    dias = max(1, (date.today() - data_inicio_obj).days)
                except (ValueError, TypeError):
                    dias = 365
                coingecko_only = (query.get('coingecko_only', [None])[0] or '').strip().lower() in ('1', 'true', 'yes')
                logger.debug("[api_benchmark_btc] ENTRADA data_inicio=%s dias=%s coingecko_only=%s", data_inicio_str, dias, coingecko_only)

                btc_cache = BTCCacheService(repo=repo)
                dados_cache = btc_cache.buscar_cache(data_inicio_obj)
                preco_por_data = {}
                bitget_data = []
                coingecko_data = []

                if coingecko_only:
                    # Exponential Coins / Alphacoins: só CoinGecko (alinhado ao notebook e à rentabilidade do portfólio)
                    cotacoes_service = CotacoesService(db_url=repo.db_url)
                    lista_hist = cotacoes_service.obter_historico_coingecko_range_chunked('bitcoin', data_inicio_obj)
                    coingecko_data = lista_hist if lista_hist else []
                    for item in lista_hist:
                        d = item.get('data')
                        if d and d >= data_inicio_str:
                            preco_por_data[d] = item['preco']
                    logger.debug("[api_benchmark_btc] COINGECKO ONLY len=%s", len(preco_por_data))
                elif btc_cache.cobre_periodo(dados_cache, data_inicio_obj):
                    for item in dados_cache:
                        d = item.get('data')
                        if d and d >= data_inicio_str:
                            preco_por_data[d] = item['preco']
                    logger.debug("[api_benchmark_btc] USANDO CACHE len=%s", len(preco_por_data))
                else:
                    cotacoes_service = CotacoesService(db_url=repo.db_url)
                    klines = cotacoes_service.obter_klines_bitget_paginado('BTCUSDT', '1Dutc', dias)
                    bitget_data = klines if klines else []
                    if klines:
                        for k in klines:
                            d = k.get('open_time')
                            if d and d >= data_inicio_str:
                                preco_por_data[d] = k['close']
                        logger.debug("[api_benchmark_btc] BITGET PAGINADO len=%s", len(preco_por_data))
                        # Complementação parcial: se Bitget não cobre data_inicio (ex.: só desde 2018), buscar pré-Bitget no CoinGecko
                        if preco_por_data:
                            min_bitget_str = min(preco_por_data.keys())
                            min_bitget = date.fromisoformat(min_bitget_str)
                            if min_bitget > data_inicio_obj:
                                data_fim_cg = min_bitget - timedelta(days=1)
                                print("[BENCHMARK] Complementando período pré-Bitget via CoinGecko:", data_inicio_obj, "→", data_fim_cg, flush=True)
                                dados_cg = cotacoes_service.obter_historico_coingecko_range_chunked(
                                    'bitcoin', data_inicio_obj, data_fim=data_fim_cg
                                )
                                for item in dados_cg:
                                    d = item.get('data')
                                    if d and d not in preco_por_data:
                                        preco_por_data[d] = item['preco']
                                coingecko_data = dados_cg
                                logger.debug("[api_benchmark_btc] COINGECKO COMPLEMENTO len=%s", len(dados_cg))
                    if not preco_por_data:
                        lista_hist = cotacoes_service.obter_historico_coingecko_range_chunked(
                            'bitcoin', data_inicio_obj
                        )
                        coingecko_data = lista_hist if lista_hist else []
                        for item in lista_hist:
                            d = item.get('data')
                            if d and d >= data_inicio_str:
                                preco_por_data[d] = item['preco']
                        logger.debug("[api_benchmark_btc] COINGECKO RANGE len=%s", len(preco_por_data))
                    if preco_por_data:
                        lista_para_cache = [{'data': d, 'preco': p} for d, p in preco_por_data.items()]
                        btc_cache.salvar_cache(lista_para_cache)

                if not preco_por_data:
                    logger.debug("[api_benchmark_btc] preco_por_data vazio, retornando []")
                    self._send_json([])
                    return
                if data_inicio_str not in preco_por_data:
                    first_avail = min(preco_por_data.keys())
                    preco_por_data[data_inicio_str] = preco_por_data[first_avail]

                print("data_inicio_obj:", data_inicio_obj, flush=True)
                print("primeira data do dict:", min(preco_por_data.keys()), flush=True)
                print("ultima data do dict:", max(preco_por_data.keys()), flush=True)
                print("hoje:", date.today(), flush=True)

                # FASE 4: garantir todos os dias de data_inicio até hoje (ffill); estender até hoje se último dia < hoje
                hoje_str = date.today().isoformat()
                datas_ordenadas_antes = sorted(preco_por_data.keys())
                d_min = datetime.strptime(data_inicio_str, "%Y-%m-%d").date()
                d_max = date.today()
                last_preco = preco_por_data[datas_ordenadas_antes[-1]]
                cur = d_min
                while cur <= d_max:
                    d_str = cur.isoformat()
                    if d_str not in preco_por_data:
                        preco_por_data[d_str] = last_preco
                    else:
                        last_preco = preco_por_data[d_str]
                    cur += timedelta(days=1)
                datas_ordenadas = sorted(preco_por_data.keys())
                print("ultima data apos ffill:", max(preco_por_data.keys()), flush=True)

                preco_base = preco_por_data[datas_ordenadas[0]]
                if not preco_base or preco_base == 0:
                    logger.debug("[api_benchmark_btc] preco_base inválido, retornando []")
                    self._send_json([])
                    return

                # Instrumentação FASE 2 + validação FASE 5
                print("DIAS SOLICITADOS:", dias, flush=True)
                print("DIAS BITGET:", len(bitget_data), flush=True)
                print("DIAS COINGECKO:", len(coingecko_data), flush=True)
                print("MIN DIA REAL:", datas_ordenadas[0], flush=True)
                print("MAX DIA REAL:", datas_ordenadas[-1], flush=True)
                print("TOTAL PONTOS:", len(datas_ordenadas), flush=True)
                print("DATA INICIAL FINAL:", datas_ordenadas[0], flush=True)
                print("DATA FINAL FINAL:", datas_ordenadas[-1], flush=True)
                print("TOTAL DIAS FINAL:", len(datas_ordenadas), flush=True)
                print("DATA INICIO SOLICITADA:", data_inicio_str, flush=True)
                faltantes = []
                for i in range(len(datas_ordenadas) - 1):
                    d1 = datetime.strptime(datas_ordenadas[i], "%Y-%m-%d")
                    d2 = datetime.strptime(datas_ordenadas[i + 1], "%Y-%m-%d")
                    if (d2 - d1).days > 1:
                        faltantes.append((d1, d2))
                print("BURACOS ENCONTRADOS:", len(faltantes), flush=True)

                serie_resposta = [
                    {'dia': d, 'rentabilidade_acumulada_pct': round(((preco_por_data[d] / preco_base) - 1) * 100, 4)}
                    for d in datas_ordenadas
                ]
                self._send_json(serie_resposta)
            except Exception as e:
                logger.debug("[api_benchmark_btc] EXCEÇÃO %s", e, exc_info=True)
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Rentabilidade histórica de uma turma
        if path.startswith('/api/turma/') and path.endswith('/historico'):
            try:
                turma_id = int(path.split('/')[3])
                rentabilidade_service = RentabilidadeService(db_url=repo.db_url)
                resultado = rentabilidade_service.obter_rentabilidade_historica(turma_id)
                if 'erro' in resultado:
                    self._send_json({'erro': resultado['erro']}, 404)
                else:
                    self._send_json(resultado)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Listar posições elegíveis para replicação em uma turma
        # GET /api/turma/posicoes-elegiveis?produto_id=1&data_inicio=2026-01-15
        if path == '/api/turma/posicoes-elegiveis':
            try:
                produto_id = int(query.get('produto_id', [0])[0])
                data_inicio = query.get('data_inicio', [None])[0]

                if not produto_id or not data_inicio:
                    self._send_json({'erro': 'produto_id e data_inicio são obrigatórios'}, 400)
                    return

                turmas_service = TurmasService(db_url=repo.db_url)
                posicoes = turmas_service.listar_posicoes_elegiveis(produto_id, data_inicio)

                self._send_json({
                    'posicoes': posicoes,
                    'total': len(posicoes),
                    'produto_id': produto_id,
                    'data_inicio': data_inicio
                })
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Obter cotação histórica de um ativo
        # GET /api/cotacao/historica?coingecko_id=bitcoin&data=2026-01-15
        # Usa o mesmo método que criar_turma para consistência de preços
        if path == '/api/cotacao/historica':
            try:
                coingecko_id = query.get('coingecko_id', [None])[0]
                data = query.get('data', [None])[0]

                if not coingecko_id or not data:
                    self._send_json({'erro': 'coingecko_id e data são obrigatórios'}, 400)
                    return

                cotacoes_service = CotacoesService(db_url=repo.db_url)
                resultado = cotacoes_service.obter_preco_historico_exato(coingecko_id, data)

                if resultado.get('status') in ('ok', 'fallback'):
                    self._send_json({
                        'coingecko_id': coingecko_id,
                        'data_solicitada': data,
                        'data_encontrada': resultado.get('data_referencia'),
                        'preco': resultado['preco'],
                        'fonte': resultado.get('fonte', 'coingecko'),
                        'status': resultado['status'],
                        'aviso': resultado.get('aviso')
                    })
                else:
                    self._send_json({
                        'erro': resultado.get('erro', f'Preço não encontrado para {coingecko_id} na data {data}'),
                        'coingecko_id': coingecko_id,
                        'data': data
                    }, 404)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Rentabilidade de todas as turmas
        if path == '/api/rentabilidade/todas':
            try:
                rentabilidade_service = RentabilidadeService(db_url=repo.db_url)
                # Parse query params
                produto_id = None
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'produto_id' in params:
                        produto_id = int(params['produto_id'])
                resultado = rentabilidade_service.obter_rentabilidade_todas_turmas(produto_id=produto_id)
                self._send_json(resultado)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Resumo de rentabilidade de todas as turmas
        if path == '/api/rentabilidade/resumo':
            try:
                rentabilidade_service = RentabilidadeService(db_url=repo.db_url)
                # Parse query params
                produto_id = None
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'produto_id' in params:
                        produto_id = int(params['produto_id'])
                resultado = rentabilidade_service.obter_rentabilidade_resumida_todas_turmas(produto_id=produto_id)
                self._send_json(resultado)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Atualizar cotações (preencher histórico faltante)
        if path == '/api/cotacoes/atualizar':
            try:
                cotacoes_service = CotacoesService(db_url=repo.db_url)
                # Parse query params
                turma_id = None
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'turma_id' in params:
                        turma_id = int(params['turma_id'])

                resultado = cotacoes_service.preencher_historico_faltante(turma_id)
                self._send_json({
                    'sucesso': True,
                    'trades_processados': resultado.get('trades_processados', 0),
                    'dias_preenchidos': resultado.get('dias_preenchidos', 0),
                    'erros': resultado.get('erros', 0)
                })
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        # API: Preencher preco_saida das posições fechadas (AC/EXC/HB/LC) via Bitget/CoinGecko
        # Roda em background thread para não travar o servidor
        if path.startswith('/api/posicoes/preencher-preco-saida'):
            try:
                produto_id_param = None
                if '?' in self.path:
                    qs = self.path.split('?')[1]
                    params = dict(p.split('=') for p in qs.split('&') if '=' in p)
                    if 'produto_id' in params:
                        produto_id_param = int(params['produto_id'])
                import threading
                bg_repo = get_repo()
                def _run():
                    try:
                        resultado = preencher_preco_saida_posicoes_fechadas(bg_repo, produto_id_param)
                        print(f"[PREENCHER-PRECO-SAIDA] Finalizado: {resultado}", flush=True)
                    except Exception as e:
                        print(f"[PREENCHER-PRECO-SAIDA] Erro na thread: {e}", flush=True)
                t = threading.Thread(target=_run, daemon=True)
                t.start()
                self._send_json({'sucesso': True, 'mensagem': 'Preenchimento iniciado em background. Acompanhe no terminal.'})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        # API: Preencher preços (entrada/saída/ambos) para qualquer produto — via dashboard
        # GET /api/posicoes/preencher-precos?produto_id=X&tipo=entrada|saida|ambos
        if path.startswith('/api/posicoes/preencher-precos'):
            try:
                params = {}
                if '?' in self.path:
                    qs = self.path.split('?')[1]
                    params = dict(p.split('=') for p in qs.split('&') if '=' in p)
                produto_id_param = int(params.get('produto_id', 0))
                tipo_param = params.get('tipo', 'ambos')
                if not produto_id_param:
                    self._send_json({'sucesso': False, 'erro': 'produto_id obrigatório'}, 400)
                    return
                if tipo_param not in ('entrada', 'saida', 'ambos'):
                    tipo_param = 'ambos'
                import threading
                bg_repo = get_repo()
                def _run_precos():
                    try:
                        resultado = preencher_precos_posicoes(bg_repo, produto_id_param, tipo_param)
                        print(f"[PREENCHER-PRECOS] Finalizado: {resultado}", flush=True)
                    except Exception as e:
                        print(f"[PREENCHER-PRECOS] Erro na thread: {e}", flush=True)
                t = threading.Thread(target=_run_precos, daemon=True)
                t.start()
                nomes_tipo = {'entrada': 'preços de entrada', 'saida': 'preços de saída', 'ambos': 'todos os preços'}
                self._send_json({
                    'sucesso': True,
                    'mensagem': f'Preenchimento de {nomes_tipo[tipo_param]} iniciado em background. Acompanhe no terminal/logs.',
                    'produto_id': produto_id_param,
                    'tipo': tipo_param
                })
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        # API: Atualizar dados (mantido para compatibilidade, usa funcao centralizada)
        if path.startswith('/api/atualizar/'):
            try:
                produto_id = int(path.split('/')[-1])
                atualizar_dados_produto(repo, produto_id)
                self._send_json({'sucesso': True, 'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        # API: Export Excel completo (rentabilidade + posições) por produto
        if path.startswith('/api/export/excel'):
            try:
                from services.export_excel_service import gerar_excel_completo

                produto_id = None
                produto_nome = 'Produto'
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'produto_id' in params:
                        produto_id = int(params['produto_id'])

                if produto_id is None:
                    self._send_json({'erro': 'produto_id obrigatório. Ex: /api/export/excel?produto_id=123'}, 400)
                    return

                produto = None
                for p in repo.listar_produtos():
                    if p.get('id') == produto_id:
                        produto = p
                        produto_nome = (p.get('nome') or 'Produto').strip()
                        break

                dados_posicoes = None
                pf_cfg = PORTFOLIO_PRODUCTS.get(produto_id)
                is_portfolio = pf_cfg and pf_cfg.get('type') in ('group', 'single')
                is_crypto = _is_produto_crypto_signals(produto)
                is_icos = _is_produto_icos(produto)

                if is_portfolio:
                    portfolio_keys = pf_cfg.get('keys', []) if pf_cfg.get('type') != 'redirect' else (PORTFOLIO_PRODUCTS.get(pf_cfg.get('target_id')) or {}).get('keys', [])
                    nomes = {'EXC': 'Principal', 'HB': 'High Beta', 'LC': 'Low Caps', 'AC': 'Alphacoins'}
                    abertas_all, fechadas_all, hist_all = [], [], []
                    for key in portfolio_keys:
                        try:
                            data, _ = get_portfolio_data(key, repo=repo)
                        except Exception:
                            continue
                        sub = nomes.get(key, key)
                        for t in data.get('posicoes_abertas', []):
                            r = dict(t)
                            if len(portfolio_keys) > 1:
                                r['sub_portfolio'] = sub
                            abertas_all.append(r)
                        for t in data.get('posicoes_fechadas', []):
                            r = dict(t)
                            if len(portfolio_keys) > 1:
                                r['sub_portfolio'] = sub
                            fechadas_all.append(r)
                    hist_all = abertas_all + fechadas_all
                    dados_posicoes = {
                        'tipo': 'portfolio',
                        'abertas': abertas_all,
                        'fechadas': fechadas_all,
                        'historico': hist_all,
                    }
                else:
                    turmas_service = TurmasService(db_url=repo.db_url)
                    turmas = turmas_service.listar_turmas(produto_id)
                    if turmas:
                        carteira_agg = []
                        tipo_produto = (produto or {}).get('tipo', '')
                        for turma in turmas:
                            carteira = turmas_service.listar_carteira_turma(turma['id'])
                            precos_atuais = _obter_precos_bitget_primeiro_coingecko_fallback(
                                carteira, tipo_produto=tipo_produto, db_url=repo.db_url
                            )
                            _enrich_carteira_trades(carteira, precos_atuais=precos_atuais, repo=repo)
                            if is_crypto:
                                _enrich_carteira_cs_attributes(carteira, repo)
                            elif is_icos:
                                _enrich_carteira_icos_attributes(carteira, repo)
                            for t in carteira:
                                t_copy = dict(t)
                                t_copy['turma'] = turma.get('nome', '')
                                carteira_agg.append(t_copy)
                        if carteira_agg:
                            dados_posicoes = {
                                'tipo': 'crypto' if is_crypto else 'icos' if is_icos else 'turmas',
                                'historico': carteira_agg,
                            }

                buffer = gerar_excel_completo(
                    produto_id=produto_id,
                    produto_nome=produto_nome,
                    db_url=repo.db_url,
                    repo=repo,
                    dados_posicoes=dados_posicoes,
                )
                safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in produto_nome)
                filename = f"export_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                self._send_excel(buffer, filename)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({'erro': str(e)}, 500)
            return

        # API: Download Excel com rentabilidade histórica
        if path == '/api/rentabilidade/excel':
            try:
                rentabilidade_service = RentabilidadeService(db_url=repo.db_url)

                # Parse query params
                produto_id = None
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'produto_id' in params:
                        produto_id = int(params['produto_id'])

                # Buscar todas as turmas
                turmas_service = TurmasService(db_url=repo.db_url)
                turmas = turmas_service.listar_turmas(produto_id)

                if not turmas:
                    self._send_json({'erro': 'Nenhuma turma encontrada'}, 404)
                    return

                # Pré-carregar preços de todos os ativos uma única vez
                turma_ids = [t['id'] for t in turmas]
                precos_cache = rentabilidade_service.construir_precos_cache(turma_ids)

                # Calcular rentabilidade histórica de cada turma
                all_data = []
                for turma in turmas:
                    turma_id = turma['id']
                    turma_nome = turma['nome']

                    # Obter série de rentabilidade (usa cache compartilhado)
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
                            'Capital em Caixa (R$)': round(portfolio.capital_em_caixa, 2)
                        })

                # Criar DataFrame
                df = pd.DataFrame(all_data)

                # Pivotar para ter turmas como colunas (formato mais útil)
                if not df.empty:
                    # Criar uma aba com dados detalhados
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        # Aba 1: Dados detalhados (long format)
                        df.to_excel(writer, sheet_name='Detalhado', index=False)

                        # Aba 2: Rentabilidade pivotada (wide format)
                        df_pivot = df.pivot_table(
                            index='Data',
                            columns='Turma',
                            values='Rentabilidade Acumulada (%)',
                            aggfunc='first'
                        ).reset_index()
                        df_pivot.to_excel(writer, sheet_name='Rentabilidade por Turma', index=False)

                        # Aba 3: Valor Total pivotado
                        df_valor = df.pivot_table(
                            index='Data',
                            columns='Turma',
                            values='Valor Total (R$)',
                            aggfunc='first'
                        ).reset_index()
                        df_valor.to_excel(writer, sheet_name='Valor por Turma', index=False)

                    buffer.seek(0)

                    # Enviar arquivo
                    filename = f"rentabilidade_turmas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    self._send_excel(buffer, filename)
                else:
                    self._send_json({'erro': 'Nenhum dado de rentabilidade encontrado'}, 404)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({'erro': str(e)}, 500)
            return

        # API: Check ATR Stops (endpoint para cron externo)
        if path == '/api/check_stops':
            try:
                from services.atr_stop_service import atualizar_stops_posicoes_abertas
                resultado = atualizar_stops_posicoes_abertas(repo, verbose=False)
                self._send_json({
                    'status': 'ok',
                    'timestamp': datetime.now().isoformat(),
                    'resultado': {
                        'updated': resultado.get('updated', 0),
                        'skipped': resultado.get('skipped', 0),
                        'unchanged': resultado.get('unchanged', 0),
                        'breached': resultado.get('breached', 0),
                        'errors': resultado.get('errors', [])
                    }
                })
            except Exception as e:
                self._send_json({'status': 'error', 'erro': str(e)}, 500)
            return

        # API: Diagnóstico de PnL - origem dos preços e dados usados (Crypto Signals / ICOs)
        if path.startswith('/api/diag/pnl'):
            try:
                produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else 0
                if not produto_id:
                    self._send_json({'erro': 'produto_id obrigatório. Ex: /api/diag/pnl?produto_id=123'}, 400)
                    return

                produto = None
                for p in repo.listar_produtos():
                    if p.get('id') == produto_id:
                        produto = p
                        break
                if not produto:
                    self._send_json({'erro': f'Produto {produto_id} não encontrado'}, 404)
                    return

                turmas_service = TurmasService(db_url=repo.db_url)
                turmas = turmas_service.listar_turmas(produto_id)
                if not turmas:
                    self._send_json({
                        'produto': produto.get('nome'),
                        'produto_id': produto_id,
                        'erro': 'Nenhuma turma encontrada',
                        'posicoes': []
                    })
                    return

                turma_id = turmas[0]['id']
                carteira = turmas_service.listar_carteira_turma(turma_id)
                precos_com_fonte = _obter_precos_com_fonte(
                    carteira, tipo_produto=produto.get('tipo', ''), db_url=repo.db_url
                )
                precos_atuais = {k: v["preco"] for k, v in precos_com_fonte.items()}
                _enrich_carteira_trades(carteira, precos_atuais, repo=repo)

                posicoes_diag = []
                for t in carteira:
                    key = _price_key(t)
                    info = precos_com_fonte.get(key) if key else None
                    fonte = info["fonte"] if info else None
                    data_remocao = t.get('data_remocao')
                    data_saida = t.get('data_saida')
                    status = (t.get('status_posicao') or '').strip().lower()
                    is_closed = bool(data_remocao or data_saida or status == 'closed')

                    if is_closed:
                        fonte = "db"  # preco_saida vem do banco (posicoes.preco_saida)
                    elif not fonte:
                        fonte = "db_fallback"  # preco_entrada quando API nao retornou preco

                    pe = t.get('preco_entrada_turma') or t.get('preco_entrada_original')
                    pa = t.get('preco_atual')
                    pnl = t.get('pnl_pct')
                    formula = ""
                    if pe and pa and pe != 0:
                        formula = f"(({pa:.6f} / {pe:.6f}) - 1) * 100 = {pnl:.2f}%"
                    elif t.get('preco_entrada_total') and t.get('preco_saida_total'):
                        pet = t['preco_entrada_total']
                        pst = t['preco_saida_total']
                        formula = f"(({pst:.6f} / {pet:.6f}) - 1) * 100 = {pnl:.2f}%"

                    posicoes_diag.append({
                        "ativo": t.get('ativo'),
                        "status": "fechado" if is_closed else "aberto",
                        "preco_entrada": float(pe) if pe is not None else None,
                        "preco_atual_ou_saida": float(pa) if pa is not None else None,
                        "pnl_pct": round(float(pnl), 2) if pnl is not None else None,
                        "fonte_preco": fonte,
                        "exchange_symbol": t.get('exchange_symbol'),
                        "coingecko_id": t.get('coingecko_id'),
                        "formula_pnl": formula or "(dados insuficientes)",
                    })

                self._send_json({
                    "produto": produto.get('nome'),
                    "produto_id": produto_id,
                    "turma_id": turma_id,
                    "ordem_apis_preco": ["bitget_perp", "bitget_spot", "coingecko", "coinmarketcap", "db"],
                    "posicoes": posicoes_diag,
                })
            except Exception as e:
                import traceback
                self._send_json({'erro': str(e), 'trace': traceback.format_exc()}, 500)
            return

        # API: Diagnóstico de preços e rentabilidade (comparar local vs nuvem)
        if path == '/api/diag/precos':
            try:
                import numpy as np
                from services.portfolio_service import TICKER_TO_COINGECKO, PortfolioService, STABLECOINS

                diag = {'timestamp': datetime.now().isoformat(), 'db_mode': 'sqlite' if repo._use_sqlite else 'postgresql'}
                produtos_diag = {}

                for pkey in ['EXC', 'HB', 'LC', 'AC']:
                    cfg = PORTFOLIO_CONFIG[pkey]
                    csv_path = _root / cfg['csv']
                    if not csv_path.exists():
                        csv_path = _root / 'data' / 'allocations' / cfg['csv']
                    if not csv_path.exists():
                        produtos_diag[pkey] = {'erro': 'CSV nao encontrado'}
                        continue

                    svc = PortfolioService(nome=cfg['nome'], csv_path=str(csv_path))
                    tickers = [c for c in svc.df_aloc.columns if TICKER_TO_COINGECKO.get(c)]
                    cg_ids = [TICKER_TO_COINGECKO[t] for t in tickers]

                    # Consultar precos_diarios do banco atual
                    ticker_info = {}
                    tickers_sem_preco = []
                    try:
                        with repo.connection() as conn:
                            import pandas as pd
                            placeholders = ','.join(['%s'] * len(cg_ids))
                            df_db = pd.read_sql_query(
                                f"SELECT coingecko_id, data, preco FROM precos_diarios WHERE coingecko_id IN ({placeholders}) ORDER BY coingecko_id, data",
                                conn, params=cg_ids
                            )

                        for t in tickers:
                            cg_id = TICKER_TO_COINGECKO[t]
                            grp = df_db[df_db['coingecko_id'] == cg_id]
                            if grp.empty:
                                tickers_sem_preco.append(t)
                                ticker_info[t] = {'cg_id': cg_id, 'registros': 0}
                            else:
                                ticker_info[t] = {
                                    'cg_id': cg_id,
                                    'registros': len(grp),
                                    'min_data': str(grp['data'].min()),
                                    'max_data': str(grp['data'].max()),
                                }
                    except Exception as e:
                        ticker_info = {'erro_db': str(e)}

                    # Calcular rentabilidade com os precos do banco
                    rent_pct = None
                    precos_carregados = 0
                    try:
                        svc2 = PortfolioService(nome=cfg['nome'], csv_path=str(csv_path))
                        svc2.carregar_precos(repo=repo)
                        precos_carregados = len(svc2.df_precos.columns)
                        svc2.calcular_retornos()
                        svc2.calcular_retornos_ponderados()
                        rent_pct = round(svc2.rentabilidade_total_pct(), 4)
                    except Exception as e:
                        rent_pct = f"ERRO: {e}"

                    prod_result = {
                        'csv': cfg['csv'],
                        'csv_periodo': f"{svc.df_aloc.index.min().date()} a {svc.df_aloc.index.max().date()}",
                        'csv_dias': len(svc.df_aloc),
                        'csv_tickers': len(svc.df_aloc.columns),
                        'tickers_com_coingecko': len(tickers),
                        'tickers_com_preco_no_db': len(tickers) - len(tickers_sem_preco),
                        'tickers_sem_preco': tickers_sem_preco,
                        'precos_carregados_final': precos_carregados,
                        'rentabilidade_pct': rent_pct,
                        'detalhe_tickers': ticker_info,
                    }
                    produtos_diag[pkey] = prod_result

                diag['produtos'] = produtos_diag

                # Totais do banco precos_diarios
                try:
                    with repo.connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT COUNT(*), COUNT(DISTINCT coingecko_id) FROM precos_diarios")
                        row = cur.fetchone()
                        if isinstance(row, dict):
                            diag['db_total_registros'] = row.get('count', row.get('COUNT(*)', None))
                        else:
                            diag['db_total_registros'] = row[0]
                            diag['db_total_ativos'] = row[1]
                except Exception as e:
                    diag['db_erro'] = str(e)

                # Imprimir no console para fácil acesso nos logs do Render
                import json as _json
                output = _json.dumps(diag, ensure_ascii=False, default=str, indent=2)
                print("=" * 60, flush=True)
                print("DIAGNOSTICO DE PRECOS E RENTABILIDADE", flush=True)
                print("=" * 60, flush=True)
                print(output, flush=True)
                print("=" * 60, flush=True)

                self._send_json(diag)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({'status': 'error', 'erro': str(e), 'trace': traceback.format_exc()}, 500)
            return

        # 404
        self._send_html("<h1>404 - Pagina nao encontrada</h1>", 404)

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer que atende cada requisicao em uma thread (melhor para deploy com multiplas requisicoes)."""
    daemon_threads = True


def _prewarm_portfolio_cache():
    """Pré-carrega dados dos portfolios em background para evitar timeout na primeira requisição."""
    import time as _time
    _time.sleep(2)
    try:
        repo = get_repo()
        for key in PORTFOLIO_CONFIG:
            try:
                print(f"[Dashboard] Pre-warming cache: {key}...", flush=True)
                get_portfolio_data(key, repo=repo)
                print(f"[Dashboard] Cache pronto: {key}", flush=True)
            except Exception as e:
                print(f"[Dashboard] Erro pre-warming {key}: {e}", flush=True)
    except Exception as e:
        print(f"[Dashboard] Erro ao iniciar pre-warm: {e}", flush=True)


def iniciar_servidor_background(host='127.0.0.1', porta=0, silent=False):
    """
    Inicia o servidor do dashboard em uma thread (para uso com Gunicorn/WSGI).
    Retorna a porta efetiva (útil quando porta=0).
    """
    servidor = ThreadedHTTPServer((host, porta), DashboardHandler)
    porta_efetiva = servidor.server_address[1]
    if not silent:
        print(f"[Dashboard] Servidor em background em {host}:{porta_efetiva}", flush=True)
    import threading
    t = threading.Thread(target=servidor.serve_forever, daemon=True)
    t.start()

    t_warm = threading.Thread(target=_prewarm_portfolio_cache, daemon=True)
    t_warm.start()

    return porta_efetiva


def iniciar_servidor(porta=8080, host='localhost'):
    """Inicia o servidor do dashboard"""
    servidor = ThreadedHTTPServer((host, porta), DashboardHandler)

    # Detectar qual banco está em uso (PostgreSQL ou SQLite fallback)
    try:
        repo = get_repo()
        if getattr(repo, '_use_sqlite', False):
            _db_info = "SQLite local (fallback)"
        else:
            import re as _re
            _url = getattr(repo, 'db_url', '') or ''
            _host_match = _re.search(r'@([^:/@]+)', _url)
            _db_info = f"PostgreSQL ({_host_match.group(1)})" if _host_match else "PostgreSQL (configurado)"
    except Exception as e:
        _db_info = f"Erro: {e}"

    print("=" * 60)
    print("  DASHBOARD WEB - Products & Positions")
    print("=" * 60)
    print(f"\n  Banco de dados: {_db_info}")
    print(f"\n  Servidor iniciado em: http://localhost:{porta}")
    print(f"\n  Rotas principais:")
    print(f"    /                  - Dashboard")
    print(f"    /menu              - Menu completo")
    print(f"    /produto/novo      - Criar produto")
    print(f"    /posicao/nova      - Criar posicao")
    print(f"    /produto/ID        - Ver produto")
    print(f"    /produto/ID/viz/X  - Visualizacao salva")
    print(f"\n  Diagnóstico PnL (origem preços/APIs):")
    print(f"    /api/diag/pnl?produto_id=ID  - Crypto Signals / ICOs")
    print(f"\n  Turmas & Rentabilidade:")
    print(f"    /turmas            - Lista de turmas")
    print(f"    /turmas/nova       - Criar turma")
    print(f"    /turmas/ID         - Detalhes da turma")
    print(f"    /turmas/comparar   - Comparar rentabilidade")
    print(f"\n  Pressione Ctrl+C para parar.")
    print("=" * 60)

    try:
        import webbrowser
        webbrowser.open(f'http://localhost:{porta}')
    except:
        pass

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServidor encerrado.")
        servidor.shutdown()


if __name__ == "__main__":
    import sys
    import os
    # Diagnóstico rentabilidade BTC: defina DEBUG_BTC_RENT=1 para ativar logs detalhados (timestamp + DEBUG)
    if os.environ.get('DEBUG_BTC_RENT'):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Migração CDN → BD: defina MIGRAR_CDN=1 nas env vars do Render para
    # deletar posições antigas de CS/ICOs e reimportar dos CSVs do CDN.
    # Após o deploy bem-sucedido, remova a variável para não rodar novamente.
    if os.environ.get('MIGRAR_CDN'):
        print("[MIGRAÇÃO] MIGRAR_CDN detectado — iniciando migração CDN → BD ...", flush=True)
        try:
            from migrar_cdn_para_bd import importar_crypto_signals, importar_icos
            _migr_repo = get_repo()
            importar_crypto_signals(_migr_repo)
            importar_icos(_migr_repo)
            print("[MIGRAÇÃO] Concluída com sucesso!", flush=True)
        except Exception as _migr_err:
            print(f"[MIGRAÇÃO] ERRO: {_migr_err}", flush=True)
            import traceback; traceback.print_exc()

    porta = int(os.environ.get('PORT', sys.argv[1] if len(sys.argv) > 1 else 8080))
    host = os.environ.get('HOST', '0.0.0.0' if os.environ.get('PORT') else 'localhost')
    iniciar_servidor(porta, host)
