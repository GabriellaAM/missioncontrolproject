"""
Bitget API Service

Handles synchronization of positions with Bitget exchange.
Supports both Perpetual (futures) and Spot products.

OPTIMIZED: Uses connection pooling and parallel requests for ~2x faster sync.
"""
import os
import hmac
import hashlib
import base64
import time
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
from storage.sqlite_repo import connect_pg
import logging

logging.basicConfig(
    level=logging.INFO,  # troque para DEBUG quando quiser mais detalhe
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# Load .env from MissionControl root
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

# Thread-local storage for HTTP sessions (thread-safe for web servers)
import threading
_thread_local = threading.local()

def _get_session() -> requests.Session:
    """Gets or creates a thread-local HTTP session for connection reuse."""
    if not hasattr(_thread_local, 'session'):
        logger.debug("Criando nova sessão HTTP para thread")
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({
            "Content-Type": "application/json",
            "locale": "en-US"
        })
    return _thread_local.session


def _normalize_product_name(nome: str) -> str:
    """
    Normalizes product name for environment variable lookup.
    Example: "Soros 3" -> "SOROS_3", "Soros Spot 2" -> "SOROS_SPOT_2"
    """
    if not nome or not isinstance(nome, str):
        return ""
    # Remove accents and special chars, replace spaces with underscore
    nome = nome.strip().upper()
    nome = re.sub(r'[^A-Z0-9_]', '_', nome)
    nome = re.sub(r'_+', '_', nome)  # Multiple underscores to single
    nome = nome.strip('_')
    return nome


def get_bitget_credentials(produto_nome: str) -> Optional[Dict[str, str]]:
    """
    Gets Bitget API credentials for a product from environment variables.

    Expected format in .env:
        BITGET_API_KEY_SOROS_3=xxx
        BITGET_SECRET_KEY_SOROS_3=xxx
        BITGET_PASSPHRASE_SOROS_3=xxx

    Fallback: if the product name ends with " 1" (e.g. "Soros Spot 1"),
    also tries without the "_1" suffix (e.g. BITGET_API_KEY_SOROS_SPOT).

    Args:
        produto_nome: Product name as stored in database

    Returns:
        Dict with api_key, secret_key, passphrase or None if not configured
    """
    logger.info(f"Buscando credenciais para produto: {produto_nome}")

    normalized = _normalize_product_name(produto_nome)

    # Try exact match first
    creds = _try_env_credentials(normalized)
    if creds:
        return creds

    # Fallback: strip trailing _1 (e.g. "Soros Spot 1" -> SOROS_SPOT_1 -> SOROS_SPOT)
    if normalized.endswith("_1"):
        fallback = normalized[:-2]
        creds = _try_env_credentials(fallback)
        if creds:
            return creds

    # Fallback: "Soros Perpétuos" -> SOROS_PERP_TUOS; .env pode ter SOROS_PERP ou MEMEBOT_PERP
    if "_PERP_TUOS" in normalized or "_PERPETUOS" in normalized:
        short_perp = normalized.replace("_PERP_TUOS", "_PERP").replace("_PERPETUOS", "_PERP")
        creds = _try_env_credentials(short_perp)
        if creds:
            return creds

    logger.warning(f"Credenciais não encontradas para: {produto_nome}")

    return None


def _try_env_credentials(suffix: str) -> Optional[Dict[str, str]]:
    """Tries to load Bitget credentials from env for a given suffix."""
    api_key = os.getenv(f"BITGET_API_KEY_{suffix}")
    secret_key = os.getenv(f"BITGET_SECRET_KEY_{suffix}")
    passphrase = os.getenv(f"BITGET_PASSPHRASE_{suffix}")

    if api_key and secret_key and passphrase:
        return {
            "api_key": api_key,
            "secret_key": secret_key,
            "passphrase": passphrase
        }
    return None


def _create_signature(timestamp: str, method: str, request_path: str, body: str, secret_key: str) -> str:
    """Creates HMAC-SHA256 signature for Bitget API authentication"""
    message = timestamp + method.upper() + request_path + body
    signature = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode('utf-8')

def _bitget_request(credentials: Dict[str, str], method: str, endpoint: str, body: str = "") -> dict:
    """Makes authenticated request to Bitget API using connection pooling."""
    
    try:
        logger.debug(f"[BITGET REQUEST] {method} {endpoint}")

        base_url = "https://api.bitget.com"
        timestamp = str(int(time.time() * 1000))

        signature = _create_signature(
            timestamp,
            method,
            endpoint,
            body,
            credentials["secret_key"]
        )

        headers = {
            "ACCESS-KEY": credentials["api_key"],
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": credentials["passphrase"],
        }

        url = base_url + endpoint
        session = _get_session()

        logger.debug(f"[BITGET REQUEST] URL: {url}")

        if method.upper() == "GET":
            response = session.get(url, headers=headers, timeout=10)
        else:
            response = session.post(url, headers=headers, data=body, timeout=10)

        logger.debug(f"[BITGET RESPONSE] Status: {response.status_code}")
        logger.debug(f"[BITGET RESPONSE] Body: {response.text[:300]}")

        response.raise_for_status()  # garante erro HTTP

        return response.json()

    except Exception as e:
        logger.exception(f"[BITGET ERROR] Erro na requisição: {str(e)}")
        raise


# Base URL para endpoints públicos (sem autenticação)
BITGET_PUBLIC_BASE = "https://api.bitget.com"


def fetch_bitget_tickers_perpetuals() -> Dict[str, float]:
    """
    Busca preços atuais (mark price) de todos os contratos perpétuos USDT na Bitget.
    Endpoint público — não requer API key.

    Returns:
        Dict[symbol, mark_price] ex: {"BTCUSDT": 69144.4, "SOLUSDT": 85.188}
    """
    url = f"{BITGET_PUBLIC_BASE}/api/v2/mix/market/tickers"
    params = {"productType": "USDT-FUTURES"}
    try:
        session = _get_session()
        r = session.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "00000":
            return {}
        result = {}
        for item in data.get("data", []):
            symbol = item.get("symbol")
            mark = item.get("markPrice")
            if symbol and mark is not None:
                try:
                    result[symbol] = float(mark)
                except (TypeError, ValueError):
                    pass
        return result
    except Exception:
        return {}


def fetch_bitget_tickers_spot() -> Dict[str, float]:
    """
    Busca preços atuais (last) de todos os pares spot na Bitget.
    Endpoint público — não requer API key.

    Returns:
        Dict[symbol, last_price] ex: {"BTCUSDT": 69108.04, "SOLUSDT": 85.02}
        Apenas pares *USDT para uso com posições spot.
    """
    url = f"{BITGET_PUBLIC_BASE}/api/v2/spot/market/tickers"
    try:
        session = _get_session()
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "00000":
            return {}
        result = {}
        for item in data.get("data", []):
            symbol = item.get("symbol")
            last = item.get("lastPr")
            if symbol and last is not None and symbol.endswith("USDT"):
                try:
                    result[symbol] = float(last)
                except (TypeError, ValueError):
                    pass
        return result
    except Exception:
        return {}


def fetch_bitget_ticker_spot(symbol: str) -> Optional[float]:
    """
    Busca preço atual (last) de um par spot na Bitget.
    Endpoint público — não requer API key.

    Args:
        symbol: Par de negociação (ex: BTCUSDT)

    Returns:
        Preço atual ou None
    """
    tickers = fetch_bitget_tickers_spot()
    return tickers.get(symbol) if symbol in tickers else None


def fetch_perpetual_positions(credentials: Dict[str, str]) -> List[Dict]:
    """
    Fetches open perpetual positions from Bitget.

    Returns list of positions with:
        - symbol: Trading pair (e.g., BTCUSDT)
        - holdSide: long or short
        - total: Total quantity held
        - averageOpenPrice: Average entry price
        - unrealizedPL: Unrealized PnL
    """

    logger.info("[PERP] Buscando posições perpétuas")

    endpoint = "/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT"
    result = _bitget_request(credentials, "GET", endpoint)

    logger.debug(f"[PERP] Resposta API: {result}")

    if result.get("code") != "00000":
        logger.error(f"[PERP] Erro API: {result}")
        raise Exception(f"Bitget API error: {result.get('msg', 'Unknown error')}")

    positions = []
    data = result.get("data", [])

    for pos in data:
        # Bitget returns positions with total > 0
        
        logger.debug(f"[PERP] Processando posição: {pos}")

        total = float(pos.get("total", 0))
        if total > 0:
            # Campo correto é openPriceAvg (não averageOpenPrice)
            entry_price = float(pos.get("openPriceAvg", 0) or 0)
            positions.append({
                "symbol": pos.get("symbol", "").replace("USDT", ""),  # BTCUSDT -> BTC
                "exchange_symbol": pos.get("symbol", ""),  # Keep original
                "side": pos.get("holdSide", "").lower(),
                "quantity": total,
                "entry_price": entry_price,
                "unrealized_pnl": float(pos.get("unrealizedPL", 0) or 0),
                "leverage": int(pos.get("leverage", 1) or 1),
                "mark_price": float(pos.get("markPrice", 0) or 0),
                "break_even_price": float(pos.get("breakEvenPrice", 0) or 0),
                # Data de abertura: cTime (criação) ou openTime; aceita ms ou segundos
                "ctime": pos.get("cTime") or pos.get("openTime") or pos.get("ctime"),
            })

    logger.info(f"[PERP] Total posições válidas: {len(positions)}")

    return positions


def fetch_spot_positions(credentials: Dict[str, str]) -> List[Dict]:
    """
    Fetches open spot positions from Bitget Copy Trading.

    Uses GET /api/v2/copy/spot-trader/order-current-track
    Paginates with idLessThan to fetch all open positions.

    Returns list of positions with:
        - symbol: Coin name (e.g., "BTC")
        - exchange_symbol: Trading pair (e.g., "BTCUSDT")
        - side: Always "long" for spot
        - quantity: Filled buy quantity (buyFillSize)
        - entry_price: Buy price (buyPrice)
        - ctime: Buy timestamp in ms (buyTime) — used as data_entrada
        - unrealized_pnl: Unrealized PnL
        - tracking_no: Unique tracking order number
    """

    logger.info("[SPOT] Buscando posições spot")

    all_positions = []
    end_id = None
    max_pages = 10  # Safety limit

    for _ in range(max_pages):
        endpoint = "/api/v2/copy/spot-trader/order-current-track?limit=50"
        if end_id:
            endpoint += f"&idLessThan={end_id}"

        result = _bitget_request(credentials, "GET", endpoint)

        if result.get("code") != "00000":
            raise Exception(f"Bitget order-current-track error: {result.get('msg', 'Unknown')}")

        data = result.get("data", {})
        tracking_list = data.get("trackingList", [])

        logger.debug(f"[SPOT] Página com {len(tracking_list)} ordens")

        if not tracking_list:
            break

        for order in tracking_list:
            symbol = order.get("symbol", "")
            qty = float(order.get("buyFillSize", 0) or 0)

            logger.debug(f"[SPOT] Ordem: {order}")

            if qty > 0:
                all_positions.append({
                    "symbol": symbol.replace("USDT", ""),
                    "exchange_symbol": symbol,
                    "side": "long",
                    "quantity": qty,
                    "entry_price": float(order.get("buyPrice", 0) or 0),
                    "ctime": order.get("buyTime"),  # timestamp ms — data real de compra
                    "unrealized_pnl": float(order.get("unrealizedPL", 0) or 0),
                    "tracking_no": order.get("trackingNo"),
                })

        # Pagination: use endId for next page
        new_end_id = data.get("endId")
        if not new_end_id or new_end_id == end_id:
            break
        end_id = new_end_id

    logger.info(f"[SPOT] Total posições: {len(all_positions)}")
    
    return all_positions


def _batch_load_quantities(repo, posicao_ids: List[int]) -> Dict[int, float]:
    """Batch load quantities for multiple positions in a single query."""

    logger.debug(f"[DB] Batch load para {len(posicao_ids)} posições")

    if not posicao_ids:
        return {}

    placeholders = ','.join(['%s' for _ in posicao_ids])
    query = f"""
        SELECT posicao_id, quantidade
        FROM posicao_atributos_produto
        WHERE posicao_id IN ({placeholders})
    """

    conn = connect_pg(repo.db_url)
    
    try:
        cursor = conn.cursor()
        logger.debug(f"[DB] Executando query batch")

        cursor.execute(query, posicao_ids)
        rows = cursor.fetchall()

        logger.debug(f"[DB] {len(rows)} linhas retornadas")

        return {row[0]: row[1] for row in rows}

    except Exception as e:
        logger.exception(f"[DB] Erro no batch load: {str(e)}")
        raise

    finally:
        conn.close()

def sync_positions_with_exchange(repo, produto_id: int, verbose: bool = True) -> Dict:
    """
    Synchronizes position quantities with Bitget exchange.

    OPTIMIZED: Uses batch DB queries and connection pooling for ~2x faster sync.
    """

    logger.info(f"[SYNC] Iniciando sync produto_id={produto_id}")

    resultado = {
        "synced": 0,
        "skipped": 0,
        "not_found": [],
        "errors": []
    }

    try:
        # Load product info
        produto = repo.carregar_produto(produto_id)
        logger.debug(f"[SYNC] Produto carregado: {produto}")

        if not produto:
            resultado["errors"].append(f"Produto {produto_id} não encontrado")
            return resultado

        produto_nome = produto["nome"]
        produto_tipo = produto["tipo"]

        # Get credentials
        credentials = get_bitget_credentials(produto_nome)
        if not credentials:
            if verbose:
                print(f"  Sem credenciais Bitget para {produto_nome}")
            resultado["skipped"] += 1
            return resultado

        logger.info(f"[SYNC] Buscando posições na exchange...")

        # Fetch positions
        if "perpétuo" in produto_tipo.lower() or "perpetuo" in produto_tipo.lower():
            exchange_positions = fetch_perpetual_positions(credentials)
        elif "spot" in produto_tipo.lower():
            exchange_positions = fetch_spot_positions(credentials)
        else:
            if verbose:
                print(f"Tipo não suportado: {produto_tipo}")
            return resultado

        logger.info(f"[SYNC] {len(exchange_positions)} posições encontradas")

        # Lookup
        exchange_lookup = {
            (p["exchange_symbol"].upper(), p["side"]): p
            for p in exchange_positions
        }

        df_posicoes = repo.carregar_posicoes_abertas(produto_id)

        if df_posicoes.empty:
            return resultado

        posicao_ids = df_posicoes['id'].tolist()
        qty_map = _batch_load_quantities(repo, posicao_ids)

        for _, db_pos in df_posicoes.iterrows():

            posicao_id = db_pos["id"]
            ativo = db_pos["ativo"]
            exchange_symbol = db_pos.get("exchange_symbol")
            side = db_pos["side"]

            logger.debug(f"[SYNC] Processando ID={posicao_id} | {ativo}")

            if not exchange_symbol:
                resultado["skipped"] += 1
                continue

            key = (exchange_symbol.upper(), side)
            exchange_pos = exchange_lookup.get(key)

            if not exchange_pos:
                resultado["not_found"].append(ativo)
                continue

            new_qty = exchange_pos["quantity"]
            updates = {"quantidade": new_qty}

            if exchange_pos.get("entry_price"):
                updates["preco_entrada_exchange"] = exchange_pos["entry_price"]

            if exchange_pos.get("unrealized_pnl") is not None:
                updates["pnl_exchange"] = exchange_pos["unrealized_pnl"]

            if exchange_pos.get("leverage") and exchange_pos["leverage"] != 1:
                updates["leverage"] = exchange_pos["leverage"]

            current_qty = qty_map.get(posicao_id)

            logger.info(f"[SYNC] {ativo}: {current_qty} -> {new_qty}")

            repo.atualizar_atributos_posicao(posicao_id, **updates)

            resultado["synced"] += 1

        logger.info(f"[SYNC] Fim sync: {resultado}")
        return resultado

    except Exception as e:
        logger.exception(f"[SYNC ERROR] {str(e)}")
        resultado["errors"].append(str(e))
        return resultado

def fetch_perpetual_history(credentials: Dict[str, str], limit: int = 100, max_pages: int = 20) -> List[Dict]:
    """
    Fetches closed perpetual positions from Bitget (last 3 months).

    Uses GET /api/v2/mix/position/history-position
    Doc: list items use "ctime" and "utime" (lowercase); pagination via idLessThan=endId.

    Returns list of closed positions with:
        - symbol, exchange_symbol, side, open_price, close_price,
          pnl, quantity, open_time, close_time
    """
    all_positions = []
    page_limit = min(limit, 100)  # API max 100 per page
    id_less_than = None

    for _ in range(max_pages):
        endpoint = f"/api/v2/mix/position/history-position?productType=USDT-FUTURES&limit={page_limit}"
        if id_less_than:
            endpoint += f"&idLessThan={id_less_than}"

        result = _bitget_request(credentials, "GET", endpoint)
        if result.get("code") != "00000":
            raise Exception(f"Bitget history-position error: {result.get('msg', 'Unknown')}")

        data_obj = result.get("data", {})
        if not isinstance(data_obj, dict):
            break
        items = data_obj.get("list", [])
        if not items:
            break

        for pos in items:
            symbol_raw = pos.get("symbol", "")
            # API doc: response uses "ctime" / "utime" (lowercase); aceitar ambos
            open_ts = pos.get("cTime") or pos.get("ctime")
            close_ts = pos.get("uTime") or pos.get("utime")
            all_positions.append({
                "symbol": symbol_raw.replace("USDT", ""),
                "exchange_symbol": symbol_raw,
                "side": pos.get("holdSide", "").lower(),
                "open_price": float(pos.get("openAvgPrice", 0)),
                "close_price": float(pos.get("closeAvgPrice", 0)),
                "pnl": float(pos.get("netProfit", 0)),
                "quantity": float(pos.get("closeTotalPos", 0)),
                "open_time": open_ts,
                "close_time": close_ts,
            })

        end_id = data_obj.get("endId")
        if not end_id:
            break
        id_less_than = end_id
        if len(all_positions) >= limit:
            break
        time.sleep(0.05)

    return all_positions


def fetch_spot_copy_history(credentials: Dict[str, str], limit: int = 100) -> List[Dict]:
    """
    Fetches closed spot copy-trading positions from Bitget.

    Uses GET /api/v2/copy/spot-trader/order-history-track
    Paginates with idLessThan to fetch all history.

    Returns list of closed positions with:
        - symbol, exchange_symbol, side, open_price, close_price,
          quantity, open_time, close_time, pnl, tracking_no
    """
    all_history = []
    end_id = None
    max_pages = 20
    page_limit = min(limit, 50)

    for _ in range(max_pages):
        endpoint = f"/api/v2/copy/spot-trader/order-history-track?limit={page_limit}"
        if end_id:
            endpoint += f"&idLessThan={end_id}"

        result = _bitget_request(credentials, "GET", endpoint)

        if result.get("code") != "00000":
            raise Exception(f"Bitget order-history-track error: {result.get('msg', 'Unknown')}")

        data = result.get("data", {})
        tracking_list = data.get("trackingList", [])

        if not tracking_list:
            break

        for order in tracking_list:
            symbol_raw = order.get("symbol", "")
            all_history.append({
                "symbol": symbol_raw.replace("USDT", ""),
                "exchange_symbol": symbol_raw,
                "side": "long",  # Spot copy is always long
                "open_price": float(order.get("buyPrice", 0) or 0),
                "close_price": float(order.get("sellPrice", 0) or 0),
                "quantity": float(order.get("fillSize", 0) or 0),
                "open_time": order.get("buyTime"),    # timestamp ms
                "close_time": order.get("sellTime"),  # timestamp ms
                "pnl": float(order.get("netProfit", 0) or 0),
                "tracking_no": order.get("trackingNo"),
            })

        # Pagination
        new_end_id = data.get("endId")
        if not new_end_id or new_end_id == end_id:
            break
        end_id = new_end_id

        if len(all_history) >= limit:
            break

    return all_history


def _timestamp_to_date_str(ts) -> Optional[str]:
    """Converte timestamp da API (ms ou segundos) para YYYY-MM-DD em data LOCAL. Retorna None se inválido."""
    from datetime import datetime
    if ts is None:
        return None
    try:
        t = int(ts)
        if t > 1e12:  # em ms
            t = t // 1000
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return None


def auto_sync_positions(repo, produto_id: int, verbose: bool = True) -> Dict:
    """
    Full position lifecycle sync with multi-exchange (Bitget + OKX):
      - OPEN: positions on exchange but not in DB -> create
      - CLOSE: positions in DB but not on exchange -> close with price/date from history
      - UPDATE: positions in both -> update quantities/PnL

    Returns:
        Dict with summary: {opened, closed, synced, skipped, errors}
    """
    from domain.posicao import Posicao
    from datetime import datetime
    import pandas as pd
    import time

    from exchanges.factory import ExchangeFactory
    from services.bitget_service import _normalize_product_name
    from services.credentials import get_exchange_credentials  # <- NOVO

    logger.info(f"[AUTO-SYNC] ===== INÍCIO produto_id={produto_id} =====")

    resultado = {
        "opened": 0,
        "closed": 0,
        "synced": 0,
        "skipped": 0,
        "errors": []
    }

    # ===== LOAD PRODUTO =====
    produto = repo.carregar_produto(produto_id)

    logger.debug(f"[AUTO-SYNC] Produto: {produto}")

    if not produto:
        resultado["errors"].append(f"Produto {produto_id} não encontrado")
        logger.error(f"[AUTO-SYNC] Produto {produto_id} NÃO encontrado")
        return resultado

    produto_nome = produto["nome"]

    # ===== CREDENTIALS (MULTI-EXCHANGE) =====
    credentials = get_exchange_credentials(produto_nome)

    logger.debug(f"[AUTO-SYNC] Credenciais encontradas: {bool(credentials)}")

    if not credentials:
        suffix = _normalize_product_name(produto_nome)
        if verbose:
            print(f"  [AUTO-SYNC] Sem credenciais para '{produto_nome}' (suffix: {suffix})")
        resultado["skipped"] += 1
        resultado["errors"].append(f"Credenciais não encontradas para {suffix}")
        return resultado

    # ===== EXCHANGE FACTORY =====
    try:
        logger.info(f"[AUTO-SYNC] Criando exchange para {produto_nome}")
        exchange = ExchangeFactory.get(produto_nome, credentials)
    except Exception as e:
        resultado["errors"].append(str(e))
        logger.exception("[AUTO-SYNC] Erro ao criar exchange")
        return resultado

    # ===== FETCH POSITIONS =====
    try:
        logger.info("[AUTO-SYNC] Buscando posições na exchange...")
        exchange_positions = exchange.fetch_positions()
    except Exception as e:
        resultado["errors"].append(f"Erro ao buscar posições: {str(e)}")
        if verbose:
            print(f"  [AUTO-SYNC] Erro ao buscar posições: {e}")
        return resultado

    if verbose:
        print(f"  [AUTO-SYNC] {produto_nome}: {len(exchange_positions)} posições na exchange")

    # ===== LOOKUP EXCHANGE =====
    exchange_lookup = {}
    for pos in exchange_positions:
        key = (pos["exchange_symbol"].upper(), pos["side"])
        exchange_lookup[key] = pos
    
    logger.debug(f"[AUTO-SYNC] Exchange lookup size: {len(exchange_lookup)}")

    # ===== LOAD DB =====
    df_posicoes = repo.carregar_posicoes_abertas(produto_id)

    logger.info(f"[AUTO-SYNC] {len(df_posicoes)} posições no banco")

    db_lookup = {}
    if not df_posicoes.empty:
        for _, db_pos in df_posicoes.iterrows():
            ex_sym = db_pos.get("exchange_symbol")
            side = db_pos.get("side", "long")
            if ex_sym and not pd.isna(ex_sym):
                key = (str(ex_sym).upper(), side)
                db_lookup[key] = db_pos

    # ===== 1. OPEN =====
    for key, ex_pos in exchange_lookup.items():
        if key not in db_lookup:
            ativo = ex_pos["symbol"]
            side = ex_pos["side"]
            entry_price = ex_pos.get("entry_price", 0)
            exchange_symbol = ex_pos.get("exchange_symbol").upper()

            logger.info(
            f"[AUTO-SYNC][OPEN] {ativo} {side} "
            f"qty={ex_pos.get('quantity')} "
            f"price={entry_price}"
            )

            data_entrada = _timestamp_to_date_str(ex_pos.get("ctime"))

            logger.debug(f"[AUTO-SYNC][OPEN] data_entrada={data_entrada}")

            if not data_entrada:
                data_entrada = datetime.now().strftime("%Y-%m-%d")

            try:
                # evitar duplicata

                logger.debug("[AUTO-SYNC][OPEN] Rechecando duplicidade no banco")

                df_recheck = repo.carregar_posicoes_abertas(produto_id)
                skip = False

                if not df_recheck.empty:
                    for _, row in df_recheck.iterrows():

                        logger.warning(
                        f"[AUTO-SYNC][OPEN] DUPLICATA detectada {ativo} "
                        f"{exchange_symbol} {data_entrada}"
                        )

                        ex_sym = row.get("exchange_symbol")
                        s = row.get("side", "long")

                        data_db = row.get("data_entrada")
                        if hasattr(data_db, "strftime"):
                            data_db = data_db.strftime("%Y-%m-%d")
                        else:
                            data_db = str(data_db)[:10]

                        if (str(ex_sym).upper(), s) == key and data_db == data_entrada:
                            posicao_id = row["id"]

                            repo.atualizar_posicao(
                                posicao_id,
                                exchange_symbol=exchange_symbol,
                                preco_entrada=entry_price if entry_price > 0 else None
                            )

                            repo.salvar_atributos_posicao(
                                posicao_id,
                                produto_id,
                                quantidade=ex_pos.get("quantity", 0),
                                preco_entrada_exchange=entry_price,
                                pnl_exchange=ex_pos.get("unrealized_pnl")
                            )

                            resultado["synced"] += 1
                            skip = True
                            break

                if skip:
                    continue

                # criar posição
                posicao = Posicao(
                    ativo=ativo,
                    side=side,
                    data_entrada=data_entrada,
                    preco_entrada=entry_price,
                    exchange_symbol=exchange_symbol,
                )

                posicao_id = repo.salvar_posicao(produto_id, posicao)
                logger.info(f"[AUTO-SYNC][OPEN] Criada posição ID={posicao_id}")
                resultado["opened"] += 1

                repo.salvar_atributos_posicao(
                    posicao_id,
                    produto_id,
                    quantidade=ex_pos.get("quantity", 0),
                    preco_entrada_exchange=entry_price,
                    pnl_exchange=ex_pos.get("unrealized_pnl"),
                )

                if verbose:
                    print(f"  [AUTO-SYNC] ABERTA: {ativo} {side} qty={ex_pos.get('quantity')}")

            except Exception as e:
                resultado["errors"].append(f"Erro ao abrir {ativo}: {str(e)}")

    # ===== 2. CLOSE =====
    history_cache = None

    for key, db_pos in db_lookup.items():
        if key not in exchange_lookup:
            posicao_id = db_pos["id"]
            ativo = db_pos["ativo"]
            side = db_pos.get("side", "long")
            exchange_symbol = db_pos.get("exchange_symbol", "")

            if verbose:
                print(f"  [AUTO-SYNC] {ativo} não está mais na exchange")

            close_price = None
            close_date = None

            try:
                if history_cache is None:
                    logger.debug("[AUTO-SYNC][CLOSE] Carregando histórico")
                    history_cache = exchange.fetch_history()
                    time.sleep(0.1)

                for hist in history_cache:
                    logger.debug(f"[AUTO-SYNC][CLOSE] Testando histórico: {hist}")
                    if (
                        hist["exchange_symbol"].upper() == exchange_symbol.upper()
                        and hist["side"] == side
                    ):
                        close_price = hist.get("close_price")
                        close_date = _timestamp_to_date_str(hist.get("close_time"))
                        break
            except Exception:
                pass

            if not close_date:
                close_date = datetime.now().strftime("%Y-%m-%d")

            try:
                repo.atualizar_posicao(
                    posicao_id,
                    status="closed",
                    data_saida=close_date,
                    preco_saida=close_price if close_price else None
                )
                resultado["closed"] += 1
                logger.info(
                f"[AUTO-SYNC][CLOSE] {ativo} fechado "
                f"price={close_price} date={close_date}"
                )

            except Exception as e:
                resultado["errors"].append(f"Erro ao fechar {ativo}: {str(e)}")

    # ===== 3. UPDATE =====
    if not df_posicoes.empty:
        posicao_ids = df_posicoes['id'].tolist()
        qty_map = _batch_load_quantities(repo, posicao_ids)

        for key, db_pos in db_lookup.items():
            if key in exchange_lookup:
                posicao_id = db_pos["id"]
                ativo = db_pos["ativo"]
                ex_pos = exchange_lookup[key]

                try:
                    repo.atualizar_posicao(
                        posicao_id,
                        exchange_symbol=ex_pos.get("exchange_symbol"),
                        preco_entrada=ex_pos.get("entry_price"),
                        data_entrada=_timestamp_to_date_str(ex_pos.get("ctime"))
                    )

                    logger.debug(
                    f"[AUTO-SYNC][UPDATE] {ativo} "
                    f"qty={ex_pos.get('quantity')} "
                    f"pnl={ex_pos.get('unrealized_pnl')}"
                    )

                    logger.debug(f"[AUTO-SYNC][UPDATE] Atualizando DB posicao_id={posicao_id}")

                    repo.salvar_atributos_posicao(
                        posicao_id,
                        produto_id,
                        quantidade=ex_pos.get("quantity"),
                        preco_entrada_exchange=ex_pos.get("entry_price"),
                        pnl_exchange=ex_pos.get("unrealized_pnl"),
                        leverage=ex_pos.get("leverage"),
                    )

                    resultado["synced"] += 1

                except Exception as e:
                    resultado["errors"].append(f"Erro ao atualizar {ativo}: {str(e)}")

    if verbose:
        print(f"  [AUTO-SYNC] Resumo: {resultado}")

    return resultado


def test_bitget_connection(produto_nome: str) -> Tuple[bool, str]:
    """
    Tests Bitget API connection for a product.

    Args:
        produto_nome: Product name

    Returns:
        Tuple of (success, message)
    """

    logger.info(f"[TEST] Testando conexão para {produto_nome}")

    credentials = get_bitget_credentials(produto_nome)
    if not credentials:
        return False, f"Credenciais não encontradas para {produto_nome}"

    try:
        # Test with a simple endpoint
        result = _bitget_request(credentials, "GET", "/api/v2/spot/account/assets")
        if result.get("code") == "00000":
            return True, "Conexão OK"
        else:
            return False, f"Erro: {result.get('msg', 'Unknown')}"
    except Exception as e:
        logger.exception("[TEST] Erro na conexão")
        return False, f"Exceção: {str(e)}"
