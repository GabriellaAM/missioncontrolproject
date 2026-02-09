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


# Load .env from MissionControl root
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

# Thread-local storage for HTTP sessions (thread-safe for web servers)
import threading
_thread_local = threading.local()

def _get_session() -> requests.Session:
    """Gets or creates a thread-local HTTP session for connection reuse."""
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({
            "Content-Type": "application/json",
            "locale": "en-US"
        })
    return _thread_local.session


def _normalize_product_name(nome: str) -> str:
    """
    Normalizes product name for environment variable lookup.
    Example: "Soros 3" -> "SOROS_3"
    """
    # Remove accents and special chars, replace spaces with underscore
    nome = nome.upper()
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

    Args:
        produto_nome: Product name as stored in database

    Returns:
        Dict with api_key, secret_key, passphrase or None if not configured
    """
    normalized = _normalize_product_name(produto_nome)

    api_key = os.getenv(f"BITGET_API_KEY_{normalized}")
    secret_key = os.getenv(f"BITGET_SECRET_KEY_{normalized}")
    passphrase = os.getenv(f"BITGET_PASSPHRASE_{normalized}")

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
    base_url = "https://api.bitget.com"
    timestamp = str(int(time.time() * 1000))

    signature = _create_signature(
        timestamp, method, endpoint, body,
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

    if method.upper() == "GET":
        response = session.get(url, headers=headers, timeout=10)
    else:
        response = session.post(url, headers=headers, data=body, timeout=10)

    return response.json()


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
    endpoint = "/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT"
    result = _bitget_request(credentials, "GET", endpoint)

    if result.get("code") != "00000":
        raise Exception(f"Bitget API error: {result.get('msg', 'Unknown error')}")

    positions = []
    data = result.get("data", [])

    for pos in data:
        # Bitget returns positions with total > 0
        total = float(pos.get("total", 0))
        if total > 0:
            positions.append({
                "symbol": pos.get("symbol", "").replace("USDT", ""),  # BTCUSDT -> BTC
                "exchange_symbol": pos.get("symbol", ""),  # Keep original
                "side": pos.get("holdSide", "").lower(),
                "quantity": total,
                "entry_price": float(pos.get("averageOpenPrice", 0)),
                "unrealized_pnl": float(pos.get("unrealizedPL", 0)),
                "leverage": int(pos.get("leverage", 1)),
            })

    return positions


def _fetch_copy_trading_positions(credentials: Dict[str, str]) -> List[Dict]:
    """Fetches positions from Copy Trading endpoint."""
    positions = []
    try:
        endpoint = "/api/v2/copy/spot-trader/order-current-track"
        result = _bitget_request(credentials, "GET", endpoint)
        if result.get("code") == "00000":
            tracking_list = result.get("data", {}).get("trackingList", [])
            for order in tracking_list:
                symbol = order.get("symbol", "")
                qty = float(order.get("buyFillSize", 0))
                if qty > 0:
                    positions.append({
                        "symbol": symbol.replace("USDT", ""),
                        "exchange_symbol": symbol,
                        "side": "long",
                        "quantity": qty,
                        "entry_price": float(order.get("buyPrice", 0)),
                        "order_id": order.get("orderId"),
                        "source": "copy_trading",
                    })
    except Exception:
        pass
    return positions


def _fetch_spot_assets(credentials: Dict[str, str]) -> List[Dict]:
    """Fetches positions from Spot Account Assets endpoint."""
    positions = []
    try:
        endpoint = "/api/v2/spot/account/assets"
        result = _bitget_request(credentials, "GET", endpoint)
        if result.get("code") == "00000":
            assets = result.get("data", [])
            for asset in assets:
                coin = asset.get("coin", "")
                available = float(asset.get("available", 0))
                frozen = float(asset.get("frozen", 0))
                total = available + frozen
                if coin in ["USDT", "USDC", "BUSD"] or total < 0.0001:
                    continue
                positions.append({
                    "symbol": coin,
                    "exchange_symbol": f"{coin}USDT",
                    "side": "long",
                    "quantity": total,
                    "entry_price": 0,
                    "source": "assets",
                })
    except Exception:
        pass
    return positions


def fetch_spot_positions(credentials: Dict[str, str]) -> List[Dict]:
    """
    Fetches spot positions from Bitget.
    OPTIMIZED: Runs endpoints in parallel for ~2x faster fetching.

    Returns list of positions with:
        - symbol: Trading pair
        - quantity: Amount held
    """
    # Run both endpoints in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_copy = executor.submit(_fetch_copy_trading_positions, credentials)
        future_assets = executor.submit(_fetch_spot_assets, credentials)

        copy_positions = future_copy.result()
        asset_positions = future_assets.result()

    # Prefer Copy Trading positions (have entry_price), fall back to assets
    if copy_positions:
        return copy_positions
    return asset_positions


def _batch_load_quantities(repo, posicao_ids: List[int]) -> Dict[int, float]:
    """Batch load quantities for multiple positions in a single query."""
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
        cursor.execute(query, posicao_ids)
        return {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        conn.close()


def sync_positions_with_exchange(repo, produto_id: int, verbose: bool = True) -> Dict:
    """
    Synchronizes position quantities with Bitget exchange.

    OPTIMIZED: Uses batch DB queries and connection pooling for ~2x faster sync.

    Args:
        repo: SQLiteRepo instance
        produto_id: Product ID to sync
        verbose: Print progress info

    Returns:
        Dict with summary: {synced: int, skipped: int, not_found: list, errors: list}
    """
    resultado = {
        "synced": 0,
        "skipped": 0,
        "not_found": [],
        "errors": []
    }

    # Load product info
    produto = repo.carregar_produto(produto_id)
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

    # Fetch positions from exchange based on product type
    try:
        if "perpétuo" in produto_tipo.lower() or "perpetuo" in produto_tipo.lower():
            exchange_positions = fetch_perpetual_positions(credentials)
        elif "spot" in produto_tipo.lower():
            exchange_positions = fetch_spot_positions(credentials)
        else:
            if verbose:
                print(f"  Tipo de produto não suportado para sync: {produto_tipo}")
            return resultado
    except Exception as e:
        resultado["errors"].append(f"Erro ao buscar posições: {str(e)}")
        return resultado

    if verbose:
        print(f"  {len(exchange_positions)} posições encontradas na exchange")

    # Create lookup by exchange_symbol + side
    exchange_lookup = {}
    for pos in exchange_positions:
        key = (pos["exchange_symbol"].upper(), pos["side"])
        exchange_lookup[key] = pos

    # Load open positions from database
    df_posicoes = repo.carregar_posicoes_abertas(produto_id)

    if df_posicoes.empty:
        if verbose:
            print("  Nenhuma posição aberta no banco")
        return resultado

    # OPTIMIZED: Batch load all quantities in single query
    posicao_ids = df_posicoes['id'].tolist()
    qty_map = _batch_load_quantities(repo, posicao_ids)

    for _, db_pos in df_posicoes.iterrows():
        posicao_id = db_pos["id"]
        ativo = db_pos["ativo"]
        exchange_symbol = db_pos.get("exchange_symbol")
        side = db_pos["side"]

        # Skip positions without exchange_symbol (manual positions)
        if not exchange_symbol:
            resultado["skipped"] += 1
            if verbose:
                print(f"  [{ativo}] Sem exchange_symbol - pulando")
            continue

        # Look up in exchange data
        key = (exchange_symbol.upper(), side)
        exchange_pos = exchange_lookup.get(key)

        if not exchange_pos:
            resultado["not_found"].append(f"{ativo} ({exchange_symbol})")
            if verbose:
                print(f"  [{ativo}] Não encontrado na exchange ({exchange_symbol} {side})")
            continue

        # Build update dict with all available exchange data
        new_qty = exchange_pos["quantity"]
        updates = {"quantidade": new_qty}

        entry_price = exchange_pos.get("entry_price")
        if entry_price:
            updates["preco_entrada_exchange"] = entry_price

        unrealized_pnl = exchange_pos.get("unrealized_pnl")
        if unrealized_pnl is not None:
            updates["pnl_exchange"] = unrealized_pnl

        leverage = exchange_pos.get("leverage")
        if leverage and leverage != 1:
            updates["leverage"] = leverage

        # Always update: pnl_exchange changes even if quantity is the same
        current_qty = qty_map.get(posicao_id)
        repo.atualizar_atributos_posicao(posicao_id, **updates)
        resultado["synced"] += 1
        if verbose:
            if current_qty and current_qty != new_qty:
                print(f"  [{ativo}] Quantidade: {current_qty} -> {new_qty}")
            else:
                print(f"  [{ativo}] Sincronizado (qty={new_qty})")

    return resultado


def test_bitget_connection(produto_nome: str) -> Tuple[bool, str]:
    """
    Tests Bitget API connection for a product.

    Args:
        produto_nome: Product name

    Returns:
        Tuple of (success, message)
    """
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
        return False, f"Exceção: {str(e)}"
