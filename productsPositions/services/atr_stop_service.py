"""
ATR Trailing Stop Service

Calculates and updates trailing stops based on ATR (Average True Range).
Replicates TradingView's ATR Trailing Stop indicator logic.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, Tuple
from dotenv import load_dotenv
from services.notificacao_service import notificar_stop_atingido

# Load environment variables for API key (from project root)
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / '.env')

# Defaults matching TradingView indicator
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULTIPLIER = 3.0


def _get_project_root() -> Path:
    """Returns project root directory"""
    return Path(__file__).parent.parent.parent


def ler_ohlc_parquet(coingecko_id: str) -> Optional[pd.DataFrame]:
    """
    Reads OHLC data from parquet file for a given asset.

    Args:
        coingecko_id: CoinGecko asset identifier

    Returns:
        DataFrame with columns: timestamp, open, high, low, close
        Returns None if file not found
    """
    project_root = _get_project_root()
    parquet_path = project_root / "data_parquet" / "crypto_data" / "coingecko" / coingecko_id / "data.parquet"

    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path)
    except ImportError:
        # pyarrow nao disponivel na nuvem - usar API como fallback
        return None

    # Ensure we have required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        return None

    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df[required_cols]


def buscar_ohlc_api(coingecko_id: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetches OHLC data from CoinGecko API for recent days.

    Args:
        coingecko_id: CoinGecko asset identifier
        days: Number of days to fetch (default: 30)
              Note: CoinGecko API only accepts specific values: 1, 7, 14, 30, 90, 180, 365

    Returns:
        DataFrame with columns: timestamp, open, high, low, close
        Returns None if API call fails
    """
    api_key = os.getenv('GECKO_API_KEY')
    if not api_key:
        print(f"[OHLC] CoinGecko: GECKO_API_KEY não configurado", flush=True)
        return None

    try:
        from pycoingecko import CoinGeckoAPI
        cg = CoinGeckoAPI(api_key=api_key)

        # CoinGecko API only accepts specific day values
        # Map requested days to nearest valid value
        valid_days = [1, 7, 14, 30, 90, 180, 365]
        api_days = min([d for d in valid_days if d >= days], default=30)

        # Fetch OHLC data for the last N days
        data = cg.get_coin_ohlc_by_id(
            id=coingecko_id,
            vs_currency='usd',
            days=api_days
        )

        if not data:
            return None

        # Convert to DataFrame
        # API returns: [[timestamp_ms, open, high, low, close], ...]
        df = pd.DataFrame(data, columns=['timestamp_ms', 'open', 'high', 'low', 'close'])

        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

        # Keep only daily data (remove intraday duplicates by taking last of each day)
        df['date'] = df['timestamp'].dt.date
        df = df.groupby('date').last().reset_index()
        df['timestamp'] = pd.to_datetime(df['date'])

        # Select required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"[OHLC] CoinGecko falhou para {coingecko_id}: {e}", flush=True)
        return None


def buscar_ohlc_bitget(exchange_symbol: str, days: int = 90, product_type: str = "perpetuos") -> Optional[pd.DataFrame]:
    """
    Fetches OHLC data from Bitget API for perpetual futures or spot.
    This provides the same data source as TradingView when using Bitget charts.

    Uses the history-candles endpoint for historical data, plus the candles endpoint
    for the current (incomplete) candle, matching TradingView's real-time behavior.

    Args:
        exchange_symbol: Bitget symbol (e.g., 'SCRTUSDT', 'BTCUSDT')
        days: Number of days to fetch (default: 90)
        product_type: 'perpetuos' for futures, 'spot' for spot market

    Returns:
        DataFrame with columns: timestamp, open, high, low, close
        Returns None if API call fails
    """
    import requests
    import time as time_module

    try:
        # Select endpoint based on product type
        if product_type == "spot":
            history_url = "https://api.bitget.com/api/v2/spot/market/history-candles"
            live_url = "https://api.bitget.com/api/v2/spot/market/candles"
        else:
            history_url = "https://api.bitget.com/api/v2/mix/market/history-candles"
            live_url = "https://api.bitget.com/api/v2/mix/market/candles"

        all_data = []

        # For spot, we need to use endTime and work backwards
        # For perpetuos, we use startTime/endTime range
        end_time = datetime.now()
        target_start = datetime.now() - timedelta(days=days)

        while end_time > target_start:
            if product_type == "spot":
                # Spot API uses endTime and limit (no startTime)
                params = {
                    "symbol": exchange_symbol,
                    "granularity": "1Dutc",  # UTC midnight candles
                    "endTime": str(int(end_time.timestamp() * 1000)),
                    "limit": "200"
                }
            else:
                # Perpetuos API uses startTime/endTime range
                start_time = end_time - timedelta(days=89)
                if start_time < target_start:
                    start_time = target_start

                params = {
                    "symbol": exchange_symbol,
                    "productType": "USDT-FUTURES",
                    "granularity": "1Dutc",  # UTC midnight candles (same as TradingView)
                    "startTime": str(int(start_time.timestamp() * 1000)),
                    "endTime": str(int(end_time.timestamp() * 1000)),
                    "limit": "200"
                }

            response = requests.get(history_url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != "00000":
                break

            data = result.get("data", [])
            if not data:
                break

            all_data.extend(data)

            if product_type == "spot":
                # For spot, get the oldest timestamp and use it as next endTime
                oldest_ts = min(int(d[0]) for d in data)
                end_time = datetime.fromtimestamp(oldest_ts / 1000) - timedelta(days=1)
            else:
                # For perpetuos, move the window back
                end_time = start_time - timedelta(days=1)

            time_module.sleep(0.1)  # Rate limit

        if not all_data:
            return None

        # Bitget returns: [timestamp, open, high, low, close, volume, quoteVolume, ...]
        # Spot has 8 columns, Perpetuos has 7 - handle both
        if len(all_data[0]) >= 7:
            df = pd.DataFrame(all_data)
            df.columns = ['timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'] + \
                         [f'col{i}' for i in range(7, len(df.columns))]

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'].astype(int), unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp'])
        df = df[['timestamp', 'open', 'high', 'low', 'close']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Fetch current (incomplete) candle using the live candles endpoint
        # This matches TradingView's real-time behavior
        try:
            if product_type == "spot":
                live_params = {
                    "symbol": exchange_symbol,
                    "granularity": "1Dutc",
                    "limit": "1"
                }
            else:
                live_params = {
                    "symbol": exchange_symbol,
                    "productType": "USDT-FUTURES",
                    "granularity": "1Dutc",
                    "limit": "1"
                }

            live_response = requests.get(live_url, params=live_params, timeout=10)
            live_response.raise_for_status()
            live_result = live_response.json()

            if live_result.get("code") == "00000" and live_result.get("data"):
                live_candle = live_result["data"][0]
                live_ts = pd.to_datetime(int(live_candle[0]), unit='ms')

                # Only append if this candle is newer than the last history candle
                if live_ts > df['timestamp'].max():
                    live_row = pd.DataFrame({
                        'timestamp': [live_ts],
                        'open': [float(live_candle[1])],
                        'high': [float(live_candle[2])],
                        'low': [float(live_candle[3])],
                        'close': [float(live_candle[4])]
                    })
                    df = pd.concat([df, live_row], ignore_index=True)
        except Exception:
            pass  # If live candle fetch fails, continue with history data only

        return df

    except Exception as e:
        print(f"[OHLC] Bitget falhou para {exchange_symbol}: {e}", flush=True)
        return None


def ler_ohlc_com_fallback(coingecko_id: str, data_entrada: str) -> Optional[pd.DataFrame]:
    """
    Reads OHLC data from Parquet, complementing with API if data is outdated.

    This is the main function to get OHLC data - it combines:
    1. Historical data from Parquet files
    2. Recent data from CoinGecko API (when Parquet is outdated)

    Args:
        coingecko_id: CoinGecko asset identifier
        data_entrada: Entry date (YYYY-MM-DD) - used to check if we need recent data

    Returns:
        DataFrame with columns: timestamp, open, high, low, close
        Returns None if no data available
    """
    data_entrada_dt = pd.to_datetime(data_entrada)
    ontem = datetime.now() - timedelta(days=1)

    # Try to load from Parquet first
    df_parquet = ler_ohlc_parquet(coingecko_id)

    if df_parquet is None or df_parquet.empty:
        # No Parquet data, try API only
        df_api = buscar_ohlc_api(coingecko_id, days=60)
        return df_api

    # Check if Parquet has data up to at least yesterday
    ultima_data_parquet = df_parquet['timestamp'].max()

    # If Parquet is up to date (has data >= data_entrada), use it directly
    if ultima_data_parquet >= data_entrada_dt:
        return df_parquet

    # Parquet is outdated, need to fetch recent data from API
    dias_faltantes = (datetime.now() - ultima_data_parquet).days + 5  # +5 for safety margin
    dias_faltantes = min(dias_faltantes, 90)  # Cap at 90 days

    df_api = buscar_ohlc_api(coingecko_id, days=dias_faltantes)

    if df_api is None or df_api.empty:
        # API failed, return Parquet data anyway (better than nothing)
        return df_parquet

    # Merge Parquet + API data
    # Keep Parquet data for historical, API for recent
    df_combined = pd.concat([df_parquet, df_api], ignore_index=True)

    # Remove duplicates (prefer API data for overlapping dates as it's more recent)
    df_combined['date'] = df_combined['timestamp'].dt.date
    df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')
    df_combined = df_combined.drop(columns=['date'])

    # Sort and reset index
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    return df_combined


def calcular_rma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates RMA (Wilder's smoothing) matching TradingView's ta.rma() exactly.

    TradingView's RMA:
    - First (period-1) bars: NA
    - Bar at index (period-1): SMA of first 'period' values
    - Subsequent bars: alpha * x + (1-alpha) * prev, where alpha = 1/period

    OPTIMIZED: Uses numpy arrays for ~80x faster calculation.
    """
    n = len(series)
    alpha = 1.0 / period
    values = series.values
    result = np.full(n, np.nan)

    if n >= period:
        # SMA seed at index period-1
        result[period - 1] = np.mean(values[:period])
        # RMA formula for subsequent values
        for i in range(period, n):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def calcular_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """
    Calculates Average True Range (ATR) matching TradingView's ta.atr() exactly.

    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = RMA of True Range (Wilder's smoothing with SMA seed)

    Args:
        df: DataFrame with high, low, close columns
        period: Lookback period for ATR

    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate True Range components
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    # True Range is max of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is RMA of True Range (matching TradingView's ta.atr)
    atr = calcular_rma(true_range, period)

    return atr


def calcular_trailing_stop(
    df: pd.DataFrame,
    side: str,
    data_entrada: str,
    atr_period: int = DEFAULT_ATR_PERIOD,
    multiplier: float = DEFAULT_ATR_MULTIPLIER
) -> Tuple[Optional[float], bool]:
    """
    Calculates ATR Trailing Stop following TradingView logic exactly.

    For longs: stop starts at close - (multiplier * ATR), trails up only
    For shorts: stop starts at close + (multiplier * ATR), trails down only

    IMPORTANT: After entry bar, uses PREVIOUS bar's close and ATR to calculate
    the candidate stop (matching TradingView's close[1] and atr[1] syntax).

    Args:
        df: DataFrame with OHLC data (must have timestamp, open, high, low, close)
        side: Position side ('long' or 'short')
        data_entrada: Entry date (YYYY-MM-DD format)
        atr_period: Period for ATR calculation
        multiplier: ATR multiplier for stop distance

    Returns:
        Tuple of (stop_value, breached)
        - stop_value: Current trailing stop value (None if breached)
        - breached: True if stop was breached
    """
    # Calculate ATR on FULL history first (matching TradingView behavior)
    df = df.copy()
    df['atr'] = calcular_atr(df, atr_period)

    # Filter from entry date (entry_bar is first bar when time >= start_time)
    data_entrada_dt = pd.to_datetime(data_entrada)
    df_filtered = df[df['timestamp'] >= data_entrada_dt].reset_index(drop=True)

    if df_filtered.empty:
        return None, False

    is_long = (side == 'long')
    trail = None
    breached = False

    hoje = pd.Timestamp.now().normalize()

    for i in range(len(df_filtered)):
        row = df_filtered.iloc[i]

        if pd.isna(row['atr']):
            continue

        close = row['close']

        if i == 0:
            # Entry bar: trail = close - mu * atr (current bar's values)
            atr = row['atr']
            if is_long:
                trail = close - (multiplier * atr)
            else:
                trail = close + (multiplier * atr)
        else:
            # Subsequent bars: cand = close[1] - mu * atr[1] (previous bar's values)
            prev_row = df_filtered.iloc[i - 1]
            prev_close = prev_row['close']
            prev_atr = prev_row['atr']

            if pd.isna(prev_atr):
                continue

            if is_long:
                candidate = prev_close - (multiplier * prev_atr)
                trail = max(trail, candidate)
                # Breach only counts on fully closed candles (not today's incomplete candle)
                candle_date = pd.Timestamp(row['timestamp']).normalize()
                if candle_date < hoje and close <= trail:
                    breached = True
                    break
            else:
                candidate = prev_close + (multiplier * prev_atr)
                trail = min(trail, candidate)
                candle_date = pd.Timestamp(row['timestamp']).normalize()
                if candle_date < hoje and close >= trail:
                    breached = True
                    break

    if breached:
        return None, True

    return trail, False


def calcular_stop_para_posicao(
    coingecko_id: str,
    side: str,
    data_entrada: str,
    atr_period: Optional[int] = None,
    atr_multiplier: Optional[float] = None,
    exchange_symbol: Optional[str] = None,
    product_type: str = "perpetuos"
) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    High-level function to calculate stop for a position.

    Args:
        coingecko_id: Asset identifier (used as fallback if exchange_symbol not provided)
        side: 'long' or 'short'
        data_entrada: Entry date (YYYY-MM-DD)
        atr_period: ATR lookback period (default: 14)
        atr_multiplier: ATR multiplier (default: 3.0)
        exchange_symbol: Bitget exchange symbol (e.g., 'SCRTUSDT') - if provided, uses Bitget data
        product_type: 'perpetuos' or 'spot' - determines which Bitget API to use

    Returns:
        Tuple of (stop_value, breached, error_message)
    """
    # Use defaults if not specified, ensure proper types
    period = int(atr_period) if atr_period is not None else DEFAULT_ATR_PERIOD
    mult = float(atr_multiplier) if atr_multiplier is not None else DEFAULT_ATR_MULTIPLIER

    # Load OHLC data - prefer Bitget if exchange_symbol is provided
    df = None
    if exchange_symbol:
        df = buscar_ohlc_bitget(exchange_symbol, days=180, product_type=product_type)

    # Fallback to CoinGecko if Bitget fails or not configured
    if (df is None or df.empty) and coingecko_id:
        df = ler_ohlc_com_fallback(coingecko_id, data_entrada)

    if df is None or df.empty:
        return None, False, f"Dados OHLC não encontrados para {exchange_symbol or coingecko_id}"

    # Calculate stop
    stop, breached = calcular_trailing_stop(
        df=df,
        side=side,
        data_entrada=data_entrada,
        atr_period=period,
        multiplier=mult
    )

    return stop, breached, None


def atualizar_stops_posicoes_abertas(repo, produto_id: Optional[int] = None, verbose: bool = True):
    """
    Updates ATR trailing stops for all open positions that have atr_multiplier configured.

    Args:
        repo: SQLiteRepo instance
        produto_id: Optional - filter by product ID
        verbose: Print progress info

    Returns:
        dict with summary: {updated: int, skipped: int, unchanged: int, breached: int, errors: list}
    """
    resultado = {
        'updated': 0,
        'skipped': 0,
        'unchanged': 0,
        'breached': 0,
        'errors': []
    }

    # Load open positions
    df_posicoes = repo.carregar_posicoes_abertas(produto_id)

    if df_posicoes.empty:
        if verbose:
            print("Nenhuma posição aberta encontrada.")
        return resultado

    hoje = date.today().strftime('%Y-%m-%d')

    for _, pos in df_posicoes.iterrows():
        posicao_id = pos['id']
        ativo = pos['ativo']
        coingecko_id = pos.get('coingecko_id')
        side = pos['side']
        data_entrada = pos['data_entrada']
        atr_data_inicio = pos.get('atr_data_inicio')

        # Get ATR config directly from position (now in posicoes table)
        atr_multiplier = pos.get('atr_multiplier')
        atr_period = pos.get('atr_period')

        # Convert to proper types (may come as float from DataFrame)
        if atr_period is not None and not pd.isna(atr_period):
            atr_period = int(atr_period)
        if atr_multiplier is not None and not pd.isna(atr_multiplier):
            atr_multiplier = float(atr_multiplier)

        # Skip if no ATR multiplier configured (manual stop management)
        if pd.isna(atr_multiplier) or atr_multiplier is None:
            resultado['skipped'] += 1
            if verbose:
                print(f"  [{ativo}] Sem atr_multiplier - pulando")
            continue

        # Get exchange_symbol for Bitget data (antes da validação de coingecko_id)
        exchange_symbol = pos.get('exchange_symbol')
        if exchange_symbol and pd.isna(exchange_symbol):
            exchange_symbol = None

        if not coingecko_id and not exchange_symbol:
            resultado['errors'].append(f"{ativo}: sem coingecko_id e sem exchange_symbol")
            print(f"  [{ativo}] Erro ATR: sem coingecko_id e sem exchange_symbol", flush=True)
            continue

        # Usar atr_data_inicio se disponível, senão data_entrada
        data_calculo = atr_data_inicio if (atr_data_inicio and not pd.isna(atr_data_inicio)) else data_entrada

        # Detect product type from produto_tipo field
        produto_tipo = pos.get('produto_tipo', '')
        if produto_tipo and not pd.isna(produto_tipo):
            produto_tipo_lower = str(produto_tipo).lower()
            if 'spot' in produto_tipo_lower:
                product_type = 'spot'
            else:
                product_type = 'perpetuos'
        else:
            product_type = 'perpetuos'  # Default

        # Calculate new stop
        stop, breached, erro = calcular_stop_para_posicao(
            coingecko_id=coingecko_id,
            side=side,
            data_entrada=data_calculo,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            exchange_symbol=exchange_symbol,
            product_type=product_type
        )

        if erro:
            resultado['errors'].append(f"{ativo}: {erro}")
            print(f"  [{ativo}] Erro ATR: {erro}", flush=True)
            continue

        if breached:
            resultado['breached'] += 1

            # Verificar se já foi notificado (último stop = -1 indica breach já registrado)
            ultimo_stop = repo.obter_ultimo_stop(posicao_id)
            ja_notificado = (ultimo_stop is not None and float(ultimo_stop) == -1.0)

            if not ja_notificado:
                # Salvar -1 para indicar que stop foi atingido (marca como notificado)
                repo.adicionar_stop_posicao(posicao_id, hoje, -1)
                if verbose:
                    print(f"  [{ativo}] STOP ATINGIDO! Enviando notificação...")
                # Enviar notificação via Telegram
                try:
                    produto_id_pos = pos.get('produto_id')
                    produto_info = repo.carregar_produto(produto_id_pos) if produto_id_pos else None
                    nome_produto = produto_info['nome'] if produto_info else "Desconhecido"
                    notificar_stop_atingido(
                        ativo=ativo,
                        side=side,
                        preco_entrada=pos.get('preco_entrada'),
                        produto_nome=nome_produto,
                        data_entrada=data_entrada
                    )
                except Exception as e:
                    if verbose:
                        print(f"  [{ativo}] Erro ao enviar notificação: {e}")
            else:
                if verbose:
                    print(f"  [{ativo}] STOP ATINGIDO (já notificado anteriormente - pulando notificação)")
            continue

        if stop is not None:
            # Check if stop changed from last saved value
            ultimo_stop = repo.obter_ultimo_stop(posicao_id)

            # Compare with tolerance (avoid floating point issues)
            if ultimo_stop is not None and abs(stop - ultimo_stop) < 0.01:
                resultado['unchanged'] += 1
                if verbose:
                    print(f"  [{ativo}] Stop inalterado: {stop:.4f}")
            else:
                # Add new stop entry only if different
                repo.adicionar_stop_posicao(posicao_id, hoje, stop)
                resultado['updated'] += 1
                if verbose:
                    if ultimo_stop is not None:
                        print(f"  [{ativo}] Stop atualizado: {ultimo_stop:.4f} -> {stop:.4f}")
                    else:
                        print(f"  [{ativo}] Stop criado: {stop:.4f}")

    return resultado
