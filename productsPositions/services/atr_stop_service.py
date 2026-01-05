"""
ATR Trailing Stop Service

Calculates and updates trailing stops based on ATR (Average True Range).
Replicates TradingView's ATR Trailing Stop indicator logic.
"""
import pandas as pd
from pathlib import Path
from datetime import date
from typing import Optional, Tuple


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

    df = pd.read_parquet(parquet_path)

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


def calcular_rma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates RMA (Wilder's smoothing) matching TradingView's ta.rma() exactly.

    TradingView's RMA:
    - First (period-1) bars: NA
    - Bar at index (period-1): SMA of first 'period' values
    - Subsequent bars: alpha * x + (1-alpha) * prev, where alpha = 1/period
    """
    alpha = 1.0 / period
    rma = pd.Series(index=series.index, dtype=float)

    # First valid RMA value is SMA of first 'period' values
    if len(series) >= period:
        rma.iloc[period - 1] = series.iloc[:period].mean()

        # Apply RMA formula for subsequent values
        for i in range(period, len(series)):
            rma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma.iloc[i - 1]

    return rma


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

    # Filter from ONE DAY BEFORE entry date (TradingView uses previous day's close/ATR for initial trail)
    data_entrada_dt = pd.to_datetime(data_entrada)
    data_inicio_dt = data_entrada_dt - pd.Timedelta(days=1)
    df_filtered = df[df['timestamp'] >= data_inicio_dt].reset_index(drop=True)

    if df_filtered.empty:
        return None, False

    is_long = (side == 'long')
    trail = None
    breached = False

    for i in range(len(df_filtered)):
        row = df_filtered.iloc[i]

        if pd.isna(row['atr']):
            continue

        close = row['close']

        if trail is None:
            # First bar (day before entry): set initial trail using this bar's close and ATR
            atr = row['atr']
            if is_long:
                trail = close - (multiplier * atr)
            else:
                trail = close + (multiplier * atr)
        else:
            # Subsequent bars (entry day onwards): use PREVIOUS bar's close and ATR
            prev_row = df_filtered.iloc[i - 1]
            prev_close = prev_row['close']
            prev_atr = prev_row['atr']

            if pd.isna(prev_atr):
                continue

            if is_long:
                candidate = prev_close - (multiplier * prev_atr)
                # Trail up only (never lower the stop for longs)
                trail = max(trail, candidate)
                # Check breach against CURRENT close
                if close <= trail:
                    breached = True
                    break
            else:
                candidate = prev_close + (multiplier * prev_atr)
                # Trail down only (never raise the stop for shorts)
                trail = min(trail, candidate)
                # Check breach against CURRENT close
                if close >= trail:
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
    atr_multiplier: Optional[float] = None
) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    High-level function to calculate stop for a position.

    Args:
        coingecko_id: Asset identifier
        side: 'long' or 'short'
        data_entrada: Entry date (YYYY-MM-DD)
        atr_period: ATR lookback period (default: 14)
        atr_multiplier: ATR multiplier (default: 3.0)

    Returns:
        Tuple of (stop_value, breached, error_message)
    """
    # Use defaults if not specified, ensure proper types
    period = int(atr_period) if atr_period is not None else DEFAULT_ATR_PERIOD
    mult = float(atr_multiplier) if atr_multiplier is not None else DEFAULT_ATR_MULTIPLIER

    # Load OHLC data
    df = ler_ohlc_parquet(coingecko_id)
    if df is None:
        return None, False, f"Dados não encontrados para {coingecko_id}"

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

        if not coingecko_id:
            resultado['errors'].append(f"{ativo}: sem coingecko_id")
            if verbose:
                print(f"  [{ativo}] Erro: sem coingecko_id")
            continue

        # Calculate new stop
        stop, breached, erro = calcular_stop_para_posicao(
            coingecko_id=coingecko_id,
            side=side,
            data_entrada=data_entrada,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier
        )

        if erro:
            resultado['errors'].append(f"{ativo}: {erro}")
            if verbose:
                print(f"  [{ativo}] Erro: {erro}")
            continue

        if breached:
            resultado['breached'] += 1
            if verbose:
                print(f"  [{ativo}] STOP BREACHED!")
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
                        print(f"  [{ativo}] Stop atualizado: {ultimo_stop:.4f} → {stop:.4f}")
                    else:
                        print(f"  [{ativo}] Stop criado: {stop:.4f}")

    return resultado
