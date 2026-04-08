import requests


def fetch_okx_tickers_spot() -> dict:
    url = "https://www.okx.com/api/v5/market/tickers"

    try:
        response = requests.get(url, params={"instType": "SPOT"}, timeout=10)

        if response.status_code != 200:
            return {}

        data = response.json().get("data", [])
        resultado = {}

        for item in data:
            inst_id = item.get("instId")  # BTC-USDT
            last = item.get("last")

            if not inst_id or not last:
                continue

            try:
                symbol = inst_id.replace("-", "")
                resultado[symbol] = float(last)
            except:
                continue

        return resultado

    except:
        return {}

def fetch_okx_ohlc(symbol: str, days: int = 30):
    import requests
    import pandas as pd

    url = "https://www.okx.com/api/v5/market/candles"

    params = {
        "instId": symbol,
        "bar": "1D",
        "limit": days
    }

    resp = requests.get(url, params=params, timeout=10)
    data = resp.json().get("data", [])

    if not data:
        return None

    df = pd.DataFrame(data, columns=[
        "ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"
    ])

    df["timestamp"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    df["close"] = df["close"].astype(float)

    return df


def fetch_okx_tickers_perpetuals() -> dict:
    url = "https://www.okx.com/api/v5/market/tickers"

    try:
        response = requests.get(url, params={"instType": "SWAP"}, timeout=10)

        if response.status_code != 200:
            return {}

        data = response.json().get("data", [])
        resultado = {}

        for item in data:
            inst_id = item.get("instId")
            last = item.get("last")

            if not inst_id or not last:
                continue

            try:
                symbol = inst_id.replace("-", "")
                resultado[symbol] = float(last)
            except:
                continue

        return resultado

    except:
        return {}