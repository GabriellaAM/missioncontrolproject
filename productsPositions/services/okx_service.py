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