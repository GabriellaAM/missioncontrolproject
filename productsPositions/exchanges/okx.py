import hmac
import base64
import hashlib
from datetime import datetime, timezone
from typing import List, Dict
import requests

from .base import BaseExchange


class OKXExchange(BaseExchange):

    BASE_URL = "https://www.okx.com"

    def __init__(self, credentials):
        self.credentials = credentials
        self.session = requests.Session()

    def _timestamp(self):
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _sign(self, timestamp, method, path, body=""):
        message = f"{timestamp}{method.upper()}{path}{body}"
        mac = hmac.new(
            self.credentials["secret_key"].encode(),
            message.encode(),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()

    def _request(self, method, path, body=""):
        ts = self._timestamp()
        sign = self._sign(ts, method, path, body)

        headers = {
            "OK-ACCESS-KEY": self.credentials["api_key"],
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.credentials["passphrase"],
            "Content-Type": "application/json"
        }

        if method == "GET":
            r = self.session.get(self.BASE_URL + path, headers=headers, timeout=10)
        else:
            r = self.session.post(self.BASE_URL + path, headers=headers, data=body, timeout=10)

        return r.json()

    # 🔥 NORMALIZAÇÃO PADRÃO (igual Bitget)
    def _parse_position(self, pos):
        qty = float(pos.get("pos", 0))

        if pos.get("posSide") == "net":
            side = "long" if qty > 0 else "short"
            qty = abs(qty)
        else:
            side = pos.get("posSide")

        symbol = pos.get("instId", "")
        exchange_symbol = symbol.replace("-", "")

        return {
            "symbol": exchange_symbol.replace("USDT", ""),
            "exchange_symbol": exchange_symbol,
            "side": side,
            "quantity": qty,
            "entry_price": float(pos.get("avgPx", 0) or 0),
            "unrealized_pnl": float(pos.get("upl", 0) or 0),
            "mark_price": float(pos.get("markPx", 0) or 0),
            "leverage": int(float(pos.get("lever", 1) or 1)),
            "ctime": pos.get("cTime"),
        }

    def fetch_positions(self) -> List[Dict]:
        res = self._request("GET", "/api/v5/account/positions")

        if res.get("code") != "0":
            raise Exception(res.get("msg"))

        positions = []
        for pos in res.get("data", []):
            parsed = self._parse_position(pos)
            if parsed["quantity"] > 0:
                positions.append(parsed)

        return positions

    def fetch_history(self) -> List[Dict]:
        res = self._request("GET", "/api/v5/account/positions-history")

        if res.get("code") != "0":
            raise Exception(res.get("msg"))

        history = []

        for pos in res.get("data", []):
            history.append({
                "symbol": pos.get("instId").replace("-", "").replace("USDT", ""),
                "exchange_symbol": pos.get("instId").replace("-", ""),
                "side": pos.get("posSide"),
                "open_price": float(pos.get("openAvgPx", 0)),
                "close_price": float(pos.get("closeAvgPx", 0)),
                "pnl": float(pos.get("pnl", 0)),
                "quantity": float(pos.get("closeTotalPos", 0)),
                "open_time": pos.get("cTime"),
                "close_time": pos.get("uTime"),
            })

        return history