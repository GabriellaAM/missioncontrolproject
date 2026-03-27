from typing import List, Dict
from .base import BaseExchange

from services.bitget_service import (
    fetch_perpetual_positions,
    fetch_perpetual_history
)


class BitgetExchange(BaseExchange):

    def __init__(self, credentials):
        self.credentials = credentials

    def fetch_positions(self) -> List[Dict]:
        return fetch_perpetual_positions(self.credentials)

    def fetch_history(self) -> List[Dict]:
        return fetch_perpetual_history(self.credentials)