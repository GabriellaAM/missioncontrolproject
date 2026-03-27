from abc import ABC, abstractmethod
from typing import List, Dict


class BaseExchange(ABC):

    @abstractmethod
    def fetch_positions(self) -> List[Dict]:
        pass

    @abstractmethod
    def fetch_history(self) -> List[Dict]:
        pass