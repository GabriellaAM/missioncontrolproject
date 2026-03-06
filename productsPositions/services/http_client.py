"""
Cliente HTTP com retry e backoff exponencial para chamadas a APIs externas.
Usado por Bitget, CoinGecko e outros serviços que precisam de controle de rate limit.
"""
import time
import logging
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)


def request_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    max_retries: int = 5,
) -> requests.Response:
    """
    Executa uma requisição GET com retry em caso de 429 ou RequestException.
    Backoff exponencial: 2 ** attempt segundos entre tentativas.
    Não silencia erros: após max_retries falhas, lança a última exceção.

    Args:
        url: URL da requisição
        params: Parâmetros de query string
        headers: Headers HTTP
        timeout: Timeout em segundos
        max_retries: Número máximo de tentativas (incluindo a primeira)

    Returns:
        requests.Response da requisição bem-sucedida

    Raises:
        requests.RequestException: Se todas as tentativas falharem
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers or {},
                timeout=timeout,
            )
            if response.status_code == 429:
                last_exception = requests.RequestException(
                    f"Rate limit (429) em {url} (tentativa {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    sleep_sec = 2 ** attempt
                    logger.warning(
                        "[request_with_retry] 429 em %s, aguardando %s s antes de retry",
                        url, sleep_sec
                    )
                    time.sleep(sleep_sec)
                    continue
                response.raise_for_status()
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                sleep_sec = 2 ** attempt
                logger.warning(
                    "[request_with_retry] Falha em %s: %s; aguardando %s s antes de retry",
                    url, e, sleep_sec
                )
                time.sleep(sleep_sec)
            else:
                raise
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("request_with_retry: nenhuma tentativa executada")
