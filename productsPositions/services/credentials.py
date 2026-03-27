import os
from typing import Optional, Dict
from services.bitget_service import _normalize_product_name


def _get_env(prefix: str, suffix: str) -> Optional[Dict[str, str]]:
    """Helper para buscar credenciais no .env"""
    api_key = os.getenv(f"{prefix}_API_KEY_{suffix}")
    secret_key = os.getenv(f"{prefix}_SECRET_KEY_{suffix}")
    passphrase = os.getenv(f"{prefix}_PASSPHRASE_{suffix}")

    if api_key and secret_key:
        return {
            "api_key": api_key,
            "secret_key": secret_key,
            "passphrase": passphrase  # pode ser None (ex: OKX mal configurado)
        }

    return None


def get_exchange_credentials(produto_nome: str) -> Optional[Dict[str, str]]:
    """
    Detecta automaticamente a exchange e retorna credenciais.

    Suporta:
    - Bitget
    - OKX
    """

    normalized = _normalize_product_name(produto_nome)

    # ===== OKX =====
    if "OKX" in produto_nome.upper():
        creds = _get_env("OKX", normalized)
        if creds:
            creds["exchange"] = "okx"
            return creds

    # ===== BITGET =====
    creds = _get_env("BITGET", normalized)
    if creds:
        creds["exchange"] = "bitget"
        return creds

    return None