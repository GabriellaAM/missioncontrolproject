import os
from typing import Optional, Dict
from services.bitget_service import _normalize_product_name


def _get_env(prefix: str, suffix: str) -> Optional[Dict[str, str]]:
    """Helper para buscar credenciais no .env com debug detalhado"""

    api_key_var = f"{prefix}_API_KEY_{suffix}"
    secret_key_var = f"{prefix}_SECRET_KEY_{suffix}"
    passphrase_var = f"{prefix}_PASSPHRASE_{suffix}"

    api_key = os.getenv(api_key_var)
    secret_key = os.getenv(secret_key_var)
    passphrase = os.getenv(passphrase_var)

    # 🔎 DEBUG FORTE (vai aparecer no log)
    print(f"[CREDENTIALS] Buscando credenciais:")
    print(f"  - {api_key_var} -> {'OK' if api_key else 'MISSING'}")
    print(f"  - {secret_key_var} -> {'OK' if secret_key else 'MISSING'}")
    print(f"  - {passphrase_var} -> {'OK' if passphrase else 'MISSING'}")

    # 🚨 Validação explícita (evita falha silenciosa)
    if not api_key:
        print(f"[CREDENTIALS][ERRO] API KEY não encontrada: {api_key_var}")
        return None

    if not secret_key:
        print(f"[CREDENTIALS][ERRO] SECRET KEY não encontrada: {secret_key_var}")
        return None

    # ✅ OK (passphrase pode ser None dependendo da exchange)
    return {
        "api_key": api_key,
        "secret_key": secret_key,
        "passphrase": passphrase
    }

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