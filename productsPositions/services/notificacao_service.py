"""
Serviço de notificação para stops atingidos via Telegram.

Configurações lidas do .env:
  - TELEGRAM_BOT_TOKEN: token do bot criado via @BotFather
  - TELEGRAM_CHAT_ID: ID do grupo/chat onde enviar as mensagens
"""
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Carregar .env do root do projeto
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / '.env')


def _montar_mensagem_telegram(ativo, side, preco_entrada, produto_nome, data_entrada=None):
    """Monta mensagem HTML para Telegram."""
    side_display = side.upper() if side else "—"
    preco_display = f"${preco_entrada:,.2f}" if preco_entrada is not None else "—"

    linhas = [
        "\U0001f6a8 <b>STOP ATINGIDO</b>",
        "",
        f"<b>Ativo:</b> {ativo}",
        f"<b>Side:</b> {side_display}",
        f"<b>Preço de Entrada:</b> {preco_display}",
        f"<b>Produto:</b> {produto_nome}",
    ]
    if data_entrada:
        linhas.append(f"<b>Data de Entrada:</b> {data_entrada}")

    return "\n".join(linhas)


def notificar_stop_atingido(ativo: str, side: str, preco_entrada: float, produto_nome: str, data_entrada: str = None):
    """
    Envia notificação de stop atingido via Telegram.

    Args:
        ativo: Nome do ativo (ex: "BTC")
        side: Direção da posição ("long" ou "short")
        preco_entrada: Preço de entrada da posição
        produto_nome: Nome do produto ao qual a posição pertence
        data_entrada: Data de entrada da posição (opcional)

    Returns:
        True se enviou com sucesso
    """
    token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '').strip()

    if not token or not chat_id:
        print("[Telegram] TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não configurado no .env.")
        return False

    mensagem = _montar_mensagem_telegram(ativo, side, preco_entrada, produto_nome, data_entrada)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": mensagem,
        "parse_mode": "HTML",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200 and resp.json().get("ok"):
            print("[Telegram] Notificação de stop atingido enviada com sucesso.")
            return True
        else:
            print(f"[Telegram] Erro: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"[Telegram] Erro ao enviar: {e}")
        return False


# Compatibilidade com imports antigos
enviar_email_stop_atingido = notificar_stop_atingido
