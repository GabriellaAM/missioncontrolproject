"""
Módulo de compatibilidade - redireciona para notificacao_service.py

O serviço de e-mail foi substituído por notificações via Telegram.
Este arquivo existe apenas para manter compatibilidade com imports antigos.
"""
from services.notificacao_service import notificar_stop_atingido

enviar_email_stop_atingido = notificar_stop_atingido
