"""
Serviço de Cotações para atualização de valores diários.

Este módulo busca cotações de:
- Binance (via API pública)
- CoinGecko (via API com key opcional)
- Parquet files locais (fallback)

E atualiza a tabela trade_valores_diarios.
"""

import psycopg2
import psycopg2.extras
import requests
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import os

# Carregar .env da raiz do projeto
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).parent.parent.parent
    load_dotenv(_project_root / '.env')
except ImportError:
    pass  # python-dotenv não instalado


class CotacoesService:
    """Serviço para buscar e atualizar cotações de ativos."""

    BINANCE_API_URL = "https://api.binance.com/api/v3"
    COINGECKO_API_URL = "https://pro-api.coingecko.com/api/v3"

    # Cache module-level de preços atuais com TTL (compartilhado entre instâncias)
    _precos_atuais_cache: Dict[str, float] = {}
    _precos_atuais_cache_ts: float = 0.0
    _precos_atuais_cache_ttl: float = 300.0  # 5 minutos

    def __init__(self, db_path: Optional[Path] = None, gecko_api_key: Optional[str] = None):
        # db_path mantido para compatibilidade de assinatura, mas não usado
        self.db_url = os.getenv('SUPABASE_DB_URL')
        if not self.db_url:
            raise ValueError("SUPABASE_DB_URL environment variable is required")
        self.gecko_api_key = gecko_api_key or os.environ.get('GECKO_API_KEY')

        # Cache de mapeamento ativo -> exchange_symbol
        self._symbol_cache: Dict[str, str] = {}
        # Cache de preço histórico (coingecko_id, data) -> resultado (reduz chamadas CoinGecko)
        self._preco_historico_cache: Dict[tuple, Dict] = {}
        self._preco_historico_cache_max = 500

    @classmethod
    def _cache_valido(cls) -> bool:
        """Verifica se o cache de preços atuais ainda é válido."""
        return (time.time() - cls._precos_atuais_cache_ts) < cls._precos_atuais_cache_ttl

    @classmethod
    def _atualizar_cache(cls, precos: Dict[str, float]):
        """Atualiza o cache module-level com novos preços."""
        cls._precos_atuais_cache.update(precos)
        cls._precos_atuais_cache_ts = time.time()

    def _get_connection(self):
        return psycopg2.connect(self.db_url)

    # ================================================================
    # BINANCE
    # ================================================================

    def obter_preco_binance(self, symbol: str) -> Optional[float]:
        """
        Obtém preço atual de um símbolo na Binance.

        Args:
            symbol: Símbolo do par (ex: BTCUSDT, ETHUSDT)

        Returns:
            Preço atual ou None se erro
        """
        try:
            response = requests.get(
                f"{self.BINANCE_API_URL}/ticker/price",
                params={"symbol": symbol},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except Exception as e:
            print(f"Erro ao buscar preço Binance para {symbol}: {e}")
        return None

    def obter_klines_binance(self, symbol: str, interval: str = "1d",
                             limit: int = 100) -> List[Dict]:
        """
        Obtém candles históricos da Binance.

        Args:
            symbol: Símbolo do par (ex: BTCUSDT)
            interval: Intervalo (1d, 4h, 1h, etc)
            limit: Número de candles

        Returns:
            Lista de dicts com open_time, close_price, etc
        """
        try:
            response = requests.get(
                f"{self.BINANCE_API_URL}/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                klines = []
                for k in data:
                    klines.append({
                        'open_time': datetime.fromtimestamp(k[0] / 1000).strftime("%Y-%m-%d"),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    })
                return klines
        except Exception as e:
            print(f"Erro ao buscar klines Binance para {symbol}: {e}")
        return []

    # ================================================================
    # COINGECKO
    # ================================================================

    def obter_preco_coingecko(self, coingecko_id: str) -> Optional[float]:
        """
        Obtém preço atual de um ativo no CoinGecko.

        Args:
            coingecko_id: ID do CoinGecko (ex: bitcoin, ethereum)

        Returns:
            Preço em USD ou None se erro
        """
        preco, _ = self.obter_preco_coingecko_com_status(coingecko_id)
        return preco

    def obter_precos_batch_coingecko(self, coingecko_ids: List[str]) -> Dict[str, float]:
        """
        Busca preços atuais de múltiplos ativos em uma única chamada CoinGecko.
        Usa cache module-level com TTL de 5 minutos para evitar chamadas repetidas.

        Args:
            coingecko_ids: Lista de IDs CoinGecko (ex: ['bitcoin', 'ethereum'])

        Returns:
            Dict mapeando coingecko_id -> preço USD. IDs não encontrados são omitidos.
        """
        if not coingecko_ids:
            return {}

        ids_unicos = list(set(coingecko_ids))

        # Verificar cache: se válido, retornar apenas os IDs solicitados do cache
        if self._cache_valido():
            ids_faltantes = [cid for cid in ids_unicos if cid not in self._precos_atuais_cache]
            if not ids_faltantes:
                return {cid: self._precos_atuais_cache[cid] for cid in ids_unicos if cid in self._precos_atuais_cache}
            # Buscar apenas os faltantes
            ids_unicos = ids_faltantes

        try:
            headers = {}
            if self.gecko_api_key:
                headers['x-cg-pro-api-key'] = self.gecko_api_key

            response = requests.get(
                f"{self.COINGECKO_API_URL}/simple/price",
                params={
                    "ids": ",".join(ids_unicos),
                    "vs_currencies": "usd"
                },
                headers=headers,
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                novos_precos = {
                    cid: float(data[cid]['usd'])
                    for cid in ids_unicos
                    if cid in data and 'usd' in data[cid]
                }
                # Atualizar cache module-level
                self._atualizar_cache(novos_precos)
                # Retornar todos os IDs solicitados (incluindo os que já estavam em cache)
                todos_ids = list(set(coingecko_ids))
                return {cid: self._precos_atuais_cache[cid] for cid in todos_ids if cid in self._precos_atuais_cache}
            return {}
        except requests.exceptions.Timeout:
            print(f"Timeout ao buscar preços batch CoinGecko para {len(ids_unicos)} ativos")
            # Retornar o que tiver em cache mesmo expirado
            return {cid: self._precos_atuais_cache[cid] for cid in set(coingecko_ids) if cid in self._precos_atuais_cache}
        except Exception as e:
            print(f"Erro ao buscar preços batch CoinGecko: {e}")
            return {cid: self._precos_atuais_cache[cid] for cid in set(coingecko_ids) if cid in self._precos_atuais_cache}

    def obter_historicos_batch(self, coingecko_ids: List[str],
                              dias: int = 365) -> Dict[str, Dict[str, float]]:
        """
        Busca histórico de preços de múltiplos ativos via CoinGecko.

        Faz uma chamada market_chart por ativo e monta um dicionário
        indexado por coingecko_id e data para consulta O(1).

        Args:
            coingecko_ids: Lista de IDs CoinGecko
            dias: Número de dias de histórico (max 365)

        Returns:
            Dict[coingecko_id, Dict[data_str, preco]]
            Ex: {'bitcoin': {'2026-01-15': 100000.0, '2026-01-16': 101000.0}}
        """
        resultado = {}
        ids_unicos = list(set(coingecko_ids))

        for cg_id in ids_unicos:
            historico = self.obter_historico_coingecko(cg_id, dias=dias)
            if historico:
                # Montar dict data->preco. Se houver duplicatas (mesmo dia),
                # a última entrada vence (geralmente o preço de fechamento)
                precos_por_data = {}
                for item in historico:
                    precos_por_data[item['data']] = item['preco']
                resultado[cg_id] = precos_por_data

        return resultado

    def obter_preco_coingecko_com_status(self, coingecko_id: str) -> tuple:
        """
        Obtém preço atual de um ativo no CoinGecko com informação de status.
        Verifica o cache module-level antes de fazer chamada HTTP.

        Args:
            coingecko_id: ID do CoinGecko (ex: bitcoin, ethereum)

        Returns:
            Tuple (preco ou None, status: 'ok', 'timeout', 'erro')
        """
        # Verificar cache module-level primeiro
        if self._cache_valido() and coingecko_id in self._precos_atuais_cache:
            return self._precos_atuais_cache[coingecko_id], 'ok'

        try:
            headers = {}
            if self.gecko_api_key:
                headers['x-cg-pro-api-key'] = self.gecko_api_key

            response = requests.get(
                f"{self.COINGECKO_API_URL}/simple/price",
                params={
                    "ids": coingecko_id,
                    "vs_currencies": "usd"
                },
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if coingecko_id in data and 'usd' in data[coingecko_id]:
                    preco = float(data[coingecko_id]['usd'])
                    # Atualizar cache
                    self._atualizar_cache({coingecko_id: preco})
                    return preco, 'ok'
            return None, 'erro'
        except requests.exceptions.Timeout:
            print(f"Timeout ao buscar preço CoinGecko para {coingecko_id}")
            # Retornar cache expirado se disponível
            if coingecko_id in self._precos_atuais_cache:
                return self._precos_atuais_cache[coingecko_id], 'timeout'
            return None, 'timeout'
        except requests.exceptions.RequestException as e:
            print(f"Erro de conexão ao buscar preço CoinGecko para {coingecko_id}: {e}")
            if coingecko_id in self._precos_atuais_cache:
                return self._precos_atuais_cache[coingecko_id], 'erro'
            return None, 'erro'
        except Exception as e:
            print(f"Erro ao buscar preço CoinGecko para {coingecko_id}: {e}")
            return None, 'erro'

    def obter_historico_coingecko(self, coingecko_id: str, dias: int = 90) -> List[Dict]:
        """
        Obtém histórico de preços do CoinGecko.

        Args:
            coingecko_id: ID do CoinGecko
            dias: Número de dias de histórico

        Returns:
            Lista de dicts com data e preco
        """
        try:
            headers = {}
            if self.gecko_api_key:
                headers['x-cg-pro-api-key'] = self.gecko_api_key

            response = requests.get(
                f"{self.COINGECKO_API_URL}/coins/{coingecko_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": dias
                },
                headers=headers,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                precos = []
                for item in data.get('prices', []):
                    timestamp_ms, preco = item
                    data_str = datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")
                    precos.append({
                        'data': data_str,
                        'preco': preco
                    })
                return precos
        except Exception as e:
            print(f"Erro ao buscar histórico CoinGecko para {coingecko_id}: {e}")
        return []

    def obter_preco_historico_exato(self, coingecko_id: str, data: str) -> Dict:
        """
        Obtém preço histórico exato de uma data específica via CoinGecko.

        Usa o endpoint /coins/{id}/history para obter o preço exato em uma data.
        Se falhar, faz fallback para /coins/{id}/market_chart e busca a data mais próxima.

        Args:
            coingecko_id: ID do CoinGecko (ex: 'bitcoin', 'ethereum')
            data: Data no formato YYYY-MM-DD

        Returns:
            Dict com:
                - preco: float (preço encontrado)
                - data_referencia: str (data efetiva do preço, pode diferir se usou fallback)
                - fonte: str ('coingecko')
                - moeda: str ('USD')
                - status: str ('ok', 'fallback', 'erro')
                - aviso: str (opcional, presente se houve fallback)
                - erro: str (opcional, presente se status='erro')
        """
        # Converter formato da data: YYYY-MM-DD -> dd-mm-yyyy (formato CoinGecko)
        try:
            data_obj = datetime.strptime(data, "%Y-%m-%d")
            data_cg = data_obj.strftime("%d-%m-%Y")
        except ValueError:
            return {
                'status': 'erro',
                'erro': f'Formato de data inválido: {data}. Use YYYY-MM-DD.'
            }

        # Validar que não é data futura
        if data_obj.date() > date.today():
            return {
                'status': 'erro',
                'erro': f'Data futura não permitida: {data}'
            }

        # Cache: evita chamadas repetidas ao CoinGecko (ex.: várias posições mesma data/ativo)
        cache_key = (coingecko_id, data)
        if cache_key in self._preco_historico_cache:
            return dict(self._preco_historico_cache[cache_key])

        # Tentar endpoint /coins/{id}/history (preço exato na data)
        try:
            headers = {}
            if self.gecko_api_key:
                headers['x-cg-pro-api-key'] = self.gecko_api_key

            response = requests.get(
                f"{self.COINGECKO_API_URL}/coins/{coingecko_id}/history",
                params={"date": data_cg, "localization": "false"},
                headers=headers,
                timeout=15
            )

            if response.status_code == 200:
                data_json = response.json()
                market_data = data_json.get('market_data', {})
                current_price = market_data.get('current_price', {})

                if 'usd' in current_price and current_price['usd'] is not None:
                    resultado = {
                        'preco': float(current_price['usd']),
                        'data_referencia': data,
                        'fonte': 'coingecko',
                        'moeda': 'USD',
                        'status': 'ok'
                    }
                    self._preco_historico_cache_set(cache_key, resultado)
                    return resultado

            # Se /history não retornou preço, tentar fallback via market_chart
            return self._fallback_market_chart(coingecko_id, data, cache_key)

        except requests.exceptions.Timeout:
            print(f"Timeout ao buscar preço histórico CoinGecko para {coingecko_id} em {data}")
            return self._fallback_market_chart(coingecko_id, data, cache_key)
        except requests.exceptions.RequestException as e:
            print(f"Erro de conexão CoinGecko para {coingecko_id} em {data}: {e}")
            return self._fallback_market_chart(coingecko_id, data, cache_key)
        except Exception as e:
            print(f"Erro ao buscar preço histórico CoinGecko para {coingecko_id} em {data}: {e}")
            return {
                'status': 'erro',
                'erro': str(e)
            }

    def _preco_historico_cache_set(self, key: tuple, valor: Dict) -> None:
        """Guarda resultado no cache de preço histórico; limita tamanho do cache."""
        while len(self._preco_historico_cache) >= self._preco_historico_cache_max and self._preco_historico_cache:
            self._preco_historico_cache.pop(next(iter(self._preco_historico_cache)))
        self._preco_historico_cache[key] = dict(valor)

    def _fallback_market_chart(self, coingecko_id: str, data: str, cache_key: Optional[tuple] = None) -> Dict:
        """
        Fallback usando market_chart quando o endpoint history não está disponível.

        Busca o histórico de 365 dias e encontra o preço mais próximo da data solicitada.
        """
        try:
            historico = self.obter_historico_coingecko(coingecko_id, dias=365)

            if not historico:
                return {
                    'status': 'erro',
                    'erro': f'Histórico não disponível para {coingecko_id}'
                }

            # Ordenar por data decrescente para encontrar a mais próxima
            historico_sorted = sorted(historico, key=lambda x: x['data'], reverse=True)

            for item in historico_sorted:
                if item['data'] <= data:
                    # Se a data encontrada é diferente da solicitada, é um fallback
                    if item['data'] != data:
                        resultado = {
                            'preco': item['preco'],
                            'data_referencia': item['data'],
                            'fonte': 'coingecko',
                            'moeda': 'USD',
                            'status': 'fallback',
                            'aviso': f"Preço de {item['data']} usado ({data} não disponível)"
                        }
                    else:
                        resultado = {
                            'preco': item['preco'],
                            'data_referencia': item['data'],
                            'fonte': 'coingecko',
                            'moeda': 'USD',
                            'status': 'ok'
                        }
                    if cache_key is not None:
                        self._preco_historico_cache_set(cache_key, resultado)
                    return resultado

            return {
                'status': 'erro',
                'erro': f'Preço não encontrado para {coingecko_id} em {data} ou antes'
            }

        except Exception as e:
            return {
                'status': 'erro',
                'erro': f'Erro no fallback market_chart: {str(e)}'
            }

    # ================================================================
    # ATUALIZAÇÃO DE VALORES DIÁRIOS
    # ================================================================

    def obter_ativo_info(self, trade_id: int) -> Optional[Dict]:
        """Obtém informações do ativo associado a um trade."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute("""
            SELECT p.ativo, p.coingecko_id, p.exchange_symbol
            FROM trades_turma tt
            JOIN posicoes p ON tt.posicao_id = p.id
            WHERE tt.id = %s
        """, (trade_id,))

        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return dict(row) if row else None

    def obter_exchange_symbol(self, ativo: str) -> str:
        """
        Converte nome do ativo para símbolo da exchange (Binance).

        Ex: BTC -> BTCUSDT, ETH -> ETHUSDT, TURBO -> TURBOUSDT
        """
        if ativo in self._symbol_cache:
            return self._symbol_cache[ativo]

        # Mapeamento especial para alguns ativos
        mapeamentos = {
            '1000BONK': 'BONKUSDT',  # Binance usa BONK, não 1000BONK
            '1000000MOG': 'MOGUSDT',  # Similar
            'NEIROCTO': 'NEIROCTOUSDT',
            'NEIROETH': 'NEIROETHUSDT',
        }

        if ativo in mapeamentos:
            symbol = mapeamentos[ativo]
        else:
            # Padrão: adicionar USDT
            symbol = f"{ativo}USDT"

        self._symbol_cache[ativo] = symbol
        return symbol

    def atualizar_cotacao_trade(self, trade_id: int, preco: float,
                                  data: Optional[str] = None,
                                  fonte: str = 'api') -> bool:
        """
        Atualiza a cotação de um trade para uma data específica.

        Args:
            trade_id: ID do trade_turma
            preco: Preço a ser registrado
            data: Data (YYYY-MM-DD), se None usa hoje
            fonte: Fonte do preço ('binance', 'coingecko', 'manual', etc)

        Returns:
            True se atualizado com sucesso
        """
        if data is None:
            data = datetime.now().strftime("%Y-%m-%d")

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO trade_valores_diarios (trade_id, data, preco, fonte)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (trade_id, data) DO UPDATE SET
                    preco = EXCLUDED.preco,
                    fonte = EXCLUDED.fonte
            """, (trade_id, data, preco, fonte))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Erro ao atualizar cotação: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def atualizar_cotacoes_turma(self, turma_id: int) -> Dict[str, Any]:
        """
        Atualiza cotações de todos os trades ativos de uma turma.

        Returns:
            Dict com estatísticas da atualização
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Buscar trades ativos
        cursor.execute("""
            SELECT DISTINCT tt.id as trade_id, p.ativo, p.coingecko_id, p.exchange_symbol
            FROM carteira_turma ct
            JOIN trades_turma tt ON ct.trade_id = tt.id
            JOIN posicoes p ON tt.posicao_id = p.id
            WHERE ct.turma_id = %s AND ct.ativo_atual = 1
        """, (turma_id,))

        trades = cursor.fetchall()
        cursor.close()
        conn.close()

        hoje = datetime.now().strftime("%Y-%m-%d")
        resultado = {
            'turma_id': turma_id,
            'data': hoje,
            'total': len(trades),
            'sucesso': 0,
            'erro': 0,
            'detalhes': []
        }

        for trade in trades:
            trade_id = trade['trade_id']
            ativo = trade['ativo']
            coingecko_id = trade['coingecko_id']
            exchange_symbol = trade['exchange_symbol']

            preco = None
            fonte = None

            # Tentar Binance primeiro (mais rápido)
            if exchange_symbol:
                preco = self.obter_preco_binance(exchange_symbol)
                fonte = 'binance'
            elif ativo:
                symbol = self.obter_exchange_symbol(ativo)
                preco = self.obter_preco_binance(symbol)
                fonte = 'binance'

            # Fallback para CoinGecko
            if preco is None and coingecko_id:
                preco = self.obter_preco_coingecko(coingecko_id)
                fonte = 'coingecko'

            if preco is not None:
                if self.atualizar_cotacao_trade(trade_id, preco, hoje, fonte):
                    resultado['sucesso'] += 1
                    resultado['detalhes'].append({
                        'trade_id': trade_id,
                        'ativo': ativo,
                        'preco': preco,
                        'fonte': fonte,
                        'status': 'ok'
                    })
                else:
                    resultado['erro'] += 1
            else:
                resultado['erro'] += 1
                resultado['detalhes'].append({
                    'trade_id': trade_id,
                    'ativo': ativo,
                    'preco': None,
                    'fonte': None,
                    'status': 'erro - preço não encontrado'
                })

            # Rate limiting
            time.sleep(0.1)

        return resultado

    def atualizar_todas_cotacoes(self) -> Dict[str, Any]:
        """
        Atualiza cotações de todas as turmas.

        Returns:
            Dict com estatísticas gerais
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute("SELECT id, nome FROM turmas")
        turmas = cursor.fetchall()
        cursor.close()
        conn.close()

        resultado = {
            'data': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'turmas_processadas': 0,
            'total_trades': 0,
            'sucesso': 0,
            'erro': 0,
            'por_turma': []
        }

        for turma in turmas:
            turma_resultado = self.atualizar_cotacoes_turma(turma['id'])
            resultado['turmas_processadas'] += 1
            resultado['total_trades'] += turma_resultado['total']
            resultado['sucesso'] += turma_resultado['sucesso']
            resultado['erro'] += turma_resultado['erro']
            resultado['por_turma'].append({
                'turma_id': turma['id'],
                'nome': turma['nome'],
                'sucesso': turma_resultado['sucesso'],
                'erro': turma_resultado['erro']
            })

        return resultado

    def preencher_historico_faltante(self, turma_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Preenche dias faltantes com preço histórico do CoinGecko.

        Verifica para cada trade ativo quais dias estão faltando entre
        a última cotação salva e ontem, e busca os preços históricos
        via CoinGecko API (market_chart).

        Args:
            turma_id: Se especificado, processa apenas essa turma

        Returns:
            Dict com estatísticas do preenchimento
        """
        from datetime import timedelta

        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        hoje = datetime.now().strftime("%Y-%m-%d")
        ontem = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Buscar trades ativos
        if turma_id:
            cursor.execute("""
                SELECT DISTINCT tt.id as trade_id, p.ativo, p.coingecko_id,
                       ct.data_insercao
                FROM carteira_turma ct
                JOIN trades_turma tt ON ct.trade_id = tt.id
                JOIN posicoes p ON tt.posicao_id = p.id
                WHERE ct.turma_id = %s AND ct.ativo_atual = 1
            """, (turma_id,))
        else:
            cursor.execute("""
                SELECT DISTINCT tt.id as trade_id, p.ativo, p.coingecko_id,
                       MIN(ct.data_insercao) as data_insercao
                FROM carteira_turma ct
                JOIN trades_turma tt ON ct.trade_id = tt.id
                JOIN posicoes p ON tt.posicao_id = p.id
                WHERE ct.ativo_atual = 1
                GROUP BY tt.id
            """)

        trades = cursor.fetchall()

        resultado = {
            'data': hoje,
            'trades_processados': 0,
            'dias_preenchidos': 0,
            'erros': 0,
            'detalhes': []
        }

        # Cache de histórico por coingecko_id para evitar chamadas duplicadas
        historico_cache: Dict[str, List[Dict]] = {}

        for trade in trades:
            trade_id = trade['trade_id']
            ativo = trade['ativo']
            coingecko_id = trade['coingecko_id']
            data_insercao = trade['data_insercao']

            # Precisa ter coingecko_id para buscar histórico
            if not coingecko_id:
                resultado['erros'] += 1
                resultado['detalhes'].append({
                    'trade_id': trade_id,
                    'ativo': ativo,
                    'status': 'erro - sem coingecko_id'
                })
                continue

            # Encontrar primeira e última cotação salva para este trade
            cursor.execute("""
                SELECT MIN(data) as primeira_data, MAX(data) as ultima_data
                FROM trade_valores_diarios
                WHERE trade_id = %s
            """, (trade_id,))
            row = cursor.fetchone()
            primeira_data = row['primeira_data'] if row and row['primeira_data'] else None
            ultima_data = row['ultima_data'] if row and row['ultima_data'] else None

            # Determinar período que precisa de preenchimento
            # Caso 1: Não tem nenhum preço - preencher de data_insercao até ontem
            # Caso 2: Tem preços mas faltam dias anteriores (desde data_insercao)
            # Caso 3: Tem preços mas faltam dias posteriores (até ontem)

            precisa_preencher = False
            data_inicio_preencher = data_insercao
            data_fim_preencher = ontem

            if primeira_data is None:
                # Caso 1: Sem preços - preencher tudo desde data_insercao
                precisa_preencher = True
            else:
                # Verificar se falta histórico anterior
                if primeira_data > data_insercao:
                    precisa_preencher = True
                    data_fim_preencher = primeira_data  # Preencher até o primeiro preço

                # Verificar se falta histórico posterior (entre ultima e ontem)
                if ultima_data and ultima_data < ontem:
                    precisa_preencher = True
                    data_inicio_preencher = ultima_data

            if not precisa_preencher:
                continue

            resultado['trades_processados'] += 1

            # Calcular quantos dias podem faltar (máximo)
            dt_inicio = datetime.strptime(data_insercao, "%Y-%m-%d")
            dt_fim = datetime.strptime(ontem, "%Y-%m-%d")
            dias_totais = (dt_fim - dt_inicio).days + 1

            if dias_totais <= 0:
                continue

            # Buscar histórico do CoinGecko (com cache)
            # Limitar a 365 dias (máximo suportado pelo CoinGecko para market_chart)
            dias_buscar = min(dias_totais + 5, 365)
            if coingecko_id not in historico_cache:
                historico = self.obter_historico_coingecko(coingecko_id, dias=dias_buscar)
                historico_cache[coingecko_id] = historico
                time.sleep(0.5)  # Rate limiting para CoinGecko

            historico = historico_cache[coingecko_id]

            if not historico:
                resultado['erros'] += 1
                resultado['detalhes'].append({
                    'trade_id': trade_id,
                    'ativo': ativo,
                    'status': 'erro - sem historico coingecko'
                })
                continue

            # Filtrar apenas os dias que precisamos (desde data_insercao até ontem)
            dias_preenchidos = 0
            for item in historico:
                item_data = item['data']

                # Só preencher dias entre data_insercao e ontem
                if item_data >= data_insercao and item_data <= ontem:
                    try:
                        cursor.execute("""
                            INSERT INTO trade_valores_diarios (trade_id, data, preco, fonte)
                            VALUES (%s, %s, %s, 'coingecko_historico')
                            ON CONFLICT (trade_id, data) DO NOTHING
                        """, (trade_id, item_data, item['preco']))
                        if cursor.rowcount > 0:
                            dias_preenchidos += 1
                    except Exception as e:
                        print(f"Erro ao inserir cotação: {e}")

            resultado['dias_preenchidos'] += dias_preenchidos
            resultado['detalhes'].append({
                'trade_id': trade_id,
                'ativo': ativo,
                'dias_preenchidos': dias_preenchidos,
                'status': 'ok'
            })

        conn.commit()
        cursor.close()
        conn.close()

        return resultado

    def obter_preco_atual(self, ativo: str, exchange_symbol: Optional[str] = None,
                          coingecko_id: Optional[str] = None) -> Optional[float]:
        """
        Obtém o preço atual de um ativo em tempo real (sem salvar no banco).

        Usa CoinGecko como fonte principal de preços.

        Args:
            ativo: Nome do ativo (ex: BTC, ETH)
            exchange_symbol: Símbolo na exchange (não usado, mantido por compatibilidade)
            coingecko_id: ID no CoinGecko (ex: bitcoin)

        Returns:
            Preço atual ou None se não encontrado
        """
        preco, _ = self.obter_preco_atual_com_status(ativo, exchange_symbol, coingecko_id)
        return preco

    def obter_preco_atual_com_status(self, ativo: str, exchange_symbol: Optional[str] = None,
                                      coingecko_id: Optional[str] = None) -> tuple:
        """
        Obtém o preço atual de um ativo em tempo real com informação de status.

        Args:
            ativo: Nome do ativo (ex: BTC, ETH)
            exchange_symbol: Símbolo na exchange (não usado, mantido por compatibilidade)
            coingecko_id: ID no CoinGecko (ex: bitcoin)

        Returns:
            Tuple (preco ou None, status: 'ok', 'timeout', 'erro', 'sem_id')
        """
        if coingecko_id:
            return self.obter_preco_coingecko_com_status(coingecko_id)
        return None, 'sem_id'


def criar_cotacoes_service() -> CotacoesService:
    """Factory function para criar uma instância do serviço."""
    return CotacoesService()


if __name__ == "__main__":
    # Teste básico
    service = criar_cotacoes_service()
    print("CotacoesService criado com sucesso")
    print(f"Database URL: {'Configurado' if service.db_url else 'AVISO: URL não encontrada'}")
    print(f"CoinGecko API Pro: {'Configurada' if service.gecko_api_key else 'AVISO: API key não encontrada'}")

    # Testar busca de preço
    print("\nTestando busca de preços:")

    # Bitcoin via Binance
    preco_btc = service.obter_preco_binance("BTCUSDT")
    print(f"  BTC (Binance): ${preco_btc:.2f}" if preco_btc else "  BTC (Binance): Erro")

    # Bitcoin via CoinGecko
    preco_btc_cg = service.obter_preco_coingecko("bitcoin")
    print(f"  BTC (CoinGecko): ${preco_btc_cg:.2f}" if preco_btc_cg else "  BTC (CoinGecko): Erro")
