"""
PortfolioService: calcula rentabilidade, alocação e posições para
produtos baseados em alocação percentual (EXC, HB, LC, Alphacoins).

Adaptado da classe PortfolioCrypto do notebook.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
ALLOCATIONS_DIR = _ROOT / "data" / "allocations"
PRICES_DIR = _ROOT / "data" / "prices"

STABLECOINS = {'USDT', 'BUSD', 'TUSD'}

TICKER_TO_COINGECKO = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'BCH': 'bitcoin-cash',
    'XRP': 'ripple', 'LTC': 'litecoin', 'XLM': 'stellar', 'XMR': 'monero',
    'ZEC': 'zcash', 'ADA': 'cardano', 'EOS': 'eos', 'ETC': 'ethereum-classic',
    'LSK': 'lisk', 'DASH': 'dash', 'OMG': 'omisego', 'SMART': 'smartcash',
    'POLY': 'polymath', 'USDT': 'tether', 'BNB': 'binancecoin', 'TUSD': 'true-usd',
    'SWM': 'swarm', 'LINK': 'chainlink', 'PAXG': 'pax-gold', 'MKR': 'maker',
    'XTZ': 'tezos', 'CRO': 'crypto-com-chain', 'KNC': 'kyber-network-crystal',
    'SNX': 'havven', 'AAVE': 'aave', 'BAND': 'band-protocol', 'WNXM': 'wrapped-nxm',
    'WBX': 'wibx', 'YFI': 'yearn-finance', 'AXS': 'axie-infinity', 'UNI': 'uniswap',
    'DODO': 'dodo', 'FLOW': 'flow', 'AUDIO': 'audius', 'BAL': 'balancer',
    'CAKE': 'pancakeswap-token', 'COMP': 'compound-governance-token',
    'SUSHI': 'sushi', 'ICP': 'internet-computer', 'POLIS': 'star-atlas-dao',
    'SPELL': 'spell-token', 'RGT': 'rari-governance-token', 'ETHDYDX': 'dydx',
    'FTM': 'fantom', 'DOT': 'polkadot', 'ATOM': 'cosmos', 'AURY': 'aurory',
    'SCRT': 'secret', 'SAND': 'the-sandbox', 'HNT': 'helium', 'SRM': 'serum',
    'RON': 'ronin', 'TRIBE': 'tribe-2', 'MATIC': 'matic-network',
    'AVAX': 'avalanche-2', 'LUNA': 'terra-luna-2', 'NEAR': 'near', 'AR': 'arweave',
    'MANA': 'decentraland', 'LDO': 'lido-dao', 'CRV': 'curve-dao-token',
    'GMX': 'gmx', 'OP': 'optimism', 'ARB': 'arbitrum', 'PENDLE': 'pendle',
    'FXS': 'frax-share', 'TIA': 'celestia', 'RUNE': 'thorchain', 'IMX': 'immutable-x',
    'AKT': 'akash-network', 'RENDER': 'render-token', 'STX': 'blockstack',
    'ONDO': 'ondo-finance', 'TON': 'the-open-network', 'AERO': 'aerodrome-finance',
    'MORPHO': 'morpho', 'ENA': 'ethena', 'VIRTUAL': 'virtual-protocol',
    'HYPE': 'hyperliquid', 'TAO': 'bittensor', 'SUI': 'sui', 'FET': 'fetch-ai',
    'SYRUP': 'syrup', 'SEI': 'sei-network', 'EUL': 'euler',
    'GALA': 'gala', 'NMR': 'numeraire', 'GNS': 'gains-network',
    'INJ': 'injective-protocol', 'RDNT': 'radiant-capital', 'KUJI': 'kujira',
    'PRIME': 'echelon-prime', 'BEAM': 'beam-2', 'MUBI': 'multibit',
    'ETHFI': 'ether-fi', 'DEEP': 'deep', 'RAY': 'raydium',
    'WELL': 'moonwell-artemis', 'AIOZ': 'aioz-network', 'ZBCN': 'zebec-network',
    'LQTY': 'liquity', 'BADGER': 'badger-dao', 'ALICE': 'my-neighbor-alice',
    'PERP': 'perpetual-protocol', 'ALPHA': 'alpha-finance',
    'YGG': 'yield-guild-games', 'GENE': 'genopets', 'ACA': 'acala',
    'RBW': 'rainbow-token-2', 'GOG': 'guild-of-guardians', 'ILV': 'illuvium',
    'CNC': 'conic-finance', 'VELA': 'vela-token', 'BOTTO': 'botto',
    'NTX': 'nunet', 'SEILOR': 'kryptonite', 'PRISMA': 'prisma-governance-token',
    'SHDW': 'genesysgo-shadow', 'NEON': 'neon', 'ML': 'mintlayer',
    'VISTA': 'ethervista', 'ANON': 'heyanon', 'YNE': 'yne',
    'GRIFFAIN': 'griffain', 'KMNO': 'kamino', 'MCADE': 'metacade',
    'MYRIA': 'myria', 'MAMO': 'mamo', 'META': 'meta-2-2',
    'BERT': 'bertram-the-pomeranian', 'AVICI': 'avici', 'UMBRA': 'umbra',
}


class PortfolioService:
    """Serviço que calcula dados de portfólio a partir de alocações percentuais."""

    def __init__(self, nome, csv_path=None, df_aloc=None, cotacoes_service=None):
        """
        Args:
            nome: nome do sub-portfólio (ex: 'Principal', 'High Beta')
            csv_path: caminho do CSV de alocação (opcional se df_aloc fornecido)
            df_aloc: DataFrame de alocação já carregado (opcional se csv_path fornecido)
            cotacoes_service: instância de CotacoesService para buscar preços
        """
        self.nome = nome
        self._cotacoes = cotacoes_service

        if df_aloc is not None:
            self.df_aloc = df_aloc
        elif csv_path:
            self.df_aloc = self._load_csv(csv_path)
        else:
            raise ValueError("csv_path ou df_aloc é obrigatório")

        self.df_aloc.columns = [c.upper() for c in self.df_aloc.columns]
        self._normalizar_percentuais()

        self.df_precos = pd.DataFrame()
        self.df_retornos = pd.DataFrame()
        self.df_retornos_ponderados = pd.DataFrame()
        self.df_posicao = pd.DataFrame()

    @staticmethod
    def _load_csv(path):
        df = pd.read_csv(path, index_col='Data', parse_dates=['Data'],
                         date_format='%Y-%m-%d')
        return df

    def _normalizar_percentuais(self):
        def parse_pct(x):
            if isinstance(x, str) and '%' in x:
                return float(x.strip('%')) / 100.0
            try:
                v = float(x)
                return v / 100.0 if v > 1.0 else v
            except (ValueError, TypeError):
                return 0.0
        self.df_aloc = self.df_aloc.map(parse_pct).fillna(0.0)

    def _coingecko_id(self, ticker):
        return TICKER_TO_COINGECKO.get(ticker)

    def carregar_precos(self, repo=None):
        """Carrega preços históricos do banco (precos_diarios) com fallback para CSVs."""
        df_precos = pd.DataFrame(index=self.df_aloc.index)
        idx_aloc_min = self.df_aloc.index.min() if not self.df_aloc.empty else None
        idx_aloc_max = self.df_aloc.index.max() if not self.df_aloc.empty else None
        logger.debug(
            "[carregar_precos] ANTES carregar index alocação min=%s max=%s len=%s",
            idx_aloc_min, idx_aloc_max, len(self.df_aloc.index)
        )

        cg_ids_needed = {}
        for ticker in self.df_aloc.columns:
            cg_id = self._coingecko_id(ticker)
            if cg_id:
                cg_ids_needed[ticker] = cg_id

        df_db = self._carregar_precos_db(repo, list(cg_ids_needed.values()))
        n_from_db = sum(1 for cg in cg_ids_needed.values() if cg in df_db)
        if cg_ids_needed:
            print(f"[PortfolioService] Preços: {n_from_db}/{len(cg_ids_needed)} ativos do SQLite (precos_diarios), resto CSV/fallback", flush=True)

        for ticker, cg_id in cg_ids_needed.items():
            df_ativo = None

            if cg_id in df_db:
                df_ativo = df_db[cg_id]
            else:
                csv_file = PRICES_DIR / f"{cg_id}.csv"
                if csv_file.exists():
                    df_ativo = pd.read_csv(csv_file, index_col='data', parse_dates=True)
                    df_ativo = df_ativo.groupby(df_ativo.index).first()
                    if 'close' not in df_ativo.columns:
                        df_ativo = None
                    else:
                        df_ativo = df_ativo[['close']].rename(columns={'close': 'preco'})

            if df_ativo is None or df_ativo.empty:
                if ticker == 'BTC':
                    logger.debug("[carregar_precos] BTC df_ativo vazio ou None, pulando")
                continue

            col = 'preco' if 'preco' in df_ativo.columns else 'close'
            idx_ativo_min = df_ativo.index.min()
            idx_ativo_max = df_ativo.index.max()
            if ticker == 'BTC':
                logger.debug(
                    "[carregar_precos] BTC ANTES reindex/ffill df_ativo len=%s index_min=%s index_max=%s nan_count=%s",
                    len(df_ativo), idx_ativo_min, idx_ativo_max,
                    df_ativo[col].isna().sum() if hasattr(df_ativo[col], 'isna') else 0
                )
            # POSSÍVEL CAUSA: reindex ao índice da alocação pode truncar se df_aloc começar depois do primeiro preço do ativo
            if df_ativo.index[0] <= self.df_aloc.index[0]:
                df_precos[ticker] = df_ativo[col].reindex(
                    self.df_aloc.index, method='ffill')
            else:
                first_price = float(df_ativo[col].iloc[0])
                df_precos[ticker] = np.nan
                df_precos.loc[df_ativo.index[0]:, ticker] = df_ativo[col]
                df_precos.loc[:df_ativo.index[0], ticker] = first_price
            df_precos[ticker] = df_precos[ticker].ffill()

            if ticker == 'BTC':
                ser = df_precos[ticker]
                logger.debug(
                    "[carregar_precos] BTC APÓS reindex/ffill len=%s index_min=%s index_max=%s nan_count=%s",
                    len(ser), ser.index.min() if not ser.empty else None, ser.index.max() if not ser.empty else None,
                    ser.isna().sum()
                )

        self.df_precos = df_precos
        logger.debug(
            "[carregar_precos] APÓS todos ativos df_precos shape=%s index_min=%s index_max=%s",
            self.df_precos.shape,
            self.df_precos.index.min() if not self.df_precos.empty else None,
            self.df_precos.index.max() if not self.df_precos.empty else None
        )
        return self.df_precos

    @staticmethod
    def _carregar_precos_db(repo, cg_ids):
        """Carrega preços da tabela precos_diarios. Retorna dict cg_id -> DataFrame."""
        if not repo or not cg_ids:
            return {}
        logger.debug(
            "[_carregar_precos_db] QUERY precos_diarios params cg_ids=%s",
            cg_ids
        )
        try:
            from storage.sqlite_repo import get_repo
            if repo is None:
                repo = get_repo()
            with repo.connection() as conn:
                placeholders = ','.join(['%s'] * len(cg_ids))
                query = f"""
                    SELECT coingecko_id, data, preco
                    FROM precos_diarios
                    WHERE coingecko_id IN ({placeholders})
                    ORDER BY coingecko_id, data
                """
                df = pd.read_sql_query(query, conn, params=cg_ids)
            logger.debug(
                "[_carregar_precos_db] APÓS read_sql_query len(df)=%s min(data)=%s max(data)=%s head5=%s tail5=%s",
                len(df),
                df['data'].min() if not df.empty and 'data' in df.columns else None,
                df['data'].max() if not df.empty and 'data' in df.columns else None,
                df.head(5).to_dict() if len(df) >= 5 else df.to_dict(),
                df.tail(5).to_dict() if len(df) >= 5 else df.to_dict()
            )
            if df.empty:
                return {}
            result = {}
            for cg_id, grp in df.groupby('coingecko_id'):
                g = grp.copy()
                g['data'] = pd.to_datetime(g['data'])
                g.set_index('data', inplace=True)
                g = g[~g.index.duplicated(keep='first')]
                result[cg_id] = g
                idx_min = g.index.min() if not g.empty else None
                idx_max = g.index.max() if not g.empty else None
                logger.debug(
                    "[_carregar_precos_db] POR ATIVO cg_id=%s rows=%s min_date=%s max_date=%s nan_count=%s head5=%s tail5=%s",
                    cg_id, len(g), idx_min, idx_max,
                    g.isna().sum().sum() if hasattr(g, 'isna') else 0,
                    g.head(5).to_dict() if len(g) >= 5 else g.to_dict(),
                    g.tail(5).to_dict() if len(g) >= 5 else g.to_dict()
                )
            return result
        except Exception as e:
            logger.debug("[_carregar_precos_db] EXCEÇÃO %s", e, exc_info=True)
            print(f"[PortfolioService] Erro ao ler precos_diarios: {e}")
            return {}

    def calcular_retornos(self):
        """Calcula retornos diários simples."""
        self.df_retornos = self.df_precos.pct_change().fillna(0)
        for coin in STABLECOINS:
            if coin in self.df_retornos.columns:
                self.df_retornos[coin] = 0
        logger.debug(
            "[calcular_retornos] APÓS pct_change fillna df_retornos shape=%s index_min=%s index_max=%s nan_count=%s",
            self.df_retornos.shape,
            self.df_retornos.index.min() if not self.df_retornos.empty else None,
            self.df_retornos.index.max() if not self.df_retornos.empty else None,
            self.df_retornos.isna().sum().sum()
        )
        return self.df_retornos

    def calcular_retornos_ponderados(self):
        """Retornos ponderados pela alocação."""
        aloc_aligned = self.df_aloc.reindex(
            columns=self.df_retornos.columns, fill_value=0)
        logger.debug(
            "[calcular_retornos_ponderados] ANTES MERGE aloc_aligned index_min=%s index_max=%s",
            aloc_aligned.index.min() if not aloc_aligned.empty else None,
            aloc_aligned.index.max() if not aloc_aligned.empty else None
        )
        self.df_retornos_ponderados = self.df_retornos.mul(aloc_aligned)
        self.df_retornos_ponderados[self.nome] = self.df_retornos_ponderados.sum(axis=1)
        col_nome = self.nome
        sr = self.df_retornos_ponderados[col_nome] if col_nome in self.df_retornos_ponderados.columns else None
        logger.debug(
            "[calcular_retornos_ponderados] APÓS mul e sum(axis=1) col=%s len=%s index_min=%s index_max=%s nan_count=%s",
            col_nome,
            len(sr) if sr is not None else 0,
            sr.index.min() if sr is not None and not sr.empty else None,
            sr.index.max() if sr is not None and not sr.empty else None,
            sr.isna().sum() if sr is not None else None
        )
        return self.df_retornos_ponderados

    def calcular_posicao(self, capital_base=10000):
        """Simula posições diárias com rebalanceamento."""
        df_aloc = self.df_aloc
        df_ret = self.df_retornos.reindex(columns=df_aloc.columns, fill_value=0)

        df_pos = pd.DataFrame(0.0, index=df_aloc.index, columns=df_aloc.columns)
        df_pos.iloc[0] = df_aloc.iloc[0] * capital_base

        for i in range(1, len(df_pos)):
            df_pos.iloc[i] = df_pos.iloc[i-1] * (1 + df_ret.iloc[i])
            if not df_aloc.iloc[i].equals(df_aloc.iloc[i-1]):
                valor_portfolio = df_pos.iloc[i-1].sum()
                df_pos.iloc[i] = df_aloc.iloc[i] * valor_portfolio * (1 + df_ret.iloc[i])

        df_pos[self.nome] = df_pos.sum(axis=1)
        self.df_posicao = df_pos
        return self.df_posicao

    def rentabilidade_acumulada(self, inicio=None, fim=None):
        """Série de rentabilidade acumulada do portfólio."""
        rp = self.df_retornos_ponderados[self.nome].copy()
        logger.debug(
            "[rentabilidade_acumulada] ANTES FILTRO inicio=%s fim=%s rp len=%s index_min=%s index_max=%s",
            inicio, fim, len(rp), rp.index.min() if not rp.empty else None, rp.index.max() if not rp.empty else None
        )
        if inicio:
            rp = rp.loc[inicio:]
            logger.debug(
                "[rentabilidade_acumulada] APÓS loc[inicio:] rp len=%s index_min=%s index_max=%s",
                len(rp), rp.index.min() if not rp.empty else None, rp.index.max() if not rp.empty else None
            )
        if fim:
            rp = rp.loc[:fim]
            logger.debug(
                "[rentabilidade_acumulada] APÓS loc[:fim] rp len=%s index_min=%s index_max=%s",
                len(rp), rp.index.min() if not rp.empty else None, rp.index.max() if not rp.empty else None
            )
        rp.iloc[0] = 0
        data_inicial_considerada = rp.index[0] if not rp.empty else None
        primeira_linha_valor = rp.iloc[0] if not rp.empty else None
        logger.debug(
            "[rentabilidade_acumulada] ANTES cumprod data_inicial_considerada=%s primeira_linha_valor=%s "
            "rp.iloc[0]=%s (zerado) len=%s sem reset_index sem iloc[0 forçando início",
            data_inicial_considerada, primeira_linha_valor, rp.iloc[0] if not rp.empty else None, len(rp)
        )
        return ((1 + rp).cumprod() - 1) * 100

    def rentabilidade_total_pct(self):
        """Rentabilidade acumulada total (%)."""
        rp = self.df_retornos_ponderados[self.nome].copy()
        rp.iloc[0] = 0
        return float(((1 + rp).prod() - 1) * 100)

    def posicoes_abertas(self):
        """Retorna lista de posições atualmente abertas."""
        df_aloc = self.df_aloc
        df_preco = self.df_precos
        ultima_linha = df_aloc.iloc[-1]
        resultados = []

        for ticker in ultima_linha.index:
            if ticker in STABLECOINS or ultima_linha[ticker] == 0:
                continue
            if ticker not in df_preco.columns:
                continue

            alocacao = df_aloc[ticker]
            preco = df_preco[ticker]
            datas_entrada = alocacao[alocacao.ne(0) & alocacao.shift(1, fill_value=0).eq(0)].index
            if len(datas_entrada) > 0:
                ultima_entrada = datas_entrada[-1]
            elif alocacao.ne(0).all():
                ultima_entrada = alocacao.index[0]
            else:
                continue

            dias = (pd.Timestamp.now() - pd.Timestamp(ultima_entrada)).days
            pe = float(preco.loc[ultima_entrada]) if ultima_entrada in preco.index else 0
            pa = float(preco.iloc[-1])
            pnl = ((pa - pe) / pe * 100) if pe != 0 else 0
            alloc_pct = float(ultima_linha[ticker]) * 100

            resultados.append({
                'ativo': ticker,
                'coingecko_id': self._coingecko_id(ticker),
                'data_entrada': str(ultima_entrada.date()),
                'dias': dias,
                'preco_entrada': round(pe, 6),
                'preco_atual': round(pa, 6),
                'pnl_pct': round(pnl, 2),
                'alocacao_pct': round(alloc_pct, 2),
                'side': 'LONG',
            })

        return sorted(resultados, key=lambda x: x['pnl_pct'], reverse=True)

    def posicoes_fechadas(self):
        """Retorna lista de posições já encerradas."""
        df_aloc = self.df_aloc
        df_preco = self.df_precos
        resultados = []

        for ticker in df_aloc.columns:
            if ticker in STABLECOINS or ticker not in df_preco.columns:
                continue

            alocacao = df_aloc[ticker]
            preco = df_preco[ticker]
            dentro = False

            for i in range(len(alocacao)):
                if alocacao.iloc[i] != 0 and not dentro:
                    data_entrada = alocacao.index[i]
                    pe = float(preco.loc[data_entrada]) if data_entrada in preco.index else 0
                    dentro = True
                elif alocacao.iloc[i] == 0 and dentro:
                    data_saida = alocacao.index[i]
                    ps = float(preco.loc[data_saida]) if data_saida in preco.index else 0
                    dias = (pd.Timestamp(data_saida) - pd.Timestamp(data_entrada)).days
                    pnl = ((ps - pe) / pe * 100) if pe != 0 else 0

                    resultados.append({
                        'ativo': ticker,
                        'coingecko_id': self._coingecko_id(ticker),
                        'data_entrada': str(data_entrada.date()),
                        'data_saida': str(data_saida.date()),
                        'dias': dias,
                        'preco_entrada': round(pe, 6),
                        'preco_saida': round(ps, 6),
                        'pnl_pct': round(pnl, 2),
                        'side': 'LONG',
                    })
                    dentro = False

        return sorted(resultados, key=lambda x: x['pnl_pct'], reverse=True)

    def rentabilidade_serie_periodo(self, inicio=None, fim=None):
        """Série de rentabilidade acumulada para um período, sem downsampling (uma entrada por dia)."""
        s = self.rentabilidade_acumulada(inicio=inicio, fim=fim)
        return [{'dia': str(d.date()), 'rentabilidade_acumulada_pct': round(float(v), 4)} for d, v in s.items()]

    @staticmethod
    def _downsample_serie(serie, max_points=500):
        """Reduz série temporal mantendo no máximo max_points entradas."""
        items = list(serie.items())
        if len(items) <= max_points:
            return [{'dia': str(d.date()), 'rentabilidade_acumulada_pct': round(v, 4)} for d, v in items]
        step = max(1, len(items) // max_points)
        sampled = items[::step]
        if items[-1] not in sampled:
            sampled.append(items[-1])
        return [{'dia': str(d.date()), 'rentabilidade_acumulada_pct': round(v, 4)} for d, v in sampled]

    @staticmethod
    def _downsample_alloc(alloc_df, max_points=500):
        """Reduz alocação histórica, mantendo apenas datas onde houve mudança ou amostradas."""
        if len(alloc_df) <= max_points:
            result = []
            for d in alloc_df.index:
                row = {'data': str(d.date())}
                for col in alloc_df.columns:
                    val = float(alloc_df.loc[d, col])
                    if val > 0:
                        row[col] = round(val, 2)
                result.append(row)
            return result

        change_mask = (alloc_df != alloc_df.shift(1)).any(axis=1)
        change_mask.iloc[0] = True
        change_mask.iloc[-1] = True
        change_dates = alloc_df.index[change_mask]

        if len(change_dates) > max_points:
            step = max(1, len(change_dates) // max_points)
            change_dates = change_dates[::step]
            if alloc_df.index[-1] not in change_dates:
                change_dates = change_dates.append(pd.DatetimeIndex([alloc_df.index[-1]]))

        result = []
        for d in change_dates:
            row = {'data': str(d.date())}
            for col in alloc_df.columns:
                val = float(alloc_df.loc[d, col])
                if val > 0:
                    row[col] = round(val, 2)
            result.append(row)
        return result

    def pnl_por_periodo(self, inicio=None, fim=None):
        """Calcula PnL acumulado por ativo em um período, incluindo ativos que saíram.

        Replica a lógica do notebook: usa máscara binária de alocação para zerar
        retornos nos dias sem exposição, depois acumula via cumprod.
        """
        df_ret = self.df_retornos.copy()
        df_aloc = self.df_aloc.copy()

        if inicio:
            df_ret = df_ret.loc[inicio:]
            df_aloc = df_aloc.loc[inicio:]
        if fim:
            df_ret = df_ret.loc[:fim]
            df_aloc = df_aloc.loc[:fim]

        if df_ret.empty or df_aloc.empty:
            return []

        cols = [c for c in df_aloc.columns if c not in STABLECOINS]
        df_aloc = df_aloc[[c for c in cols if c in df_aloc.columns]]
        mask = (df_aloc != 0).astype(int)

        df_ret_aligned = df_ret.reindex(columns=mask.columns, fill_value=0)
        df_masked = df_ret_aligned.mul(mask)

        cols_com_aloc = mask.columns[(mask != 0).any(axis=0)]
        df_masked = df_masked[cols_com_aloc]

        if df_masked.empty:
            return []

        df_masked.iloc[0] = 0
        acum = ((1 + df_masked).cumprod() - 1) * 100

        last_row = acum.iloc[-1].sort_values()
        result = []
        for ticker, pnl in last_row.items():
            result.append({
                'ativo': ticker,
                'pnl_pct': round(float(pnl), 2),
            })
        return result

    def alocacao_historica(self):
        """Retorna df_aloc como percentuais (0-100) para o chart."""
        return (self.df_aloc * 100).round(2)



# Configuração dos portfólios conhecidos
PORTFOLIO_CONFIG = {
    'EXC': {'csv': 'EXC.csv', 'nome': 'Principal'},
    'HB':  {'csv': 'HB.csv',  'nome': 'High Beta'},
    'LC':  {'csv': 'LC.csv',  'nome': 'Low Caps'},
    'AC':  {'csv': 'AC.csv',  'nome': 'Alphacoins'},
}


def _get_svc(portfolio_key, repo=None):
    """Instancia e prepara um PortfolioService para o portfolio_key dado."""
    config = PORTFOLIO_CONFIG.get(portfolio_key)
    if not config:
        raise ValueError(f"Portfolio desconhecido: {portfolio_key}")

    csv_path = _ROOT / config['csv']
    if not csv_path.exists():
        csv_path = ALLOCATIONS_DIR / config['csv']

    if repo is None:
        try:
            from storage.sqlite_repo import get_repo
            repo = get_repo()
        except Exception:
            repo = None

    svc = PortfolioService(nome=config['nome'], csv_path=str(csv_path))
    svc.carregar_precos(repo=repo)
    svc.calcular_retornos()
    return svc


def get_portfolio_data(portfolio_key, capital_base=10000, repo=None):
    """Carrega e calcula dados de um portfólio pelo seu key."""
    svc = _get_svc(portfolio_key, repo=repo)
    svc.calcular_retornos_ponderados()
    svc.calcular_posicao(capital_base)

    rent_total = svc.rentabilidade_total_pct()
    rent_serie = svc.rentabilidade_acumulada()
    abertas = svc.posicoes_abertas()
    fechadas = svc.posicoes_fechadas()

    valor_total = float(svc.df_posicao[svc.nome].iloc[-1]) if not svc.df_posicao.empty else capital_base
    capital_alocado = valor_total * (1.0 - float(svc.df_aloc.iloc[-1].get('USDT', 0)))

    rent_serie_list = svc._downsample_serie(rent_serie)

    alloc_hist = svc.alocacao_historica()
    alloc_list = svc._downsample_alloc(alloc_hist)
    # Caixa não é exibido para EXC, HB, LC, AC (apenas para produtos com API key: Soros, Memebot)

    data = {
        'resumo': {
            'rentabilidade_acumulada_pct': round(rent_total, 2),
            'valor_total': round(valor_total, 2),
            'capital_alocado': round(capital_alocado, 2),
            'capital_em_caixa': round(valor_total - capital_alocado, 2),
            'capital_base': capital_base,
            'trades_ativos': len(abertas),
            'trades_fechados': len(fechadas),
        },
        'rentabilidade_serie': rent_serie_list,
        'posicoes_abertas': abertas,
        'posicoes_fechadas': fechadas,
        'alocacao_historica': alloc_list,
    }
    return data, svc


def get_portfolio_pnl(portfolio_key, inicio=None, fim=None, repo=None):
    """Retorna PnL por ativo para um período específico."""
    svc = _get_svc(portfolio_key, repo=repo)
    return svc.pnl_por_periodo(inicio=inicio, fim=fim)


def get_portfolio_rentabilidade_serie(portfolio_key, inicio=None, fim=None, repo=None):
    """Retorna série de rentabilidade acumulada para um período, dia a dia (sem downsampling)."""
    svc = _get_svc(portfolio_key, repo=repo)
    svc.calcular_retornos_ponderados()
    return svc.rentabilidade_serie_periodo(inicio=inicio, fim=fim)
