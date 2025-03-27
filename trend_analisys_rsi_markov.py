import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import os
from pycoingecko import CoinGeckoAPI
from scipy.stats import skew
import plotly.graph_objects as go
import warnings
from tabulate import tabulate
from markov_volatility import MarkovVolatility  # Import the MarkovVolatility class
from plotly.subplots import make_subplots

warnings.filterwarnings(action='ignore')

class TrendAnalyzer:
    def __init__(self, asset_ids, data_path, btc_data_path, use_btc_adjusted=True, verbose=True, 
                 lookback_days=90, trend_metrics_lookback='all', markov_model_name=None, 
                 ssr_data_path=None):
        self.asset_ids = asset_ids
        self.data_path = data_path
        self.btc_data_path = btc_data_path
        self.use_btc_adjusted = use_btc_adjusted
        self.verbose = verbose
        self.lookback_days = lookback_days
        self.trend_metrics_lookback = trend_metrics_lookback
        self.ma_periods = {
            'Short Term': [3, 5, 7, 14],
            'Medium Term': [21, 30, 45, 63],
            'Long Term': [84, 100, 120, 150, 200, 252, 365]
        }
        self.ticker_mapping = self._get_ticker_mapping()
        self.asset_data = {}
        self.static_computed = False
        self.portfolios = {}  # Dictionary to store different portfolios
        self.ssr_data_path = ssr_data_path
        self.ssr_data = None
        
        # Initialize logger first
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler('trend_analyzer.log')]
        )
        self.logger = logging.getLogger(__name__)
        
        # Then load Markov volatility model if provided
        self.markov_model = None
        self.volatility_cache = {}  # Cache for volatility states
        if markov_model_name:
            try:
                self.markov_model = MarkovVolatility.load_model(markov_model_name)
                self.logger.info(f"Loaded Markov volatility model: {markov_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load Markov volatility model: {e}")
                
        # Load SSR data if path provided
        if ssr_data_path:
            self._load_ssr_data()

    def _get_ticker_mapping(self):
        cg = CoinGeckoAPI()
        coins_list = cg.get_coins_list()
        coins_df = pd.DataFrame(coins_list)
        known_mappings = {'VIRTUAL': 'virtual-protocol', 'HYPE': 'hyperliquid', 'YNE': 'yesnoerror'}
        mapping = {ticker: coin_id for ticker, coin_id in known_mappings.items() if coin_id in self.asset_ids}
        filtered_coins_df = coins_df[coins_df['id'].isin(self.asset_ids)]
        for _, row in filtered_coins_df.iterrows():
            ticker = row['symbol'].upper()
            if ticker not in mapping:
                mapping[ticker] = row['id']
        return mapping

    def _load_data(self, asset_id):
        data_path = f"{self.data_path}{asset_id}_candles.csv"
        if not os.path.exists(data_path):
            self.logger.warning(f"File not found for {asset_id}: {data_path}")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(data_path)
            data['date'] = pd.to_datetime(data['date'])
            data.dropna(subset=['close'], inplace=True)
            if data.empty:
                self.logger.warning(f"No valid data for {asset_id} after dropping NaN in 'close'.")
                return pd.DataFrame()
            
            if asset_id != 'bitcoin':
                try:
                    btc_data = pd.read_csv(self.btc_data_path)
                    btc_data['date'] = pd.to_datetime(btc_data['date'])
                    
                    # Include OHLC from BTC data
                    btc_data = btc_data[['date', 'open', 'high', 'low', 'close']].rename(
                        columns={'open': 'btc_open', 'high': 'btc_high', 'low': 'btc_low', 'close': 'btc_close'}
                    )
                    btc_data.dropna(subset=['btc_close'], inplace=True)
                    data = data.merge(btc_data, on='date', how='inner')
                    if data.empty:
                        self.logger.warning(f"No overlapping dates between {asset_id} and Bitcoin data after merge.")
                        return pd.DataFrame()
                    # Calculate BTC-adjusted OHLC prices
                    data[f'{asset_id}_btc_open'] = data['open'] / data['btc_open']
                    data[f'{asset_id}_btc_high'] = data['high'] / data['btc_high']
                    data[f'{asset_id}_btc_low'] = data['low'] / data['btc_low']
                    data[f'{asset_id}_btc'] = data['close'] / data['btc_close']  # Existing close vs BTC
                    # Drop BTC columns to keep DataFrame clean
                    data = data.drop(columns=['btc_open', 'btc_high', 'btc_low', 'btc_close'], errors='ignore')
                except Exception as e:
                    self.logger.error(f"Error merging Bitcoin data for {asset_id}: {e}")
                    return pd.DataFrame()
            self.logger.info(f"Successfully loaded data for {asset_id} with {len(data)} rows.")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data for {asset_id}: {e}")
            return pd.DataFrame()
        
    def _rsi(self, prices):
        """Helper function to calculate RSI for a price series."""
        deltas = np.diff(prices)
        if len(deltas) == 0:
            return np.nan
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain) if len(gain) > 0 else 0
        avg_loss = np.mean(loss) if len(loss) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        return 100 - (100 / (1 + rs)) if rs != np.inf else 100
    
    def _calculate_smooth_rsi(self, data, price_column, rsi_length=28, roc_length=28):
        """
        Calculate Smooth RSI and output a binary signal (+1 for Bull, -1 for Bear).
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data
            price_column (str): Column name to base RSI on (e.g., 'close', 'ethereum_btc')
            rsi_length (int): RSI period
            roc_length (int): RoC period
        
        Returns:
            pd.DataFrame: DataFrame with RSI signal
        """
        if data.empty or price_column not in data.columns:
            return data
        
        data = data.copy()
        # Determine if this is a BTC-adjusted price column
        if price_column.endswith('_btc'):
            asset_prefix = price_column.split('_btc')[0]
            open_col = f'{asset_prefix}_btc_open'
            high_col = f'{asset_prefix}_btc_high'
            low_col = f'{asset_prefix}_btc_low'
            close_col = price_column  # e.g., 'ethereum_btc'
        else:
            open_col = 'open'
            high_col = 'high'
            low_col = 'low'
            close_col = price_column  # e.g., 'close'
        
        # Ensure numeric data
        for col in [open_col, high_col, low_col, close_col]:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Calculate RSI for OHLC
        rsi_high = data[high_col].rolling(rsi_length).apply(
            lambda x: self._rsi(x), raw=True).fillna(method='bfill')
        rsi_open = data[open_col].rolling(rsi_length).apply(
            lambda x: self._rsi(x), raw=True).fillna(method='bfill')
        rsi_low = data[low_col].rolling(rsi_length).apply(
            lambda x: self._rsi(x), raw=True).fillna(method='bfill')
        rsi_close = data[close_col].rolling(rsi_length).apply(
            lambda x: self._rsi(x), raw=True).fillna(method='bfill')
        
        # Smooth RSI as average
        smooth_rsi = (rsi_high + rsi_open + rsi_low + rsi_close) / 4
        
        # Calculate RoC for OHLC
        roc_high = (data[high_col] - data[high_col].shift(roc_length)) / data[high_col].shift(roc_length) * 100
        roc_open = (data[open_col] - data[open_col].shift(roc_length)) / data[open_col].shift(roc_length) * 100
        roc_low = (data[low_col] - data[low_col].shift(roc_length)) / data[low_col].shift(roc_length) * 100
        roc_close = (data[close_col] - data[close_col].shift(roc_length)) / data[close_col].shift(roc_length) * 100
        
        # Smooth RoC as average
        smooth_roc = (roc_high + roc_open + roc_low + roc_close) / 4
        
        # Binary signal: +1 for Bull (RSI > 50 AND RoC > 0), -1 for Bear
        data[f'RSI_Signal_{price_column}'] = ((smooth_rsi > 50) & (smooth_roc > 0)).astype(int) * 2 - 1
        
        return data

    def _calculate_indicators(self, data, price_column, asset_id, suffix=''):
        if price_column not in data.columns or data.empty:
            self.logger.warning(f"Column '{price_column}' not found for {asset_id} or data is empty.")
            return
        data[price_column] = pd.to_numeric(data[price_column], errors='coerce')
        data.dropna(subset=[price_column], inplace=True)
        for term, periods in self.ma_periods.items():
            for period in periods:
                if len(data) >= period:
                    data[f'EMA_{suffix}_{period}'] = data[price_column].ewm(span=period, min_periods=period, adjust=False).mean()
                    data[f'RoC_{suffix}_{period}'] = (data[price_column] - data[price_column].shift(period)) / data[price_column].shift(period) * 100

    def classify_trend(self, mas, rocs, data, suffix):
        n_mas = len(mas.dropna())
        n_rocs = len(rocs.dropna())
        
        # If we don't have enough data to calculate trends, return Neutral
        if n_rocs == 0:
            return "Neutral"
        
        # Use only RoC values to determine trend (ignore EMA metrics)
        pos_rocs = sum(1 for roc in rocs if not np.isnan(roc) and roc > 0) / n_rocs * 100
        
        # Classify trend based solely on RoC values
        if pos_rocs >= 75:
            return "Strong Bull"
        elif pos_rocs >= 50:
            return "Weak Bull"
        elif pos_rocs <= 25:
            return "Strong Bear"
        elif pos_rocs < 50:
            return "Weak Bear"
        return "Neutral"

    def _compute_overall_classification(self, trends):
        valid_trends = [t for t in trends if t not in [None, np.nan, 'N/A']]
        if not valid_trends:
            return "Neutral"
        overall_pos = sum(1 for c in valid_trends if c in ["Strong Bull", "Weak Bull"]) / len(valid_trends) * 100
        if overall_pos >= 75:
            return "Strong Bull"
        elif overall_pos >= 50:
            return "Weak Bull"
        elif overall_pos <= 25:
            return "Strong Bear"
        elif overall_pos < 50:
            return "Weak Bear"
        return "Neutral"

    def _create_classified_data(self, data, asset_id):
        if data.empty:
            return pd.DataFrame()
        classified_data = data.copy()
        classified_data['returns_usd'] = classified_data['close'].pct_change()
        if asset_id != 'bitcoin' and f'{asset_id}_btc' in classified_data.columns:
            classified_data['returns_btc'] = classified_data[f'{asset_id}_btc'].pct_change()

        self._calculate_indicators(classified_data, 'close', asset_id, suffix='USD')
        for term in self.ma_periods:
            mas = [f'EMA_USD_{p}' for p in self.ma_periods[term] if f'EMA_USD_{p}' in classified_data.columns]
            rocs = [f'RoC_USD_{p}' for p in self.ma_periods[term] if f'RoC_USD_{p}' in classified_data.columns]
            classified_data[f'{term} (USD)'] = classified_data.apply(
                lambda row: self.classify_trend(row[mas], row[rocs], classified_data, 'USD'), axis=1
            )
        classified_data['Overall (USD)'] = classified_data.apply(
            lambda row: self._compute_overall_classification([row[f'{term} (USD)'] for term in self.ma_periods]), axis=1
        )

        # Add Smooth RSI signal for USD
        classified_data = self._calculate_smooth_rsi(classified_data, 'close')

        if asset_id != 'bitcoin' and f'{asset_id}_btc' in classified_data.columns:
            self._calculate_indicators(classified_data, f'{asset_id}_btc', asset_id, suffix='BTC')
            for term in self.ma_periods:
                mas = [f'EMA_BTC_{p}' for p in self.ma_periods[term] if f'EMA_BTC_{p}' in classified_data.columns]
                rocs = [f'RoC_BTC_{p}' for p in self.ma_periods[term] if f'RoC_BTC_{p}' in classified_data.columns]
                classified_data[f'{term} (BTC)'] = classified_data.apply(
                    lambda row: self.classify_trend(row[mas], row[rocs], classified_data, 'BTC'), axis=1
                )
            classified_data['Overall (BTC)'] = classified_data.apply(
                lambda row: self._compute_overall_classification([row[f'{term} (BTC)'] for term in self.ma_periods]), axis=1
            )
            # Add Smooth RSI signal for BTC-adjusted price using the close column
            classified_data = self._calculate_smooth_rsi(classified_data, f'{asset_id}_btc')
        else:
            for term in self.ma_periods:
                classified_data[f'{term} (BTC)'] = pd.NA
            classified_data['Overall (BTC)'] = pd.NA
        
        return classified_data

    def calculate_sharpe_sortino(self, data, returns_column):
        if data.empty or len(data) < 2 or returns_column not in data.columns:
            return 0.0, 0.0
        returns = data[returns_column].pct_change().dropna()
        self.logger.debug(f"Returns for {returns_column}: {returns.head().to_list()}")
        if len(returns) < 2:
            return 0.0, 0.0
        mean_return = returns.mean() * 365
        std_return = returns.std() * np.sqrt(365)
        sharpe = mean_return / std_return if std_return > 0 else (0.0 if mean_return == 0 else float('-inf' if mean_return < 0 else 'inf'))
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(365) if not downside_returns.empty else 0.0
        sortino = mean_return / downside_deviation if downside_deviation > 0 else (0.0 if mean_return == 0 else float('-inf' if mean_return < 0 else 'inf'))
        self.logger.debug(f"Sharpe calc: mean={mean_return:.4f}, std={std_return:.4f}, sharpe={sharpe:.2f}")
        self.logger.debug(f"Sortino calc: downside_std={downside_deviation:.4f}, sortino={sortino:.2f}")
        return sharpe, sortino

    def calculate_trend_type_metrics(self, data, trend_type, lookback):
        if data.empty or trend_type not in data.columns:
            return {}
        if lookback != 'all':
            latest_date = data['date'].max()
            start_date = latest_date - timedelta(days=lookback)
            data = data[data['date'] >= start_date].copy()
        if data.empty or len(data) < 2:
            return {}
        metrics = {'USD': {}, 'BTC': {}}
        trend_types = ['Strong Bull', 'Weak Bull', 'Strong Bear', 'Weak Bear', 'Neutral']
        for t in trend_types:
            trend_data = data[data[trend_type] == t].copy()
            if trend_data.empty:
                metrics['USD'][t] = metrics['BTC'][t] = {'sharpe': 0.0, 'sortino': 0.0, 'mean': 0.0, 'median': 0.0, 'skew': 0.0}
                continue
            sharpe_usd, sortino_usd = self.calculate_sharpe_sortino(trend_data, 'returns_usd')
            metrics['USD'][t] = {
                'sharpe': sharpe_usd, 'sortino': sortino_usd,
                'mean': trend_data['returns_usd'].mean() or 0.0,
                'median': trend_data['returns_usd'].median() or 0.0,
                'skew': skew(trend_data['returns_usd'].dropna(), nan_policy='omit') if len(trend_data['returns_usd'].dropna()) > 0 else 0.0
            }
            if 'returns_btc' in trend_data.columns:
                sharpe_btc, sortino_btc = self.calculate_sharpe_sortino(trend_data, 'returns_btc')
                metrics['BTC'][t] = {
                    'sharpe': sharpe_btc, 'sortino': sortino_btc,
                    'mean': trend_data['returns_btc'].mean() or 0.0,
                    'median': trend_data['returns_btc'].median() or 0.0,
                    'skew': skew(trend_data['returns_btc'].dropna(), nan_policy='omit') if len(trend_data['returns_btc'].dropna()) > 0 else 0.0
                }
            else:
                metrics['BTC'][t] = {'sharpe': 0.0, 'sortino': 0.0, 'mean': 0.0, 'median': 0.0, 'skew': 0.0}
        return metrics

    def calculate_transition_probabilities(self, data, trend_type, lookback_days):
        if data.empty or len(data) < 2:
            return 0.0
        latest_date = data['date'].max()
        start_date = latest_date - timedelta(days=lookback_days)
        data = data[data['date'] >= start_date].copy()
        if len(data) < 2:
            return 0.0
        current_trend = data[trend_type].iloc[-1]
        is_bullish = current_trend in ['Strong Bull', 'Weak Bull']
        prev_trend = data[trend_type].shift(1)
        transitions = prev_trend + " -> " + data[trend_type]
        transitions = transitions.dropna()
        if transitions.empty:
            return 0.0
        total_bullish = len(data[data[trend_type].isin(['Strong Bull', 'Weak Bull'])]) - 1
        if total_bullish <= 0:
            return 0.0
        if is_bullish:
            bull_to_bull = len(transitions[transitions.str.contains('Bull ->.*Bull', na=False)])
            return bull_to_bull / total_bullish if total_bullish > 0 else 0.0
        bear_to_bear = len(transitions[transitions.str.contains('Bear ->.*Bear', na=False)])
        total_bearish = len(data) - total_bullish - 1
        return bear_to_bear / total_bearish if total_bearish > 0 else 0.0

    def _compute_lookback_metrics(self):
        results = []
        reverse_mapping = {v: k for k, v in self.ticker_mapping.items()}
        for asset_id in tqdm(self.asset_ids, desc="Computing lookback metrics", disable=not self.verbose):
            classified_data = self.asset_data.get(asset_id, {}).get('classified_data', pd.DataFrame())
            raw_data = self.asset_data.get(asset_id, {}).get('raw_data', pd.DataFrame())
            latest_classifications = classified_data.iloc[-1].to_dict() if not classified_data.empty else {
                f'{term} (USD)': 'N/A' for term in ['Short Term', 'Medium Term', 'Long Term', 'Overall']
            } | {f'{term} (BTC)': 'N/A' for term in ['Short Term', 'Medium Term', 'Long Term', 'Overall']}
            sharpe_usd, sortino_usd = self.calculate_sharpe_sortino(raw_data, 'close')
            sharpe_btc, sortino_btc = (self.calculate_sharpe_sortino(raw_data, f'{asset_id}_btc') 
                                       if asset_id != 'bitcoin' and f'{asset_id}_btc' in raw_data.columns else (0.0, 0.0))
            trans_prob_usd = self.calculate_transition_probabilities(classified_data, 'Overall (USD)', self.lookback_days)
            trans_prob_btc = self.calculate_transition_probabilities(classified_data, 'Overall (BTC)', self.lookback_days) if asset_id != 'bitcoin' else 0.0
            trend_metrics = {trend_type: self.calculate_trend_type_metrics(classified_data, trend_type, self.trend_metrics_lookback)
                            for trend_type in [f'{term} (USD)' for term in ['Short Term', 'Medium Term', 'Long Term', 'Overall']] +
                            [f'{term} (BTC)' for term in ['Short Term', 'Medium Term', 'Long Term', 'Overall']]}
            self.asset_data[asset_id]['trend_metrics'] = trend_metrics
            current_trend_usd = latest_classifications.get('Overall (USD)', 'N/A')
            current_trend_btc = latest_classifications.get('Overall (BTC)', 'N/A')
            trend_metrics_usd = trend_metrics['Overall (USD)'].get(current_trend_usd, {'sharpe': 0.0, 'sortino': 0.0, 'mean': 0.0, 'median': 0.0, 'skew': 0.0})
            trend_metrics_btc = trend_metrics['Overall (BTC)'].get(current_trend_btc, {'sharpe': 0.0, 'sortino': 0.0, 'mean': 0.0, 'median': 0.0, 'skew': 0.0})
            ticker = reverse_mapping.get(asset_id, asset_id.upper())
            results.append({
                'Ticker': ticker,
                'Short Term Trend (USD)': latest_classifications.get('Short Term (USD)', 'N/A'),
                'Medium Term Trend (USD)': latest_classifications.get('Medium Term (USD)', 'N/A'),
                'Long Term Trend (USD)': latest_classifications.get('Long Term (USD)', 'N/A'),
                'Overall Trend (USD)': latest_classifications.get('Overall (USD)', 'N/A'),
                'Short Term Trend (BTC)': latest_classifications.get('Short Term (BTC)', 'N/A'),
                'Medium Term Trend (BTC)': latest_classifications.get('Medium Term (BTC)', 'N/A'),
                'Long Term Trend (BTC)': latest_classifications.get('Long Term (BTC)', 'N/A'),
                'Overall Trend (BTC)': latest_classifications.get('Overall (BTC)', 'N/A'),
                'Sharpe Ratio (USD)': sharpe_usd,
                'Sortino Ratio (USD)': sortino_usd,
                'Sharpe Ratio (BTC)': sharpe_btc,
                'Sortino Ratio (BTC)': sortino_btc,
                'Trend Persistence (USD)': trans_prob_usd,
                'Trend Persistence (BTC)': trans_prob_btc,
                'Current Trend Sharpe (USD)': trend_metrics_usd['sharpe'],
                'Current Trend Sortino (USD)': trend_metrics_usd['sortino'],
                'Current Trend Mean Return (USD)': trend_metrics_usd['mean'],
                'Current Trend Median Return (USD)': trend_metrics_usd['median'],
                'Current Trend Skew (USD)': trend_metrics_usd['skew'],
                'Current Trend Sharpe (BTC)': trend_metrics_btc['sharpe'],
                'Current Trend Sortino (BTC)': trend_metrics_btc['sortino'],
                'Current Trend Mean Return (BTC)': trend_metrics_btc['mean'],
                'Current Trend Median Return (BTC)': trend_metrics_btc['median'],
                'Current Trend Skew (BTC)': trend_metrics_btc['skew'],
                'Latest Date': classified_data['date'].iloc[-1].date() if not classified_data.empty else None
            })
        return pd.DataFrame(results)

    def analyze_multiple_assets(self):
        self.asset_data = {}
        for asset_id in tqdm(self.asset_ids, desc="Analyzing assets (static)", disable=not self.verbose):
            raw_data = self._load_data(asset_id)
            classified_data = self._create_classified_data(raw_data, asset_id)
            self.asset_data[asset_id] = {
                'raw_data': raw_data.copy(),
                'classified_data': classified_data.copy(),
                'price_column_usd': 'close',
                'price_column_btc': f'{asset_id}_btc' if asset_id != 'bitcoin' and f'{asset_id}_btc' in classified_data.columns else None
            }
        self.static_computed = True
        summary_df = self._compute_lookback_metrics()
        self.logger.info("Trend analysis completed for multiple assets.")
        return summary_df

    def set_basket_criteria(self, criteria):
        valid_trends = ['Strong Bull', 'Weak Bull', 'Strong Bear', 'Weak Bear', 'Neutral']
        for trend_type, allowed in criteria.items():
            if trend_type not in [f'{term} (USD)' for term in ['Short Term', 'Medium Term', 'Long Term', 'Overall']] + \
                                [f'{term} (BTC)' for term in ['Short Term', 'Medium Term', 'Long Term', 'Overall']]:
                self.logger.error(f"Invalid trend type: {trend_type}")
                raise ValueError(f"Trend type {trend_type} not recognized.")
            if not all(val in valid_trends for val in allowed):
                self.logger.error(f"Invalid trend values: {allowed}")
                raise ValueError(f"Trend values must be in {valid_trends}")
        self.basket_criteria = criteria
        self.logger.info(f"Basket criteria set: {criteria}")

    def filter_basket_historical(self, dates):
        if not self.static_computed:
            self.logger.error("Static computations not performed.")
            raise ValueError("Run analyze_multiple_assets() first.")
        if not hasattr(self, 'basket_criteria'):
            self.logger.warning("No basket criteria set. Using default: Overall (BTC) = Strong Bull")
            self.basket_criteria = {'Overall (BTC)': ['Strong Bull']}

        historical_baskets = []
        for date in dates:
            date = pd.to_datetime(date)
            basket = []
            for asset_id in self.asset_ids:
                if asset_id == 'bitcoin':  # Skip Bitcoin for baskets
                    continue
                classified_data = self.asset_data.get(asset_id, {}).get('classified_data', pd.DataFrame())
                if classified_data.empty or classified_data['date'].max() < date:
                    continue
                data = classified_data[classified_data['date'] <= date].copy()
                if data.empty:
                    continue
                latest = data.iloc[-1]
                meets_criteria = all(
                    str(latest.get(trend_type, 'N/A')) in allowed
                    for trend_type, allowed in self.basket_criteria.items()
                )
                if meets_criteria:
                    ticker = [k for k, v in self.ticker_mapping.items() if v == asset_id][0]
                    basket.append({
                        'Date': date.date(),
                        'Ticker': ticker,
                        'Short Term (USD)': latest.get('Short Term (USD)', 'N/A'),
                        'Medium Term (USD)': latest.get('Medium Term (USD)', 'N/A'),
                        'Long Term (USD)': latest.get('Long Term (USD)', 'N/A'),
                        'Overall (USD)': latest.get('Overall (USD)', 'N/A'),
                        'Short Term (BTC)': latest.get('Short Term (BTC)', 'N/A'),
                        'Medium Term (BTC)': latest.get('Medium Term (BTC)', 'N/A'),
                        'Long Term (BTC)': latest.get('Long Term (BTC)', 'N/A'),
                        'Overall (BTC)': latest.get('Overall (BTC)', 'N/A')
                    })
            historical_baskets.extend(basket)
        
        basket_df = pd.DataFrame(historical_baskets)
        if basket_df.empty:
            self.logger.info(f"No assets met criteria for dates: {dates}")
        else:
            self.logger.info(f"Basket constructed with {len(basket_df)} entries for dates: {dates}")
        return basket_df

    def _max_drawdown(self, series):
        roll_max = series.cummax()
        drawdowns = (series - roll_max) / roll_max
        return drawdowns.min() if not drawdowns.empty else 0.0

    def _load_ssr_data(self):
        """Load SSR oscillator data from CSV file"""
        if not self.ssr_data_path:
            self.logger.warning("No SSR data path provided. SSR signal will not be available.")
            return
            
        try:
            data = pd.read_csv(self.ssr_data_path)
            if 'date' not in data.columns or 'ssr_oscillator' not in data.columns:
                self.logger.error("SSR data file must contain 'date' and 'ssr_oscillator' columns.")
                return
                
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            # Set date as index for easier lookup
            self.ssr_data = data.set_index('date')
            self.logger.info(f"Successfully loaded SSR data with {len(data)} rows.")
        except Exception as e:
            self.logger.error(f"Failed to load SSR data: {e}")
            
    def _get_ssr_signal(self, date):
        """
        Get SSR signal for a specific date.
        
        Args:
            date: Date to get SSR signal for
            
        Returns:
            int: 1 if ssr_oscillator > 0, 0 otherwise
        """
        if self.ssr_data is None:
            return 0  # Neutral if no data is available
            
        # Convert date to timestamp if not already
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
            
        try:
            # Find the value for this date or the last available date
            available_dates = self.ssr_data.index[self.ssr_data.index <= date]
            if len(available_dates) == 0:
                return 0
                
            latest_date = available_dates[-1]
            ssr_value = self.ssr_data.loc[latest_date, 'ssr_oscillator']
            
            # Return 1 if positive, 0 otherwise
            return 1 if ssr_value > 0 else -1
        except Exception as e:
            self.logger.warning(f"Error getting SSR signal for {date}: {e}")
            return 0  # Neutral on error

    def create_portfolio(self, portfolio_name, btc_trend_gating=None, usd_conditions=None, btc_conditions=None, 
                         rsi_conditions_usd=False, rsi_conditions_btc=False, btc_only=False, 
                         use_volatility_filter=False, volatility_weight=1.0, use_ssr_signal=False,
                         use_ssr_gate=False, btc_rsi_gate=False, use_btc_rsi_signal=False,
                         follow_portfolio=None):
        """
        Create a new portfolio with specified trend, RSI, volatility and SSR conditions.
        
        Args:
            portfolio_name (str): Name to identify the portfolio
            btc_trend_gating (dict, optional): BTC trend condition that gates all others
            usd_conditions (dict, optional): USD trend conditions
            btc_conditions (dict, optional): BTC trend conditions
            rsi_conditions_usd (bool): If True, requires RSI signal vs USD to be 1
            rsi_conditions_btc (bool): If True, requires RSI signal vs BTC to be 1 (altcoins only)
            btc_only (bool): If True, portfolio only includes Bitcoin
            use_volatility_filter (bool): If True, include volatility signals in decision making
            volatility_weight (float): Weight of volatility signal (1.0 = equal to other signals)
            use_ssr_signal (bool): If True, include SSR oscillator signal in decision making
            use_ssr_gate (bool): If True, use SSR signal as a gate rather than an additive signal
            btc_rsi_gate (bool): If True, use BTC's RSI signal as a gate for altcoin selection
            use_btc_rsi_signal (bool): If True, include BTC's RSI signal as a component (not a gate)
            follow_portfolio (str, optional): Name of another portfolio to follow signals from
        """
        if not self.static_computed:
            self.logger.error("Static computations not performed.")
            raise ValueError("Run analyze_multiple_assets() first.")
            
        if portfolio_name in self.portfolios:
            self.logger.warning(f"Portfolio {portfolio_name} already exists. Overwriting.")
        
        # Validate parameters
        if not isinstance(rsi_conditions_usd, bool) or not isinstance(rsi_conditions_btc, bool):
            raise ValueError("rsi_conditions_usd and rsi_conditions_btc must be boolean")
            
        if not isinstance(use_volatility_filter, bool):
            raise ValueError("use_volatility_filter must be boolean")
            
        if not isinstance(use_ssr_signal, bool):
            raise ValueError("use_ssr_signal must be boolean")
            
        if not isinstance(use_ssr_gate, bool):
            raise ValueError("use_ssr_gate must be boolean")
            
        if not isinstance(btc_rsi_gate, bool):
            raise ValueError("btc_rsi_gate must be boolean")
            
        if not isinstance(use_btc_rsi_signal, bool):
            raise ValueError("use_btc_rsi_signal must be boolean")
            
        # Validate that the followed portfolio exists
        if follow_portfolio is not None and follow_portfolio not in self.portfolios:
            self.logger.error(f"Portfolio '{follow_portfolio}' does not exist. Cannot follow.")
            raise ValueError(f"Portfolio '{follow_portfolio}' not found.")
            
        # Cannot use SSR as both signal and gate
        if use_ssr_signal and use_ssr_gate:
            self.logger.warning("Cannot use SSR as both signal and gate. Using as gate only.")
            use_ssr_signal = False
            
        # Cannot use BTC RSI as both signal and gate
        if use_btc_rsi_signal and btc_rsi_gate:
            self.logger.warning("Cannot use BTC RSI as both signal and gate. Using as gate only.")
            use_btc_rsi_signal = False
            
        if not isinstance(volatility_weight, (int, float)) or volatility_weight < 0:
            raise ValueError("volatility_weight must be a non-negative number")
        
        self.portfolios[portfolio_name] = {
            'btc_trend_gating': btc_trend_gating,
            'usd_conditions': usd_conditions or {},
            'btc_conditions': btc_conditions or {},
            'rsi_conditions_usd': rsi_conditions_usd,
            'rsi_conditions_btc': rsi_conditions_btc,
            'btc_only': btc_only,
            'use_volatility_filter': use_volatility_filter,
            'volatility_weight': volatility_weight,
            'use_ssr_signal': use_ssr_signal,
            'use_ssr_gate': use_ssr_gate,
            'btc_rsi_gate': btc_rsi_gate,
            'use_btc_rsi_signal': use_btc_rsi_signal,
            'follow_portfolio': follow_portfolio,
            'creation_date': datetime.now(),
            'backtest_results': None
        }
        
        self.logger.info(f"Portfolio '{portfolio_name}' created with criteria: BTC gating={btc_trend_gating}, "
                        f"USD conditions={usd_conditions}, BTC conditions={btc_conditions}, "
                        f"RSI USD={rsi_conditions_usd}, RSI BTC={rsi_conditions_btc}, "
                        f"volatility filter={use_volatility_filter}, volatility weight={volatility_weight}, "
                        f"SSR signal={use_ssr_signal}, SSR gate={use_ssr_gate}, "
                        f"BTC RSI gate={btc_rsi_gate}, BTC RSI signal={use_btc_rsi_signal}, "
                        f"Follow portfolio={follow_portfolio}, "
                        f"BTC only={btc_only}")
        return portfolio_name

    def list_portfolios(self):
        """Return a list of all stored portfolios with their criteria"""
        portfolio_list = []
        for name, details in self.portfolios.items():
            portfolio_list.append({
                'name': name,
                'btc_trend_gating': details['btc_trend_gating'],
                'usd_conditions': details['usd_conditions'],
                'btc_conditions': details['btc_conditions'],
                'btc_only': details['btc_only'],
                'rsi_conditions_usd': details.get('rsi_conditions_usd', False),
                'rsi_conditions_btc': details.get('rsi_conditions_btc', False),
                'use_volatility_filter': details.get('use_volatility_filter', False),
                'volatility_weight': details.get('volatility_weight', 1.0),
                'use_ssr_signal': details.get('use_ssr_signal', False),
                'use_ssr_gate': details.get('use_ssr_gate', False),
                'btc_rsi_gate': details.get('btc_rsi_gate', False),
                'use_btc_rsi_signal': details.get('use_btc_rsi_signal', False),
                'follow_portfolio': details.get('follow_portfolio', None),
                'creation_date': details['creation_date'],
                'has_backtest': details['backtest_results'] is not None
            })
        return pd.DataFrame(portfolio_list)
        
    def delete_portfolio(self, portfolio_name):
        """Delete a stored portfolio"""
        if portfolio_name not in self.portfolios:
            self.logger.warning(f"Portfolio '{portfolio_name}' does not exist.")
            return False
            
        del self.portfolios[portfolio_name]
        self.logger.info(f"Portfolio '{portfolio_name}' deleted.")
        return True
        
    def get_portfolio_details(self, portfolio_name):
        """Get detailed information about a specific portfolio"""
        if portfolio_name not in self.portfolios:
            self.logger.warning(f"Portfolio '{portfolio_name}' does not exist.")
            return None
            
        return self.portfolios[portfolio_name]

    def _get_ticker_from_id(self, asset_id):
        """
        Convert an asset ID back to its ticker symbol.
        
        Args:
            asset_id (str): The asset ID to convert
            
        Returns:
            str: The corresponding ticker symbol
        """
        # First check if it's in our reverse mapping
        reverse_mapping = {v: k for k, v in self.ticker_mapping.items()}
        return reverse_mapping.get(asset_id, asset_id.upper())

    def backtest_portfolio(self, portfolio_name, start_date, end_date, initial_capital=10000, alt_cost=0.005, btc_cost=0.001, signal_threshold=75):
        """
        Backtest a portfolio strategy. For BTC-only portfolios, this implements trend following.
        For altcoin portfolios, this tracks signals and decisions but does not manage a portfolio
        (use plot_individual_asset_performance for that).
        
        Args:
            portfolio_name (str): Name of the portfolio to backtest
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            initial_capital (float): Initial capital amount
            alt_cost (float): Transaction cost for altcoin trades (as decimal)
            btc_cost (float): Transaction cost for BTC trades (as decimal)
            signal_threshold (float): Threshold for combined signal to generate buy decision (default: 75)
        """
        if portfolio_name not in self.portfolios:
            self.logger.error(f"Portfolio '{portfolio_name}' does not exist.")
            raise ValueError(f"Portfolio '{portfolio_name}' not found.")
        
        portfolio = self.portfolios[portfolio_name]
        btc_trend_gating = portfolio['btc_trend_gating']
        usd_conditions = portfolio['usd_conditions']
        btc_conditions = portfolio['btc_conditions']
        rsi_conditions_usd = portfolio['rsi_conditions_usd']
        rsi_conditions_btc = portfolio['rsi_conditions_btc']
        btc_only = portfolio['btc_only']
        use_volatility_filter = portfolio.get('use_volatility_filter', False)
        volatility_weight = portfolio.get('volatility_weight', 1.0)
        use_ssr_signal = portfolio.get('use_ssr_signal', False)
        use_ssr_gate = portfolio.get('use_ssr_gate', False)
        btc_rsi_gate = portfolio.get('btc_rsi_gate', False)
        use_btc_rsi_signal = portfolio.get('use_btc_rsi_signal', False)
        follow_portfolio = portfolio.get('follow_portfolio', None)
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        dates = pd.date_range(start_date, end_date, freq='D', inclusive='both')
        
        # Get signals from followed portfolio if applicable
        followed_portfolio_signals = {}
        if follow_portfolio is not None:
            # Check if the followed portfolio has backtest results for the date range
            if follow_portfolio not in self.portfolios:
                self.logger.error(f"Portfolio '{follow_portfolio}' does not exist. Cannot follow.")
                raise ValueError(f"Portfolio '{follow_portfolio}' not found.")
                
            followed_backtest = self.portfolios[follow_portfolio].get('backtest_results')
            if followed_backtest is None:
                self.logger.info(f"Running backtest for followed portfolio '{follow_portfolio}'")
                # Run backtest for the followed portfolio first
                self.backtest_portfolio(
                    portfolio_name=follow_portfolio, 
                    start_date=start_date, 
                    end_date=end_date,
                    initial_capital=initial_capital, 
                    alt_cost=alt_cost, 
                    btc_cost=btc_cost,
                    signal_threshold=signal_threshold
                )
                followed_backtest = self.portfolios[follow_portfolio].get('backtest_results')
                
            # Extract signals from the followed portfolio
            followed_signals_df = followed_backtest['signals_df']
            # If it's a BTC-only portfolio, extract decisions for BTC
            if self.portfolios[follow_portfolio]['btc_only']:
                btc_signals = followed_signals_df[followed_signals_df['asset'] == 'BTC']
                for idx, row in btc_signals.iterrows():
                    followed_portfolio_signals[idx.to_pydatetime()] = row['final_decision']
                self.logger.info(f"Using signals from BTC-only portfolio '{follow_portfolio}'")
            # Otherwise extract all asset decisions
            else:
                for idx, row in followed_signals_df.iterrows():
                    date_key = idx.to_pydatetime()
                    if date_key not in followed_portfolio_signals:
                        followed_portfolio_signals[date_key] = {}
                    followed_portfolio_signals[date_key][row['asset']] = row['final_decision']
                self.logger.info(f"Using signals from altcoin portfolio '{follow_portfolio}'")
        
        # Initialize signal trackers
        signal_data = {
            'date': [],
            'asset': [],
            'usd_trend_signal': [],
            'btc_trend_signal': [],
            'rsi_usd_signal': [],
            'rsi_btc_signal': [],
            'volatility_signal': [],
            'ssr_signal': [],
            'ssr_gate': [],
            'btc_gate': [],
            'btc_rsi_gate': [],
            'btc_rsi_signal': [],
            'followed_portfolio_signal': [],
            'combined_signal': [],
            'final_decision': []
        }
        
        # Get BTC data and calculate BTC signals if needed for gating
        btc_signals = {}
        if btc_trend_gating:
            btc_classified = self.asset_data['bitcoin']['classified_data']
            btc_signals = {trend_type: {} for trend_type in btc_trend_gating}
            for date in dates:
                btc_date_data = btc_classified[btc_classified['date'] <= date]
                if not btc_date_data.empty:
                    latest_btc = btc_date_data.iloc[-1]
                    for trend_type in btc_trend_gating:
                        btc_signals[trend_type][date] = str(latest_btc.get(trend_type, 'N/A')) in btc_trend_gating[trend_type]
        
        # Get BTC RSI signals if needed for RSI gating or signaling
        btc_rsi_signals = {}
        if btc_rsi_gate or use_btc_rsi_signal:
            btc_classified = self.asset_data['bitcoin']['classified_data']
            for date in dates:
                btc_date_data = btc_classified[btc_classified['date'] <= date]
                if not btc_date_data.empty:
                    latest_btc = btc_date_data.iloc[-1]
                    # Store the actual RSI signal value (1 or -1) for use as a signal component
                    btc_rsi_signals[date] = latest_btc.get('RSI_Signal_close', 0)
        
        if btc_only:
            btc_position = False
            btc_value = initial_capital  # Initialize with initial capital
            btc_values = pd.Series(index=dates, dtype=float)  # Use Series instead of dict for consistency
            btc_values.iloc[0] = initial_capital  # Set initial capital
            btc_returns = self.asset_data['bitcoin']['raw_data'].set_index('date')['close'].pct_change().reindex(dates, fill_value=0)
            previous_signal = False
            
            for i, date in enumerate(dates):
                day_transaction_cost = 0
                if i == 0:
                    btc_position = False
                    continue  # Skip first day but keep initial capital
                
                btc_gate_passed = True
                if btc_trend_gating:
                    btc_gate_passed = all(btc_signals[trend_type][date] for trend_type in btc_trend_gating)
                
                btc_classified = self.asset_data['bitcoin']['classified_data']
                btc_date_data = btc_classified[btc_classified['date'] <= date]
                if btc_date_data.empty:
                    btc_values.iloc[i] = btc_values.iloc[i-1]  # Use iloc for Series
                    continue
                
                latest_btc = btc_date_data.iloc[-1]
                usd_condition = all(str(latest_btc.get(trend_type, 'N/A')) in allowed for trend_type, allowed in usd_conditions.items()) if usd_conditions else True
                rsi_usd_signal = latest_btc.get('RSI_Signal_close', 0) if rsi_conditions_usd else 1
                
                # Get volatility signal if requested
                volatility_signal = 0
                if use_volatility_filter and self.markov_model is not None:
                    volatility_signal = self._get_volatility_state(date)
                
                # Get SSR signal if requested
                ssr_signal = 0
                if use_ssr_signal:
                    ssr_signal = self._get_ssr_signal(date)
                
                # Calculate BTC gate (0 or 1 multiplier)
                btc_gate_multiplier = 1 if btc_gate_passed else 0
                
                # Calculate USD trend signal
                usd_trend_signal = 1 if usd_condition else -1 if usd_conditions else 0
                
                # Calculate combined signal with all components
                signal_components = []
                
                # Add trend signal only if conditions were specified and passed
                if usd_conditions and usd_trend_signal > 0:
                    signal_components.append(1)  # Trend conditions passed
                    
                # Add RSI signal if required
                if rsi_conditions_usd:
                    rsi_signal = latest_btc.get('RSI_Signal_close', 0)
                    signal_components.append(rsi_signal)
                    
                # Add volatility signal with weight if enabled
                if use_volatility_filter:
                    signal_components.append(volatility_signal * volatility_weight)
                    
                # Add SSR signal if enabled
                if use_ssr_signal:
                    signal_components.append(ssr_signal)
                
                # Calculate combined signal with BTC gating as multiplier
                combined_signal = sum(signal_components)
                if btc_trend_gating:
                    combined_signal *= btc_gate_multiplier
                
                # Get signal from followed portfolio if applicable
                followed_portfolio_signal = None
                if follow_portfolio is not None and date in followed_portfolio_signals:
                    followed_portfolio_signal = followed_portfolio_signals[date]
                    # Override combined signal with followed portfolio signal if following
                    # This ensures we make the same decisions as the followed portfolio
                    current_signal = followed_portfolio_signal > 0
                else:
                    current_signal = combined_signal > 0
                
                # Record signals
                signal_data['date'].append(date)
                signal_data['asset'].append('BTC')
                signal_data['usd_trend_signal'].append(usd_trend_signal if usd_conditions else None)
                signal_data['btc_trend_signal'].append(None)  # Always None for BTC-only portfolios
                signal_data['rsi_usd_signal'].append(rsi_usd_signal if rsi_conditions_usd else None)
                signal_data['rsi_btc_signal'].append(None)  # Not applicable for BTC
                signal_data['volatility_signal'].append(volatility_signal if use_volatility_filter else None)
                signal_data['ssr_signal'].append(ssr_signal if use_ssr_signal else None)
                signal_data['ssr_gate'].append(None)  # Not applicable for BTC
                signal_data['btc_gate'].append(btc_gate_multiplier if btc_trend_gating else None)
                signal_data['btc_rsi_gate'].append(None)  # Not applicable for BTC-only portfolio
                signal_data['btc_rsi_signal'].append(None)  # Not applicable for BTC-only portfolio
                signal_data['followed_portfolio_signal'].append(followed_portfolio_signal)
                signal_data['combined_signal'].append(combined_signal)
                signal_data['final_decision'].append(1 if current_signal else 0)
                
                if i > 1:
                    new_btc_position = previous_signal
                    if new_btc_position != btc_position:
                        prev_value = btc_values.iloc[i-1]  # Use iloc for Series
                        if new_btc_position:
                            day_transaction_cost = btc_cost * prev_value
                            self.logger.info(f"Entering BTC position on {date}: Cost = {day_transaction_cost:.2f}")
                        else:
                            day_transaction_cost = btc_cost * prev_value
                            self.logger.info(f"Exiting BTC position on {date}: Cost = {day_transaction_cost:.2f}")
                        btc_position = new_btc_position
                
                previous_signal = current_signal
                if btc_position:
                    btc_daily_return = btc_returns[date]
                    if btc_value == initial_capital:  # Changed condition
                        btc_value = btc_values.iloc[i-1] - day_transaction_cost  # Use iloc for Series
                    else:
                        btc_value *= (1 + btc_daily_return)
                    btc_values.iloc[i] = btc_value  # Use iloc for Series
                else:
                    btc_values.iloc[i] = btc_values.iloc[i-1] - day_transaction_cost  # Use iloc for Series
            
            # Create results DataFrame for BTC-only portfolio
            results_df = pd.DataFrame({
                'Date': dates,
                'Portfolio_Value': btc_values.values  # Use values from Series
            }).set_index('Date')
            
            # Calculate performance metrics
            total_return = (results_df['Portfolio_Value'].iloc[-1] / initial_capital) - 1
            max_drawdown = self._max_drawdown(results_df['Portfolio_Value'])
            
            backtest_results = {
                'results_df': results_df,
                'signals_df': pd.DataFrame(signal_data).set_index('date'),
                'metrics': {
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'initial_capital': initial_capital,
                    'final_value': results_df['Portfolio_Value'].iloc[-1],
                    'sharpe_ratio': self.calculate_sharpe_sortino(results_df, 'Portfolio_Value')[0],
                    'sortino_ratio': self.calculate_sharpe_sortino(results_df, 'Portfolio_Value')[1],
                }
            }
        else:
            # For altcoin portfolios, just track signals
            for date in dates:
                for asset_id in self.asset_ids:
                    if asset_id == 'bitcoin':
                        continue
                        
                    classified_data = self.asset_data.get(asset_id, {}).get('classified_data', pd.DataFrame())
                    if classified_data.empty:
                        continue
                        
                    asset_date_data = classified_data[classified_data['date'] <= date]
                    if asset_date_data.empty:
                        continue
                        
                    ticker = self._get_ticker_from_id(asset_id)
                    
                    latest = asset_date_data.iloc[-1]
                    usd_condition = all(str(latest.get(trend_type, 'N/A')) in allowed for trend_type, allowed in usd_conditions.items()) if usd_conditions else True
                    btc_condition = all(str(latest.get(trend_type, 'N/A')) in allowed for trend_type, allowed in btc_conditions.items()) if btc_conditions else True
                    rsi_usd_signal = latest.get('RSI_Signal_close', 0) if rsi_conditions_usd else 1
                    rsi_btc_signal = latest.get(f'RSI_Signal_{asset_id}_btc', 0) if rsi_conditions_btc else 1
                    
                    # Calculate USD and BTC trend signals separately
                    usd_trend_signal = 1 if usd_condition else -1 if usd_conditions else 0
                    btc_trend_signal = 1 if btc_condition else -1 if btc_conditions else 0
                    
                    # Get volatility signal if requested
                    volatility_signal = 0
                    if use_volatility_filter and self.markov_model is not None:
                        volatility_signal = self._get_volatility_state(date)
                    
                    # Get SSR signal if requested
                    ssr_signal = 0
                    ssr_gate_multiplier = 1  # Default to 1 (pass-through)
                    if use_ssr_signal or use_ssr_gate:
                        ssr_signal = self._get_ssr_signal(date)
                        if use_ssr_gate:
                            ssr_gate_multiplier = 1 if ssr_signal > 0 else 0
                    
                    # Calculate BTC gate as a multiplier (0 or 1)
                    btc_gate_passed = True
                    if btc_trend_gating:
                        btc_gate_passed = all(btc_signals[trend_type][date] for trend_type in btc_trend_gating)
                    btc_gate_multiplier = 1 if btc_gate_passed else 0
                    
                    # Calculate BTC RSI gate as a multiplier (0 or 1)
                    btc_rsi_gate_multiplier = 1  # Default to 1 (pass-through)
                    if btc_rsi_gate and date in btc_rsi_signals:
                        btc_rsi_gate_multiplier = 1 if btc_rsi_signals[date] > 0 else 0
                    
                    # Get BTC RSI signal if requested as a component
                    btc_rsi_signal = 0
                    if use_btc_rsi_signal and date in btc_rsi_signals:
                        btc_rsi_signal = btc_rsi_signals[date]
                    
                    # Calculate combined signal
                    signal_components = []
                    
                    # Add trend signals if conditions were specified
                    if usd_conditions:
                        signal_components.append(usd_trend_signal)
                    if btc_conditions:
                        signal_components.append(btc_trend_signal)
                    
                    # Add RSI signals if enabled
                    if rsi_conditions_usd:
                        signal_components.append(rsi_usd_signal)
                    if rsi_conditions_btc:
                        signal_components.append(rsi_btc_signal)
                    
                    # Add BTC RSI signal if enabled as a component
                    if use_btc_rsi_signal:
                        signal_components.append(btc_rsi_signal)
                    
                    # Add volatility signal if enabled
                    if use_volatility_filter:
                        signal_components.append(volatility_signal)
                        
                    # Add SSR signal if enabled
                    if use_ssr_signal:
                        signal_components.append(ssr_signal)
                    
                    # Calculate percentage of positive signals
                    if signal_components:
                        positive_signals = sum(1 for signal in signal_components if signal > 0)
                        combined_signal = (positive_signals / len(signal_components)) * 100
                    else:
                        combined_signal = 0
                    
                    # Apply gates
                    if btc_trend_gating:
                        combined_signal *= btc_gate_multiplier
                    if use_ssr_gate:
                        combined_signal *= ssr_gate_multiplier
                    if btc_rsi_gate:
                        combined_signal *= btc_rsi_gate_multiplier
                    
                    # Get signal from followed portfolio if applicable
                    followed_portfolio_signal = None
                    # Calculate the altcoin's own decision based on its signals
                    altcoin_decision = 1 if combined_signal > signal_threshold else 0
                    
                    # If following a BTC-only portfolio, use its decision for Bitcoin
                    if follow_portfolio is not None and date in followed_portfolio_signals:
                        if isinstance(followed_portfolio_signals[date], dict):
                            # If following an altcoin portfolio, use asset-specific decision if available
                            if ticker in followed_portfolio_signals[date]:
                                followed_portfolio_signal = followed_portfolio_signals[date][ticker]
                            else:
                                followed_portfolio_signal = 0  # No signal for this asset
                        else:
                            # If following a BTC-only portfolio, use the BTC decision for all altcoins
                            followed_portfolio_signal = followed_portfolio_signals[date]
                            
                        # Use followed portfolio as a gate - both signals must be positive
                        # This preserves the altcoin's own signal while gating with the followed portfolio
                        final_decision = 1 if (altcoin_decision == 1 and followed_portfolio_signal == 1) else 0
                    else:
                        # If not following a portfolio, just use the altcoin's own decision
                        final_decision = altcoin_decision
                    
                    # Record signals
                    signal_data['date'].append(date)
                    signal_data['asset'].append(ticker)
                    signal_data['usd_trend_signal'].append(usd_trend_signal if usd_conditions else None)
                    signal_data['btc_trend_signal'].append(btc_trend_signal if btc_conditions else None)
                    signal_data['rsi_usd_signal'].append(rsi_usd_signal if rsi_conditions_usd else None)
                    signal_data['rsi_btc_signal'].append(rsi_btc_signal if rsi_conditions_btc else None)
                    signal_data['volatility_signal'].append(volatility_signal if use_volatility_filter else None)
                    signal_data['ssr_signal'].append(ssr_signal if use_ssr_signal else None)
                    signal_data['ssr_gate'].append(ssr_gate_multiplier if use_ssr_gate else None)
                    signal_data['btc_gate'].append(btc_gate_multiplier if btc_trend_gating else None)
                    signal_data['btc_rsi_gate'].append(btc_rsi_gate_multiplier if btc_rsi_gate else None)
                    signal_data['btc_rsi_signal'].append(btc_rsi_signal if use_btc_rsi_signal else None)
                    signal_data['followed_portfolio_signal'].append(followed_portfolio_signal)
                    signal_data['combined_signal'].append(combined_signal)
                    signal_data['final_decision'].append(final_decision)
            
            # Create empty results DataFrame for altcoin portfolio (signals only)
            results_df = pd.DataFrame(index=dates)
            
            backtest_results = {
                'results_df': results_df,
                'signals_df': pd.DataFrame(signal_data).set_index('date'),
                'metrics': {
                    'total_return': 0,
                    'max_drawdown': 0,
                    'initial_capital': initial_capital,
                    'final_value': initial_capital,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                }
            }
        
        self.portfolios[portfolio_name]['backtest_results'] = backtest_results
        return backtest_results

    def plot_portfolio_with_volatility(self, portfolio_names, btc_trend_portfolio_name, start_date, end_date, initial_capital=1000, show_plot=True):
        """
        Plot performance of portfolios against BTC buy-and-hold, including volatility regime visualization.
        
        Args:
            portfolio_names (str or list): Name(s) of the portfolio(s) to plot
            btc_trend_portfolio_name (str): Name of the BTC trend-following portfolio
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            initial_capital (float): Initial capital amount (default: 1000)
            show_plot (bool): If True, display the plot; if False, return the figure without displaying
            
        Returns:
            plotly.graph_objects.Figure: The generated plot
        """
        # Ensure we have a model loaded
        if self.markov_model is None:
            self.logger.error("No volatility model loaded. Cannot visualize volatility regimes.")
            raise ValueError("This method requires a loaded MarkovVolatility model.")
            
        if isinstance(portfolio_names, str):
            portfolio_names = [portfolio_names]
            
        # Validate portfolio names
        all_portfolios = portfolio_names + [btc_trend_portfolio_name]
        for portfolio_name in all_portfolios:
            if portfolio_name not in self.portfolios:
                self.logger.error(f"Portfolio '{portfolio_name}' does not exist.")
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
        
        # Ensure btc_trend_portfolio_name is a BTC-only portfolio
        if not self.portfolios[btc_trend_portfolio_name]['btc_only']:
            self.logger.error(f"Portfolio '{btc_trend_portfolio_name}' must be a BTC-only portfolio.")
            raise ValueError(f"Portfolio '{btc_trend_portfolio_name}' must be created with btc_only=True")
        
        # Recalculate backtests for the specified date range
        for portfolio_name in all_portfolios:
            self.logger.info(f"Recalculating backtest for portfolio '{portfolio_name}' from {start_date} to {end_date}.")
            self.backtest_portfolio(
                portfolio_name=portfolio_name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                alt_cost=0.005  # Assuming default transaction cost
            )
        
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Get BTC data for buy-and-hold comparison
        btc_data = self.asset_data.get('bitcoin', {}).get('raw_data', pd.DataFrame())
        if btc_data.empty:
            self.logger.error("Bitcoin data not available for comparison.")
            raise ValueError("Bitcoin data required for comparison.")
        
        # Calculate BTC buy-and-hold
        btc_price_series = btc_data.set_index('date')['close'].reindex(dates, method='ffill')
        if btc_price_series.empty:
            self.logger.error("No Bitcoin price data available for the date range.")
            raise ValueError("Bitcoin price data required for the date range.")
        
        # Calculate returns from price series
        btc_returns = btc_price_series.pct_change().fillna(0)
        
        # Initialize buy-and-hold portfolio with initial capital on first day
        btc_buy_hold = pd.Series(index=dates)
        btc_buy_hold.iloc[0] = initial_capital  # Explicitly set first day to initial capital
        
        # Calculate cumulative value for remaining days
        for i in range(1, len(dates)):
            btc_buy_hold.iloc[i] = btc_buy_hold.iloc[i-1] * (1 + btc_returns.iloc[i])
        
        # Get volatility states for all dates
        volatility_states = {}
        for date in dates:
            volatility_states[date] = self._get_volatility_state(date)
        
        # Create plot with subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            row_heights=[0.8, 0.2],
                            specs=[[{"type": "scatter"}],
                                   [{"type": "scatter"}]],
                            subplot_titles=("Portfolio Performance", "Volatility Signal"))
        
        # Add BTC buy-and-hold
        fig.add_trace(go.Scatter(x=dates, y=btc_buy_hold, mode='lines', 
                                name='BTC Buy & Hold', line=dict(color='blue')),
                     row=1, col=1)
        
        # Add BTC trend-following portfolio
        btc_trend_backtest = self.portfolios[btc_trend_portfolio_name]['backtest_results']
        btc_trend_value = btc_trend_backtest['results_df']['Portfolio_Value'].reindex(dates, method='ffill')
        fig.add_trace(go.Scatter(x=dates, y=btc_trend_value, mode='lines', 
                                name=f'BTC Trend: {btc_trend_portfolio_name}', line=dict(color='green')),
                     row=1, col=1)
        
        # Add each portfolio
        colors = ['orange', 'purple', 'red', 'cyan', 'magenta']  # Add more colors if needed
        for idx, portfolio_name in enumerate(portfolio_names):
            backtest = self.portfolios[portfolio_name]['backtest_results']
            results_df = backtest['results_df']
            portfolio_value = results_df['Portfolio_Value'].reindex(dates, method='ffill')
            fig.add_trace(go.Scatter(x=dates, y=portfolio_value, mode='lines', 
                                    name=f'Portfolio: {portfolio_name}', 
                                    line=dict(color=colors[idx % len(colors)])),
                         row=1, col=1)
        
        # Add volatility state to the second subplot
        vol_states_df = pd.DataFrame({'state': volatility_states}, index=dates)
        
        # Add low volatility periods as green background
        for i in range(len(vol_states_df)):
            if i == 0 or vol_states_df['state'].iloc[i] != vol_states_df['state'].iloc[i-1]:
                start_date = vol_states_df.index[i]
                state = vol_states_df['state'].iloc[i]
                
                # Find end date of current state
                j = i
                while j < len(vol_states_df)-1 and vol_states_df['state'].iloc[j] == vol_states_df['state'].iloc[j+1]:
                    j += 1
                end_date = vol_states_df.index[j]
                
                # Color based on volatility state
                color = 'rgba(0,255,0,0.3)' if state == 1 else 'rgba(255,0,0,0.3)'  # Green for low vol, red for high vol
                
                # Add colored background for this volatility period
                fig.add_vrect(
                    x0=start_date, x1=end_date,
                    fillcolor=color, opacity=0.5,
                    layer="below", line_width=0,
                    row=1, col=1
                )
        
        # Add volatility signal line (1 for low, -1 for high)
        fig.add_trace(go.Scatter(x=dates, y=vol_states_df['state'], mode='lines',
                               name='Volatility Signal (1=Low, -1=High)', 
                               line=dict(color='black', width=2)),
                     row=2, col=1)
        
        # Add horizontal line at 0 for reference
        fig.add_trace(go.Scatter(x=dates, y=[0]*len(dates), mode='lines',
                               name='Neutral', line=dict(color='gray', dash='dash')),
                     row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance with Volatility Regimes',
            xaxis_title='Date',
            yaxis_title='Value',
            yaxis2_title='Signal',
            legend=dict(x=0, y=1),
            template='plotly_white',
            height=800
        )
        
        # Display the plot if show_plot is True
        if show_plot:
            fig.show()
        
        # Display metrics for each portfolio
        print("\nPerformance Metrics:")
        print(f"BTC Buy & Hold: Return = {(btc_buy_hold.iloc[-1] / initial_capital - 1):.2%}, "
              f"Max Drawdown = {self._max_drawdown(btc_buy_hold):.2%}")
        
        btc_trend_metrics = btc_trend_backtest['metrics']
        print(f"BTC Trend ({btc_trend_portfolio_name}): Return = {btc_trend_metrics['total_return']:.2%}, "
              f"Max Drawdown = {btc_trend_metrics['max_drawdown']:.2%}")
        
        for portfolio_name in portfolio_names:
            backtest = self.portfolios[portfolio_name]['backtest_results']
            metrics = backtest['metrics']
            print(f"\nPortfolio '{portfolio_name}':")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Total Transaction Costs: {metrics['total_costs']:.2f}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        
        return fig

    # Add a new helper method to get volatility states
    def _get_volatility_state(self, date):
        """
        Get volatility state for a specific date.
        
        Args:
            date: Date to get volatility state for
            
        Returns:
            int: +1 for low volatility (desired), -1 for high volatility (undesired)
        """
        if self.markov_model is None:
            return 0  # Neutral if no model is available
            
        # Convert date to string format if needed
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
            
        # Use cache to avoid redundant calls
        if date_str in self.volatility_cache:
            state = self.volatility_cache[date_str]
        else:
            try:
                state = self.markov_model.get_volatility_state(date_str)
                self.volatility_cache[date_str] = state
            except Exception as e:
                self.logger.warning(f"Error getting volatility state for {date_str}: {e}")
                return 0  # Neutral on error
                
        # Convert state to signal: low volatility (+1), high volatility (-1)
        return 1 if state == 'low' else -1

    def plot_individual_asset_performance(self, portfolio_name, btc_trend_portfolio_name, start_date, end_date, initial_capital=10000, asset_filter=None, show_plot=True):
        """
        Plot performance of individual assets when selected by the strategy, alongside BTC buy & hold and trend following.
        Each asset maintains its own continuous portfolio value, starting from initial_capital, with daily returns applied.
        
        Args:
            portfolio_name (str): Name of the portfolio to analyze
            btc_trend_portfolio_name (str): Name of the BTC trend-following portfolio
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            initial_capital (float): Initial capital amount (default: 10000)
            asset_filter (list, optional): List of asset tickers to include in the plot
            show_plot (bool): If True, display the plot
        
        Returns:
            plotly.graph_objects.Figure: The generated plot
        """
        if portfolio_name not in self.portfolios:
            self.logger.error(f"Portfolio '{portfolio_name}' does not exist.")
            raise ValueError(f"Portfolio '{portfolio_name}' not found")
            
        if btc_trend_portfolio_name not in self.portfolios:
            self.logger.error(f"Portfolio '{btc_trend_portfolio_name}' does not exist.")
            raise ValueError(f"Portfolio '{btc_trend_portfolio_name}' not found")
            
        # Ensure btc_trend_portfolio_name is a BTC-only portfolio
        if not self.portfolios[btc_trend_portfolio_name]['btc_only']:
            self.logger.error(f"Portfolio '{btc_trend_portfolio_name}' must be a BTC-only portfolio.")
            raise ValueError(f"Portfolio '{btc_trend_portfolio_name}' must be created with btc_only=True")
        
        # Recalculate backtests for the specified date range
        self.logger.info(f"Recalculating backtest for portfolio '{portfolio_name}' from {start_date} to {end_date}.")
        self.backtest_portfolio(
            portfolio_name=portfolio_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            alt_cost=0.005  # 0.5% transaction cost
        )
        
        self.logger.info(f"Recalculating backtest for portfolio '{btc_trend_portfolio_name}' from {start_date} to {end_date}.")
        self.backtest_portfolio(
            portfolio_name=btc_trend_portfolio_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            btc_cost=0.001  # 0.1% transaction cost
        )
        
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Get BTC data for buy-and-hold comparison
        btc_data = self.asset_data.get('bitcoin', {}).get('raw_data', pd.DataFrame())
        if btc_data.empty:
            self.logger.error("Bitcoin data not available for comparison.")
            raise ValueError("Bitcoin data required for comparison.")
        
        # Calculate BTC buy-and-hold
        btc_price_series = btc_data.set_index('date')['close'].reindex(dates, method='ffill')
        btc_returns = btc_price_series.pct_change().fillna(0)
        btc_buy_hold = pd.Series(index=dates, dtype=float)
        btc_buy_hold.iloc[0] = initial_capital
        for i in range(1, len(dates)):
            btc_buy_hold.iloc[i] = btc_buy_hold.iloc[i-1] * (1 + btc_returns.iloc[i])
        
        # Get BTC trend-following portfolio performance
        btc_trend_backtest = self.portfolios[btc_trend_portfolio_name]['backtest_results']
        btc_trend_value = btc_trend_backtest['results_df']['Portfolio_Value'].reindex(dates, method='ffill')
        
        # Get signals from the strategy
        strategy_backtest = self.portfolios[portfolio_name]['backtest_results']
        signals_df = strategy_backtest['signals_df']
        
        # Create plot
        fig = go.Figure()
        
        # Add BTC buy-and-hold
        fig.add_trace(go.Scatter(x=dates, y=btc_buy_hold, mode='lines', 
                                name='BTC Buy & Hold', line=dict(color='blue')))
        
        # Add BTC trend-following portfolio
        fig.add_trace(go.Scatter(x=dates, y=btc_trend_value, mode='lines', 
                                name=f'BTC Trend: {btc_trend_portfolio_name}', line=dict(color='green')))
        
        # Get unique assets that were selected at any point
        selected_assets = signals_df['asset'].unique()
        selected_assets = [asset for asset in selected_assets if asset != 'BTC']
        
        # Apply asset filter if provided
        if asset_filter is not None:
            asset_filter = [ticker.upper() for ticker in asset_filter]
            selected_assets = [asset for asset in selected_assets if asset in asset_filter]
            if not selected_assets:
                self.logger.warning("No selected assets match the provided filter.")
                return fig
        
        # Store metrics and trade history for each asset
        asset_metrics = {}
        trade_history = {}
        
        # Calculate and plot performance for each selected asset
        colors = ['orange', 'purple', 'red', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
        for idx, asset in enumerate(selected_assets):
            # Get dates when this asset was selected
            asset_signals = signals_df[signals_df['asset'] == asset]
            selected_dates = asset_signals[asset_signals['final_decision'] == 1].index
            
            if len(selected_dates) > 0:
                # Get asset price data
                asset_id = self.ticker_mapping.get(asset, asset.lower())
                asset_data = self.asset_data.get(asset_id, {}).get('raw_data', pd.DataFrame())
                if not asset_data.empty:
                    # Calculate asset prices and daily returns
                    asset_price = asset_data.set_index('date')['close'].reindex(dates, method='ffill')
                    asset_returns = asset_price.pct_change().fillna(0)
                    
                    # Initialize tracking variables
                    trade_count = 0
                    total_costs = 0
                    in_position = False
                    daily_returns = []
                    current_trade = None
                    trades = []
                    
                    # Initialize portfolio value tracking
                    portfolio_value = pd.Series(index=dates, dtype=float)
                    portfolio_value.iloc[0] = initial_capital
                    current_value = initial_capital
                    
                    # Track the initial capital invested for each trade
                    initial_invested = initial_capital
                    
                    # Track the value through time
                    for i in range(1, len(dates)):
                        current_date = dates[i]
                        was_in_position = in_position
                        in_position = current_date in selected_dates
                        
                        # Check for position changes
                        if in_position and not was_in_position:
                            # Enter new position
                            trade_count += 1
                            
                            # Calculate and apply entry cost
                            entry_cost = 0.005 * current_value
                            total_costs += entry_cost
                            initial_invested = current_value  # Record the capital before cost
                            current_value -= entry_cost
                            
                            # Record entry price data
                            entry_price = asset_price.iloc[i]
                            original_entry_value = current_value  # Value after costs
                            
                            # Record trade entry data
                            current_trade = {
                                'trade_id': trade_count,
                                'entry_date': current_date,
                                'exit_date': None,
                                'holding_days': 0,
                                'entry_price': entry_price,
                                'exit_price': None,
                                'price_return': None,
                                'entry_value': original_entry_value,
                                'entry_cost': -entry_cost,
                                'exit_value': None,
                                'exit_cost': None,
                                'trade_return': None,
                                'is_open': True
                            }
                        
                        # Apply daily returns during holding period
                        if in_position:
                            daily_return = asset_returns.iloc[i]
                            current_value *= (1 + daily_return)
                            daily_returns.append(daily_return)
                        
                        # Check for position exit
                        if was_in_position and not in_position:
                            # Exit existing position
                            exit_price = asset_price.iloc[i]
                            
                            if current_trade is not None:
                                # Calculate price return
                                price_return = (exit_price / current_trade['entry_price']) - 1
                                
                                # Calculate the expected portfolio value based on price_return
                                expected_value = current_trade['entry_value'] * (1 + price_return)
                                
                                # Adjust current_value to match the expected value (correct for any drift)
                                current_value = expected_value
                                
                                # Calculate and apply exit cost
                                exit_cost = 0.005 * current_value
                                total_costs += exit_cost
                                current_value -= exit_cost
                                
                                # Calculate trade return
                                trade_return = (current_value - initial_invested) / initial_invested
                                
                                holding_days = (current_date - current_trade['entry_date']).days
                                
                                current_trade.update({
                                    'exit_date': current_date,
                                    'holding_days': holding_days,
                                    'exit_price': exit_price,
                                    'price_return': price_return,
                                    'exit_value': current_value,
                                    'exit_cost': -exit_cost,
                                    'trade_return': trade_return,
                                    'is_open': False
                                })
                                trades.append(current_trade)
                                current_trade = None
                                initial_invested = current_value  # Reset for the next trade
                        
                        # Update portfolio value for this date
                        portfolio_value.iloc[i] = current_value
                    
                    # Handle open trade at end of period
                    if current_trade is not None:
                        exit_price = asset_price.iloc[-1]
                        price_return = (exit_price / current_trade['entry_price']) - 1
                        
                        # Adjust current_value to match the price_return
                        current_value = current_trade['entry_value'] * (1 + price_return)
                        # No exit cost for open trades
                        exit_cost = 0
                        
                        # Calculate trade return
                        trade_return = (current_value - initial_invested) / initial_invested
                        
                        holding_days = (dates[-1] - current_trade['entry_date']).days
                        
                        current_trade.update({
                            'exit_date': dates[-1],
                            'holding_days': holding_days,
                            'exit_price': exit_price,
                            'price_return': price_return,
                            'exit_value': current_value,
                            'exit_cost': 0,
                            'trade_return': trade_return,
                            'is_open': True
                        })
                        trades.append(current_trade)
                    
                    # Store trade history with reordered columns
                    if trades:
                        trades_df = pd.DataFrame(trades)
                        column_order = [
                            'trade_id', 'entry_date', 'exit_date', 'holding_days',
                            'entry_price', 'exit_price', 'price_return',
                            'entry_value', 'entry_cost', 'exit_value', 'exit_cost',
                            'trade_return', 'is_open'
                        ]
                        trade_history[asset] = trades_df[column_order]
                    
                    # Calculate metrics
                    total_return = (portfolio_value.iloc[-1] / initial_capital) - 1
                    max_drawdown = self._max_drawdown(portfolio_value)
                    
                    # Use the recorded daily returns for metrics
                    daily_returns_series = pd.Series(daily_returns)
                    if len(daily_returns_series) > 1:
                        sharpe = np.sqrt(365) * (daily_returns_series.mean() / daily_returns_series.std())
                        downside_returns = daily_returns_series[daily_returns_series < 0]
                        sortino = np.sqrt(365) * (daily_returns_series.mean() / downside_returns.std()) if len(downside_returns) > 1 else 0
                    else:
                        sharpe, sortino = 0, 0
                    
                    # Calculate win rate and average trade metrics
                    if trades:
                        trades_df = pd.DataFrame(trades)
                        avg_trade_return = trades_df['trade_return'].mean()
                        winning_trades = trades_df[trades_df['trade_return'] > 0]
                        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
                        avg_win = winning_trades['trade_return'].mean() if len(winning_trades) > 0 else 0
                        avg_loss = trades_df[trades_df['trade_return'] <= 0]['trade_return'].mean() if len(trades_df[trades_df['trade_return'] <= 0]) > 0 else 0
                        avg_holding_days = trades_df['holding_days'].mean()
                    else:
                        avg_trade_return = 0
                        win_rate = avg_win = avg_loss = avg_holding_days = 0
                    
                    asset_metrics[asset] = {
                        'Total Return': total_return,
                        'Max Drawdown': max_drawdown,
                        'Sharpe Ratio': sharpe,
                        'Sortino Ratio': sortino,
                        'Number of Trades': trade_count,
                        'Win Rate': win_rate,
                        'Avg Win': avg_win,
                        'Avg Loss': avg_loss,
                        'Avg Holding Days': avg_holding_days,
                        'Total Costs': total_costs,
                        'First Trade': selected_dates[0],
                        'Last Trade': selected_dates[-1],
                        'Trading Days': len(selected_dates)
                    }
                    
                    # Add to plot with average trade return in the name
                    fig.add_trace(go.Scatter(x=dates, y=portfolio_value, mode='lines', 
                                        name=f'{asset} ({trade_count} trades, Avg Trade Return: {avg_trade_return:.1%})', 
                                        line=dict(color=colors[idx % len(colors)])))
        
        # Store trade history and metrics in the portfolio
        self.portfolios[portfolio_name]['trade_history'] = trade_history
        self.portfolios[portfolio_name]['asset_metrics'] = asset_metrics
        
        # Update layout
        fig.update_layout(
            title='Individual Asset Performance When Selected by Strategy',
            xaxis_title='Date',
            yaxis_title='Value',
            legend=dict(x=0, y=1),
            template='plotly_white'
        )
        
        # Display the plot if show_plot is True
        if show_plot:
            fig.show()
        
        # Display metrics
        print("\nPerformance Metrics:")
        print(f"BTC Buy & Hold: Return = {(btc_buy_hold.iloc[-1] / initial_capital - 1):.2%}, "
            f"Max Drawdown = {self._max_drawdown(btc_buy_hold):.2%}")
        
        btc_trend_metrics = btc_trend_backtest['metrics']
        print(f"BTC Trend ({btc_trend_portfolio_name}): Return = {btc_trend_metrics['total_return']:.2%}, "
            f"Max Drawdown = {btc_trend_metrics['max_drawdown']:.2%}")
        
        print("\nIndividual Asset Performance:")
        metrics_df = pd.DataFrame.from_dict(asset_metrics, orient='index')
        metrics_df = metrics_df.sort_values('Total Return', ascending=False)
        
        # Format metrics for display
        metrics_df['Total Return'] = metrics_df['Total Return'].map('{:.2%}'.format)
        metrics_df['Max Drawdown'] = metrics_df['Max Drawdown'].map('{:.2%}'.format)
        metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].map('{:.2f}'.format)
        metrics_df['Sortino Ratio'] = metrics_df['Sortino Ratio'].map('{:.2f}'.format)
        metrics_df['Win Rate'] = metrics_df['Win Rate'].map('{:.2%}'.format)
        metrics_df['Avg Win'] = metrics_df['Avg Win'].map('{:.2%}'.format)
        metrics_df['Avg Loss'] = metrics_df['Avg Loss'].map('{:.2%}'.format)
        metrics_df['Avg Holding Days'] = metrics_df['Avg Holding Days'].map('{:.1f}'.format)
        metrics_df['Total Costs'] = metrics_df['Total Costs'].map('${:,.2f}'.format)
        
        print("\n" + tabulate(metrics_df, headers='keys', tablefmt='pipe', showindex=True))
        
        # Print detailed trade history for each asset
        print("\nDetailed Trade History:")
        for asset, trades_df in trade_history.items():
            print(f"\n{asset} Trades:")
            # Format trade returns and price returns as percentages
            trades_df['trade_return'] = trades_df['trade_return'].map('{:.2%}'.format)
            trades_df['price_return'] = trades_df['price_return'].map('{:.2%}'.format)
            print(tabulate(trades_df, headers='keys', tablefmt='pipe', showindex=False))
        
        return fig

    def create_roc_based_portfolio(self, portfolio_name, btc_trend_gating=None, usd_conditions=None, btc_conditions=None, 
                             rsi_conditions_usd=False, rsi_conditions_btc=False, btc_only=False, 
                             use_volatility_filter=False, volatility_weight=1.0, use_ssr_signal=False,
                             use_ssr_gate=False, btc_rsi_gate=False, use_btc_rsi_signal=False,
                             follow_portfolio=None, short_term_condition=True, 
                             medium_term_condition=False, long_term_condition=False,
                             trend_conditions=None, use_btc_adjusted=False, **kwargs):
        """
        Create a portfolio that uses only Rate of Change (RoC) metrics for trend analysis.
        Accepts the same parameters as create_portfolio for compatibility.
        
        Args:
            portfolio_name (str): Name to identify the portfolio
            btc_trend_gating (dict, optional): BTC trend condition that gates all others
            usd_conditions (dict, optional): USD trend conditions
            btc_conditions (dict, optional): BTC trend conditions
            rsi_conditions_usd (bool): If True, requires RSI signal vs USD to be 1
            rsi_conditions_btc (bool): If True, requires RSI signal vs BTC to be 1 (altcoins only)
            btc_only (bool): If True, creates a BTC-only portfolio
            use_volatility_filter (bool): If True, include volatility signals in decision making
            volatility_weight (float): Weight of volatility signal (1.0 = equal to other signals)
            use_ssr_signal (bool): If True, include SSR oscillator signal in decision making
            use_ssr_gate (bool): If True, use SSR signal as a gate rather than an additive signal
            btc_rsi_gate (bool): If True, use BTC's RSI signal as a gate for altcoin selection
            use_btc_rsi_signal (bool): If True, include BTC's RSI signal as a component (not a gate)
            follow_portfolio (str, optional): Name of another portfolio to follow signals from
            short_term_condition (bool): If True, include short-term trends in conditions
            medium_term_condition (bool): If True, include medium-term trends in conditions
            long_term_condition (bool): If True, include long-term trends in conditions
            trend_conditions (list): List of acceptable trend conditions (e.g., ["Strong Bull", "Weak Bull"])
                                    Default: ["Strong Bull", "Weak Bull"]
            use_btc_adjusted (bool): If True, use BTC-adjusted prices for altcoins
            
        Returns:
            str: The portfolio name
        """
        if not self.static_computed:
            self.logger.error("Static computations not performed.")
            raise ValueError("Run analyze_multiple_assets() first.")
        
        # If specific trend conditions already provided via usd_conditions or btc_conditions, use those
        if usd_conditions or btc_conditions:
            self.logger.info("Using provided trend conditions from usd_conditions/btc_conditions parameters")
            return self.create_portfolio(
                portfolio_name=portfolio_name,
                btc_trend_gating=btc_trend_gating,
                usd_conditions=usd_conditions,
                btc_conditions=btc_conditions,
                rsi_conditions_usd=rsi_conditions_usd,
                rsi_conditions_btc=rsi_conditions_btc,
                btc_only=btc_only,
                use_volatility_filter=use_volatility_filter,
                volatility_weight=volatility_weight,
                use_ssr_signal=use_ssr_signal,
                use_ssr_gate=use_ssr_gate,
                btc_rsi_gate=btc_rsi_gate,
                use_btc_rsi_signal=use_btc_rsi_signal,
                follow_portfolio=follow_portfolio
            )
            
        # Otherwise, build RoC-based conditions
        if trend_conditions is None:
            trend_conditions = ["Strong Bull", "Weak Bull"]
            
        # Determine which conditions to include based on timeframes
        conditions = {}
        if short_term_condition:
            if use_btc_adjusted and not btc_only:
                conditions["Short Term (BTC)"] = trend_conditions
            else:
                conditions["Short Term (USD)"] = trend_conditions
                
        if medium_term_condition:
            if use_btc_adjusted and not btc_only:
                conditions["Medium Term (BTC)"] = trend_conditions
            else:
                conditions["Medium Term (USD)"] = trend_conditions
                
        if long_term_condition:
            if use_btc_adjusted and not btc_only:
                conditions["Long Term (BTC)"] = trend_conditions
            else:
                conditions["Long Term (USD)"] = trend_conditions
        
        # If no conditions specified, use at least one
        if not conditions:
            if use_btc_adjusted and not btc_only:
                conditions["Short Term (BTC)"] = trend_conditions
            else:
                conditions["Short Term (USD)"] = trend_conditions
        
        # Create portfolio with appropriate conditions
        if btc_only:
            return self.create_portfolio(
                portfolio_name=portfolio_name,
                usd_conditions=conditions,
                btc_only=True,
                rsi_conditions_usd=rsi_conditions_usd,
                use_volatility_filter=use_volatility_filter,
                volatility_weight=volatility_weight,
                use_ssr_signal=use_ssr_signal,
                use_ssr_gate=use_ssr_gate,
                btc_rsi_gate=btc_rsi_gate,
                use_btc_rsi_signal=use_btc_rsi_signal,
                follow_portfolio=follow_portfolio
            )
        else:
            if use_btc_adjusted:
                return self.create_portfolio(
                    portfolio_name=portfolio_name,
                    btc_conditions=conditions,
                    btc_only=False,
                    rsi_conditions_usd=rsi_conditions_usd,
                    rsi_conditions_btc=rsi_conditions_btc,
                    use_volatility_filter=use_volatility_filter,
                    volatility_weight=volatility_weight,
                    use_ssr_signal=use_ssr_signal,
                    use_ssr_gate=use_ssr_gate,
                    btc_rsi_gate=btc_rsi_gate,
                    use_btc_rsi_signal=use_btc_rsi_signal,
                    follow_portfolio=follow_portfolio
                )
            else:
                return self.create_portfolio(
                    portfolio_name=portfolio_name,
                    usd_conditions=conditions,
                    btc_only=False,
                    rsi_conditions_usd=rsi_conditions_usd,
                    rsi_conditions_btc=rsi_conditions_btc,
                    use_volatility_filter=use_volatility_filter,
                    volatility_weight=volatility_weight,
                    use_ssr_signal=use_ssr_signal,
                    use_ssr_gate=use_ssr_gate,
                    btc_rsi_gate=btc_rsi_gate,
                    use_btc_rsi_signal=use_btc_rsi_signal,
                    follow_portfolio=follow_portfolio
                )