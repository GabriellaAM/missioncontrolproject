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

warnings.filterwarnings(action='ignore')

class TrendAnalyzer:
    def __init__(self, asset_ids, data_path, btc_data_path, use_btc_adjusted=True, verbose=True, lookback_days=90, trend_metrics_lookback='all'):
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
            'Long Term': [84, 120, 150, 200, 252, 365]
        }
        self.ticker_mapping = self._get_ticker_mapping()
        self.asset_data = {}
        self.static_computed = False
        self.portfolios = {}  # Dictionary to store different portfolios
        
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler('trend_analyzer.log')]
        )
        self.logger = logging.getLogger(__name__)

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
                    btc_data = btc_data[['date', 'close']].rename(columns={'close': 'btc_close'})
                    btc_data.dropna(subset=['btc_close'], inplace=True)
                    data = data.merge(btc_data, on='date', how='inner')
                    if data.empty:
                        self.logger.warning(f"No overlapping dates between {asset_id} and Bitcoin data after merge.")
                        return pd.DataFrame()
                    data[f'{asset_id}_btc'] = data['close'] / data['btc_close']
                    data = data.drop(columns=['btc_close'], errors='ignore')
                except Exception as e:
                    self.logger.error(f"Error merging Bitcoin data for {asset_id}: {e}")
                    return pd.DataFrame()
            self.logger.info(f"Successfully loaded data for {asset_id} with {len(data)} rows.")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data for {asset_id}: {e}")
            return pd.DataFrame()

    def _calculate_indicators(self, data, price_column, asset_id, suffix=''):
        if price_column not in data.columns or data.empty:
            self.logger.warning(f"Column '{price_column}' not found for {asset_id} or data is empty.")
            return
        data[price_column] = pd.to_numeric(data[price_column], errors='coerce')
        data.dropna(subset=[price_column], inplace=True)
        for term, periods in self.ma_periods.items():
            for period in periods:
                if len(data) >= period:
                    data[f'EMA_{suffix}_{period}'] = data[price_column].rolling(window=period, min_periods=period).mean()
                    data[f'RoC_{suffix}_{period}'] = (data[price_column] - data[price_column].shift(period)) / data[price_column].shift(period) * 100

    def classify_trend(self, mas, rocs, data, suffix):
        n_mas = len(mas.dropna())
        n_rocs = len(rocs.dropna())
        if n_mas == 0 or n_rocs == 0:
            return "Neutral"
        pos_mas = sum(
            1 for col in mas.index 
            if not np.isnan(data[f'EMA_{suffix}_{col.split("_")[-1]}'].iloc[-1]) 
            and not np.isnan(data[f'EMA_{suffix}_{col.split("_")[-1]}'].iloc[-2]) 
            and data[f'EMA_{suffix}_{col.split("_")[-1]}'].iloc[-1] > data[f'EMA_{suffix}_{col.split("_")[-1]}'].iloc[-2]
        ) / n_mas * 100
        pos_rocs = sum(1 for roc in rocs if not np.isnan(roc) and roc > 0) / n_rocs * 100
        avg_pos = (pos_mas + pos_rocs) / 2
        if avg_pos >= 75:
            return "Strong Bull"
        elif avg_pos >= 50:
            return "Weak Bull"
        elif avg_pos <= 25:
            return "Strong Bear"
        elif avg_pos <= 40:
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
        elif overall_pos <= 40:
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

    def create_portfolio(self, portfolio_name, btc_trend_gating=None, usd_conditions=None, btc_conditions=None, btc_only=False):
        """
        Create a new portfolio with specified trend conditions.
        
        Args:
            portfolio_name (str): Name to identify the portfolio
            btc_trend_gating (dict, optional): BTC trend condition that gates all others.
                                              E.g., {'Overall (USD)': ['Strong Bull']}
            usd_conditions (dict, optional): USD trend conditions. E.g., {'Short Term (USD)': ['Strong Bull', 'Weak Bull']}
            btc_conditions (dict, optional): BTC trend conditions. E.g., {'Overall (BTC)': ['Strong Bull']}
            btc_only (bool): If True, the portfolio will only include Bitcoin and follow the specified conditions.
        """
        if not self.static_computed:
            self.logger.error("Static computations not performed.")
            raise ValueError("Run analyze_multiple_assets() first.")
            
        if portfolio_name in self.portfolios:
            self.logger.warning(f"Portfolio {portfolio_name} already exists. Overwriting.")
            
        self.portfolios[portfolio_name] = {
            'btc_trend_gating': btc_trend_gating,
            'usd_conditions': usd_conditions or {},
            'btc_conditions': btc_conditions or {},
            'btc_only': btc_only,
            'creation_date': datetime.now(),
            'backtest_results': None
        }
        
        self.logger.info(f"Portfolio '{portfolio_name}' created with criteria: BTC gating={btc_trend_gating}, "
                         f"USD conditions={usd_conditions}, BTC conditions={btc_conditions}, BTC only={btc_only}")
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

    def backtest_portfolio(self, portfolio_name, start_date, end_date, initial_capital=1000, alt_cost=0.005, btc_cost=0.001):
        if portfolio_name not in self.portfolios:
            self.logger.error(f"Portfolio '{portfolio_name}' does not exist.")
            raise ValueError(f"Portfolio '{portfolio_name}' not found.")
        
        portfolio = self.portfolios[portfolio_name]
        btc_trend_gating = portfolio['btc_trend_gating']
        usd_conditions = portfolio['usd_conditions']
        btc_conditions = portfolio['btc_conditions']
        btc_only = portfolio['btc_only']
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Pre-calculate returns for all assets
        asset_returns = {}
        for asset_id in self.asset_ids:
            if asset_id == 'bitcoin' and not btc_only:
                continue
            raw_data = self.asset_data.get(asset_id, {}).get('raw_data', pd.DataFrame())
            if not raw_data.empty:
                price_series = raw_data.set_index('date')['close'].reindex(dates, method='ffill')
                asset_returns[asset_id] = price_series.pct_change().fillna(0)
        
        portfolio_df = pd.DataFrame(index=dates)
        btc_data = self.asset_data.get('bitcoin', {}).get('classified_data', pd.DataFrame())
        btc_signals = {}
        if btc_trend_gating:
            for trend_type, allowed_trends in btc_trend_gating.items():
                trend_series = btc_data.set_index('date')[trend_type].reindex(dates, method='ffill')
                btc_signals[trend_type] = trend_series.isin(allowed_trends)
        
        portfolio_values = {date: initial_capital for date in dates}
        portfolio_composition = {date: [] for date in dates}
        transaction_costs = {date: 0 for date in dates}
        assets_held = {date: 0 for date in dates}
        
        if btc_only:
            # BTC-only portfolio logic (unchanged for brevity, but ensure it works)
            btc_position = False
            btc_value = 0
            btc_values = {date: 0 for date in dates}
            btc_returns = self.asset_data['bitcoin']['raw_data'].set_index('date')['close'].pct_change().reindex(dates, fill_value=0)
            previous_signal = False
            
            for i, date in enumerate(dates):
                day_transaction_cost = 0
                if i == 0:
                    btc_position = False
                    btc_values[date] = 0
                    portfolio_composition[date] = []
                    transaction_costs[date] = 0
                    assets_held[date] = 0
                    portfolio_values[date] = initial_capital
                    continue
                
                btc_gate_passed = True
                if btc_trend_gating:
                    btc_gate_passed = all(btc_signals[trend_type][date] for trend_type in btc_trend_gating)
                
                btc_classified = self.asset_data['bitcoin']['classified_data']
                btc_date_data = btc_classified[btc_classified['date'] <= date]
                if btc_date_data.empty:
                    btc_values[date] = btc_values[dates[i-1]]
                    portfolio_values[date] = portfolio_values[dates[i-1]]
                    portfolio_composition[date] = []
                    assets_held[date] = 0
                    continue
                
                latest_btc = btc_date_data.iloc[-1]
                usd_condition = all(str(latest_btc.get(trend_type, 'N/A')) in allowed for trend_type, allowed in usd_conditions.items()) if usd_conditions else True
                btc_condition = all(str(latest_btc.get(trend_type, 'N/A')) in allowed for trend_type, allowed in btc_conditions.items()) if btc_conditions else True
                
                current_signal = btc_gate_passed and usd_condition and btc_condition
                if i > 1:
                    new_btc_position = previous_signal
                    if new_btc_position != btc_position:
                        prev_value = portfolio_values[dates[i-1]]
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
                    if btc_value == 0:
                        btc_value = portfolio_values[dates[i-1]] - day_transaction_cost
                    else:
                        btc_value *= (1 + btc_daily_return)
                    btc_values[date] = btc_value
                else:
                    btc_values[date] = 0
                
                portfolio_composition[date] = ['BTC'] if btc_position else []
                assets_held[date] = 1 if btc_position else 0
                transaction_costs[date] = day_transaction_cost
                portfolio_values[date] = btc_values[date] if btc_position else (portfolio_values[dates[i-1]] - day_transaction_cost)
        
        else:
            altcoin_positions = {asset_id: False for asset_id in self.asset_ids if asset_id != 'bitcoin'}
            altcoin_values = {asset_id: 0 for asset_id in self.asset_ids if asset_id != 'bitcoin'}
            previous_eligible_assets = []
            
            for i, date in enumerate(dates):
                day_transaction_cost = 0
                if i == 0:
                    portfolio_values[date] = initial_capital
                    portfolio_composition[date] = []
                    transaction_costs[date] = 0
                    assets_held[date] = 0
                    continue
                
                btc_gate_passed = True
                if btc_trend_gating:
                    btc_gate_passed = all(btc_signals[trend_type][date] for trend_type in btc_trend_gating)
                
                current_eligible_assets = []
                for asset_id in self.asset_ids:
                    if asset_id == 'bitcoin':
                        continue
                    classified_data = self.asset_data.get(asset_id, {}).get('classified_data', pd.DataFrame())
                    if classified_data.empty:
                        continue
                    asset_date_data = classified_data[classified_data['date'] <= date]
                    if asset_date_data.empty:
                        continue
                    latest = asset_date_data.iloc[-1]
                    
                    usd_condition = all(str(latest.get(trend_type, 'N/A')) in allowed for trend_type, allowed in usd_conditions.items()) if usd_conditions else True
                    btc_condition = all(str(latest.get(trend_type, 'N/A')) in allowed for trend_type, allowed in btc_conditions.items()) if btc_conditions else True
                    
                    if btc_gate_passed and usd_condition and btc_condition:
                        current_eligible_assets.append(asset_id)
                
                eligible_assets = previous_eligible_assets if i > 1 else []
                num_eligible = len(eligible_assets)
                assets_held[date] = num_eligible
                current_portfolio = sorted([self._get_ticker_from_id(asset_id) for asset_id in eligible_assets])
                prev_portfolio_composition = portfolio_composition[dates[i-1]] if i > 0 else []
                # Fix: Compare lists directly
                portfolio_changed = (num_eligible != len(prev_portfolio_composition)) or (set(current_portfolio) != set(prev_portfolio_composition))
                
                # Apply returns to existing positions
                for asset_id in eligible_assets:
                    if asset_id in asset_returns:
                        daily_return = asset_returns[asset_id][date]
                        if altcoin_values[asset_id] > 0:
                            altcoin_values[asset_id] *= (1 + daily_return)
                            self.logger.debug(f"{date}: {asset_id} return={daily_return:.4f}, new_value={altcoin_values[asset_id]:.2f}")
                
                pre_rebalance_value = sum(altcoin_values.values())
                if portfolio_changed:
                    prev_value = portfolio_values[dates[i-1]]
                    day_transaction_cost = alt_cost * prev_value
                    new_total_value = (prev_value if pre_rebalance_value == 0 else pre_rebalance_value) - day_transaction_cost
                    
                    if num_eligible > 0 and new_total_value > 0:
                        per_asset_value = new_total_value / num_eligible
                        for asset_id in self.asset_ids:
                            if asset_id == 'bitcoin':
                                continue
                            if asset_id in eligible_assets:
                                altcoin_values[asset_id] = per_asset_value * (1 + asset_returns.get(asset_id, pd.Series())[date])
                                self.logger.debug(f"{date}: Enter {asset_id} at {altcoin_values[asset_id]:.2f} with return {asset_returns.get(asset_id, pd.Series())[date]:.4f}")
                            else:
                                altcoin_values[asset_id] = 0
                    elif num_eligible == 0:
                        for asset_id in self.asset_ids:
                            if asset_id != 'bitcoin':
                                altcoin_values[asset_id] = 0
                
                total_alt_value = sum(altcoin_values.values())
                portfolio_composition[date] = current_portfolio  # Store as list
                transaction_costs[date] = day_transaction_cost
                portfolio_values[date] = total_alt_value if num_eligible > 0 else (portfolio_values[dates[i-1]] - day_transaction_cost)
                previous_eligible_assets = current_eligible_assets.copy()
                self.logger.debug(f"{date}: Value={portfolio_values[date]:.2f}, Assets={current_portfolio}, Cost={day_transaction_cost:.2f}")
        
        # Join list into string only when building results_df
        results_df = pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': [portfolio_values[date] for date in dates],
            'Transaction_Cost': [transaction_costs[date] for date in dates],
            'Assets_Held': [assets_held[date] for date in dates],
            'Portfolio_Composition': [', '.join(composition) if composition else 'Cash' for composition in portfolio_composition.values()]
        }).set_index('Date')
        
        returns = results_df['Portfolio_Value'].pct_change().fillna(0)
        total_return = (results_df['Portfolio_Value'].iloc[-1] / initial_capital) - 1
        max_drawdown = self._max_drawdown(results_df['Portfolio_Value'])
        total_costs = results_df['Transaction_Cost'].sum()
        
        backtest_results = {
            'results_df': results_df,
            'metrics': {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'total_costs': total_costs,
                'initial_capital': initial_capital,
                'final_value': results_df['Portfolio_Value'].iloc[-1],
                'sharpe_ratio': self.calculate_sharpe_sortino(results_df, 'Portfolio_Value')[0],
                'sortino_ratio': self.calculate_sharpe_sortino(results_df, 'Portfolio_Value')[1],
            }
        }
        
        self.portfolios[portfolio_name]['backtest_results'] = backtest_results
        self.logger.info(f"Backtest completed: Return={total_return:.2%}, Max DD={max_drawdown:.2%}")
        return backtest_results
    
    def _get_ticker_from_id(self, asset_id):
        """Helper to get ticker from asset ID"""
        for ticker, id_val in self.ticker_mapping.items():
            if id_val == asset_id:
                return ticker
        return asset_id.upper()
        
    def plot_portfolio_performance(self, portfolio_names, btc_trend_portfolio_name, start_date, end_date, initial_capital=1000, show_plot=True):
        """
        Plot performance of one or more portfolios against BTC buy-and-hold and a user-defined BTC trend-following portfolio.
        
        Args:
            portfolio_names (str or list): Name(s) of the portfolio(s) to plot
            btc_trend_portfolio_name (str): Name of the BTC trend-following portfolio
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            initial_capital (float): Initial capital amount (default: 1000)
            show_plot (bool): If True, display the plot; if False, return the figure without displaying (default: True)
            
        Returns:
            plotly.graph_objects.Figure: The generated plot
        """
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
        
        # Plot
        fig = go.Figure()
        
        # Add BTC buy-and-hold
        fig.add_trace(go.Scatter(x=dates, y=btc_buy_hold, mode='lines', 
                                name='BTC Buy & Hold', line=dict(color='blue')))
        
        # Add BTC trend-following portfolio
        btc_trend_backtest = self.portfolios[btc_trend_portfolio_name]['backtest_results']
        btc_trend_value = btc_trend_backtest['results_df']['Portfolio_Value'].reindex(dates, method='ffill')
        fig.add_trace(go.Scatter(x=dates, y=btc_trend_value, mode='lines', 
                                name=f'BTC Trend: {btc_trend_portfolio_name}', line=dict(color='green')))
        
        # Add each portfolio
        colors = ['orange', 'purple', 'red', 'cyan', 'magenta']  # Add more colors if needed
        for idx, portfolio_name in enumerate(portfolio_names):
            backtest = self.portfolios[portfolio_name]['backtest_results']
            results_df = backtest['results_df']
            portfolio_value = results_df['Portfolio_Value'].reindex(dates, method='ffill')
            fig.add_trace(go.Scatter(x=dates, y=portfolio_value, mode='lines', 
                                    name=f'Portfolio: {portfolio_name}', 
                                    line=dict(color=colors[idx % len(colors)])))
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Value',
            legend=dict(x=0, y=1),
            template='plotly_white'
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