import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelBinarizer
from ta import add_all_ta_features
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
from sklearn.feature_selection import SelectKBest, f_classif  
import optuna
import pickle
import os
from datetime import datetime
import glob
import logging

class KMRF:
    """
    Implementation of the KMRF (KAMA+MSR+Random Forest) model for Bitcoin trading,
    based on the paper "Improving Portfolio Performance Using a Novel Method for
    Predicting Financial Regimes" by Piotr Pomorski and Denise Gorse.

    This class:
    - Detects regimes using KAMA+MSR (Markov-Switching Regression for volatility, KAMA for trends).
    - Reclassifies regimes into three labels: Bullish, Bearish, Other.
    - Extracts features (technical, on-chain, and MSR probabilities).
    - Selects features using BorutaShap with PGTS cross-validation.
    - Trains a Random Forest classifier with Optuna hyperparameter optimization using PGTS.
    - Implements a contrarian trading strategy based on RF predictions.
    """

    def __init__(self, price_data_path=None, onchain_data_path=None, train_test_split=0.85, init_date='2018-01-01', embargo_pct=0.01):
        """
        Initialize the KMRF model.

        Parameters:
        -----------
        price_data_path : str, optional
            Path to the CSV file containing Bitcoin price data.
            Default: '/Users/valter.rebelo/MissionControl/data/micro/candleData/bitcoin_candles.csv'
        onchain_data_path : str, optional
            Path to the directory containing on-chain data CSV files.
            Default: '/Users/valter.rebelo/MissionControl/data/onchainData'
        train_test_split : float
            Fraction of data to use for training (default: 0.85).
        init_date : str
            Start date for the data (default: '2018-01-01' to ensure all on-chain features are available).
        embargo_pct : float
            Percentage of data to use as an embargo period (default: 0.01).
        """
        self.price_data_path = price_data_path if price_data_path else '/Users/valter.rebelo/MissionControl/data/micro/candleData/bitcoin_candles.csv'
        self.onchain_data_path = onchain_data_path if onchain_data_path else '/Users/valter.rebelo/MissionControl/data/onchainData'
        self.train_test_split = train_test_split
        self.init_date = init_date  # Set to 2018-01-01 to ensure all on-chain features are available
        self.embargo_pct = embargo_pct
        self.load_data()

    def load_data(self):
        """
        Load and prepare the Bitcoin price and on-chain data.
        Combines price data with on-chain metrics.
        """
        try:
            # Load price data
            price_data = pd.read_csv(self.price_data_path)
            price_data['date'] = pd.to_datetime(price_data['date'])
            price_data.set_index('date', inplace=True)
            price_data = price_data[price_data.index >= self.init_date]
            price_data = price_data.dropna()

            # Load on-chain data
            onchain_data = self.load_onchain_data(symbol='bitcoin', data_path=self.onchain_data_path, verbose=True)

            # Join price and on-chain data on date index
            self.data = price_data.join(onchain_data, how='left')
            self.data = self.data.dropna(subset=['close', 'open', 'high', 'low'])  # Ensure price data is complete

            # Split into training and test sets
            split_idx = int(len(self.data) * self.train_test_split)
            self.train_data = self.data.iloc[:split_idx]
            self.test_data = self.data.iloc[split_idx:]

            # Extract closing prices
            self.asset_data = self.data['close']
            self.train_asset_data = self.train_data['close']
            self.test_asset_data = self.test_data['close']

            print(f"Data loaded successfully: {len(self.data)} records")
            print(f"Training set: {len(self.train_data)} records")
            print(f"Test set: {len(self.test_data)} records")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def load_onchain_data(self, symbol: str, data_path: str, verbose: bool = False) -> pd.DataFrame:
        """
        Load on-chain metrics for a specific cryptocurrency with improved handling of sparse data.
        Adapted from your provided function.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'bitcoin')
            data_path: Path to on-chain data files
            verbose: Whether to print progress
            
        Returns:
            DataFrame with on-chain metrics
        """
        if verbose:
            print(f"Loading on-chain data for {symbol}...")

        ticker_map = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "solana": "SOL"
        }

        if symbol not in ticker_map:
            if verbose:
                print(f"No on-chain data available for: {symbol}")
            return pd.DataFrame()

        ticker = ticker_map[symbol]
        files = glob.glob(f"{data_path}/{ticker}_*.csv")

        if not files:
            if verbose:
                print(f"No on-chain data files found for {symbol}")
            return pd.DataFrame()

        all_dfs = []
        for file in files:
            try:
                metric_name = os.path.basename(file).replace(f"{ticker}_", "").replace(".csv", "")
                df = pd.read_csv(file)
                if 'date' not in df.columns:
                    if verbose:
                        print(f"No date column in {file}, skipping")
                    continue

                df.set_index(pd.to_datetime(df['date']), inplace=True)
                df.drop(columns=['date'], inplace=True)

                for col in df.columns:
                    if df[col].isna().all():
                        if verbose:
                            print(f"Column {col} in {file} is all NaN, skipping")
                        continue

                    non_nan_count = df[col].count()
                    if non_nan_count < 30:
                        if verbose:
                            print(f"Column {col} in {file} has only {non_nan_count} non-NaN values, skipping")
                        continue

                    data_span = df.index.max() - df.index.min()
                    if data_span.days < 365:
                        if verbose:
                            print(f"Column {col} in {file} has insufficient data span ({data_span.days} days), skipping")
                        continue

                    col_df = pd.DataFrame()
                    col_df[col] = df[col]
                    if not col_df.empty:
                        all_dfs.append(col_df)

            except Exception as e:
                if verbose:
                    print(f"Error loading {file}: {str(e)}")

        if not all_dfs:
            if verbose:
                print(f"No valid on-chain data loaded for {symbol}")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, axis=1)
        if verbose:
            print(f"Loaded {combined_df.shape[1]} on-chain metrics")

        return combined_df

    def derive_onchain_features(self, df: pd.DataFrame, onchain_columns: list) -> pd.DataFrame:
        """
        Generate derived features from on-chain metrics with robust handling of edge cases.
        Adapted from your provided function.
        
        Args:
            df: DataFrame with on-chain metrics
            onchain_columns: List of on-chain column names
            
        Returns:
            DataFrame with derived features added
        """
        df = df.copy()
        processed_columns = []

        for col in onchain_columns:
            if col not in df.columns:
                continue

            if isinstance(df[col], pd.DataFrame):
                print(f"Column {col} is a DataFrame, not a Series. Skipping this column.")
                continue

            processed_columns.append(col)

            if pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"Failed to convert column {col} to numeric: {str(e)}")
                    continue

            valid_count = df[col].notna().sum()
            if valid_count <= 30:
                print(f"Column {col} has only {valid_count} non-NaN values, insufficient for rolling calculations")
                continue

            df[f'{col}_ma7'] = df[col].rolling(window=7).mean()
            df[f'{col}_ma30'] = df[col].rolling(window=30).mean()
            df[f'{col}_std30'] = df[col].rolling(window=30).std()
            df[f'{col}_mom7'] = df[col].diff(7)
            df[f'{col}_mom30'] = df[col].diff(30)
            df[f'{col}_zscore'] = (df[col] - df[f'{col}_ma30']) / df[f'{col}_std30']

        print(f"Processed {len(processed_columns)} on-chain metrics out of {len(onchain_columns)} total")
        return df

    def get_markov(self, no_regimes=2, return_data=False):
        """
        Apply Markov-Switching Regression to detect volatility regimes.
        """
        try:
            log_returns_full = np.log(self.asset_data).diff().dropna()
            log_returns_full.index = self.data.index[1:]

            split_idx = int(len(self.data) * self.train_test_split)
            embargo_size = int(len(self.data) * self.embargo_pct)
            train_end_idx = split_idx - embargo_size
            log_returns_train = log_returns_full.iloc[:train_end_idx]

            markov_model = sm.tsa.MarkovRegression(
                log_returns_train.iloc[1:],
                k_regimes=no_regimes,
                switching_variance=True,
                exog=log_returns_train.iloc[:-1]
            )
            self.msreg_results = markov_model.fit()

            predict_exog = log_returns_full.iloc[:-1]
            full_model = sm.tsa.MarkovRegression(
                log_returns_full.iloc[1:],
                k_regimes=no_regimes,
                switching_variance=True,
                exog=predict_exog
            )
            full_results = full_model.smooth(self.msreg_results.params)

            low_var = full_results.smoothed_marginal_probabilities[0]
            high_var = full_results.smoothed_marginal_probabilities[1]
            markov_data = pd.concat([low_var, high_var], axis=1)
            markov_data.columns = ['Low_var', 'High_var']
            markov_data['dataset'] = 1
            markov_data.iloc[train_end_idx:split_idx, -1] = 0
            markov_data.iloc[split_idx:, -1] = 2
            self.markov = markov_data

            print("Markov Switching Regression model fitted successfully.")
            print(f"Training data: {train_end_idx} points")
            print(f"Embargo data: {embargo_size} points")
            print(f"Test data: {len(markov_data) - split_idx} points")

            if return_data:
                return low_var, high_var

        except Exception as e:
            print(f"Error fitting Markov model: {e}")
            raise

    def kama(self, close, length=10, fast=2, slow=30):
        """
        Calculate Kaufman's Adaptive Moving Average.
        """
        change = close.diff(length).abs()
        volatility = close.diff().abs().rolling(window=length).sum()
        er = pd.Series(np.where(volatility != 0, change / volatility, 0), index=close.index)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = pd.Series(np.square(er * (fast_sc - slow_sc) + slow_sc), index=close.index)
        kama = pd.Series(close, index=close.index)

        for i in range(length, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])

        return kama

    def get_kama(self, n_window=13, pow1=2, pow2=30, gamma=0.15, return_data=False):
        """
        Calculate KAMA and its filter for trend detection.
        """
        try:
            log_price = np.log(self.asset_data)
            kama_ = self.kama(close=log_price, length=n_window, fast=pow1, slow=pow2)
            kama_diff = kama_.diff(n_window)
            filtr = gamma * np.std(kama_diff)
            period_low = kama_.rolling(window=n_window, min_periods=n_window).min()
            period_high = kama_.rolling(window=n_window, min_periods=n_window).max()
            periods_low_high = pd.concat([period_low, period_high], axis=1)
            periods_low_high.columns = ['Period_low', 'Period_high']
            self.kama_values = kama_
            self.filtr = filtr
            self.periods_low_high = periods_low_high

            print("KAMA calculation completed successfully.")

            if return_data:
                return filtr, period_low, period_high, kama_

        except Exception as e:
            print(f"Error calculating KAMA: {e}")
            raise

    def get_classes(self, return_data=True, drop_empty_class=True):
        """
        Combine MSR and KAMA to classify market into four regimes.
        """
        try:
            if not hasattr(self, 'markov') or not hasattr(self, 'kama_values'):
                raise ValueError("Must run get_markov() and get_kama() before get_classes()")

            all_dates = self.asset_data.index
            log_price = np.log(self.asset_data)
            log_returns = log_price.diff()
            classes = pd.DataFrame('out of bounds', index=all_dates, columns=['label'])

            for attr in ['markov', 'kama_values', 'periods_low_high']:
                if hasattr(self, attr):
                    setattr(self, attr, getattr(self, attr).reindex(all_dates, method='ffill'))

            low_var_class = self.markov['Low_var'] > 0.5
            high_var_class = self.markov['High_var'] > 0.5
            bullish_class = self.kama_values > self.periods_low_high['Period_low'].shift() + self.filtr
            bearish_class = self.kama_values < self.periods_low_high['Period_high'].shift() - self.filtr

            classes.loc[high_var_class & bullish_class, 'label'] = 'Bullish_High_Var'
            classes.loc[low_var_class & bullish_class, 'label'] = 'Bullish_Low_Var'
            classes.loc[high_var_class & bearish_class, 'label'] = 'Bearish_High_Var'
            classes.loc[low_var_class & bearish_class, 'label'] = 'Bearish_Low_Var'

            unclassified = classes['label'] == 'out of bounds'
            if unclassified.any():
                unclassified_idx = classes[unclassified].index
                vol_regime = pd.Series('high', index=unclassified_idx)
                vol_regime[self.markov.loc[unclassified_idx, 'Low_var'] > 0.5] = 'low'
                returns_ma = log_returns.rolling(window=5).mean().fillna(0)
                trend = pd.Series('neutral', index=unclassified_idx)
                trend[returns_ma[unclassified_idx] > 0] = 'bullish'
                trend[returns_ma[unclassified_idx] < 0] = 'bearish'

                for idx in unclassified_idx:
                    if idx in vol_regime.index and idx in trend.index:
                        if vol_regime[idx] == 'low' and trend[idx] == 'bullish':
                            classes.loc[idx, 'label'] = 'Bullish_Low_Var'
                        elif vol_regime[idx] == 'low' and trend[idx] == 'bearish':
                            classes.loc[idx, 'label'] = 'Bearish_Low_Var'
                        elif vol_regime[idx] == 'high' and trend[idx] == 'bullish':
                            classes.loc[idx, 'label'] = 'Bullish_High_Var'
                        elif vol_regime[idx] == 'high' and trend[idx] == 'bearish':
                            classes.loc[idx, 'label'] = 'Bearish_High_Var'

            classes['Log_BTC'] = log_price
            classes['Log_BTC_returns'] = log_returns
            classes['KAMA'] = self.kama_values

            if drop_empty_class:
                classes = classes[classes['label'] != 'out of bounds']

            label2color = {
                'Bullish_High_Var': 'yellow',
                'Bullish_Low_Var': 'green',
                'Bearish_High_Var': 'red',
                'Bearish_Low_Var': 'orange'
            }
            classes['color'] = classes['label'].apply(lambda label: label2color.get(label, 'gray'))
            classes['id'] = classes.groupby((classes['label'] != classes['label'].shift(1)).cumsum()).ngroup()
            self.aligned_classes = classes

            print("Regime classification completed successfully.")

            if return_data:
                return classes

        except Exception as e:
            print(f"Error classifying regimes: {e}")
            raise

    def reclassify_regimes(self, classes):
        """
        Reclassify KAMA+MSR regimes into three labels: Bullish, Bearish, Other.
        Implements Section 3.5 of the KMRF paper.
        """
        try:
            reclassified = pd.DataFrame(index=classes.index, columns=['label'])
            reclassified['label'] = 'Other'

            classes['regime_change'] = (classes['label'] != classes['label'].shift(1)).cumsum()
            regime_groups = classes.groupby('regime_change')

            current_state = None
            for _, group in regime_groups:
                regime = group['label'].iloc[0]
                log_price = group['Log_BTC']

                if regime == 'Bullish_Low_Var':
                    current_state = 'Bullish'
                    reclassified.loc[group.index, 'label'] = 'Bullish'
                elif regime == 'Bullish_High_Var' and current_state == 'Bullish':
                    peak_idx = log_price.idxmax()
                    reclassified.loc[group.index[:group.index.get_loc(peak_idx) + 1], 'label'] = 'Bullish'
                    current_state = None
                elif regime == 'Bearish_High_Var':
                    current_state = 'Bearish'
                    reclassified.loc[group.index, 'label'] = 'Bearish'
                elif regime == 'Bearish_Low_Var' and current_state == 'Bearish':
                    trough_idx = log_price.idxmin()
                    reclassified.loc[group.index[:group.index.get_loc(trough_idx) + 1], 'label'] = 'Bearish'
                    current_state = None

            transaction_cost = 0.001
            reclassified['returns'] = classes['Log_BTC'].diff()
            reclassified['profitable'] = reclassified['returns'].abs() > transaction_cost
            reclassified.loc[(reclassified['label'] != 'Other') & (~reclassified['profitable']), 'label'] = 'Other'

            print("Regime reclassification completed successfully.")
            return reclassified

        except Exception as e:
            print(f"Error reclassifying regimes: {e}")
            raise

    def extract_features(self, data):
        """
        Extract technical, on-chain, and MSR features for RF prediction.
        Implements Section 3.1.2 of the KMRF paper.
        """
        try:
            # Create a copy of the data to avoid modifying the original
            data = data.copy()

            # Import volume data from @bitcoin.csv
            try:
                volume_data = pd.read_csv('/Users/valter.rebelo/MissionControl/data/micro/assetData/bitcoin.csv')
                volume_data['date'] = pd.to_datetime(volume_data['date'])
                volume_data.set_index('date', inplace=True)
                
                # Join volume data with our existing data
                if 'total_volume' in volume_data.columns:
                    data = data.join(volume_data[['total_volume']], how='left')
                    # Rename to 'volume' as expected by add_all_ta_features
                    data['volume'] = data['total_volume']
                    # Drop the temporary total_volume column
                    data = data.drop(columns=['total_volume'], errors='ignore')
                else:
                    print("Warning: 'total_volume' column not found in @bitcoin.csv")
                    # Create a dummy volume column as fallback
                    data['volume'] = data['close'] * 1000  # Dummy volume
            except Exception as e:
                print(f"Error loading volume data: {e}")
                # Create a dummy volume column as fallback
                data['volume'] = data['close'] * 1000  # Dummy volume

            # Technical features using ta
            tech_features = add_all_ta_features(
                data,
                open="open", high="high", low="low", close="close", volume="volume",
                fillna=True
            )

            # On-chain features
            onchain_columns = [
                'btc_hash_rate', 'cdd_account_based', 'entity_adj_dormancy_flow', 'entity_adj_nupl',
                'exchanges_net_pit', 'mvrv', 'mvrv_lth', 'mvrv_sth', 'net_realized_profit_loss',
                'pct_supply_in_profit', 'price_drawdown_relative', 'puell_multiple', 'realized_price',
                'realized_profits_to_value_ratio', 'relative_unrealized_loss', 'relative_unrealized_profit',
                'reserve_risk', 'ssr_oscillator', 'sth_sopr', 'supply_active_1y_2y', 'supply_active_3m_6m',
                'utxo_loss_count', 'utxo_profit_count'
            ]


            # Derive onchain features
            onchain_data = self.derive_onchain_features(data, onchain_columns)

            # Select derived on-chain features (excluding raw columns to avoid redundancy)
            derived_columns = [col for col in onchain_data.columns if any(suffix in col for suffix in ['_ma7', '_ma30', '_std30', '_mom7', '_mom30', '_zscore'])]
            onchain_features = onchain_data[derived_columns]

            # MSR features (Low_var, High_var)
            msr_features = self.markov[['Low_var', 'High_var']].reindex(data.index, method='ffill')

            # Combine features
            features = pd.concat([tech_features, onchain_features, msr_features], axis=1)

            # Debug: Check for NaN values in features
            nan_columns = features.columns[features.isna().any()].tolist()
            if nan_columns:
                print("Columns with NaN values before imputation:", nan_columns)
                for col in nan_columns:
                    print(f"Number of NaNs in {col}: {features[col].isna().sum()}")

            # Impute NaN values to satisfy tsfresh requirements
            # Use forward fill followed by backward fill for time-series data
            features = features.fillna(method='ffill').fillna(method='bfill')

            # Verify that all NaNs are removed
            if features.isna().any().any():
                raise ValueError(f"Features still contain NaN values after imputation: {features.columns[features.isna().any()].tolist()}")

            # Extract tsfresh features using MinimalFCParameters
            tsfresh_features = extract_features(
                features.reset_index(),
                column_id='date',
                column_sort='date',
                column_value=None,
                default_fc_parameters=MinimalFCParameters()  # Use MinimalFCParameters instead of 'minimal'
            )
            tsfresh_features.index = features.index

            print("Feature extraction completed successfully.")
            return pd.concat([features, tsfresh_features], axis=1)

        except Exception as e:
            print(f"Error extracting features: {e}")
            raise


    def purged_group_time_series_split(self, n_splits=5, embargo_pct=0.01):
        """
        Implement PGTS cross-validation for time-series data.
        Implements Section 3.3 of the KMRF paper.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        embargo_size = int(len(self.train_data) * embargo_pct)

        for train_idx, val_idx in tscv.split(self.train_data):
            val_start = val_idx[0]
            val_end = val_idx[-1]
            embargo_end = min(val_start + embargo_size, len(self.train_data) - 1)
            train_idx = train_idx[train_idx < (val_start - embargo_size)]
            val_idx = val_idx[val_idx >= embargo_end]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

    

    def select_features(self, features, labels):
        """
        Select important features using SelectKBest with PGTS cross-validation.
        Runs SelectKBest on each fold and aggregates results to avoid data leakage.
        """
        try:
            # Preprocess features: remove columns that are all NaN, constant, or have excessive NaNs
            features_clean = features.dropna(axis=1, how='all')  # Remove columns that are all NaN
            features_clean = features_clean.loc[:, (features_clean != features_clean.iloc[0]).any()]  # Remove constant columns
            nan_threshold = 0.5 * len(features_clean)  # Remove columns with more than 50% NaNs
            features_clean = features_clean.loc[:, features_clean.isna().sum() <= nan_threshold]

            print(f"Features after preprocessing: {features_clean.columns.tolist()}")

            # Run SelectKBest on each PGTS fold
            selected_features_per_fold = []
            for fold_idx, (train_idx, val_idx) in enumerate(self.purged_group_time_series_split(n_splits=5)):
                print(f"Running SelectKBest on fold {fold_idx + 1}...")
                X_train, y_train = features_clean.iloc[train_idx], labels.iloc[train_idx]

                # Use SelectKBest to select the top 50 features
                selector = SelectKBest(score_func=f_classif, k=50)
                selector.fit(X_train, y_train)

                # Get the selected features for this fold
                selected_indices = selector.get_support()
                selected_features = X_train.columns[selected_indices].tolist()
                selected_features_per_fold.append(set(selected_features))
                print(f"Fold {fold_idx + 1} selected features: {selected_features}")

            # Aggregate features: select features that appear in at least 3 out of 5 folds
            all_features = set().union(*selected_features_per_fold)
            feature_counts = {feature: 0 for feature in all_features}
            for fold_features in selected_features_per_fold:
                for feature in fold_features:
                    feature_counts[feature] += 1

            # Select features that appear in at least 3 folds
            selected_features = [feature for feature, count in feature_counts.items() if count >= 3]
            print(f"Final selected features (appearing in at least 3 folds): {selected_features}")

            # Return the original features DataFrame with only the selected columns
            return features[selected_features]

        except Exception as e:
            print(f"Error selecting features: {e}")
            raise

    def train_rf_classifier(self, features, labels, evals=500):
        """
        Train RF classifier with Optuna hyperparameter optimization using PGTS.
        Implements Section 3.3 of the KMRF paper.
        """
        try:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                    'max_depth': trial.suggest_int('max_depth', 1, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),  # Changed from 1 to 2
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
                    'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),
                    'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.05),
                    'max_features': trial.suggest_float('max_features', 0.2, 1.0)
                }

                rf = RandomForestClassifier(**params, random_state=42)
                mcc_scores = []

                for train_idx, val_idx in self.purged_group_time_series_split(n_splits=5):
                    X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                    y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

                    rf.fit(X_train, y_train)
                    preds = rf.predict(X_val)
                    mcc = matthews_corrcoef(y_val, preds)
                    mcc_scores.append(mcc)

                return -np.mean(mcc_scores)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=evals)

            best_params = study.best_params
            self.rf_model = RandomForestClassifier(**best_params, random_state=42)
            self.rf_model.fit(features, labels)

            print(f"RF training completed. Best parameters: {best_params}")
            return best_params

        except Exception as e:
            print(f"Error training RF classifier: {e}")
            raise

    def predict_regimes(self, features):
        """
        Predict regimes using the trained RF model.
        """
        try:
            if not hasattr(self, 'rf_model'):
                raise ValueError("Must train RF model before predicting regimes.")
            return pd.Series(self.rf_model.predict(features), index=features.index)

        except Exception as e:
            print(f"Error predicting regimes: {e}")
            raise

    def implement_trading_strategy(self, predicted_regimes, initial_capital=1000, test_only=True):
        """
        Implement a contrarian trading strategy based on RF predictions.
        Implements Section 3.6 of the KMRF paper.
        """
        try:
            if test_only and hasattr(self, 'markov') and 'dataset' in self.markov.columns:
                test_indices = self.markov[self.markov['dataset'] == 2].index
                predicted_regimes = predicted_regimes[predicted_regimes.index.isin(test_indices)]

            portfolio = pd.DataFrame(index=predicted_regimes.index)
            portfolio['btc_price'] = self.data['close'].reindex(predicted_regimes.index)
            portfolio['regime'] = predicted_regimes
            portfolio['next_open'] = self.data['open'].reindex(predicted_regimes.index).shift(-1)

            portfolio['trading_state'] = 0  # 1 = long, -1 = short, 0 = flat
            portfolio['cash'] = initial_capital
            portfolio['btc_holdings'] = 0
            portfolio['portfolio_value'] = initial_capital
            portfolio['trades'] = 0
            portfolio['trading_costs'] = 0
            portfolio['cumulative_costs'] = 0

            trading_cost_pct = 0.001

            for i in range(len(portfolio) - 1):
                curr_regime = portfolio.iloc[i]['regime']
                curr_state = portfolio.iloc[i]['trading_state']
                curr_cash = portfolio.iloc[i]['cash']
                curr_btc = portfolio.iloc[i]['btc_holdings']
                curr_value = portfolio.iloc[i]['portfolio_value']
                next_open = portfolio.iloc[i]['next_open']

                if curr_regime == 'Bullish':
                    target_state = 1  # Short (overbought)
                elif curr_regime == 'Bearish':
                    target_state = 0  # Long (oversold)
                else:
                    target_state = 0  # Flat

                next_idx = portfolio.index[i + 1]
                portfolio.loc[next_idx, 'trading_state'] = curr_state
                portfolio.loc[next_idx, 'cash'] = curr_cash
                portfolio.loc[next_idx, 'btc_holdings'] = curr_btc

                if curr_state != target_state:
                    trading_cost = curr_value * trading_cost_pct if next_open is not None else 0

                    if target_state == 1:
                        btc_to_buy = (curr_cash - trading_cost) / next_open if next_open is not None else 0
                        portfolio.loc[next_idx, 'trading_state'] = 1
                        portfolio.loc[next_idx, 'cash'] = 0
                        portfolio.loc[next_idx, 'btc_holdings'] = btc_to_buy
                    elif target_state == -1:
                        btc_to_short = (curr_value - trading_cost) / next_open if next_open is not None else 0
                        portfolio.loc[next_idx, 'trading_state'] = -1
                        portfolio.loc[next_idx, 'cash'] = curr_value + btc_to_short * next_open
                        portfolio.loc[next_idx, 'btc_holdings'] = -btc_to_short
                    else:
                        if curr_state == 1:
                            cash_from_sale = curr_btc * next_open - trading_cost if next_open is not None else 0
                            portfolio.loc[next_idx, 'cash'] = cash_from_sale
                        elif curr_state == -1:
                            cash_after_cover = curr_cash - (curr_btc * next_open + trading_cost) if next_open is not None else curr_cash
                            portfolio.loc[next_idx, 'cash'] = cash_after_cover
                        portfolio.loc[next_idx, 'trading_state'] = 0
                        portfolio.loc[next_idx, 'btc_holdings'] = 0

                    portfolio.loc[next_idx, 'trades'] = 1
                    portfolio.loc[next_idx, 'trading_costs'] = trading_cost

            portfolio['portfolio_value'] = portfolio['cash'] + (portfolio['btc_holdings'] * portfolio['btc_price'])
            portfolio['cumulative_costs'] = portfolio['trading_costs'].cumsum()
            portfolio['strategy_returns'] = portfolio['portfolio_value'].pct_change()
            portfolio['btc_returns'] = portfolio['btc_price'].pct_change()

            initial_btc_holdings = initial_capital / portfolio['btc_price'].iloc[0]
            portfolio['buy_hold_value'] = initial_btc_holdings * portfolio['btc_price']
            portfolio['strategy_normalized'] = portfolio['portfolio_value'] / initial_capital
            portfolio['buy_hold_normalized'] = portfolio['buy_hold_value'] / initial_capital

            print("Trading strategy implemented successfully.")
            return portfolio

        except Exception as e:
            print(f"Error implementing trading strategy: {e}")
            raise

    def calculate_performance_metrics(self, portfolio, test_only=True):
        """
        Calculate performance metrics for the trading strategy.
        Implements Section 3.7.1 of the KMRF paper.
        """
        try:
            if test_only and 'dataset' in self.markov.columns:
                test_indices = self.markov[self.markov['dataset'] == 2].index
                portfolio = portfolio[portfolio.index.isin(test_indices)]

            annualize_factor = 365
            btc_returns = portfolio['btc_returns'].dropna()
            strategy_returns = portfolio['strategy_returns'].dropna()

            btc_annual_return = (1 + btc_returns.mean()) ** annualize_factor - 1
            strategy_annual_return = (1 + strategy_returns.mean()) ** annualize_factor - 1
            btc_volatility = btc_returns.std() * np.sqrt(annualize_factor)
            strategy_volatility = strategy_returns.std() * np.sqrt(annualize_factor)
            btc_sharpe = btc_annual_return / btc_volatility if btc_volatility != 0 else 0
            strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility != 0 else 0
            btc_downside = btc_returns[btc_returns < 0].std() * np.sqrt(annualize_factor)
            strategy_downside = strategy_returns[strategy_returns < 0].std() * np.sqrt(annualize_factor)
            btc_sortino = btc_annual_return / btc_downside if btc_downside != 0 else 0
            strategy_sortino = strategy_annual_return / strategy_downside if strategy_downside != 0 else 0

            excess_returns = strategy_returns - btc_returns
            tracking_error = excess_returns.std() * np.sqrt(annualize_factor)
            ir = (strategy_annual_return - btc_annual_return) / tracking_error if tracking_error != 0 else 0

            btc_cum_returns = (1 + btc_returns).cumprod()
            strategy_cum_returns = (1 + strategy_returns).cumprod()
            btc_peak = btc_cum_returns.cummax()
            strategy_peak = strategy_cum_returns.cummax()
            btc_drawdown = (btc_cum_returns / btc_peak - 1)
            strategy_drawdown = (strategy_cum_returns / strategy_peak - 1)
            btc_max_drawdown = btc_drawdown.min()
            strategy_max_drawdown = strategy_drawdown.min()
            num_trades = portfolio['trades'].sum()
            total_trading_costs = portfolio['cumulative_costs'].iloc[-1] if len(portfolio) > 0 else 0
            final_btc_value = portfolio['buy_hold_normalized'].iloc[-1] if len(portfolio) > 0 else 1
            final_strategy_value = portfolio['strategy_normalized'].iloc[-1] if len(portfolio) > 0 else 1

            metrics = {
                'BTC Annual Return': btc_annual_return,
                'Strategy Annual Return': strategy_annual_return,
                'BTC Volatility': btc_volatility,
                'Strategy Volatility': strategy_volatility,
                'BTC Sharpe Ratio': btc_sharpe,
                'Strategy Sharpe Ratio': strategy_sharpe,
                'BTC Sortino Ratio': btc_sortino,
                'Strategy Sortino Ratio': strategy_sortino,
                'BTC Max Drawdown': btc_max_drawdown,
                'Strategy Max Drawdown': strategy_max_drawdown,
                'Information Ratio': ir,
                'Number of Trades': num_trades,
                'Total Trading Costs': total_trading_costs,
                'Final Buy & Hold Value': final_btc_value,
                'Final Strategy Value': final_strategy_value,
                'Outperformance': final_strategy_value - final_btc_value
            }

            print("Performance metrics calculated successfully.")
            return metrics

        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            raise

    def calculate_mcc(self, true_labels, predicted_labels):
        """
        Calculate Matthews Correlation Coefficient for each class.
        Implements Section 3.7.2 of the KMRF paper.
        """
        try:
            lb = LabelBinarizer()
            true_binary = lb.fit_transform(true_labels)
            pred_binary = lb.transform(predicted_labels)

            mcc_scores = {}
            for i, class_name in enumerate(lb.classes_):
                mcc = matthews_corrcoef(true_binary[:, i], pred_binary[:, i])
                mcc_scores[class_name] = mcc

            avg_bull_bear = np.mean([mcc_scores.get('Bullish', 0), mcc_scores.get('Bearish', 0)])
            mcc_scores['Avg Bullish & Bearish'] = avg_bull_bear
            return mcc_scores

        except Exception as e:
            print(f"Error calculating MCC: {e}")
            raise

    def run_pipeline(self):
        """
        Run the full KMRF pipeline: regime detection, feature extraction, feature selection,
        RF training, prediction, trading, and evaluation.
        """
        try:
            # Step 1: Detect regimes with KAMA+MSR
            self.get_markov()
            self.get_kama()
            classes = self.get_classes()

            # Step 2: Reclassify regimes into Bullish, Bearish, Other
            reclassified = self.reclassify_regimes(classes)

            # Step 3: Extract features
            features = self.extract_features(self.data)

            # Step 4: Split features and labels into training and test sets
            split_idx = int(len(self.data) * self.train_test_split)
            train_features = features.iloc[:split_idx]
            test_features = features.iloc[split_idx:]
            train_labels = reclassified['label'].iloc[:split_idx]
            test_labels = reclassified['label'].iloc[split_idx:]

            # Align test_labels and test_features on common indices
            common_indices = test_labels.index.intersection(test_features.index)
            test_labels = test_labels.loc[common_indices]
            test_features = test_features.loc[common_indices]

            # Step 5: Feature selection on training data
            selected_train_features = self.select_features(train_features, train_labels)
            selected_test_features = test_features[selected_train_features.columns]

            # Step 6: Train RF classifier on training data
            self.train_rf_classifier(selected_train_features, train_labels, evals=100)

            # Step 7: Predict regimes on test data
            predicted_regimes = self.predict_regimes(selected_test_features)

            # Step 8: Evaluate classification performance (MCC)
            mcc_scores = self.calculate_mcc(test_labels, predicted_regimes)
            print("MCC Scores:", mcc_scores)

            # Step 9: Implement trading strategy
            portfolio = self.implement_trading_strategy(predicted_regimes)

            # Step 10: Calculate performance metrics
            metrics = self.calculate_performance_metrics(portfolio)
            print("Performance Metrics:", metrics)

            return portfolio, metrics, mcc_scores

        except Exception as e:
            print(f"Error running pipeline: {e}")
            raise

    def save_model(self, filename=None):
        """
        Save the model to a file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kmrf_model_{timestamp}.pkl"

        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        with open(filename, 'wb') as f:
            pickle.dump({
                'markov': self.markov,
                'kama_values': self.kama_values,
                'filtr': self.filtr,
                'periods_low_high': self.periods_low_high,
                'aligned_classes': self.aligned_classes,
                'msreg_results': self.msreg_results,
                'rf_model': self.rf_model
            }, f)

        print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, price_data_path, onchain_data_path, model_path):
        """
        Load a saved model.
        """
        instance = cls(price_data_path, onchain_data_path)
        with open(model_path, 'rb') as f:
            saved_model = pickle.load(f)

        for key, value in saved_model.items():
            setattr(instance, key, value)

        print(f"Model loaded from {model_path}")
        return instance