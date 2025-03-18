import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import logging
import warnings
import feature_selection as fs
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import joblib
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RegimeSwitchModel:
    def __init__(self, symbol, interval='1d', train_pct=0.8, strategy='long-only', confidence_threshold=0.75):
        self.symbol = symbol
        self.interval = interval
        self.train_pct = train_pct
        if strategy not in ['long-only', 'long-short']:
            raise ValueError("Strategy must be either 'long-only' or 'long-short'")
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.data = None
        self.features = None
        self.selected_features = None
        self.train_data = None
        self.state_probabilities = None
        self.transition_probs = None
        self.hmm = None
        self.onchain_data_path = "/Users/valter.rebelo/MissionControl/data/onchainData"
        self.macro_data_path = "/Users/valter.rebelo/MissionControl/data/macro/fredData"
        self.macro_features = [
            'treasury5YInflationExpectation', 
            'treasury5YInflationForwardRate', 
            'creditSpreads', 
            'vix', 
            'sp500', 
            'globalCbLiquidity',
            'move'
        ]

    def load_data(self):
        """Load price data without caching."""
        try:
            df_1 = pd.read_csv(f"/Users/valter.rebelo/MissionControl/data/micro/candleData/{self.symbol}_candles.csv")
            df_1['date'] = pd.to_datetime(df_1['date'])
            df_1.set_index('date', inplace=True)
            df_2 = pd.read_csv(f"/Users/valter.rebelo/MissionControl/data/micro/assetData/{self.symbol}.csv")
            df_2['date'] = pd.to_datetime(df_2['date'])
            df_2.set_index('date', inplace=True)
            btc_df = pd.read_csv("/Users/valter.rebelo/MissionControl/data/micro/candleData/bitcoin_candles.csv")
            btc_df['date'] = pd.to_datetime(btc_df['date'])
            btc_df.set_index('date', inplace=True)

            data = pd.merge(df_1, df_2[['total_volume', 'market_cap']], on='date', how='inner')
            data.rename(columns={'total_volume': 'Volume', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
            data.index.name = 'Date'
            if self.symbol != "bitcoin":
                data['close_btc'] = (data['Close'] / btc_df['close']) * 100
                data.dropna(inplace=True)
            if self.symbol == "bitcoin":
                data = data[data.index >= '2019-01-01']
            
            self.data = data
            logging.info(f"Loaded {len(data)} rows of basic data for {self.symbol}")
            return self.data
        
        except Exception as e:
            logging.error(f"Data loading failed for {self.symbol}: {str(e)}")
            raise

    def generate_all_features(self):
        """Generate all features using the feature_selection module."""
        if self.data is None:
            raise ValueError("Basic data must be loaded first. Call load_data() before generate_all_features().")
        
        try:
            feature_results = fs.generate_features(
                data=self.data,
                symbol=self.symbol,
                include_technical=True,
                include_onchain=True,
                include_macro=True,
                onchain_data_path=self.onchain_data_path,
                macro_data_path=self.macro_data_path,
                macro_features=self.macro_features,
                verbose=True,
                show_plots=False
            )
            self.features = feature_results['processed_data']
            logging.info(f"Generated {len(self.features.columns)} features")
            return self.features
        
        except Exception as e:
            logging.error(f"Feature generation failed: {str(e)}")
            raise

    def set_manual_features(self, feature_list):
        """Manually set the features to use for modeling."""
        if self.features is None:
            raise ValueError("Features must be generated first. Call generate_all_features() before set_manual_features().")
        
        available_features = self.features.columns.tolist()
        valid_features = [f for f in feature_list if f in available_features]
        
        if len(valid_features) == 0:
            raise ValueError("None of the specified features exist in the data.")
        if len(valid_features) < len(feature_list):
            missing = set(feature_list) - set(valid_features)
            logging.warning(f"Some requested features are not available: {missing}")
        
        self.selected_features = valid_features
        logging.info(f"Manually set {len(valid_features)} features: {valid_features}")
        
        if 'log_close' not in self.features.columns:
            self.features['log_close'] = np.log(self.features['Close'])
        self.model_data = self.features[['log_return', 'log_close', 'Open', 'Close', 'Volume'] + self.selected_features]
        
        return valid_features

    def split_data(self, data, embargo_percent=0.01):
        """Split data into train, embargo, and test sets."""
        try:
            total_rows = len(data)
            train_end_idx = int(total_rows * self.train_pct)
            embargo_end_idx = int(train_end_idx + (total_rows * embargo_percent))
            
            train = data.iloc[:train_end_idx]
            embargo = data.iloc[train_end_idx:embargo_end_idx]
            test = data.iloc[embargo_end_idx:]
            
            self.train_data = train
            if len(train) < 50 or len(test) < 50:
                raise ValueError("Train or test set too small (<50 rows).")
            logging.info(f"Split data: train={len(train)}, embargo={len(embargo)}, test={len(test)}")
            return train, embargo, test
        
        except Exception as e:
            logging.error(f"Data splitting failed: {str(e)}")
            raise

    def normalize_features(self, train, test, features, train_mean=None, train_std=None):
        """Normalize features using provided or computed means and stds."""
        if train_mean is None or train_std is None:
            train_mean = train[features].mean()
            train_std = train[features].std()
        train_normalized = (train[features] - train_mean) / train_std
        test_normalized = (test[features] - train_mean) / train_std
        return (pd.DataFrame(train_normalized, index=train.index, columns=features),
                pd.DataFrame(test_normalized, index=test.index, columns=features))

    def train_hmm(self, train, features):
        """Train a Hidden Markov Model (HMM) on the provided training data."""
        try:
            if train is None:
                raise ValueError("Training data is None. Please run load_data, generate_all_features, set_manual_features, and split_data first.")
            
            f_train_normalized, _ = self.normalize_features(train, train, features)
            hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=1000, random_state=42)
            hmm.fit(f_train_normalized)
            if not hmm.monitor_.converged:
                logging.warning("HMM training did not converge.")
            
            self.transition_probs = hmm.transmat_
            self.hmm = hmm
            logging.info(f"Transition probabilities extracted: \n{self.transition_probs}")
            logging.info("HMM trained successfully")
            return hmm
        except Exception as e:
            logging.error(f"HMM training failed: {str(e)}")
            raise

    def predict_states(self, hmm, f_test, optimal_states, test_data):
        """Predict states using a single HMM and map to trading actions based on confidence_threshold."""
        hidden_states = hmm.predict(f_test)
        state_probs = hmm.predict_proba(f_test)
        confidences = [state_probs[i, state] for i, state in enumerate(hidden_states)]

        raw_states = pd.Series(hidden_states, index=test_data.index)
        predicted_states = []
        for state, confidence in zip(hidden_states, confidences):
            if confidence < self.confidence_threshold:
                predicted_states.append('Flat')
            else:
                predicted_states.append(optimal_states.get(state, 'Flat'))

        if len(predicted_states) != len(test_data):
            logging.error(f"Mismatch: predicted_states length ({len(predicted_states)}) does not match test_data length ({len(test_data)})")
            raise ValueError(f"Length mismatch: {len(predicted_states)} predictions vs {len(test_data)} test rows")

        state_series = pd.Series(predicted_states, index=test_data.index)
        confidence_series = pd.Series(confidences, index=test_data.index)

        self.state_probabilities = pd.DataFrame({
            'confidence': confidence_series,
            'state': state_series,
            'raw_state': raw_states
        })

        return state_series

    def find_optimal_states(self, hmm, train_data, features):
        """Find optimal state mappings that maximize Sharpe ratio."""
        try:
            f_train_normalized, _ = self.normalize_features(train_data, train_data, features)
            hidden_states = hmm.predict(f_train_normalized)
            unique_states = np.unique(hidden_states)
            
            state_returns = {}
            for state in unique_states:
                state_mask = (hidden_states == state)
                if sum(state_mask) > 0:
                    if 'log_return' in train_data.columns:
                        returns = train_data.loc[state_mask, 'log_return'].values
                    else:
                        returns = np.diff(np.log(train_data.loc[state_mask, 'Close'].values))
                        
                    state_returns[state] = {
                        'mean': np.mean(returns),
                        'std': np.std(returns) if len(returns) > 1 else 1e-6,
                        'sharpe': np.mean(returns) / (np.std(returns) if len(returns) > 1 else 1e-6),
                        'count': len(returns)
                    }
            
            optimal_states = {}
            sorted_states = sorted(state_returns.items(), key=lambda x: x[1]['sharpe'], reverse=True)
            
            if self.strategy == 'long-only':
                for i, (state, _) in enumerate(sorted_states):
                    if i == 0 and state_returns[state]['mean'] > 0:
                        optimal_states[state] = 'Long'
                    else:
                        optimal_states[state] = 'Flat'
            else:
                for i, (state, metrics) in enumerate(sorted_states):
                    if i == 0 and metrics['mean'] > 0:
                        optimal_states[state] = 'Long'
                    elif i == len(sorted_states) - 1 and metrics['mean'] < 0:
                        optimal_states[state] = 'Short'
                    else:
                        optimal_states[state] = 'Flat'
            
            logging.info(f"Optimal state mappings based on Sharpe ratio: {optimal_states}")
            for state, action in optimal_states.items():
                if state in state_returns:
                    metrics = state_returns[state]
                    logging.info(f"State {state} -> {action}: Sharpe={metrics['sharpe']:.2f}, Mean={metrics['mean']:.4f}, Count={metrics['count']}")
            
            return optimal_states
        except Exception as e:
            logging.error(f"Optimal state finding failed: {str(e)}")
            state_means = hmm.means_[:, 0]
            default_mapping = {}
            for state in range(len(state_means)):
                if state == np.argmax(state_means):
                    default_mapping[state] = 'Long'
                elif self.strategy == 'long-short' and state == np.argmin(state_means):
                    default_mapping[state] = 'Short'
                else:
                    default_mapping[state] = 'Flat'
            logging.warning(f"Using fallback state mapping: {default_mapping}")
            return default_mapping

    def run_pipeline(self, test, features):
        """Train a single HMM and predict states."""
        try:
            hmm = self.train_hmm(self.train_data, features)
            optimal_states = self.find_optimal_states(hmm, self.train_data, features)
            f_train_normalized, f_test_normalized = self.normalize_features(self.train_data, test, features)
            states = self.predict_states(hmm, f_test_normalized, optimal_states, test)
            logging.info("Prediction completed")
            return states
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def save_model(self, classification_metrics=None, base_filepath="hmm_model", base_metadata_filepath="model_metadata", save_dir="/Users/valter.rebelo/MissionControl/models"):
        """Save the trained HMM model and its metadata with classification metrics."""
        if not hasattr(self, 'hmm') or self.hmm is None:
            raise ValueError("Model must be trained before saving. Run run_pipeline() first.")
        
        # Compute normalization stats from training data
        features = self.selected_features
        train_mean = self.train_data[features].mean().to_dict()
        train_std = self.train_data[features].std().to_dict()

        # Create version based on date and strategy
        strategy_prefix = "long_only" if self.strategy == "long-only" else "long_short"
        version = datetime.now().strftime('%Y%m%d')
        filepath = f"{save_dir}/{base_filepath}_{strategy_prefix}_v{version}.pkl"
        metadata_filepath = f"{save_dir}/{base_metadata_filepath}_{strategy_prefix}_v{version}.json"
        
   
        os.makedirs(save_dir, exist_ok=True)

        # Use provided classification metrics if available, otherwise set to default
        if classification_metrics is None:
            classification_metrics = {
                'confusion_matrix': {},
                'sensitivity': {},
                'specificity': {},
                'accuracy': 0.0
            }

        # Save HMM parameters
        joblib.dump(self.hmm, filepath)
        logging.info(f"Model saved to {filepath}")

        # Gather metadata
        metadata = {
            "symbol": self.symbol,
            "interval": self.interval,
            "train_pct": self.train_pct,
            "strategy": self.strategy,
            "confidence_threshold": self.confidence_threshold,
            "selected_features": self.selected_features,
            "training_date_range": {
                "start": self.train_data.index.min().strftime('%Y-%m-%d'),
                "end": self.train_data.index.max().strftime('%Y-%m-%d')
            },
            "transition_probs": self.transition_probs.tolist(),
            "normalization": {
                "means": train_mean,
                "stds": train_std
            },
            "classification_metrics": classification_metrics,
            "save_date": datetime.now().strftime('%Y-%m-%d'),
            "version": version
        }
        
        # Save metadata
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Metadata saved to {metadata_filepath}")

        return filepath, metadata_filepath

    def load_model_and_predict(self, base_model_filepath="hmm_model", base_metadata_filepath="model_metadata", 
                         save_dir="/Users/valter.rebelo/MissionControl/models", version=None, start_date=None, 
                         confidence_threshold=None):
        """Load a specified saved model and predict states for new data, returning a DataFrame with recommendations."""
        try:
            # Enforce version specification
            if version is None:
                raise ValueError("Please specify a version to load a specific model (e.g., '20250317')")
            
            # Construct file paths with specified version and check for matching strategy
            strategy_prefix = "long_only" if self.strategy == "long-only" else "long_short"
            model_filepath = f"{save_dir}/{base_model_filepath}_{strategy_prefix}_v{version}.pkl"
            metadata_filepath = f"{save_dir}/{base_metadata_filepath}_{strategy_prefix}_v{version}.json"

            # Verify file existence
            if not os.path.exists(model_filepath):
                raise FileNotFoundError(f"No model found for {strategy_prefix} strategy with version {version} in {save_dir}")
            
            # Load model
            self.hmm = joblib.load(model_filepath)
            logging.info(f"Model loaded from {model_filepath}")

            # Load metadata and validate strategy
            with open(metadata_filepath, 'r') as f:
                metadata = json.load(f)
            loaded_strategy = metadata["strategy"]
            if loaded_strategy != self.strategy:
                raise ValueError(f"Loaded model strategy ({loaded_strategy}) does not match object strategy ({self.strategy})")
            self.selected_features = metadata["selected_features"]
            self.transition_probs = np.array(metadata["transition_probs"])
            self.confidence_threshold = metadata["confidence_threshold"] if confidence_threshold is None else confidence_threshold
            if not 0 <= self.confidence_threshold <= 1:
                raise ValueError("confidence_threshold must be between 0 and 1")
            training_end_date = pd.to_datetime(metadata["training_date_range"]["end"])
            self.train_mean = pd.Series(metadata["normalization"]["means"])
            self.train_std = pd.Series(metadata["normalization"]["stds"])
            logging.info(f"Metadata loaded from {metadata_filepath} with confidence_threshold={self.confidence_threshold}")

            # Load the latest data
            if self.data is None:
                self.load_data()

            # Generate features for the new data using the feature_selection module
            self.generate_all_features()
            if self.features is None:
                raise ValueError("Failed to generate features for new data")

            # Ensure log_close is computed (should be handled by generate_all_features, but double-check)
            if 'log_close' not in self.features.columns:
                self.features['log_close'] = np.log(self.features['Close'])

            # Prepare model_data with all necessary columns
            required_columns = ['log_return', 'log_close', 'Open', 'Close', 'Volume'] + self.selected_features
            model_data = self.features[required_columns].copy()
            self.model_data = model_data
            logging.info(f"Initial model_data columns: {model_data.columns.tolist()}")

            # Filter data to start from the specified date or the day after training end date
            if start_date is None:
                start_date = training_end_date + pd.Timedelta(days=1)
            logging.info(f"Predicting states from {start_date}")
            new_data_subset = model_data[model_data.index >= start_date].copy()

            if new_data_subset.empty:
                logging.warning("No new data to predict after the start date")
                return pd.DataFrame(columns=['Date', 'Decision', 'Confidence'])

            # Remove any potential duplicate dates
            new_data_subset = new_data_subset[~new_data_subset.index.duplicated(keep='first')]

            # Normalize new data using stored training stats for selected features only
            _, f_new_normalized = self.normalize_features(
                new_data_subset, new_data_subset, self.selected_features,
                train_mean=self.train_mean, train_std=self.train_std
            )

            # Predict states (no retraining)
            optimal_states = self.find_optimal_states(self.hmm, self.train_data, self.selected_features)
            states = self.predict_states(self.hmm, f_new_normalized, optimal_states, new_data_subset)
            logging.info("States predicted for new data")

            # Format recommendations as a DataFrame with index as Date
            recommendations_df = pd.DataFrame({
                'Decision': states,
                'Confidence': self.state_probabilities['confidence']
            }, index=new_data_subset.index)
            recommendations_df.index.name = 'Date'

            logging.info(f"Generated recommendations for new data from {new_data_subset.index[0]} to {new_data_subset.index[-1]}:\n{recommendations_df}")
            return recommendations_df

        except Exception as e:
            logging.error(f"Loading or prediction failed: {str(e)}")
            raise

    def simulate_trading(self, recommendations_df=None, test=None, states=None, initial_capital=1000, save_path=None):
        """Simulate trading based on predicted recommendations or states."""
        try:
            # Handle the case for daily predictions (recommendations_df)
            if recommendations_df is not None:
                # Ensure the recommendations_df has the expected structure
                if not all(col in recommendations_df.columns for col in ['Decision', 'Confidence']):
                    raise ValueError("recommendations_df must contain 'Decision' and 'Confidence' columns")
                
                # Align with new data (assuming model_data is already set)
                if self.model_data is None or self.model_data.empty:
                    raise ValueError("model_data must be initialized. Run load_model_and_predict first.")
                
                # Filter model_data to match the dates in recommendations_df
                new_data_subset = self.model_data[self.model_data.index.isin(recommendations_df.index)].copy()
                if new_data_subset.empty:
                    raise ValueError("No matching data found for the recommendations dates")

                # Ensure index alignment
                new_data_subset = new_data_subset.reindex(recommendations_df.index)
                shifted_decisions = recommendations_df['Decision'].shift(1).fillna('Flat')
                position_multiplier = (shifted_decisions == 'Long').astype(int) - (shifted_decisions == 'Short').astype(int)
                confidences = recommendations_df['Confidence']
            
            # Handle the case for training (test and states)
            elif test is not None and states is not None:
                # Ensure states and test are aligned
                if not states.index.equals(test.index):
                    logging.error(f"Index mismatch: states index ({states.index}) does not match test index ({test.index})")
                    raise ValueError("States index must match test index")

                new_data_subset = test.copy()
                shifted_decisions = states.shift(1).fillna('Flat')
                position_multiplier = (shifted_decisions == 'Long').astype(int) - (shifted_decisions == 'Short').astype(int)
                # In training, confidences might not be directly available; use state_probabilities if set
                confidences = self.state_probabilities['confidence'] if self.state_probabilities is not None else pd.Series(1.0, index=test.index)
            else:
                raise ValueError("Must provide either recommendations_df or both test and states")

            raw_returns = new_data_subset['Open'].pct_change().fillna(0)

            # Initialize portfolio values and costs
            portfolio_value = pd.Series(index=new_data_subset.index, dtype=float)
            portfolio_value.iloc[0] = initial_capital
            cumulative_costs = pd.Series(0.0, index=new_data_subset.index)
            transaction_costs = pd.Series(0.0, index=new_data_subset.index)
            returns = pd.Series(0.0, index=new_data_subset.index)
            trading_cost_rate = 0.001

            position_changes = position_multiplier.diff().fillna(0).abs()

            for t in range(1, len(new_data_subset)):
                prev_portfolio_value = portfolio_value.iloc[t-1]
                raw_return = raw_returns.iloc[t] * position_multiplier.iloc[t-1]
                portfolio_value.iloc[t] = prev_portfolio_value * (1 + raw_return)

                if position_changes.iloc[t] > 0:
                    transaction_cost_dollars = prev_portfolio_value * position_changes.iloc[t] * trading_cost_rate
                    portfolio_value.iloc[t] -= transaction_cost_dollars
                    transaction_costs.iloc[t] = transaction_cost_dollars
                else:
                    transaction_costs.iloc[t] = 0.0

                cumulative_costs.iloc[t] = cumulative_costs.iloc[t-1] + transaction_costs.iloc[t]
                if portfolio_value.iloc[t-1] > 0:
                    returns.iloc[t] = (portfolio_value.iloc[t] / portfolio_value.iloc[t-1]) - 1
                else:
                    returns.iloc[t] = 0.0

            cum_returns = (portfolio_value / initial_capital)
            bnh_returns = new_data_subset['Close'].pct_change().fillna(0.0)
            bnh_cum_returns = (1 + bnh_returns).cumprod()
            bnh_portfolio_value = initial_capital * bnh_cum_returns

            # Construct the result DataFrame
            result = pd.DataFrame({
                'Close': new_data_subset['Close'],
                'state': recommendations_df['Decision'] if recommendations_df is not None else states,
                'raw_state': self.state_probabilities['raw_state'].reindex(new_data_subset.index) if self.state_probabilities is not None else pd.Series(np.nan, index=new_data_subset.index),
                'returns': returns,
                'cum_returns': cum_returns,
                'portfolio_value': portfolio_value,
                'transaction_costs': transaction_costs,
                'cumulative_costs': cumulative_costs,
                'bnh_returns': bnh_returns,
                'bnh_cum_returns': bnh_cum_returns,
                'bnh_portfolio_value': bnh_portfolio_value
            })

            # Calculate state metrics if possible
            unique_states = result['state'].unique()
            for state in unique_states:
                if state != 'Flat':
                    state_returns = returns[result['state'] == state]
                    if len(state_returns) > 1:
                        sharpe = (state_returns.mean() / state_returns.std()) * np.sqrt(365)
                        mean_ret = state_returns.mean()
                        logging.info(f"Test State {state}: Sharpe={sharpe:.2f}, Mean={mean_ret:.4f}, Count={len(state_returns)}")

            if save_path:
                result.to_csv(save_path)
                logging.info(f"Backtest results saved to {save_path}")
            logging.info(f"Final portfolio value: ${portfolio_value.iloc[-1]:.2f}, Total trading costs: ${cumulative_costs.iloc[-1]:.2f}")

            self.plot_backtest(result, initial_capital)
            return result

        except Exception as e:
            logging.error(f"Trading simulation failed: {str(e)}")
            raise

    def plot_backtest(self, result, initial_capital=1000):
        """Plot backtest results with trading costs, confidence levels, and regime-colored price."""
        try:
            fig = make_subplots(rows=5, cols=1, 
                              shared_xaxes=True, 
                              vertical_spacing=0.08,
                              subplot_titles=('Portfolio Performance', 'Price with Regimes', 'Trading State', 
                                            'Predicted State Confidence', 'Cumulative Trading Costs'),
                              row_heights=[0.35, 0.25, 0.15, 0.15, 0.1])

            fig.add_trace(go.Scatter(x=result.index, y=result['portfolio_value'], 
                                   mode='lines', name='Strategy', line=dict(color='blue', width=1)),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=result.index, y=result['bnh_portfolio_value'], 
                                   mode='lines', name='Buy & Hold', line=dict(color='gray', width=1)),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=[result.index[0], result.index[-1]], 
                                   y=[initial_capital, initial_capital],
                                   mode='lines', name='Initial Capital', 
                                   line=dict(color='black', dash='dash', width=0.8)),
                        row=1, col=1)

            state_colors = {}
            for raw_state in result['raw_state'].unique():
                mask = result['raw_state'] == raw_state
                if mask.any():
                    most_common_state = result.loc[mask, 'state'].mode()[0]
                    if most_common_state == 'Long':
                        state_colors[raw_state] = 'darkgreen'
                    elif most_common_state == 'Flat':
                        state_colors[raw_state] = 'darkred' if self.strategy == 'long-only' else 'grey'
                    else:
                        state_colors[raw_state] = 'darkred'

            for state in result['raw_state'].unique():
                mask = result['raw_state'] == state
                state_label = f"State {state} ({result.loc[mask, 'state'].mode()[0]})"
                fig.add_trace(go.Scatter(x=result.index[mask], y=result['Close'][mask],
                                      mode='markers', name=state_label,
                                      marker=dict(color=state_colors.get(state, 'black'), size=4)),
                            row=2, col=1)

            state_numeric = result['state'].map({'Long': 1, 'Flat': 0, 'Short': -1})
            fig.add_trace(go.Scatter(x=result.index, y=state_numeric, mode='lines', 
                                   name='Trading State', line=dict(color='green', width=1)),
                        row=3, col=1)

            if self.state_probabilities is not None and 'confidence' in self.state_probabilities.columns:
                fig.add_trace(go.Scatter(x=self.state_probabilities.index, 
                                       y=self.state_probabilities['confidence'],
                                       mode='lines', name='Predicted Confidence',
                                       line=dict(color='purple', width=1)),
                            row=4, col=1)
                fig.add_trace(go.Scatter(x=[self.state_probabilities.index[0], self.state_probabilities.index[-1]], 
                                       y=[self.confidence_threshold, self.confidence_threshold], 
                                       mode='lines', name='Threshold',
                                       line=dict(color="red", width=0.8, dash="dash")),
                            row=4, col=1)

            fig.add_trace(go.Scatter(x=result.index, y=result['cumulative_costs'], 
                                   mode='lines', name='Trading Costs', 
                                   line=dict(color='red', width=1)),
                        row=5, col=1)

            latest_state = result['state'].iloc[-1] 
            recommendation = f"Current Position: {latest_state} (as of {result.index[-1].strftime('%Y-%m-%d')})"
            fig.add_annotation(
                text=recommendation,
                xref="paper", yref="paper",
                x=0.5, y=0.95, showarrow=False,
                font=dict(size=12, color="black"),
                align="center"
            )

            fig.update_layout(height=1300,
                            width=1000,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(t=150, b=60, l=50, r=50))

            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, gridcolor='lightgray', zeroline=False, showline=True, linewidth=0.5, linecolor='lightgray')
            fig.update_yaxes(title_text="Price ($)", row=2, col=1, gridcolor='lightgray', zeroline=False, showline=True, linewidth=0.5, linecolor='lightgray')
            fig.update_yaxes(title_text="State", tickvals=[-1, 0, 1], ticktext=['Short', 'Flat', 'Long'], row=3, col=1, gridcolor='lightgray', zeroline=False, showline=True, linewidth=0.5, linecolor='lightgray')
            fig.update_yaxes(title_text="Confidence", range=[0, 1], row=4, col=1, gridcolor='lightgray', zeroline=False, showline=True, linewidth=0.5, linecolor='lightgray')
            fig.update_yaxes(title_text="Trading Costs ($)", row=5, col=1, gridcolor='lightgray', zeroline=False, showline=True, linewidth=0.5, linecolor='lightgray')
            fig.update_xaxes(title_text="Date", row=5, col=1, gridcolor='lightgray', showline=True, linewidth=0.5, linecolor='lightgray')

            fig.show()
            return fig
        except Exception as e:
            logging.error(f"Plotting backtest failed: {str(e)}")
            raise

    def evaluate(self, results):
        """Evaluate the performance of the trading strategy, using cumulative returns over state periods for classification metrics."""
        try:
            cum_returns = results['cum_returns'].fillna(1.0)
            returns = results['returns'].fillna(0.0)
            
            # Standard performance metrics
            ann_ret = (cum_returns.iloc[-1] ** (365/len(results))) - 1
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
            neg_returns = returns[returns < 0]
            sortino = (returns.mean() / neg_returns.std()) * np.sqrt(365) if len(neg_returns) > 0 else np.inf
            ann_vol = returns.std() * np.sqrt(365)
            drawdowns = cum_returns / cum_returns.cummax() - 1
            max_dd = drawdowns.min()
            state = results['state']
            switches = sum(state.iloc[i] != state.iloc[i-1] for i in range(1, len(state)))
            total_costs = results['cumulative_costs'].iloc[-1]
            
            bnh_returns = results['bnh_returns']
            bnh_cum_returns = results['bnh_cum_returns']
            bnh_ann_ret = (bnh_cum_returns.iloc[-1] ** (365/len(results))) - 1
            bnh_sharpe = (bnh_returns.mean() / bnh_returns.std()) * np.sqrt(365)
            bnh_neg_returns = bnh_returns[bnh_returns < 0]
            bnh_sortino = (bnh_returns.mean() / bnh_neg_returns.std()) * np.sqrt(365) if len(bnh_neg_returns) > 0 else np.inf
            bnh_ann_vol = bnh_returns.std() * np.sqrt(365)
            bnh_drawdowns = bnh_cum_returns / bnh_cum_returns.cummax() - 1
            bnh_max_dd = bnh_drawdowns.min()

            # Classification metrics: Evaluate based on cumulative returns over state periods
            # Step 1: Identify state periods (consecutive days with the same state)
            state_changes = (state != state.shift(1)).cumsum()
            # Create a list of periods
            periods = []
            current_state = state.iloc[0]
            start_idx = 0
            for i in range(1, len(state)):
                if state.iloc[i] != current_state:
                    periods.append((current_state, results.index[start_idx], results.index[i-1]))
                    current_state = state.iloc[i]
                    start_idx = i
            periods.append((current_state, results.index[start_idx], results.index[-1]))
            logging.info(f"Number of state periods evaluated: {len(periods)}")

            # Step 2: Compute cumulative returns for each state period
            cumulative_returns = []
            predicted_states = []
            for state_val, start_date, end_date in periods:
                # Cumulative return from start to end of the state period
                period_returns = results['returns'].loc[start_date:end_date]
                period_cum_return = (period_returns + 1).prod() - 1
                cumulative_returns.append(period_cum_return)
                predicted_states.append(state_val)

            # Step 3: Define ground truth based on cumulative returns over the state period
            return_threshold = 0.005  # Small threshold for Flat (e.g., ±0.5% cumulative)
            actual_labels = []
            for cum_ret in cumulative_returns:
                if cum_ret > return_threshold:
                    actual_labels.append('Long')
                elif cum_ret < -return_threshold:
                    actual_labels.append('Short' if self.strategy == 'long-short' else 'Flat')
                else:
                    actual_labels.append('Flat')

            # Step 4: Compute confusion matrix
            from sklearn.metrics import confusion_matrix
            labels = ['Long', 'Flat', 'Short'] if self.strategy == 'long-short' else ['Long', 'Flat']
            cm = confusion_matrix(actual_labels, predicted_states, labels=labels)
            cm_dict = {label: cm[i].tolist() for i, label in enumerate(labels)}

            # Step 5: Compute sensitivity, specificity, and accuracy
            if self.strategy == 'long-short':
                # For multi-class, compute metrics per class (one-vs-rest)
                sensitivity = {}
                specificity = {}
                for i, label in enumerate(labels):
                    true_positives = cm[i, i]
                    false_negatives = sum(cm[i, :]) - true_positives
                    false_positives = sum(cm[:, i]) - true_positives
                    true_negatives = cm.sum() - (true_positives + false_negatives + false_positives)
                    
                    sensitivity[label] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    specificity[label] = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            else:
                # For long-only (binary: Long vs. Flat)
                true_positives = cm[0, 0]  # Long predicted as Long
                false_negatives = cm[0, 1]  # Long predicted as Flat
                false_positives = cm[1, 0]  # Flat predicted as Long
                true_negatives = cm[1, 1]  # Flat predicted as Flat
                
                sensitivity = {'Long': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0}
                specificity = {'Flat': true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0}

            # Accuracy
            accuracy = (pd.Series(actual_labels) == pd.Series(predicted_states)).mean()

            # Create metrics table
            metrics_df = pd.DataFrame({
                'Metric': ['Annualized Return', 'Sharpe Ratio', 'Sortino Ratio', 'Annualized Volatility', 
                        'Maximum Drawdown', 'Final Cum Return', 'Switches', 'Total Trading Costs', 
                        'Accuracy'],
                f'HMM {self.strategy} Strategy': [ann_ret, sharpe, sortino, ann_vol, max_dd, cum_returns.iloc[-1], switches, total_costs, accuracy],
                'Buy & Hold': [bnh_ann_ret, bnh_sharpe, bnh_sortino, bnh_ann_vol, bnh_max_dd, bnh_cum_returns.iloc[-1], 'N/A', 'N/A', 'N/A']
            })

            # Add sensitivity and specificity for each relevant label using pd.concat
            additional_metrics = []
            for label in sensitivity:
                additional_metrics.append(pd.DataFrame({
                    'Metric': [f'Sensitivity ({label})'],
                    f'HMM {self.strategy} Strategy': [sensitivity[label]],
                    'Buy & Hold': ['N/A']
                }))
            for label in specificity:
                additional_metrics.append(pd.DataFrame({
                    'Metric': [f'Specificity ({label})'],
                    f'HMM {self.strategy} Strategy': [specificity[label]],
                    'Buy & Hold': ['N/A']
                }))
            metrics_df = pd.concat([metrics_df] + additional_metrics, ignore_index=True)

            metrics_df.set_index('Metric', inplace=True)
            logging.info(f"Evaluation metrics:\n{metrics_df}")

            # Prepare classification metrics for metadata
            classification_metrics = {
                'confusion_matrix': cm_dict,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'accuracy': accuracy
            }

            return metrics_df, classification_metrics

        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")
            raise