"""
Signal data container for Soros System.

This module provides a container for signal-specific information,
including signal values, evaluation results, and meta-labeling capabilities.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import pickle
from sklearn.ensemble import RandomForestClassifier
import joblib


class SignalData:
    """
    Container for signal-specific information.
    
    Stores signal values, activation dates, evaluation results,
    and optimal decay period information.
    """
    
    def __init__(
        self, 
        signal_name: str, 
        asset_id: str, 
        values: Optional[pd.Series] = None
    ):
        """Initialize the signal data container.
        
        Args:
            signal_name: Name of the signal
            asset_id: ID of the asset this signal is for
            values: Series with signal values (index should be dates)
        """
        self.logger = logging.getLogger(__name__)
        self.signal_name = signal_name
        self.asset_id = asset_id
        self.values = values
        
        # Signal evaluation results
        self.is_effective = False
        self.weight = 0.0
        self.optimal_decay = 0
        self.evaluation_results = {}
        
        # Activation tracking
        self.activation_dates = []
        
        # Meta-labeling
        self.meta_model = None
        self.meta_features = []
        self.use_meta_labeling = False
        
    @property
    def name(self) -> str:
        """Alias for signal_name for compatibility.
        
        Returns:
            str: The name of the signal
        """
        return self.signal_name
        
    def set_values(self, values: pd.Series) -> None:
        """Set or update the signal values.
        
        Args:
            values: Series with signal values (index should be dates)
        """
        self.values = values
        self.logger.debug(f"Updated values for signal {self.signal_name} on {self.asset_id}")
        
        # Extract activation dates (0->1 transitions)
        self.extract_activation_dates()
        
    def extract_activation_dates(self) -> List[datetime]:
        """Extract dates where the signal activates (transitions from 0 to 1).
        
        Returns:
            list: List of activation dates
        """
        if self.values is None or len(self.values) < 2:
            self.activation_dates = []
            return []
            
        # Find transitions from 0 to 1
        transitions = (self.values.shift(1) == 0) & (self.values == 1)
        
        # Get dates where transitions occur
        activation_dates = self.values.index[transitions].tolist()
        self.activation_dates = activation_dates
        
        self.logger.debug(
            f"Extracted {len(activation_dates)} activation dates for signal {self.signal_name} on {self.asset_id}"
        )
        
        return activation_dates
        
    def set_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Set or update the evaluation results for this signal.
        
        Args:
            results: Dictionary with evaluation results from SignalEvaluator
        """
        self.evaluation_results = results
        
        # Extract key metrics
        self.is_effective = results.get('overall_effectiveness', False)
        self.weight = results.get('weight', 0.0)
        self.optimal_decay = results.get('optimal_decay', 0)
        
        self.logger.debug(
            f"Updated evaluation for signal {self.signal_name} on {self.asset_id}: "
            f"effective={self.is_effective}, weight={self.weight:.4f}, "
            f"decay={self.optimal_decay}"
        )
        
    def get_activations_in_range(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[datetime]:
        """Get activation dates within a specified range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            list: List of activation dates within the range
        """
        if not self.activation_dates:
            return []
            
        filtered_dates = self.activation_dates
        
        if start_date is not None:
            filtered_dates = [d for d in filtered_dates if d >= start_date]
            
        if end_date is not None:
            filtered_dates = [d for d in filtered_dates if d <= end_date]
            
        return filtered_dates
        
    def train_meta_model(
        self, 
        price_data: pd.DataFrame,
        feature_columns: List[str],
        forward_returns_window: int = None,
        min_return_threshold: float = 0.0,
        verbose: bool = False
    ) -> Tuple[float, int]:
        """Train a meta-labeling model to filter signal activations.
        
        Args:
            price_data: DataFrame with price data
            feature_columns: List of column names to use as features
            forward_returns_window: Window size for forward returns (defaults to optimal_decay)
            min_return_threshold: Minimum return to consider a trade successful
            verbose: Whether to print detailed information
            
        Returns:
            tuple: (accuracy, sample_count) of the trained model
        """
        if not self.is_effective:
            self.logger.warning(
                f"Cannot train meta-model for ineffective signal {self.signal_name} on {self.asset_id}"
            )
            return 0.0, 0
            
        if forward_returns_window is None:
            forward_returns_window = self.optimal_decay
            
        if forward_returns_window <= 0:
            self.logger.warning(
                f"Invalid forward returns window for signal {self.signal_name} on {self.asset_id}"
            )
            return 0.0, 0
            
        self.logger.info(
            f"Training meta-model for signal {self.signal_name} on {self.asset_id} "
            f"with {len(feature_columns)} features"
        )
        
        # Get activation dates (where signal = 1)
        activations = self.values[self.values == 1]
        
        if len(activations) < 10:
            self.logger.warning(
                f"Insufficient activations ({len(activations)}) for meta-labeling "
                f"signal {self.signal_name} on {self.asset_id}"
            )
            return 0.0, 0
            
        # Calculate forward returns for each activation
        activation_dates = activations.index
        forward_returns = []
        
        for date in activation_dates:
            try:
                # Find the price at activation
                entry_price = price_data.loc[date, 'close']
                
                # Find the forward price after decay days
                forward_date_idx = price_data.index.get_loc(date) + forward_returns_window
                if forward_date_idx < len(price_data):
                    forward_date = price_data.index[forward_date_idx]
                    exit_price = price_data.loc[forward_date, 'close']
                    
                    # Calculate return
                    ret = (exit_price / entry_price) - 1
                    forward_returns.append(ret)
                else:
                    forward_returns.append(np.nan)
            except (KeyError, IndexError):
                forward_returns.append(np.nan)
                
        # Create target labels (1 if return > threshold, 0 otherwise)
        forward_returns = pd.Series(forward_returns, index=activation_dates)
        labels = (forward_returns > min_return_threshold).astype(int)
        
        # Drop NaNs
        labels = labels.dropna()
        
        if len(labels) < 10:
            self.logger.warning(
                f"Insufficient labels ({len(labels)}) after dropping NaNs for meta-labeling "
                f"signal {self.signal_name} on {self.asset_id}"
            )
            return 0.0, 0
        
        # Extract features for each activation date
        features = []
        valid_dates = []
        
        for date in labels.index:
            try:
                feature_values = price_data.loc[date, feature_columns].values
                if not np.isnan(feature_values).any():
                    features.append(feature_values)
                    valid_dates.append(date)
            except (KeyError, IndexError):
                continue
                
        if len(features) < 10:
            self.logger.warning(
                f"Insufficient features ({len(features)}) for meta-labeling "
                f"signal {self.signal_name} on {self.asset_id}"
            )
            return 0.0, 0
            
        # Convert to numpy arrays for modeling
        features = np.array(features)
        labels = labels.loc[valid_dates].values
        
        if verbose:
            self.logger.info(f"Training with {len(features)} samples, {len(feature_columns)} features")
            self.logger.info(f"Class distribution: {np.bincount(labels)}")
            
        # Train random forest classifier
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=3, 
            random_state=42
        )
        
        try:
            model.fit(features, labels)
            accuracy = model.score(features, labels)
            
            # Store the model and features
            self.meta_model = model
            self.meta_features = feature_columns
            self.use_meta_labeling = True
            
            if verbose:
                self.logger.info(
                    f"Meta-model for {self.signal_name} on {self.asset_id} - "
                    f"In-sample accuracy: {accuracy:.4f}"
                )
                
                # Print feature importances
                importances = model.feature_importances_
                feature_importances = pd.Series(importances, index=feature_columns)
                top_features = feature_importances.sort_values(ascending=False).head(5)
                
                self.logger.info(f"Top 5 features: {top_features}")
                
            return accuracy, len(features)
            
        except Exception as e:
            self.logger.error(
                f"Error training meta-model for {self.signal_name} on {self.asset_id}: {e}"
            )
            self.use_meta_labeling = False
            return 0.0, 0
        
    def predict_meta_label(self, date: datetime, price_data: pd.DataFrame) -> bool:
        """Predict whether a signal activation should be taken using meta-labeling.
        
        Args:
            date: Date of the signal activation
            price_data: DataFrame with price data
            
        Returns:
            bool: True if the signal should be taken, False otherwise
        """
        if not self.use_meta_labeling or self.meta_model is None:
            return True
            
        try:
            # Extract features for the date
            features = price_data.loc[date, self.meta_features].values.reshape(1, -1)
            
            # Predict
            prediction = self.meta_model.predict(features)[0]
            probability = self.meta_model.predict_proba(features)[0, 1]
            
            self.logger.debug(
                f"Meta-label for {self.signal_name} on {date}: "
                f"{'TAKE' if prediction == 1 else 'SKIP'} (probability={probability:.4f})"
            )
            
            return prediction == 1
            
        except Exception as e:
            self.logger.error(
                f"Error predicting meta-label for {self.signal_name} on {date}: {e}"
            )
            return True
        
    def save_meta_model(self, path: str) -> bool:
        """Save the meta-model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.meta_model:
            self.logger.warning(f"No meta-model to save for {self.signal_name} on {self.asset_id}")
            return False
            
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            joblib.dump(
                {
                    'model': self.meta_model,
                    'features': self.meta_features,
                    'signal_name': self.signal_name,
                    'asset_id': self.asset_id,
                    'timestamp': datetime.now().isoformat()
                },
                path
            )
            
            self.logger.info(f"Saved meta-model for {self.signal_name} on {self.asset_id} to {path}")
            return True
            
        except Exception as e:
            self.logger.error(
                f"Error saving meta-model for {self.signal_name} on {self.asset_id}: {e}"
            )
            return False
    
    def load_meta_model(self, path: str) -> bool:
        """Load a meta-model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load model
            data = joblib.load(path)
            
            # Verify signal and asset
            if data['signal_name'] != self.signal_name or data['asset_id'] != self.asset_id:
                self.logger.warning(
                    f"Model mismatch: expected {self.signal_name}/{self.asset_id}, "
                    f"got {data['signal_name']}/{data['asset_id']}"
                )
                
            # Set model and features
            self.meta_model = data['model']
            self.meta_features = data['features']
            self.use_meta_labeling = True
            
            self.logger.info(
                f"Loaded meta-model for {self.signal_name} on {self.asset_id} from {path} "
                f"(saved at {data['timestamp']})"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Error loading meta-model for {self.signal_name} on {self.asset_id}: {e}"
            )
            return False 