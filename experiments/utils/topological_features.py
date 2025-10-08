"""
Topological Feature Extraction Utilities

This module provides tools for extracting topological features from time series data
using time delay embedding and persistent homology. All functions are designed to
prevent data leakage by ensuring features at time t only use data up to time t.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from ripser import ripser
from scipy.spatial.distance import cdist


def time_delay_embedding(series: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    Create time delay embedding of the series - FIXED to prevent future data leakage.

    Parameters:
    -----------
    series : np.ndarray
        Input time series
    tau : int
        Time delay
    m : int
        Embedding dimension

    Returns:
    --------
    np.ndarray
        Time delay embedded data of shape (embedded_length, m)
    """
    n = len(series)
    if n < (m - 1) * tau + 1:
        return np.array([])

    # CRITICAL FIX: Embed backwards in time to prevent future data leakage
    # Instead of looking forward, we look backward from each point
    embedded_length = n - (m - 1) * tau
    embedded = np.zeros((embedded_length, m))

    for i in range(embedded_length):
        for j in range(m):
            # FIXED: Look backward in time instead of forward
            # This ensures we only use past data for each embedded point
            embedded[i, j] = series[i + (m - 1 - j) * tau]

    return embedded


def extract_topological_features(
    data: pd.DataFrame,
    window_length: int,
    selected_cols: List[str],
    tau: int = 3,
    embedding_dim: int = 3,
    max_dimension: int = 2
) -> pd.DataFrame:
    """
    Extract topological features from sliding windows of univariate time series data using time delay embedding.

    IMPORTANT: This function is designed to prevent future data leakage for ML applications.
    Features for window i are computed using only data up to and including time i.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with timestamp index
    window_length : int
        Length of sliding window
    selected_cols : List[str]
        List of column names to use for topological analysis (should be single column for univariate)
    tau : int, default=3
        Time delay for embedding (fixed value)
    embedding_dim : int, default=3
        Embedding dimension
    max_dimension : int, default=1
        Maximum homology dimension to compute

    Returns:
    --------
    pd.DataFrame
        DataFrame with topological features for each window, indexed by the END timestamp of each window
    """

    # Select the univariate data
    if len(selected_cols) > 1:
        print("Warning: Multiple columns provided, using only the first one for univariate analysis")

    selected_data = data[selected_cols[0]].dropna()

    # CRITICAL FIX: Ensure no future data leakage
    # For each window ending at time t, we use data from [t-window_length+1, t]
    # This means features computed at time t use only data available up to time t
    n_windows = len(selected_data) - window_length + 1

    results = []
    previous_diagrams = None

    for i in range(n_windows):
        # FIXED: Window now correctly uses data ending at time i+window_length-1
        # This ensures that features at timestamp t use only data up to and including t
        window_series = selected_data.iloc[i:i+window_length].values

        # Get the timestamp for the END of this window (when features become available)
        window_end_timestamp = selected_data.index[i + window_length - 1]

        # Normalize the window (z-score normalization)
        window_mean = np.mean(window_series)
        window_std = np.std(window_series)
        if window_std > 0:
            normalized_window = (window_series - window_mean) / window_std
        else:
            normalized_window = window_series - window_mean

        # Create time delay embedding with fixed tau
        embedded_data = time_delay_embedding(normalized_window, tau, embedding_dim)

        # Skip if embedding is too small
        if len(embedded_data) < 4:  # Need at least 4 points for meaningful topology
            continue

        # Compute persistent homology
        diagrams = ripser(embedded_data, maxdim=max_dimension)['dgms']

        # Initialize features for this window with proper timestamp
        features = {
            'timestamp': window_end_timestamp,  # When these features become available
        }

        # Extract features for each dimension
        for dim in range(len(diagrams)):
            diagram = diagrams[dim]

            if len(diagram) > 0:
                # Remove infinite persistence points for finite calculations
                finite_diagram = diagram[diagram[:, 1] != np.inf]

                # Number of holes (Betti number)
                num_holes = len(finite_diagram)
                features[f'num_holes_{dim}'] = num_holes
                features[f'betti_{dim}'] = num_holes  # Compatibility

                if len(finite_diagram) > 0:
                    # Persistence (death - birth) = hole lifetimes
                    persistence = finite_diagram[:, 1] - finite_diagram[:, 0]

                    # Maximum hole lifetime
                    max_hole_lifetime = np.max(persistence)
                    features[f'max_hole_lifetime_{dim}'] = max_hole_lifetime

                    # Average lifetime of all holes
                    avg_hole_lifetime = np.mean(persistence)
                    features[f'avg_hole_lifetime_{dim}'] = avg_hole_lifetime

                    # L1 norm of persistence
                    l1_norm = np.sum(persistence)
                    features[f'l1_norm_{dim}'] = l1_norm

                    # L2 norm of persistence
                    l2_norm = np.sqrt(np.sum(persistence**2))
                    features[f'l2_norm_{dim}'] = l2_norm

                    # Sum of hole lifetimes (same as L1 norm, but explicit)
                    sum_hole_lifetimes = np.sum(persistence)
                    features[f'sum_hole_lifetimes_{dim}'] = sum_hole_lifetimes

                    # Persistence entropy
                    if l1_norm > 0:
                        p_i = persistence / l1_norm
                        # Avoid log(0) by filtering out zero probabilities
                        p_i_nonzero = p_i[p_i > 0]
                        if len(p_i_nonzero) > 0:
                            persistence_entropy = -np.sum(p_i_nonzero * np.log(p_i_nonzero))
                        else:
                            persistence_entropy = 0
                    else:
                        persistence_entropy = 0
                    features[f'persistence_entropy_{dim}'] = persistence_entropy

                    # Standard deviation of persistence (hole lifetime variability)
                    std_persistence = np.std(persistence)
                    features[f'std_persistence_{dim}'] = std_persistence
                else:
                    # No finite persistence points
                    _set_zero_features(features, dim)
            else:
                # No topological features found
                _set_zero_features(features, dim)

            # Calculate Wasserstein distance from previous window
            if previous_diagrams is not None and dim < len(previous_diagrams):
                wasserstein_dist = _calculate_wasserstein_distance(
                    previous_diagrams[dim], diagram
                )
                features[f'wasserstein_{dim}'] = wasserstein_dist
            else:
                # First window or dimension doesn't exist in previous
                features[f'wasserstein_{dim}'] = 0

        # Store current diagrams for next iteration
        previous_diagrams = diagrams
        results.append(features)

    # Convert to DataFrame and set proper timestamp index
    df = pd.DataFrame(results)

    # CRITICAL: Set timestamp as index to ensure proper alignment with original data
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)

    return df


def _set_zero_features(features: Dict[str, Any], dim: int) -> None:
    """Set all topological features to zero for a given dimension."""
    features[f'num_holes_{dim}'] = 0
    features[f'betti_{dim}'] = 0
    features[f'max_hole_lifetime_{dim}'] = 0
    features[f'avg_hole_lifetime_{dim}'] = 0
    features[f'l1_norm_{dim}'] = 0
    features[f'l2_norm_{dim}'] = 0
    features[f'sum_hole_lifetimes_{dim}'] = 0
    features[f'persistence_entropy_{dim}'] = 0
    features[f'std_persistence_{dim}'] = 0


def _calculate_wasserstein_distance(prev_diagram: np.ndarray, curr_diagram: np.ndarray) -> float:
    """
    Calculate Wasserstein distance between two persistence diagrams using optimal transport.

    Parameters:
    -----------
    prev_diagram : np.ndarray
        Previous persistence diagram
    curr_diagram : np.ndarray
        Current persistence diagram

    Returns:
    --------
    float
        Wasserstein distance
    """
    try:
        from persim import wasserstein
        
        # Remove infinite persistence points for both diagrams
        prev_finite = prev_diagram[prev_diagram[:, 1] != np.inf]
        curr_finite = curr_diagram[curr_diagram[:, 1] != np.inf]
        
        # Calculate actual Wasserstein distance using persim
        if len(prev_finite) == 0 and len(curr_finite) == 0:
            return 0.0
        elif len(prev_finite) == 0:
            # Distance from empty diagram to curr_finite
            return wasserstein(np.array([]).reshape(0, 2), curr_finite)
        elif len(curr_finite) == 0:
            # Distance from prev_finite to empty diagram
            return wasserstein(prev_finite, np.array([]).reshape(0, 2))
        else:
            # Both diagrams have points
            return wasserstein(prev_finite, curr_finite)
            
    except ImportError:
        # Fallback to scipy's implementation if persim is not available
        from scipy.optimize import linear_sum_assignment
        
        # Remove infinite persistence points for both diagrams
        prev_finite = prev_diagram[prev_diagram[:, 1] != np.inf]
        curr_finite = curr_diagram[curr_diagram[:, 1] != np.inf]
        
        if len(prev_finite) == 0 and len(curr_finite) == 0:
            return 0.0
        
        # Add diagonal projections for unmatched points
        # For each point (b, d), its diagonal projection is ((b+d)/2, (b+d)/2)
        prev_diag = np.array([[(p[0] + p[1])/2, (p[0] + p[1])/2] for p in prev_finite])
        curr_diag = np.array([[(p[0] + p[1])/2, (p[0] + p[1])/2] for p in curr_finite])
        
        # Create extended diagrams with diagonal projections
        if len(prev_finite) > 0 and len(curr_finite) > 0:
            extended_prev = np.vstack([prev_finite, curr_diag])
            extended_curr = np.vstack([curr_finite, prev_diag])
        elif len(prev_finite) > 0:
            # Only previous has points
            extended_prev = prev_finite
            extended_curr = prev_diag
        else:
            # Only current has points
            extended_prev = curr_diag
            extended_curr = curr_finite
        
        # Calculate cost matrix (L-infinity norm)
        cost_matrix = cdist(extended_prev, extended_curr, metric='chebyshev')
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Return total cost
        return cost_matrix[row_indices, col_indices].sum()


def extract_multi_series_topological_features(
    data: pd.DataFrame,
    series_configs: Dict[str, Dict[str, Any]],
    default_window_length: int = 21,
    default_tau: int = 3,
    default_embedding_dim: int = 3,
    default_max_dimension: int = 1
) -> pd.DataFrame:
    """
    Extract topological features for multiple univariate series with individual configurations.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with timestamp index
    series_configs : Dict[str, Dict[str, Any]]
        Configuration for each series. Key is column name, value is config dict.
        Example: {
            'bitcoin_close': {'window_length': 21, 'tau': 3},
            'vix_close': {'window_length': 15, 'tau': 2}
        }
    default_* : Default values for parameters not specified in series_configs

    Returns:
    --------
    pd.DataFrame
        Combined topological features for all series
    """
    all_features = []

    for series_name, config in series_configs.items():
        if series_name not in data.columns:
            print(f"Warning: Series '{series_name}' not found in data. Skipping.")
            continue

        # Use series-specific config or defaults
        window_length = config.get('window_length', default_window_length)
        tau = config.get('tau', default_tau)
        embedding_dim = config.get('embedding_dim', default_embedding_dim)
        max_dimension = config.get('max_dimension', default_max_dimension)

        # Extract features for this series
        series_features = extract_topological_features(
            data=data,
            window_length=window_length,
            selected_cols=[series_name],
            tau=tau,
            embedding_dim=embedding_dim,
            max_dimension=max_dimension
        )

        # Add prefix to avoid column name collisions
        series_features = series_features.add_prefix(f'{series_name}_topo_')
        all_features.append(series_features)

    # Combine all features
    if all_features:
        # Start with the first feature set
        combined_features = all_features[0]

        # Merge with the rest
        for features in all_features[1:]:
            combined_features = combined_features.merge(
                features,
                left_index=True,
                right_index=True,
                how='outer'
            )

        return combined_features
    else:
        return pd.DataFrame()