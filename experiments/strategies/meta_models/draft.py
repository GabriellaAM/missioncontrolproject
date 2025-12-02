#%%
import pandas as pd 
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import after adding to path
from utils.feature_loader import FeatureLoader

def fractional_diff(series, d):
    """
    Compute fractional differentiation of a time series.
    
    Parameters:
    series: pandas Series, the time series to differentiate
    d: float, the fractional differentiation parameter (0 < d < 1)
    
    Returns:
    pandas Series with fractionally differentiated values
    """
    from scipy.special import gamma
    
    # Convert to numpy array for computation
    x = series.values
    n = len(x)
    
    # Compute binomial coefficients for fractional differentiation
    weights = np.zeros(n)
    weights[0] = 1
    
    for k in range(1, n):
        weights[k] = weights[k-1] * (d - k + 1) / k
    
    # Apply fractional differentiation
    frac_diff = np.zeros(n)
    for i in range(n):
        frac_diff[i] = np.sum(weights[:i+1] * x[i::-1])
    
    # Return as pandas Series with original index
    return pd.Series(frac_diff, index=series.index)

asset = ['bitcoin']

feats = FeatureLoader(start_date='2020-01-01', end_date='2025-11-26')

feats = feats.build_feature_set(
    crypto_assets=asset[0],
    fred_indicators=['creditSpreads', 'treasury5YInflationExpectation'],
    yahoo_tickers=['vix', 'move'],
    calculated_features={'yieldCurveRegime': ['regime', 'spread_2s10s'],
                         'rty_ym_ratio': 'rty'}
)
#feats.rename(columns={'value': 'rty_ym_ratio'}, inplace=True)
feats.set_index('timestamp', inplace=True)

feats['bitcoin_close_log_return_1'] = np.log(feats['bitcoin_close'] / feats['bitcoin_close'].shift(1))

# Add fractionally differentiated bitcoin_close with parameter 0.4
feats['bitcoin_close_fracdiff'] = fractional_diff(feats['bitcoin_close'], 0.4)

# Generate lagged features for vix_close, move_close, creditspreads, treasury5yinflationexpectations
lag_features = ['bitcoin_close', 'vix_close', 'move_close', 'creditspreads', 'treasury5yinflationexpectation', 'spread_2s10s']
lags = [1, 10, 30]

for feature in lag_features:
    if feature in feats.columns:
        for lag in lags:
            # Calculate log returns and handle potential inf/nan values
            log_returns = np.log(feats[feature] / feats[feature].shift(lag))
            # Replace inf and -inf with NaN, but keep zeros as they are valid log returns
            log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
            # Forward fill any NaN values to prevent gaps in the date index
            log_returns = log_returns.fillna(method='ffill')
            feats[f'{feature}_log_return_{lag}'] = log_returns

feats.dropna(inplace=True)

feats['ema_10'] = feats['bitcoin_close'].ewm(span=10, adjust=False).mean()


# %% 

from ripser import ripser
from persim import plot_diagrams
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_topological_features(data, window_length, selected_cols, tau=3, embedding_dim=3, max_dimension=1):
    """
    Extract topological features from sliding windows of univariate time series data using time delay embedding.
    
    IMPORTANT: This function is designed to prevent future data leakage for ML applications.
    Features for window i are computed using only data up to and including time i.
    
    Parameters:
    data: pandas DataFrame with time series data
    window_length: int, length of sliding window
    selected_cols: list of column names to use for topological analysis (should be single column for univariate)
    tau: int, time delay for embedding (fixed value, default=3)
    embedding_dim: int, embedding dimension (default=3)
    max_dimension: int, maximum homology dimension to compute (default=1)
    
    Returns:
    pandas DataFrame with topological features for each window, indexed by the END timestamp of each window
    """
    
    def time_delay_embedding(series, tau, m):
        """Create time delay embedding of the series"""
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
            'window_start_idx': i, 
            'window_end_idx': i + window_length - 1,
            'tau_used': tau
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
                
                # Also keep the original betti number name for compatibility
                features[f'betti_{dim}'] = num_holes
                
                if len(finite_diagram) > 0:
                    # Persistence (death - birth) = hole lifetimes
                    persistence = finite_diagram[:, 1] - finite_diagram[:, 0]
                    
                    # Sum of persistences
                    sum_persistence = np.sum(persistence)
                    features[f'sum_persistence_{dim}'] = sum_persistence
                    
                    # Maximum hole lifetime
                    max_hole_lifetime = np.max(persistence)
                    features[f'max_hole_lifetime_{dim}'] = max_hole_lifetime
                    
                    # Average lifetime of all holes
                    avg_hole_lifetime = np.mean(persistence)
                    features[f'avg_hole_lifetime_{dim}'] = avg_hole_lifetime
                    
                    # Number of relevant holes (persistence > threshold)
                    # Use a threshold based on the standard deviation of persistence values
                    if len(persistence) > 1:
                        persistence_threshold = np.mean(persistence)# + 0.5 * np.std(persistence)
                    else:
                        persistence_threshold = np.mean(persistence)
                    
                    num_relevant_holes = np.sum(persistence > persistence_threshold)
                    features[f'num_relevant_holes_{dim}'] = num_relevant_holes
                    
                    # L1 norm of persistence
                    l1_norm = np.sum(persistence)
                    features[f'l1_norm_{dim}'] = l1_norm
                    
                    # L2 norm of persistence
                    l2_norm = np.sqrt(np.sum(persistence**2))
                    features[f'l2_norm_{dim}'] = l2_norm
                    
                    # L3 norm of persistence
                    l3_norm = np.power(np.sum(persistence**3), 1/3)
                    features[f'l3_norm_{dim}'] = l3_norm
                    
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
                    
                    # Maximum persistence (same as max_hole_lifetime, kept for compatibility)
                    features[f'max_persistence_{dim}'] = max_hole_lifetime
                    
                    # Mean persistence (same as avg_hole_lifetime, kept for compatibility)
                    features[f'mean_persistence_{dim}'] = avg_hole_lifetime
                    
                    # Standard deviation of persistence (hole lifetime variability)
                    std_persistence = np.std(persistence)
                    features[f'std_persistence_{dim}'] = std_persistence
                else:
                    # No finite persistence points
                    features[f'sum_persistence_{dim}'] = 0
                    features[f'max_hole_lifetime_{dim}'] = 0
                    features[f'avg_hole_lifetime_{dim}'] = 0
                    features[f'num_relevant_holes_{dim}'] = 0
                    features[f'l1_norm_{dim}'] = 0
                    features[f'l2_norm_{dim}'] = 0
                    features[f'l3_norm_{dim}'] = 0
                    features[f'persistence_entropy_{dim}'] = 0
                    features[f'max_persistence_{dim}'] = 0
                    features[f'mean_persistence_{dim}'] = 0
                    features[f'std_persistence_{dim}'] = 0
            else:
                # No topological features found
                features[f'num_holes_{dim}'] = 0
                features[f'betti_{dim}'] = 0
                features[f'sum_persistence_{dim}'] = 0
                features[f'max_hole_lifetime_{dim}'] = 0
                features[f'avg_hole_lifetime_{dim}'] = 0
                features[f'num_relevant_holes_{dim}'] = 0
                features[f'l1_norm_{dim}'] = 0
                features[f'l2_norm_{dim}'] = 0
                features[f'l3_norm_{dim}'] = 0
                features[f'persistence_entropy_{dim}'] = 0
                features[f'max_persistence_{dim}'] = 0
                features[f'mean_persistence_{dim}'] = 0
                features[f'std_persistence_{dim}'] = 0
            
            # Calculate Wasserstein distance from previous window
            if previous_diagrams is not None and dim < len(previous_diagrams):
                prev_diagram = previous_diagrams[dim]
                curr_diagram = diagram
                
                # Remove infinite persistence points for both diagrams
                prev_finite = prev_diagram[prev_diagram[:, 1] != np.inf]
                curr_finite = curr_diagram[curr_diagram[:, 1] != np.inf]
                
                # Calculate Wasserstein distance (1-Wasserstein distance)
                if len(prev_finite) > 0 and len(curr_finite) > 0:
                    # Simple approximation of 1-Wasserstein distance
                    from scipy.spatial.distance import cdist
                    
                    # If diagrams have different sizes, pad the smaller one
                    if len(prev_finite) != len(curr_finite):
                        # Add points on the diagonal for padding
                        if len(prev_finite) < len(curr_finite):
                            # Pad previous diagram
                            n_pad = len(curr_finite) - len(prev_finite)
                            diagonal_points = np.array([[0, 0]] * n_pad)
                            prev_finite = np.vstack([prev_finite, diagonal_points])
                        else:
                            # Pad current diagram
                            n_pad = len(prev_finite) - len(curr_finite)
                            diagonal_points = np.array([[0, 0]] * n_pad)
                            curr_finite = np.vstack([curr_finite, diagonal_points])
                    
                    # Calculate pairwise distances and find minimum matching
                    distances = cdist(prev_finite, curr_finite, metric='euclidean')
                    # Simple approximation: sum of minimum distances
                    wasserstein_dist = np.sum(np.min(distances, axis=1))
                elif len(prev_finite) == 0 and len(curr_finite) == 0:
                    wasserstein_dist = 0
                else:
                    # One diagram is empty, distance is the sum of persistence of the non-empty one
                    if len(curr_finite) > 0:
                        wasserstein_dist = np.sum(curr_finite[:, 1] - curr_finite[:, 0])
                    else:
                        wasserstein_dist = np.sum(prev_finite[:, 1] - prev_finite[:, 0])
                
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
    
    # FIXED: Moving averages must also prevent future data leakage
    # Use only past data for rolling calculations
    for dim in range(max_dimension + 1):
        if f'num_holes_{dim}' in df.columns:
            # Rolling with min_periods ensures we don't use future data
            df[f'num_holes_{dim}_ma5'] = df[f'num_holes_{dim}'].rolling(window=5, min_periods=1).mean()
            df[f'num_holes_{dim}_ma10'] = df[f'num_holes_{dim}'].rolling(window=10, min_periods=1).mean()
        if f'sum_persistence_{dim}' in df.columns:
            df[f'sum_persistence_{dim}_ma5'] = df[f'sum_persistence_{dim}'].rolling(window=5, min_periods=1).mean()
            df[f'sum_persistence_{dim}_ma10'] = df[f'sum_persistence_{dim}'].rolling(window=10, min_periods=1).mean()
        if f'max_hole_lifetime_{dim}' in df.columns:
            df[f'max_hole_lifetime_{dim}_ma5'] = df[f'max_hole_lifetime_{dim}'].rolling(window=5, min_periods=1).mean()
        if f'avg_hole_lifetime_{dim}' in df.columns:
            df[f'avg_hole_lifetime_{dim}_ma5'] = df[f'avg_hole_lifetime_{dim}'].rolling(window=5, min_periods=1).mean()
        if f'num_relevant_holes_{dim}' in df.columns:
            df[f'num_relevant_holes_{dim}_ma5'] = df[f'num_relevant_holes_{dim}'].rolling(window=5, min_periods=1).mean()
        if f'l1_norm_{dim}' in df.columns:
            df[f'l1_norm_{dim}_ma5'] = df[f'l1_norm_{dim}'].rolling(window=5, min_periods=1).mean()
            df[f'l1_norm_{dim}_ma10'] = df[f'l1_norm_{dim}'].rolling(window=10, min_periods=1).mean()
        if f'persistence_entropy_{dim}' in df.columns:
            df[f'persistence_entropy_{dim}_ma5'] = df[f'persistence_entropy_{dim}'].rolling(window=5, min_periods=1).mean()
    
    # Calculate differences (first differences) for dimension 1 features
    if 1 <= max_dimension:
        if f'sum_persistence_1' in df.columns:
            df[f'sum_persistence_1_diff'] = df[f'sum_persistence_1'].diff()
        if f'l1_norm_1' in df.columns:
            df[f'l1_norm_1_diff'] = df[f'l1_norm_1'].diff()
        if f'l2_norm_1' in df.columns:
            df[f'l2_norm_1_diff'] = df[f'l2_norm_1'].diff()
        if f'avg_hole_lifetime_1' in df.columns:
            df[f'avg_hole_lifetime_1_diff'] = df[f'avg_hole_lifetime_1'].diff()
    
    return df

# Example usage with user-defined parameters
window_length = 50  # User can modify this

feats['log_ohlc'] = np.log((feats['bitcoin_open'] + feats['bitcoin_high'] + feats['bitcoin_low'] + feats['bitcoin_close'])/4)
feats['log_ohlc_return_1'] = np.log(feats['bitcoin_close'] / feats['bitcoin_close'].shift(1))

selected_columns = ['ema_10']  # User can modify this

# Extract topological features with fixed tau
topo_features = extract_topological_features(
    data=feats,
    window_length=window_length,
    selected_cols=selected_columns,
    tau=4,  # Fixed tau value
    max_dimension=2
)

print("Topological features extracted:")
print(topo_features.head())
print(f"\nFeature columns: {topo_features.columns.tolist()}")


################################################################################
################################################################################    
################################################################################
################################################################################


# %%

from ripser import ripser
from persim import PersistenceImager
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import ipywidgets as widgets
from IPython.display import display

def compute_persistence_images(data, window_length, selected_cols, tau=3, embedding_dim=3, max_dimension=1, 
                             pixel_size=0.1, birth_range=(0, 10), pers_range=(0, 5), kernel_params={'sigma': 1.0}):
    """
    Compute persistence images from sliding windows of univariate time series data using time delay embedding.
    
    Parameters:
    data: pandas DataFrame with time series data
    window_length: int, length of sliding window
    selected_cols: list of column names to use for topological analysis
    tau: int, time delay for embedding (default=3)
    embedding_dim: int, embedding dimension (default=3)
    max_dimension: int, maximum homology dimension to compute (default=1)
    pixel_size: float, resolution of persistence image (default=0.1)
    birth_range: tuple, range for birth times (default=(0, 10))
    pers_range: tuple, range for persistence values (default=(0, 5))
    kernel_params: dict, parameters for Gaussian kernel (default={'sigma': 1.0})
    
    Returns:
    dict containing persistence images for each dimension and window
    """
    
    def time_delay_embedding(series, tau, m):
        """Create time delay embedding of the series"""
        n = len(series)
        if n < (m - 1) * tau + 1:
            return np.array([])
        
        embedded_length = n - (m - 1) * tau
        embedded = np.zeros((embedded_length, m))
        
        for i in range(embedded_length):
            for j in range(m):
                embedded[i, j] = series[i + (m - 1 - j) * tau]
        
        return embedded
    
    # Select the univariate data
    if len(selected_cols) > 1:
        print("Warning: Multiple columns provided, using only the first one for univariate analysis")
    
    selected_data = data[selected_cols[0]].dropna()
    
    # Initialize persistence imager
    pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=birth_range, 
                             pers_range=pers_range, kernel_params=kernel_params)
    
    # Store results - only for dimensions 0 and 1
    persistence_images = {'dim_0': [], 'dim_1': []}
    timestamps = []
    
    # Sliding window analysis
    for i in range(window_length, len(selected_data) + 1):
        window_data = selected_data.iloc[i-window_length:i]
        
        # Create time delay embedding
        embedded_data = time_delay_embedding(window_data.values, tau, embedding_dim)
        
        if len(embedded_data) == 0:
            # If embedding fails, append zeros
            persistence_images['dim_0'].append(np.zeros((int(pers_range[1]/pixel_size), int(birth_range[1]/pixel_size))))
            persistence_images['dim_1'].append(np.zeros((int(pers_range[1]/pixel_size), int(birth_range[1]/pixel_size))))
            timestamps.append(window_data.index[-1])
            continue
        
        # Standardize the embedded data
        scaler = StandardScaler()
        embedded_data_scaled = scaler.fit_transform(embedded_data)
        
        try:
            # Compute persistent homology
            diagrams = ripser(embedded_data_scaled, maxdim=max_dimension)['dgms']
            
            # Compute persistence images for dimensions 0 and 1 only
            for dim in [0, 1]:
                if dim < len(diagrams) and len(diagrams[dim]) > 0:
                    # Filter out infinite persistence points
                    finite_diagram = diagrams[dim][diagrams[dim][:, 1] != np.inf]
                    
                    if len(finite_diagram) > 0:
                        # Compute persistence image
                        pimg = pimgr.transform([finite_diagram])
                        persistence_images[f'dim_{dim}'].append(pimg[0])
                    else:
                        # No finite persistence points
                        persistence_images[f'dim_{dim}'].append(np.zeros((int(pers_range[1]/pixel_size), int(birth_range[1]/pixel_size))))
                else:
                    # No persistence points for this dimension
                    persistence_images[f'dim_{dim}'].append(np.zeros((int(pers_range[1]/pixel_size), int(birth_range[1]/pixel_size))))
            
        except Exception as e:
            print(f"Error computing persistence for window ending at {window_data.index[-1]}: {e}")
            # Append zeros for dimensions 0 and 1
            persistence_images['dim_0'].append(np.zeros((int(pers_range[1]/pixel_size), int(birth_range[1]/pixel_size))))
            persistence_images['dim_1'].append(np.zeros((int(pers_range[1]/pixel_size), int(birth_range[1]/pixel_size))))
        
        timestamps.append(window_data.index[-1])
    
    return persistence_images, timestamps

def extract_persistence_image_features(persistence_images, timestamps):
    """
    Extract statistical features from persistence images.
    
    Parameters:
    persistence_images: dict containing persistence images for each dimension
    timestamps: list of timestamps corresponding to each image
    
    Returns:
    pandas DataFrame with extracted features
    """
    features_list = []
    
    for i, timestamp in enumerate(timestamps):
        feature_dict = {'timestamp': timestamp}
        
        for dim_key, images in persistence_images.items():
            img = images[i]
            
            # Extract statistical features from the persistence image
            feature_dict[f'{dim_key}_mean'] = np.mean(img)
            feature_dict[f'{dim_key}_std'] = np.std(img)
            feature_dict[f'{dim_key}_max'] = np.max(img)
            feature_dict[f'{dim_key}_sum'] = np.sum(img)
            feature_dict[f'{dim_key}_entropy'] = -np.sum(img * np.log(img + 1e-10))  # Add small epsilon to avoid log(0)
            
            # Compute moments
            feature_dict[f'{dim_key}_skewness'] = stats.skew(img.flatten())
            feature_dict[f'{dim_key}_kurtosis'] = stats.kurtosis(img.flatten())
            
            # Compute percentiles
            flat_img = img.flatten()
            feature_dict[f'{dim_key}_p25'] = np.percentile(flat_img, 25)
            feature_dict[f'{dim_key}_p50'] = np.percentile(flat_img, 50)
            feature_dict[f'{dim_key}_p75'] = np.percentile(flat_img, 75)
            feature_dict[f'{dim_key}_p90'] = np.percentile(flat_img, 90)
            feature_dict[f'{dim_key}_p95'] = np.percentile(flat_img, 95)
            
            # Count non-zero pixels (active regions)
            feature_dict[f'{dim_key}_nonzero_count'] = np.count_nonzero(img)
            feature_dict[f'{dim_key}_nonzero_ratio'] = np.count_nonzero(img) / img.size
        
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list)

# Enable matplotlib widget backend for interactive plots
%matplotlib widget

def create_interactive_persistence_viewer(persistence_images, timestamps, bitcoin_prices=None):
    """
    Create an interactive viewer for persistence images with a slider.
    
    Parameters:
    persistence_images: dict containing persistence images for each dimension
    timestamps: list of timestamps corresponding to each image
    bitcoin_prices: optional pandas Series of bitcoin prices aligned with timestamps
    """
    
    # Adjust figure layout based on whether bitcoin prices are provided
    if bitcoin_prices is not None:
        fig = plt.figure(figsize=(15, 10))
        # Create grid layout with more space between subplots
        gs = fig.add_gridspec(2, len(persistence_images), height_ratios=[2, 1], hspace=0.4)
        
        # Create persistence image subplots in the top row
        axes = []
        for i in range(len(persistence_images)):
            ax = fig.add_subplot(gs[0, i])
            axes.append(ax)
        
        # Create bitcoin price subplot in the bottom row, spanning all columns
        price_ax = fig.add_subplot(gs[1, :])
    else:
        fig, axes = plt.subplots(1, len(persistence_images), figsize=(15, 5))
        if len(persistence_images) == 1:
            axes = [axes]
    
    # Initialize persistence image plots
    im_objects = []
    for i, (dim_key, images) in enumerate(persistence_images.items()):
        im = axes[i].imshow(images[0], cmap='viridis', origin='lower', aspect='auto')
        axes[i].set_title(f'{dim_key.replace("_", " ").title()}')
        axes[i].set_xlabel('Birth')
        axes[i].set_ylabel('Persistence')
        im_objects.append(im)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    # Add bitcoin price subplot if provided
    if bitcoin_prices is not None:
        price_line, = price_ax.plot(timestamps, bitcoin_prices, 'b-', alpha=0.7)
        price_marker, = price_ax.plot(timestamps[0], bitcoin_prices.iloc[0], 'ro', markersize=8)
        price_ax.set_xlabel('Date')
        price_ax.set_ylabel('Bitcoin Price')
        price_ax.set_title('Bitcoin Price Timeline')
        price_ax.grid(True, alpha=0.3)
    
    # Create slider
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(timestamps) - 1,
        step=1,
        description='Time Step:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    
    # Create date display
    date_display = widgets.HTML(
        value=f"<b>Date:</b> {timestamps[0].strftime('%Y-%m-%d')}"
    )
    
    # Update function
    def update_images(change):
        idx = change['new']
        
        # Update persistence images
        for i, (dim_key, images) in enumerate(persistence_images.items()):
            im_objects[i].set_array(images[idx])
            im_objects[i].set_clim(vmin=images[idx].min(), vmax=images[idx].max())
        
        # Update date display
        date_display.value = f"<b>Date:</b> {timestamps[idx].strftime('%Y-%m-%d')}"
        
        # Update bitcoin price marker if provided
        if bitcoin_prices is not None:
            price_marker.set_data([timestamps[idx]], [bitcoin_prices.iloc[idx]])
        
        fig.canvas.draw()
    
    # Connect slider to update function
    slider.observe(update_images, names='value')
    
    # Display widgets
    display(widgets.VBox([slider, date_display]))
    
    plt.tight_layout()
    plt.show()
    
    return fig, slider

selected_columns = ['bitcoin_close']  # User can modify this
window_length = 60
# Compute persistence images
print("Computing persistence images...")
persistence_images, pi_timestamps = compute_persistence_images(
    data=feats,
    window_length=window_length,
    selected_cols=selected_columns,
    tau=5,
    max_dimension=3,
    pixel_size=0.08,
    birth_range=(0, 3),
    pers_range=(0, 3),
    kernel_params={'sigma': 0.1}
)

# %%

# Extract features from persistence images
pi_features = extract_persistence_image_features(persistence_images, pi_timestamps)
pi_features.set_index('timestamp', inplace=True)

print("Persistence image features extracted:")
print(pi_features.head())
print(f"\nFeature columns: {pi_features.columns.tolist()}")

# Get aligned bitcoin prices for the interactive viewer
bitcoin_aligned = feats['bitcoin_close'].iloc[window_length-1:window_length-1+len(pi_timestamps)]

# Create interactive persistence image viewer
print("\nCreating interactive persistence image viewer...")
fig, slider = create_interactive_persistence_viewer(
    persistence_images, 
    pi_timestamps, 
    bitcoin_aligned
)

################################################################################
################################################################################
################################################################################


# %% 

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Get the aligned timestamps for x-axis
aligned_timestamps = feats.index[window_length-1:window_length-1+len(topo_features)]


topo_features['l2_norm_1_ma7'] = topo_features['avg_hole_lifetime_1'].rolling(window=7, min_periods=1).mean()
topo_features['l1_norm_1_diff'] = topo_features['avg_hole_lifetime_0'].diff(7)

# Calculate first differences of the norms
l2_norm_diff = topo_features['wasserstein_1']
l1_norm_diff = topo_features['wasserstein_0']

# Calculate quantiles for both norms
l2_quantiles = np.quantile(l2_norm_diff.dropna(), [0.2, 0.4, 0.6, 0.8])
l1_quantiles = np.quantile(l1_norm_diff.dropna(), [0.2, 0.4, 0.6, 0.8])

# Create subplots with secondary y-axes
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('L2 Norm (Dimension 1) First Difference vs Bitcoin Close Price',
                   'L1 Norm (Dimension 1) First Difference vs Bitcoin Close Price'),
    specs=[[{"secondary_y": True}],
           [{"secondary_y": True}]],
    vertical_spacing=0.1
)

# Get bitcoin aligned data
bitcoin_aligned = feats['bitcoin_close'].iloc[window_length-1:window_length-1+len(topo_features)]

# Plot 1: L2 Norm first difference for dimension 1
fig.add_trace(
    go.Scatter(x=aligned_timestamps, y=l2_norm_diff,
               mode='lines+markers',
               name='L2 Norm Diff (Dim 1)',
               line=dict(color='red', width=1.5),
               marker=dict(size=2)),
    row=1, col=1, secondary_y=False
)

# Add L2 quantile lines to first subplot
for i, quantile in enumerate(l2_quantiles):
    fig.add_hline(y=quantile, 
                  line_dash="dash", 
                  line_color="green", 
                  opacity=0.5,
                  annotation_text=f"Q{i+2}",
                  annotation_position="right",
                  row=1, col=1)

# Add Bitcoin price to first subplot
fig.add_trace(
    go.Scatter(x=aligned_timestamps, y=bitcoin_aligned.values,
               mode='lines',
               name='Bitcoin Close Price',
               line=dict(color='blue', width=1.5),
               opacity=0.7),
    row=1, col=1, secondary_y=True
)

# Plot 2: L1 Norm first difference for dimension 1
fig.add_trace(
    go.Scatter(x=aligned_timestamps, y=l1_norm_diff,
               mode='lines+markers',
               name='L1 Norm Diff (Dim 1)',
               line=dict(color='green', width=1.5),
               marker=dict(size=2, symbol='square')),
    row=2, col=1, secondary_y=False
)

# Add L1 quantile lines to second subplot
for i, quantile in enumerate(l1_quantiles):
    fig.add_hline(y=quantile, 
                  line_dash="dash", 
                  line_color="red", 
                  opacity=0.5,
                  annotation_text=f"Q{i+2}",
                  annotation_position="right",
                  row=2, col=1)

# Add Bitcoin price to second subplot
fig.add_trace(
    go.Scatter(x=aligned_timestamps, y=bitcoin_aligned.values,
               mode='lines',
               name='Bitcoin Close Price',
               line=dict(color='blue', width=1.5),
               opacity=0.7,
               showlegend=False),
    row=2, col=1, secondary_y=True
)

# Update y-axes labels
fig.update_yaxes(title_text="L2 Norm First Difference (Dimension 1)", title_font_color="red", 
                 row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Bitcoin Close Price", title_font_color="blue", 
                 row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="L1 Norm First Difference (Dimension 1)", title_font_color="green", 
                 row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="Bitcoin Close Price", title_font_color="blue", 
                 row=2, col=1, secondary_y=True)

# Update x-axis label
fig.update_xaxes(title_text="Date", row=2, col=1)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    title_text="Topological Features First Differences vs Bitcoin Price"
)

fig.show()

################################################################################
################################################################################
################################################################################

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Get the aligned timestamps for x-axis
aligned_timestamps = feats.index[window_length-1:window_length-1+len(topo_features)]
topo_features['norm_persistence'] = topo_features['mean_persistence_1'] / topo_features['std_persistence_1']

# Get the original norms (not differences)
l2_norm = topo_features['avg_hole_lifetime_0'].copy()
l1_norm = topo_features['l2_norm_1'].copy()

# Define rolling window for quintile calculation
rolling_window = 30  # 1 year of trading days

# Calculate rolling quintiles
l2_rolling_quintiles = pd.Series(index=l2_norm.index, dtype=float)
l1_rolling_quintiles = pd.Series(index=l1_norm.index, dtype=float)

for i in range(rolling_window, len(l2_norm)):
    # Get rolling window data
    l2_window = l2_norm.iloc[i-rolling_window:i]
    l1_window = l1_norm.iloc[i-rolling_window:i]
    
    # Calculate quintiles for the window
    l2_quintile_thresholds = l2_window.quantile([0.2, 0.4, 0.6, 0.8])
    l1_quintile_thresholds = l1_window.quantile([0.2, 0.4, 0.6, 0.8])
    
    # Classify current value
    current_l2 = l2_norm.iloc[i]
    current_l1 = l1_norm.iloc[i]
    
    # Assign quintile (1-5)
    if current_l2 <= l2_quintile_thresholds[0.2]:
        l2_rolling_quintiles.iloc[i] = 1
    elif current_l2 <= l2_quintile_thresholds[0.4]:
        l2_rolling_quintiles.iloc[i] = 2
    elif current_l2 <= l2_quintile_thresholds[0.6]:
        l2_rolling_quintiles.iloc[i] = 3
    elif current_l2 <= l2_quintile_thresholds[0.8]:
        l2_rolling_quintiles.iloc[i] = 4
    else:
        l2_rolling_quintiles.iloc[i] = 5
    
    if current_l1 <= l1_quintile_thresholds[0.2]:
        l1_rolling_quintiles.iloc[i] = 1
    elif current_l1 <= l1_quintile_thresholds[0.4]:
        l1_rolling_quintiles.iloc[i] = 2
    elif current_l1 <= l1_quintile_thresholds[0.6]:
        l1_rolling_quintiles.iloc[i] = 3
    elif current_l1 <= l1_quintile_thresholds[0.8]:
        l1_rolling_quintiles.iloc[i] = 4
    else:
        l1_rolling_quintiles.iloc[i] = 5

# Create new series where only 4th and 5th quintile values are kept, others set to 0
l2_filtered_q5 = l2_norm.copy()
l2_filtered_q4 = l2_norm.copy()
l1_filtered_q5 = l1_norm.copy()
l1_filtered_q4 = l1_norm.copy()

# Set all non-5th quintile values to 0 for Q5 series
l2_filtered_q5[l2_rolling_quintiles != 5] = 0
l1_filtered_q5[l1_rolling_quintiles != 5] = 0

# Set all non-4th quintile values to 0 for Q4 series
l2_filtered_q4[l2_rolling_quintiles != 4] = 0
l1_filtered_q4[l1_rolling_quintiles != 4] = 0

# Create subplots with secondary y-axes
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('L2 Norm (Dimension 1) - 4th & 5th Rolling Quintiles vs Bitcoin Close Price',
                   'L1 Norm (Dimension 1) - 4th & 5th Rolling Quintiles vs Bitcoin Close Price'),
    specs=[[{"secondary_y": True}],
           [{"secondary_y": True}]],
    vertical_spacing=0.1
)

# Get bitcoin aligned data
bitcoin_aligned = feats['bitcoin_close'].iloc[window_length-1:window_length-1+len(topo_features)]

# Plot 1: L2 Norm for dimension 1 (5th quintile) as columns
fig.add_trace(
    go.Bar(x=aligned_timestamps, y=l2_filtered_q5,
           name='L2 Norm (Dim 1) - Q5',
           marker_color='red',
           opacity=0.9),
    row=1, col=1, secondary_y=False
)

# Plot 1: L2 Norm for dimension 1 (4th quintile) as columns
fig.add_trace(
    go.Bar(x=aligned_timestamps, y=l2_filtered_q4,
           name='L2 Norm (Dim 1) - Q4',
           marker_color='orange',
           opacity=0.7),
    row=1, col=1, secondary_y=False
)

# Add Bitcoin price to first subplot
fig.add_trace(
    go.Scatter(x=aligned_timestamps, y=bitcoin_aligned.values,
               mode='lines',
               name='Bitcoin Close Price',
               line=dict(color='blue', width=1.5),
               opacity=0.9),
    row=1, col=1, secondary_y=True
)

# Plot 2: L1 Norm for dimension 1 (5th quintile) as columns
fig.add_trace(
    go.Bar(x=aligned_timestamps, y=l1_filtered_q5,
           name='L1 Norm (Dim 1) - Q5',
           marker_color='green',
           opacity=0.9),
    row=2, col=1, secondary_y=False
)

# Plot 2: L1 Norm for dimension 1 (4th quintile) as columns
fig.add_trace(
    go.Bar(x=aligned_timestamps, y=l1_filtered_q4,
           name='L1 Norm (Dim 1) - Q4',
           marker_color='lightgreen',
           opacity=0.7),
    row=2, col=1, secondary_y=False
)

# Add Bitcoin price to second subplot
fig.add_trace(
    go.Scatter(x=aligned_timestamps, y=bitcoin_aligned.values,
               mode='lines',
               name='Bitcoin Close Price',
               line=dict(color='blue', width=1.5),
               opacity=0.9,
               showlegend=False),
    row=2, col=1, secondary_y=True
)

# Update y-axes labels
fig.update_yaxes(title_text="L2 Norm (Dimension 1) - Q4 & Q5", title_font_color="red", 
                 row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Bitcoin Close Price", title_font_color="blue", 
                 row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="L1 Norm (Dimension 1) - Q4 & Q5", title_font_color="green", 
                 row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="Bitcoin Close Price", title_font_color="blue", 
                 row=2, col=1, secondary_y=True)

# Update x-axis label
fig.update_xaxes(title_text="Date", row=2, col=1)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    title_text="Topological Features vs Bitcoin Price (4th & 5th Rolling Quintiles)",
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.show()

# %% 

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats

# --- USER CONFIGURABLE SECTION ---
# Choose the topological feature to analyze (must be a column in topo_features)
# Examples: 'l2_norm_1', 'avg_hole_lifetime_1', 'std_persistence_1', etc.
topo_feature_col = 'num_relevant_holes_1'  # <-- Change this to any topological feature column you want

# Label for the feature (for axis and legend)
topo_feature_label = topo_feature_col.replace('_', ' ').title()

# Number of quantiles (quintiles by default)
n_quantiles = 5

# Confidence level for error bars
confidence_level = 0.95

# Forward return horizons (in days)
forward_days = [1, 5, 10, 20]
# --- END USER CONFIGURABLE SECTION ---

scatter_data = []

for days in forward_days:
    # Calculate forward percentage change
    forward_pct_change = feats['bitcoin_close'].pct_change(periods=-days) * 100  # Negative for forward-looking

    # Align with topological features
    aligned_forward_change = forward_pct_change.iloc[window_length-1:window_length-1+len(topo_features)]

    # Create scatter plot data
    for i, (feat_val, pct_change) in enumerate(zip(topo_features[topo_feature_col], aligned_forward_change)):
        if not pd.isna(pct_change) and not pd.isna(feat_val):
            scatter_data.append({
                'Topo_Feature': feat_val,
                'Forward_Pct_Change': pct_change,
                'Days_Forward': f'{days} days',
                'Date': aligned_timestamps[i]
            })

# Convert to DataFrame for easier plotting
scatter_df = pd.DataFrame(scatter_data)

# Bin the chosen topological feature by quantiles for each forward period
quantile_data = []

for days in forward_days:
    subset = scatter_df[scatter_df['Days_Forward'] == f'{days} days'].copy()
    if len(subset) > 0:
        # Compute quantile edges
        try:
            quantile_edges = np.unique(np.nanpercentile(subset['Topo_Feature'], np.linspace(0, 100, n_quantiles + 1)))
        except Exception as e:
            print(f"Error computing quantile edges: {e}")
            quantile_edges = None

        # If not enough unique edges, fallback to no binning
        if quantile_edges is not None and len(quantile_edges) > 1:
            n_bins = len(quantile_edges) - 1
            labels = [f'Q{i+1}' for i in range(n_bins)]
            subset['Topo_Quantile'] = pd.cut(
                subset['Topo_Feature'],
                bins=quantile_edges,
                labels=labels,
                include_lowest=True,
                duplicates='drop'
            )
        else:
            subset['Topo_Quantile'] = 'Q1'

        # Calculate statistics for each quantile including confidence intervals
        quantile_stats = []
        for quantile in subset['Topo_Quantile'].unique():
            if pd.notna(quantile):
                quantile_data_subset = subset[subset['Topo_Quantile'] == quantile]['Forward_Pct_Change']

                mean_val = quantile_data_subset.mean()
                std_val = quantile_data_subset.std()
                count_val = len(quantile_data_subset)

                # Calculate confidence interval
                if count_val > 1:
                    alpha = 1 - confidence_level
                    t_critical = stats.t.ppf(1 - alpha/2, df=count_val-1)
                    margin_of_error = t_critical * (std_val / np.sqrt(count_val))
                    ci_lower = mean_val - margin_of_error
                    ci_upper = mean_val + margin_of_error
                else:
                    ci_lower = mean_val
                    ci_upper = mean_val

                quantile_stats.append({
                    'Topo_Quantile': quantile,
                    'mean': mean_val,
                    'std': std_val,
                    'count': count_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'Days_Forward': f'{days} days'
                })

        quantile_data.extend(quantile_stats)

# Convert to DataFrame
quantile_df = pd.DataFrame(quantile_data)

# Create bar plot showing mean returns by quantile with confidence intervals
fig = go.Figure()

# Add bars for each forward period
colors = px.colors.qualitative.Set1
for i, days in enumerate(forward_days):
    subset = quantile_df[quantile_df['Days_Forward'] == f'{days} days']

    # Calculate error bars (confidence interval bounds)
    error_y_upper = subset['ci_upper'] - subset['mean']
    error_y_lower = subset['mean'] - subset['ci_lower']

    fig.add_trace(go.Bar(
        name=f'{days} days',
        x=subset['Topo_Quantile'],
        y=subset['mean'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_y_upper,
            arrayminus=error_y_lower,
            visible=True
        ),
        marker_color=colors[i % len(colors)],
        offsetgroup=i
    ))

# Update layout
fig.update_layout(
    title=f'Mean Forward Bitcoin Returns by {topo_feature_label} Quintiles<br><sub>{confidence_level*100:.0f}% Confidence Intervals</sub>',
    xaxis_title=f"{topo_feature_label} Quintile",
    yaxis_title="Mean Forward Bitcoin Return (%)",
    barmode='group',
    height=600,
    showlegend=True
)

fig.show()

# Print quantile statistics with confidence intervals
print(f"\nQuantile Analysis - Mean Forward Returns by {topo_feature_label} Quintiles ({confidence_level*100:.0f}% CI):")
print("=" * 80)
for days in forward_days:
    print(f"\n{days} days forward:")
    subset = quantile_df[quantile_df['Days_Forward'] == f'{days} days']
    for _, row in subset.iterrows():
        print(f"  {row['Topo_Quantile']}: {row['mean']:.2f}% [{row['ci_lower']:.2f}%, {row['ci_upper']:.2f}%] (±{row['std']:.2f}%, n={row['count']})")

# Calculate correlation coefficients for each forward period
print("\n" + "=" * 80)
print(f"Correlation between {topo_feature_label} and Forward Price Changes:")
for days in forward_days:
    subset = scatter_df[scatter_df['Days_Forward'] == f'{days} days']
    if len(subset) > 1:
        correlation = subset['Topo_Feature'].corr(subset['Forward_Pct_Change'])
        print(f"{days} days forward: {correlation:.4f}")

# Calculate and show Sharpe ratio by quantile for each forward period
print("\n" + "=" * 80)
print("Sharpe Ratio by Quantile (Forward Returns):")
for days in forward_days:
    print(f"\n{days} days forward:")
    subset = quantile_df[quantile_df['Days_Forward'] == f'{days} days']
    for _, row in subset.iterrows():
        if row['std'] > 0:  # Avoid division by zero
            sharpe_ratio = row['mean'] / row['std']
            print(f"  {row['Topo_Quantile']}: {sharpe_ratio:.4f} (mean: {row['mean']:.2f}%, std: {row['std']:.2f}%, n={row['count']})")
        else:
            print(f"  {row['Topo_Quantile']}: N/A (std=0, mean: {row['mean']:.2f}%, n={row['count']})")


# %%

# %%

%matplotlib widget


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(12, 6))

# Get the aligned timestamps for x-axis
aligned_timestamps = feats.index[window_length-1:window_length-1+len(topo_features)]

# Plot topological feature on left y-axis
color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('L2 Norm (Dimension 1)', color=color)
ax1.plot(aligned_timestamps, topo_features['l1_norm_1'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

# Create second y-axis for bitcoin price
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Bitcoin Close Price', color=color)

# Plot bitcoin close price (aligned with topological features)
# Since topo_features starts from window_length, we need to align the indices
bitcoin_aligned = feats['bitcoin_close'].iloc[window_length-1:window_length-1+len(topo_features)]
ax2.plot(aligned_timestamps, bitcoin_aligned.values, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Topological Feature vs Bitcoin Close Price')
plt.tight_layout()
plt.show()

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_information_coefficient(topo_features, target_data, feature_names, horizons=[5, 10, 20, 30, 40, 50], vol_window=20):
    """
    Calculate and plot the Information Coefficient (IC) between multiple topological features
    and future returns over multiple horizons using non-overlapping returns and volatility normalization.
    Each feature gets a subplot with a line chart showing IC vs. return horizons.
    
    Parameters:
    topo_features: DataFrame containing the topological features
    target_data: DataFrame containing the target data (e.g., Bitcoin close prices)
    feature_names: list of str, names of the topological features to analyze
    horizons: list, number of days forward to calculate returns for
    vol_window: int, window size for calculating rolling volatility for normalization
    """
    # Calculate log returns for each horizon
    returns_dict = {}
    sample_counts = {}
    for h in horizons:
        # Non-overlapping returns by taking every h-th return
        log_returns = np.log(target_data['bitcoin_close'] / target_data['bitcoin_close'].shift(h))
        # Select every h-th return to avoid overlap
        log_returns = log_returns[::h]
        returns_dict[f'return_{h}d'] = log_returns
        sample_counts[h] = log_returns.dropna().count()
    
    # Calculate rolling volatility for normalization
    vol = np.log(target_data['bitcoin_close'] / target_data['bitcoin_close'].shift(1)).rolling(window=vol_window).std() * np.sqrt(252)
    
    # Set up subplots - one for each feature
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features), sharex=True)
    
    # If there's only one feature, wrap axes in a list for iteration
    if n_features == 1:
        axes = [axes]
    
    # Calculate and plot IC for each feature
    for idx, feature_name in enumerate(feature_names):
        # Align the feature data with returns
        ic_results = {}
        for h in horizons:
            returns = returns_dict[f'return_{h}d']
            # Align the feature with the returns by reindexing
            aligned_feature = topo_features[feature_name].reindex(returns.index, method='ffill')
            aligned_vol = vol.reindex(returns.index, method='ffill')
            
            # Volatility normalize the returns
            normalized_returns = returns / aligned_vol
            
            # Calculate correlation (Information Coefficient)
            ic = aligned_feature.corr(normalized_returns)
            ic_results[h] = ic
        
        # Plotting as a line chart
        horizons_list = list(ic_results.keys())
        ic_values = list(ic_results.values())
        
        axes[idx].plot(horizons_list, ic_values, marker='o', color='skyblue', linewidth=2, markersize=8)
        axes[idx].set_ylabel('Information Coefficient')
        axes[idx].set_title(f'IC of {feature_name} vs Vol-Normalized Bitcoin Returns')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on top of points
        for x, y in zip(horizons_list, ic_values):
            axes[idx].text(x, y, f'{y:.3f}', ha='center', va='bottom' if y >= 0 else 'top')
        
        # Adjust y-axis limits to make room for labels
        y_abs_max = max(abs(min(ic_values)), abs(max(ic_values)))
        axes[idx].set_ylim(-y_abs_max*1.2, y_abs_max*1.2)
        
        # Print results for this feature
        print(f"Information Coefficients for {feature_name}:")
        for h, ic in ic_results.items():
            print(f"  {h}-day horizon: {ic:.3f} (samples: {sample_counts[h]})")
    
    # Set x-label on the bottom subplot
    axes[-1].set_xlabel('Return Horizon (Days)')
    
    # Add text box with sample counts in the upper left corner of the last subplot to avoid collisions
    sample_text = "Sample Counts:\n" + "\n".join([f"{h}d: {sample_counts[h]}" for h in horizons])
    axes[-1].text(0.02, 0.98, sample_text, transform=axes[-1].transAxes, 
                  verticalalignment='top', horizontalalignment='left', 
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Choose topological features to analyze
features_to_analyze = ['num_relevant_holes_1', 'l1_norm_1', 'persistence_entropy_1']  # Add more features as needed
plot_information_coefficient(topo_features, feats, features_to_analyze)

# %%

# Simulate an advanced trading strategy based on the Information Coefficient (IC) of num_relevant_holes_1
# Given the IC of -0.29 on a 30-day horizon, this suggests a negative correlation, 
# meaning higher values of num_relevant_holes_1 are associated with lower future returns.
# Strategy: Implement a dynamic position sizing strategy with volatility adjustment and stop-loss/take-profit levels.
# Additionally, combine with a trend filter (EMA) to avoid trading against the major trend, enhancing signal quality.

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Define the feature to use for the strategy
feature_name = 'persistence_entropy_1'
horizon = 30  # 30-day horizon based on IC analysis

# Create a copy of the topological features DataFrame for strategy simulation
strategy_df = topo_features[[feature_name]].copy()

# Align Bitcoin prices with the topological features
aligned_bitcoin_prices = feats['bitcoin_close'].reindex(strategy_df.index, method='ffill')

# Calculate forward returns for the horizon (30 days forward) - for analysis purposes
forward_returns = aligned_bitcoin_prices.pct_change(periods=-horizon) * 100  # Negative for forward-looking

# Calculate a trend filter using EMA (50-day EMA as a simple trend indicator)
ema_50 = aligned_bitcoin_prices.rolling(window=50, min_periods=1).mean()

# Strategy logic: Use rolling quintiles to determine high/low values of the feature
rolling_window = 30  # Use a 30-day rolling window for quintile calculation
quintiles = pd.Series(index=strategy_df.index, dtype=float)

for i in range(rolling_window, len(strategy_df)):
    window_data = strategy_df[feature_name].iloc[i-rolling_window:i]
    quintile_thresholds = window_data.quantile([0.2, 0.4, 0.6, 0.8])
    current_value = strategy_df[feature_name].iloc[i]
    
    # Assign quintile (1-5)
    if current_value <= quintile_thresholds[0.2]:
        quintiles.iloc[i] = 1
    elif current_value <= quintile_thresholds[0.4]:
        quintiles.iloc[i] = 2
    elif current_value <= quintile_thresholds[0.6]:
        quintiles.iloc[i] = 3
    elif current_value <= quintile_thresholds[0.8]:
        quintiles.iloc[i] = 4
    else:
        quintiles.iloc[i] = 5

# Calculate rolling volatility for position sizing (20-day rolling standard deviation of returns)
volatility = aligned_bitcoin_prices.pct_change().rolling(window=20, min_periods=1).std() * np.sqrt(252)  # Annualized volatility
volatility_target = 0.15  # Target annualized volatility of 15% for position sizing

# Generate signals based on quintiles with dynamic position sizing and trend filter
signals = pd.Series(index=strategy_df.index, dtype=float)
position_sizes = pd.Series(index=strategy_df.index, dtype=float)

for i in range(len(strategy_df)):
    if i < rolling_window:
        signals.iloc[i] = 0
        position_sizes.iloc[i] = 0
        continue
    
    current_price = aligned_bitcoin_prices.iloc[i]
    current_ema = ema_50.iloc[i]
    current_vol = volatility.iloc[i]
    current_quintile = quintiles.iloc[i]
    
    # Calculate position size based on volatility targeting (inverse volatility weighting)
    if current_vol > 0:
        position_size = volatility_target / current_vol
        position_size = min(position_size, 1.0)  # Cap at 100% exposure to avoid excessive leverage
    else:
        position_size = 0
    
    # Determine signal direction based on quintile and trend filter
    if current_quintile == 5 and current_price < current_ema:  # Short only if below EMA (bearish trend)
        signals.iloc[i] = -1
        position_sizes.iloc[i] = position_size
    elif current_quintile == 1 and current_price > current_ema:  # Long only if above EMA (bullish trend)
        signals.iloc[i] = 1
        position_sizes.iloc[i] = position_size
    else:
        signals.iloc[i] = 0
        position_sizes.iloc[i] = 0

# Calculate strategy returns with dynamic position sizing
# Shift signals and position sizes by 1 to avoid look-ahead bias (trade on next day's close)
daily_returns = aligned_bitcoin_prices.pct_change()
strategy_returns = signals.shift(1) * position_sizes.shift(1) * daily_returns

# Implement stop-loss and take-profit logic (5% stop-loss, 10% take-profit per trade)
stop_loss = 0.05
take_profit = 0.10
adjusted_strategy_returns = strategy_returns.copy()
active_position = 0
entry_price = 0

for i in range(1, len(strategy_df)):
    if active_position == 0:  # No position
        if signals.iloc[i-1] != 0:  # New position initiated
            active_position = signals.iloc[i-1]
            entry_price = aligned_bitcoin_prices.iloc[i]
        adjusted_strategy_returns.iloc[i] = strategy_returns.iloc[i]
    else:  # Active position
        current_price = aligned_bitcoin_prices.iloc[i]
        pct_change_since_entry = (current_price - entry_price) / entry_price
        
        # Check stop-loss or take-profit for long position
        if active_position == 1:
            if pct_change_since_entry <= -stop_loss:  # Stop-loss triggered
                adjusted_strategy_returns.iloc[i] = -stop_loss * position_sizes.iloc[i-1]
                active_position = 0
            elif pct_change_since_entry >= take_profit:  # Take-profit triggered
                adjusted_strategy_returns.iloc[i] = take_profit * position_sizes.iloc[i-1]
                active_position = 0
            else:
                adjusted_strategy_returns.iloc[i] = strategy_returns.iloc[i]
        # Check stop-loss or take-profit for short position
        elif active_position == -1:
            if pct_change_since_entry >= stop_loss:  # Stop-loss triggered
                adjusted_strategy_returns.iloc[i] = -stop_loss * position_sizes.iloc[i-1]
                active_position = 0
            elif pct_change_since_entry <= -take_profit:  # Take-profit triggered
                adjusted_strategy_returns.iloc[i] = take_profit * position_sizes.iloc[i-1]
                active_position = 0
            else:
                adjusted_strategy_returns.iloc[i] = strategy_returns.iloc[i]
                
        # Check if signal changes (close position)
        if signals.iloc[i-1] != active_position:
            active_position = 0

# Calculate cumulative returns for the strategy
cumulative_strategy_returns = (1 + adjusted_strategy_returns).cumprod() - 1
cumulative_bitcoin_returns = (1 + aligned_bitcoin_prices.pct_change()).cumprod() - 1

# Plot the strategy performance
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cumulative_strategy_returns.index,
    y=cumulative_strategy_returns * 100,
    mode='lines',
    name='Strategy Returns (Vol-Adjusted + Stops)',
    line=dict(color='green', width=2)
))
fig.add_trace(go.Scatter(
    x=cumulative_bitcoin_returns.index,
    y=cumulative_bitcoin_returns * 100,
    mode='lines',
    name='Bitcoin Buy & Hold',
    line=dict(color='blue', width=2)
))

# Update layout
fig.update_layout(
    title=f'Advanced Strategy Performance: {feature_name} (30-Day Horizon, IC: -0.29)',
    xaxis_title='Date',
    yaxis_title='Cumulative Return (%)',
    height=600,
    showlegend=True,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.show()

# Calculate and display performance metrics
annualized_return = adjusted_strategy_returns.mean() * 252 * 100  # Assuming 252 trading days in a year
annualized_volatility = adjusted_strategy_returns.std() * np.sqrt(252) * 100
sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
total_return = cumulative_strategy_returns.iloc[-1] * 100
bitcoin_total_return = cumulative_bitcoin_returns.iloc[-1] * 100

# Additional metrics: Win rate and average trade duration
trades = signals.diff().abs().dropna()
num_trades = len(trades[trades != 0])
trade_returns = adjusted_strategy_returns[signals.shift(1).abs() > 0]
win_rate = len(trade_returns[trade_returns > 0]) / len(trade_returns) if len(trade_returns) > 0 else 0

print(f"Advanced Strategy Performance Metrics for {feature_name} (30-Day Horizon):")
print(f"Total Return: {total_return:.2f}%")
print(f"Annualized Return: {annualized_return:.2f}%")
print(f"Annualized Volatility: {annualized_volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Number of Trades: {num_trades}")
print(f"Bitcoin Buy & Hold Total Return: {bitcoin_total_return:.2f}%")