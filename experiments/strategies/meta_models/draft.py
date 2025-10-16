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

feats = FeatureLoader(start_date='2017-01-01', end_date='2025-10-14')

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
l2_norm = topo_features['l1_norm_1'].copy()
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
topo_feature_col = 'l1_norm_1'  # <-- Change this to any topological feature column you want

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
ax1.plot(aligned_timestamps, topo_features['std_persistence_1'], color=color)
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

import kmapper as km
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go

# Prepare data for KMapper analysis
# Get L2 norm dimension 1, L1 norm dimension 1, and forward 5-day returns
l2_norm_dim1 = topo_features['l2_norm_1'].copy()
l1_norm_dim1 = topo_features['l1_norm_1'].copy()

# Calculate forward 5-day returns
forward_5d_returns = feats['bitcoin_close'].pct_change(periods=-5) * 100  # Negative for forward-looking
aligned_forward_returns = forward_5d_returns.iloc[window_length-1:window_length-1+len(topo_features)]

# Create combined dataset, removing NaN values
combined_data = pd.DataFrame({
    'l2_norm_dim1': l2_norm_dim1,
    'l1_norm_dim1': l1_norm_dim1,
    'forward_5d_returns': aligned_forward_returns
}).dropna()

print(f"KMapper analysis with {len(combined_data)} data points")
print(f"L2 Norm Dim1 - Mean: {combined_data['l2_norm_dim1'].mean():.4f}, Std: {combined_data['l2_norm_dim1'].std():.4f}")
print(f"L1 Norm Dim1 - Mean: {combined_data['l1_norm_dim1'].mean():.4f}, Std: {combined_data['l1_norm_dim1'].std():.4f}")
print(f"Forward 5d Returns - Mean: {combined_data['forward_5d_returns'].mean():.4f}, Std: {combined_data['forward_5d_returns'].std():.4f}")

# Prepare data for KMapper - now including both L1 and L2 norms
X = combined_data[['l2_norm_dim1', 'l1_norm_dim1', 'forward_5d_returns']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize KMapper
mapper = km.KeplerMapper(verbose=1)

# Create lens function - we'll use both L2 and L1 norms as the lens
lens = X_scaled[:, [0, 1]]  # L2 norm and L1 norm dimensions 1 as lens

# Create the topological network
graph = mapper.map(lens, 
                   X_scaled,
                   cover=km.Cover(n_cubes=10, perc_overlap=0.3),
                   clusterer=DBSCAN(eps=0.5, min_samples=3))

# Create visualization
html_file = "kmapper_l2l1norm_forward_returns.html"
mapper.visualize(graph, 
                 path_html=html_file,
                 title="KMapper: L2 & L1 Norm Dim1 vs Forward 5-Day Returns",
                 custom_tooltips=combined_data['forward_5d_returns'].values)

print(f"KMapper visualization saved to: {html_file}")

# Analyze the graph structure
print(f"\nGraph Analysis:")
print(f"Number of nodes: {len(graph['nodes'])}")
print(f"Number of edges: {len(graph['links'])}")

# Analyze node statistics
node_stats = []
for node_id, node_members in graph['nodes'].items():
    node_l2_norms = combined_data.iloc[node_members]['l2_norm_dim1']
    node_l1_norms = combined_data.iloc[node_members]['l1_norm_dim1']
    node_returns = combined_data.iloc[node_members]['forward_5d_returns']
    
    node_stats.append({
        'node_id': node_id,
        'size': len(node_members),
        'mean_l2_norm': node_l2_norms.mean(),
        'mean_l1_norm': node_l1_norms.mean(),
        'mean_forward_return': node_returns.mean(),
        'std_forward_return': node_returns.std()
    })

node_stats_df = pd.DataFrame(node_stats)
print(f"\nTop 5 nodes by size:")
print(node_stats_df.nlargest(5, 'size')[['node_id', 'size', 'mean_l2_norm', 'mean_l1_norm', 'mean_forward_return']])

print(f"\nNodes with highest mean forward returns:")
print(node_stats_df.nlargest(5, 'mean_forward_return')[['node_id', 'size', 'mean_l2_norm', 'mean_l1_norm', 'mean_forward_return']])

print(f"\nNodes with lowest mean forward returns:")
print(node_stats_df.nsmallest(5, 'mean_forward_return')[['node_id', 'size', 'mean_l2_norm', 'mean_l1_norm', 'mean_forward_return']])

# Create a scatter plot colored by node membership (L2 vs L1 norms)
fig = go.Figure()

# Color each point by its node membership
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
node_colors = {}

for i, (node_id, node_members) in enumerate(graph['nodes'].items()):
    color = colors[i % len(colors)]
    node_colors[node_id] = color
    
    node_data = combined_data.iloc[node_members]
    
    fig.add_trace(go.Scatter(
        x=node_data['l2_norm_dim1'],
        y=node_data['l1_norm_dim1'],
        mode='markers',
        name=f'Node {node_id} (n={len(node_members)})',
        marker=dict(color=color, size=8, opacity=0.7),
        text=[f'Node: {node_id}<br>L2 Norm: {x:.4f}<br>L1 Norm: {y:.4f}<br>Return: {z:.2f}%' 
              for x, y, z in zip(node_data['l2_norm_dim1'], node_data['l1_norm_dim1'], node_data['forward_5d_returns'])],
        hovertemplate='%{text}<extra></extra>'
    ))

fig.update_layout(
    title="KMapper Node Clustering: L2 Norm vs L1 Norm (Dim1)",
    xaxis_title="L2 Norm (Dimension 1)",
    yaxis_title="L1 Norm (Dimension 1)",
    height=600,
    showlegend=True
)

fig.show()

# Create a second scatter plot: L2 norm vs forward returns
fig2 = go.Figure()

for i, (node_id, node_members) in enumerate(graph['nodes'].items()):
    color = colors[i % len(colors)]
    node_data = combined_data.iloc[node_members]
    
    fig2.add_trace(go.Scatter(
        x=node_data['l2_norm_dim1'],
        y=node_data['forward_5d_returns'],
        mode='markers',
        name=f'Node {node_id} (n={len(node_members)})',
        marker=dict(color=color, size=8, opacity=0.7),
        text=[f'Node: {node_id}<br>L2 Norm: {x:.4f}<br>L1 Norm: {y:.4f}<br>Return: {z:.2f}%' 
              for x, y, z in zip(node_data['l2_norm_dim1'], node_data['l1_norm_dim1'], node_data['forward_5d_returns'])],
        hovertemplate='%{text}<extra></extra>'
    ))

fig2.update_layout(
    title="KMapper Node Clustering: L2 Norm Dim1 vs Forward 5-Day Returns",
    xaxis_title="L2 Norm (Dimension 1)",
    yaxis_title="Forward 5-Day Returns (%)",
    height=600,
    showlegend=True
)

fig2.show()

# Create a third scatter plot: L1 norm vs forward returns
fig3 = go.Figure()

for i, (node_id, node_members) in enumerate(graph['nodes'].items()):
    color = colors[i % len(colors)]
    node_data = combined_data.iloc[node_members]
    
    fig3.add_trace(go.Scatter(
        x=node_data['l1_norm_dim1'],
        y=node_data['forward_5d_returns'],
        mode='markers',
        name=f'Node {node_id} (n={len(node_members)})',
        marker=dict(color=color, size=8, opacity=0.7),
        text=[f'Node: {node_id}<br>L2 Norm: {x:.4f}<br>L1 Norm: {y:.4f}<br>Return: {z:.2f}%' 
              for x, y, z in zip(node_data['l2_norm_dim1'], node_data['l1_norm_dim1'], node_data['forward_5d_returns'])],
        hovertemplate='%{text}<extra></extra>'
    ))

fig3.update_layout(
    title="KMapper Node Clustering: L1 Norm Dim1 vs Forward 5-Day Returns",
    xaxis_title="L1 Norm (Dimension 1)",
    yaxis_title="Forward 5-Day Returns (%)",
    height=600,
    showlegend=True
)

fig3.show()

# Additional analysis: correlation within nodes
print(f"\nCorrelation analysis within nodes:")
for node_id, node_members in graph['nodes'].items():
    if len(node_members) > 3:  # Only analyze nodes with sufficient data
        node_data = combined_data.iloc[node_members]
        l2_correlation = node_data['l2_norm_dim1'].corr(node_data['forward_5d_returns'])
        l1_correlation = node_data['l1_norm_dim1'].corr(node_data['forward_5d_returns'])
        l1_l2_correlation = node_data['l1_norm_dim1'].corr(node_data['l2_norm_dim1'])
        print(f"Node {node_id} (n={len(node_members)}): L2-Returns corr = {l2_correlation:.4f}, L1-Returns corr = {l1_correlation:.4f}, L1-L2 corr = {l1_l2_correlation:.4f}")


# %% 
%matplotlib widget

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from matplotlib.collections import LineCollection

# --- Step 1: Select single univariate series for time delay embedding ---
col = 'bitcoin_close'  # Single column for univariate analysis
series_data = feats[col].copy()
dates = feats.index.to_numpy()  # assumes feats has DateTimeIndex

# Convert to numpy array and handle NaN values
series = series_data.values
series = pd.Series(series).fillna(method='ffill').values  # Forward fill NaN values

# EWMA smoothing
def ewma_normalize(data, span):
    df = pd.Series(data)
    return df.ewm(span=span).mean().values

span = 21
series = ewma_normalize(series, span)

# Standardize the series
scaler = StandardScaler()
series = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# --- Step 2: Time delay embedding ---
def time_delay_embedding(data, tau, embedding_dim):
    """
    Create time delay embedding of univariate time series.
    
    Parameters:
    data: 1D array of time series data
    tau: time delay
    embedding_dim: embedding dimension (2 or 3)
    
    Returns:
    embedded: array of shape (n_samples - (embedding_dim-1)*tau, embedding_dim)
    """
    n = len(data)
    n_embedded = n - (embedding_dim - 1) * tau
    
    if n_embedded <= 0:
        raise ValueError("Time series too short for given tau and embedding_dim")
    
    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = data[i * tau:i * tau + n_embedded]
    
    return embedded

# --- Step 3: Sliding window on time delay embedded data ---
def sliding_window_embedding(embedded_data, w):
    """
    Apply sliding window to already time-delay embedded data.
    """
    n_samples, n_features = embedded_data.shape
    windowed = np.zeros((n_samples - w + 1, w * n_features))
    for i in range(n_samples - w + 1):
        windowed[i] = embedded_data[i:i+w].flatten()
    return windowed

# Parameters for time delay embedding
tau = 3  # time delay
embedding_dim = 3  # can be 2 or 3
w = 21  # sliding window size

# Create time delay embedding
embedded_series = time_delay_embedding(series, tau, embedding_dim)

# Apply sliding window to embedded data
windowed_embedded = sliding_window_embedding(embedded_series, w)

# Pre-compute global bounds for reset functionality
all_windows = windowed_embedded.reshape(-1, w, embedding_dim)
global_min = np.min(all_windows, axis=(0, 1))
global_max = np.max(all_windows, axis=(0, 1))
margin = 0.1 * (global_max - global_min)
global_bounds = {
    'x': (global_min[0] - margin[0], global_max[0] + margin[0]),
    'y': (global_min[1] - margin[1], global_max[1] + margin[1])
}
if embedding_dim == 3:
    global_bounds['z'] = (global_min[2] - margin[2], global_max[2] + margin[2])

# --- Step 4: Interactive visualization with sliders ---
idx0 = 0
eps0 = 0.3

def get_window_dates(idx):
    # Account for the offset due to time delay embedding and sliding window
    start_idx = idx + (embedding_dim - 1) * tau
    end_idx = start_idx + w - 1
    start = pd.to_datetime(dates[start_idx]).strftime("%Y-%m-%d")
    end = pd.to_datetime(dates[end_idx]).strftime("%Y-%m-%d")
    return start, end

def get_color_by_time(window_data):
    """
    Color points based on their temporal position within the window.
    Earlier points are darker, later points are lighter.
    """
    n_points = len(window_data)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    return colors

window0 = windowed_embedded[idx0].reshape(w, embedding_dim)
start_date, end_date = get_window_dates(idx0)
point_colors0 = get_color_by_time(window0)

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('black')

if embedding_dim == 2:
    ax = fig.add_subplot(111, facecolor='black')
    scatter = ax.scatter(window0[:, 0], window0[:, 1], s=30, c=point_colors0, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_title(f"2D Time Delay Embedding (τ={tau}, window {idx0}, {start_date} → {end_date}, ε={eps0})\nColors: dark=early, light=late", color='white')
    ax.set_xlabel(f"{col}(t)", color='white')
    ax.set_ylabel(f"{col}(t-{tau})", color='white')
    ax.tick_params(colors='white')
elif embedding_dim == 3:
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    scatter = ax.scatter(window0[:, 0], window0[:, 1], window0[:, 2], s=30, c=point_colors0, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_title(f"3D Time Delay Embedding (τ={tau}, window {idx0}, {start_date} → {end_date}, ε={eps0})\nColors: dark=early, light=late", color='white')
    ax.set_xlabel(f"{col}(t)", color='white')
    ax.set_ylabel(f"{col}(t-{tau})", color='white')
    ax.set_zlabel(f"{col}(t-{2*tau})", color='white')
    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

edge_lines = []

# --- Sliders ---
max_slider_val = windowed_embedded.shape[0] - 1
ax_slider_w = plt.axes([0.15, 0.02, 0.7, 0.03])
slider_w = Slider(ax_slider_w, "Window", 0, max_slider_val, valinit=idx0, valstep=1)

ax_slider_eps = plt.axes([0.15, 0.06, 0.7, 0.03])
slider_eps = Slider(ax_slider_eps, "Epsilon", 0.01, 1.0, valinit=eps0, valstep=0.01)

resetax = plt.axes([0.8, 0.92, 0.15, 0.05])
button = Button(resetax, 'Reset Zoom')

def set_consistent_bounds():
    if embedding_dim == 2:
        ax.set_xlim(global_bounds['x'])
        ax.set_ylim(global_bounds['y'])
    elif embedding_dim == 3:
        ax.set_xlim(global_bounds['x'])
        ax.set_ylim(global_bounds['y'])
        ax.set_zlim(global_bounds['z'])

def draw_complex(window, eps):
    global edge_lines
    for line in edge_lines:
        line.remove()
    edge_lines.clear()

    D = squareform(pdist(window))
    i_indices, j_indices = np.where((D <= eps) & (D > 0))
    mask = i_indices < j_indices
    i_indices, j_indices = i_indices[mask], j_indices[mask]

    if len(i_indices) > 0:
        if embedding_dim == 2:
            segments = np.stack([window[i_indices], window[j_indices]], axis=1)
            line_collection = LineCollection(segments, colors='yellow', linewidths=2, alpha=0.8)
            ax.add_collection(line_collection)
            edge_lines.append(line_collection)
        elif embedding_dim == 3:
            for i, j in zip(i_indices, j_indices):
                line, = ax.plot([window[i, 0], window[j, 0]],
                                [window[i, 1], window[j, 1]],
                                [window[i, 2], window[j, 2]],
                                c="yellow", lw=2, alpha=0.8)
                edge_lines.append(line)

def update(val):
    idx = int(slider_w.val)
    eps = slider_eps.val
    window = windowed_embedded[idx].reshape(w, embedding_dim)
    start_date, end_date = get_window_dates(idx)
    
    # Update colors based on temporal position
    point_colors = get_color_by_time(window)

    if embedding_dim == 2:
        scatter.set_offsets(window)
        scatter.set_color(point_colors)
        ax.set_title(f"2D Time Delay Embedding (τ={tau}, window {idx}, {start_date} → {end_date}, ε={eps:.2f})\nColors: dark=early, light=late", color='white')
    elif embedding_dim == 3:
        scatter._offsets3d = (window[:, 0], window[:, 1], window[:, 2])
        scatter.set_color(point_colors)
        ax.set_title(f"3D Time Delay Embedding (τ={tau}, window {idx}, {start_date} → {end_date}, ε={eps:.2f})\nColors: dark=early, light=late", color='white')

    draw_complex(window, eps)
    fig.canvas.draw_idle()

def reset(event):
    set_consistent_bounds()
    fig.canvas.draw_idle()

slider_w.on_changed(update)
slider_eps.on_changed(update)
button.on_clicked(reset)

# Initial setup
set_consistent_bounds()
draw_complex(window0, eps0)
plt.tight_layout()
plt.show()

# Print information about the embedding
print(f"\nTime Delay Embedding Information:")
print(f"Original series: {col}")
print(f"Time delay (τ): {tau}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Sliding window size: {w}")
print(f"Total embedded windows: {windowed_embedded.shape[0]}")

# %%
