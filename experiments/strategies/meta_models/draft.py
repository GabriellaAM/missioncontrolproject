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

asset = ['bitcoin']

feats = FeatureLoader(start_date='2019-01-01', end_date='2025-05-30')

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
        """Create time delay embedding of the series - FIXED to prevent future data leakage"""
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
                    
                    # Maximum hole lifetime
                    max_hole_lifetime = np.max(persistence)
                    features[f'max_hole_lifetime_{dim}'] = max_hole_lifetime
                    
                    # Average lifetime of all holes
                    avg_hole_lifetime = np.mean(persistence)
                    features[f'avg_hole_lifetime_{dim}'] = avg_hole_lifetime
                    
                    # Number of relevant holes (persistence > threshold)
                    # Use a threshold based on the standard deviation of persistence values
                    if len(persistence) > 1:
                        persistence_threshold = np.mean(persistence) + 0.5 * np.std(persistence)
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
        if f'l1_norm_1' in df.columns:
            df[f'l1_norm_1_diff'] = df[f'l1_norm_1'].diff()
        if f'l2_norm_1' in df.columns:
            df[f'l2_norm_1_diff'] = df[f'l2_norm_1'].diff()
        if f'avg_hole_lifetime_1' in df.columns:
            df[f'avg_hole_lifetime_1_diff'] = df[f'avg_hole_lifetime_1'].diff()
    
    return df

# Example usage with user-defined parameters
window_length = 21  # User can modify this

selected_columns = ['bitcoin_close']  # User can modify this

# Extract topological features with fixed tau
topo_features = extract_topological_features(
    data=feats,
    window_length=window_length,
    selected_cols=selected_columns,
    tau=3,  # Fixed tau value
    max_dimension=3
)

print("Topological features extracted:")
print(topo_features.head())
print(f"\nFeature columns: {topo_features.columns.tolist()}")



# %%

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Get the aligned timestamps for x-axis
aligned_timestamps = feats.index[window_length-1:window_length-1+len(topo_features)]



# Plot 1: Average hole lifetime for dimension 1
color1 = 'tab:red'
ax1.set_ylabel('Average Hole Lifetime (Dimension 1)', color=color1)
ax1.plot(aligned_timestamps, topo_features['avg_hole_lifetime_1'], 
         color=color1, linewidth=1.5, marker='o', markersize=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Create second y-axis for bitcoin price on first subplot
ax1_twin = ax1.twinx()
color_btc = 'tab:blue'
ax1_twin.set_ylabel('Bitcoin Close Price', color=color_btc)
bitcoin_aligned = feats['bitcoin_close'].iloc[window_length-1:window_length-1+len(topo_features)]
ax1_twin.plot(aligned_timestamps, bitcoin_aligned.values, color=color_btc, alpha=0.7)
ax1_twin.tick_params(axis='y', labelcolor=color_btc)

ax1.set_title('Average Hole Lifetime (Dimension 1) Through Time vs Bitcoin Close Price')

# Plot 2: Number of significant holes for dimension 1
color2 = 'tab:green'
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Significant Holes (Dimension 1)', color=color2)
ax2.plot(aligned_timestamps, topo_features['num_relevant_holes_1'], 
         color=color2, linewidth=1.5, marker='s', markersize=2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(True, alpha=0.3)

# Create second y-axis for bitcoin price on second subplot
ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('Bitcoin Close Price', color=color_btc)
ax2_twin.plot(aligned_timestamps, bitcoin_aligned.values, color=color_btc, alpha=0.7)
ax2_twin.tick_params(axis='y', labelcolor=color_btc)

ax2.set_title('Number of Significant Holes (Dimension 1) Through Time vs Bitcoin Close Price')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print statistics for dimension 1
print("\nDimension 1 Topological Features Statistics:")
print(f"Average Hole Lifetime:")
avg_lifetime = topo_features['avg_hole_lifetime_1']
print(f"  Mean: {avg_lifetime.mean():.4f}")
print(f"  Std: {avg_lifetime.std():.4f}")
print(f"  Min: {avg_lifetime.min():.4f}")
print(f"  Max: {avg_lifetime.max():.4f}")
print(f"  Median: {avg_lifetime.median():.4f}")

print(f"\nNumber of Significant Holes:")
num_holes = topo_features['num_relevant_holes_1']
print(f"  Mean: {num_holes.mean():.2f}")
print(f"  Std: {num_holes.std():.2f}")
print(f"  Min: {int(num_holes.min())}")
print(f"  Max: {int(num_holes.max())}")
print(f"  Median: {num_holes.median():.2f}")
print(f"  Total periods with >0 holes: {(num_holes > 0).sum()}")


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
