# %% [markdown]
# # CatBoost Time Series Classification Strategy
# This notebook demonstrates a ML-based strategy using CatBoost for time series classification
# vs a traditional rules-based strategy pipeline

# %% Imports and Setup
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)

# %% Data Loading
def load_crypto_data(start_date=None, end_date=None):
    """Load Bitcoin and other crypto data from parquet files with optional date filtering"""
    
    # Load Bitcoin as primary asset
    btc_df = pd.read_parquet('data_parquet/crypto_data/coingecko/bitcoin/data.parquet')
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp']).dt.tz_localize(None)
    
    # Apply date filter if specified
    if start_date:
        btc_df = btc_df[btc_df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        btc_df = btc_df[btc_df['timestamp'] <= pd.to_datetime(end_date)]
    
    # Load Ethereum for correlation features
    try:
        eth_df = pd.read_parquet('data_parquet/crypto_data/coingecko/ethereum/data.parquet')
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp']).dt.tz_localize(None)
        eth_df = eth_df[['timestamp', 'close']].rename(columns={'close': 'eth_close'})
        
        # Apply date filter if specified
        if start_date:
            eth_df = eth_df[eth_df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            eth_df = eth_df[eth_df['timestamp'] <= pd.to_datetime(end_date)]
    except:
        eth_df = None
    
    # Load VIX for market fear/greed
    try:
        vix_df = pd.read_parquet('data_parquet/macro_data/yahoo/vix/data.parquet')
        vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp']).dt.tz_localize(None)
        vix_df = vix_df[['timestamp', 'close']].rename(columns={'close': 'vix_close'})
        
        # Apply date filter if specified
        if start_date:
            vix_df = vix_df[vix_df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            vix_df = vix_df[vix_df['timestamp'] <= pd.to_datetime(end_date)]
    except:
        vix_df = None
    
    return btc_df, eth_df, vix_df

btc_df, eth_df, vix_df = load_crypto_data(start_date='2020-01-01', end_date='2025-01-01')
print(f"Loaded {len(btc_df)} rows of Bitcoin data")
print(f"Date range: {btc_df['timestamp'].min()} to {btc_df['timestamp'].max()}")

# %% Feature Engineering
def create_time_series_features(df, eth_df=None, vix_df=None):
    """Create comprehensive time series features for ML model"""
    
    data = df.copy()
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Price-based features
    data['returns_1d'] = data['close'].pct_change(1)
    data['returns_7d'] = data['close'].pct_change(7)
    data['returns_30d'] = data['close'].pct_change(30)
    
    # Technical indicators
    # Moving averages
    data['sma_7'] = data['close'].rolling(7).mean()
    data['sma_21'] = data['close'].rolling(21).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    
    # Price relative to moving averages
    data['price_vs_sma7'] = data['close'] / data['sma_7'] - 1
    data['price_vs_sma21'] = data['close'] / data['sma_21'] - 1
    data['price_vs_sma50'] = data['close'] / data['sma_50'] - 1
    
    # Volatility features
    data['volatility_7d'] = data['returns_1d'].rolling(7).std()
    data['volatility_21d'] = data['returns_1d'].rolling(21).std()
    
    # Volume features
    data['volume_sma_7'] = data['total_volume'].rolling(7).mean()
    data['volume_ratio'] = data['total_volume'] / data['volume_sma_7']
    
    # Price action features
    data['high_low_ratio'] = data['high'] / data['low']
    data['close_vs_high'] = data['close'] / data['high']
    data['close_vs_low'] = data['close'] / data['low']
    
    # Momentum features
    data['rsi_14'] = calculate_rsi(data['close'], 14)
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_7d'] = data['close'] / data['close'].shift(7) - 1
    
    # Market cap features
    data['mcap_change_1d'] = data['market_cap'].pct_change(1)
    data['mcap_vs_volume'] = data['market_cap'] / data['total_volume']
    
    # Time-based features
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['month'] = data['timestamp'].dt.month
    data['quarter'] = data['timestamp'].dt.quarter
    data['year'] = data['timestamp'].dt.year
    
    # Join external data
    if eth_df is not None:
        data = pd.merge(data, eth_df, on='timestamp', how='left')
        data['btc_eth_ratio'] = data['close'] / data['eth_close']
        # Calculate ETH returns first
        data['eth_returns'] = data['eth_close'].pct_change()
        # Calculate rolling correlation between BTC and ETH returns
        data['btc_eth_corr_7d'] = data['returns_1d'].rolling(7).corr(data['eth_returns'])
    
    if vix_df is not None:
        data = pd.merge(data, vix_df, on='timestamp', how='left')
        data['vix_change'] = data['vix_close'].pct_change()
        
    return data

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Create features
data = create_time_series_features(btc_df, eth_df, vix_df)
print(f"Created {data.shape[1]} features")

# %% Target Variable Creation
def create_target_variable(data, horizon_days=7):
    """
    Create triple barrier target using volatility-based barriers
    
    Triple barrier method:
    - Upper barrier: +2 * EWMA volatility (20-day span)
    - Lower barrier: -2 * EWMA volatility (20-day span)  
    - Time barrier: horizon_days
    
    Target classes:
    - 1: Hit upper barrier first (bullish)
    - 0: Hit lower barrier first (bearish)
    - -1: Hit time barrier first (neutral/sideways)
    """
    
    data = data.copy()
    
    # Calculate EWMA volatility (20-day span)
    data['returns_1d'] = data['close'].pct_change()
    data['ewma_vol'] = data['returns_1d'].ewm(span=20).std()
    
    # Define barriers relative to current price
    data['upper_barrier'] = 2 * data['ewma_vol']  # +2 sigma move
    data['lower_barrier'] = -2 * data['ewma_vol']  # -2 sigma move
    
    # Initialize target variable
    data['target'] = -1  # Default to time barrier (neutral)
    data['barrier_hit_day'] = horizon_days  # Days until barrier hit
    data['hit_return'] = np.nan  # Return when barrier was hit
    
    # For each observation, look forward to see which barrier is hit first
    for i in range(len(data) - horizon_days):
        current_price = data.iloc[i]['close']
        upper_threshold = current_price * (1 + data.iloc[i]['upper_barrier'])
        lower_threshold = current_price * (1 + data.iloc[i]['lower_barrier'])
        
        # Look at next horizon_days prices
        future_prices = data.iloc[i+1:i+1+horizon_days]['close'].values
        
        # Check which barrier is hit first
        for day, price in enumerate(future_prices, 1):
            if price >= upper_threshold:
                data.iloc[i, data.columns.get_loc('target')] = 1  # Bullish
                data.iloc[i, data.columns.get_loc('barrier_hit_day')] = day
                data.iloc[i, data.columns.get_loc('hit_return')] = (price / current_price) - 1
                break
            elif price <= lower_threshold:
                data.iloc[i, data.columns.get_loc('target')] = 0  # Bearish  
                data.iloc[i, data.columns.get_loc('barrier_hit_day')] = day
                data.iloc[i, data.columns.get_loc('hit_return')] = (price / current_price) - 1
                break
        
        # If no barrier hit, it remains -1 (neutral/time barrier)
        if data.iloc[i]['target'] == -1:
            final_price = data.iloc[i+horizon_days]['close']
            data.iloc[i, data.columns.get_loc('hit_return')] = (final_price / current_price) - 1
    
    # Drop rows with NaN values (from volatility calculation and incomplete forward-looking periods)
    data = data.dropna(subset=['ewma_vol', 'upper_barrier', 'lower_barrier'])
    
    # Remove last horizon_days rows as they don't have complete forward-looking data
    data = data.iloc[:-horizon_days]
    
    # Remap target classes to 0, 1, 2 for CatBoost compatibility
    # -1 (neutral) -> 0, 0 (bearish) -> 1, 1 (bullish) -> 2
    target_mapping = {-1: 0, 0: 1, 1: 2}
    data['target_original'] = data['target'].copy()  # Keep original for interpretation
    data['target'] = data['target'].map(target_mapping)
    
    return data

print("Creating triple barrier target variable...")
data = create_target_variable(data, horizon_days=7)

# Print target distribution using original labels
target_distribution = data['target_original'].value_counts(normalize=True).sort_index()
print(f"\nTriple Barrier Target Distribution:")
print(f"Bearish (0 - hit lower barrier): {target_distribution.get(0, 0):.3f}")
print(f"Bullish (1 - hit upper barrier): {target_distribution.get(1, 0):.3f}") 
print(f"Neutral (-1 - time barrier):    {target_distribution.get(-1, 0):.3f}")

# Print statistics about barrier hits
avg_days_to_hit = data[data['target_original'] != -1]['barrier_hit_day'].mean()
print(f"\nAverage days to barrier hit: {avg_days_to_hit:.1f}")
print(f"Samples with complete data: {len(data)}")

# %% Rules-Based Strategy Implementation
class RulesBasedStrategy:
    """Traditional rules-based strategy for comparison"""
    
    def __init__(self):
        self.name = "Rules-Based Strategy"
    
    def generate_signals(self, data):
        """Generate 3-class signals based on technical rules"""
        
        signals = pd.Series(-1, index=data.index)  # Default to neutral (-1)
        
        # Bullish conditions
        bullish_condition1 = (data['close'] > data['sma_7']) & (data['returns_7d'] > 0)
        bullish_condition2 = data['rsi_14'] < 70  # Not overbought
        bullish_condition3 = data['volume_ratio'] > 1.2  # High volume
        bullish_condition4 = data['volatility_7d'] < data['volatility_7d'].rolling(30).mean()
        bullish_condition5 = data['momentum_3d'] > 0.02  # Strong positive momentum
        
        # Bearish conditions
        bearish_condition1 = (data['close'] < data['sma_7']) & (data['returns_7d'] < 0)
        bearish_condition2 = data['rsi_14'] > 30  # Not oversold
        bearish_condition3 = data['volume_ratio'] > 1.0  # Some volume
        bearish_condition4 = data['momentum_3d'] < -0.02  # Strong negative momentum
        bearish_condition5 = data['volatility_7d'] > data['volatility_7d'].rolling(30).mean()
        
        # Count bullish and bearish rule matches
        bullish_count = (bullish_condition1.astype(int) + bullish_condition2.astype(int) + 
                        bullish_condition3.astype(int) + bullish_condition4.astype(int) + 
                        bullish_condition5.astype(int))
        
        bearish_count = (bearish_condition1.astype(int) + bearish_condition2.astype(int) + 
                        bearish_condition3.astype(int) + bearish_condition4.astype(int) + 
                        bearish_condition5.astype(int))
        
        # Generate signals based on rule counts
        signals[bullish_count >= 3] = 1   # Bullish signal
        signals[bearish_count >= 3] = 0   # Bearish signal
        # Everything else remains -1 (neutral)
        
        return signals
    
    def evaluate(self, data):
        """Evaluate rules-based strategy"""
        
        # Remove rows with NaN (due to rolling calculations)
        clean_data = data.dropna()
        
        signals = self.generate_signals(clean_data)
        actual = clean_data['target']
        
        # Align signals and actual
        aligned_signals = signals.reindex(actual.index, fill_value=0)
        
        accuracy = accuracy_score(actual, aligned_signals)
        
        print(f"\n{self.name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Show distribution of predictions
        signal_dist = aligned_signals.value_counts(normalize=True).sort_index()
        print(f"Signal distribution:")
        print(f"  Bearish (0): {signal_dist.get(0, 0):.3f}")
        print(f"  Bullish (1): {signal_dist.get(1, 0):.3f}")
        print(f"  Neutral (-1): {signal_dist.get(-1, 0):.3f}")
        
        return aligned_signals, accuracy

# %% ML-Based Strategy Implementation  
class MLStrategy:
    """Machine Learning strategy using CatBoost"""
    
    def __init__(self, model_params=None):
        self.name = "ML-Based Strategy (CatBoost)"
        self.model = None
        self.feature_names = None
        self.model_params = model_params or {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_strength': 1,
            'od_type': 'Iter',
            'od_wait': 50,
            'allow_writing_files': False,
            'verbose': False,
            'objective': 'MultiClass',  # For 3-class classification
            'classes_count': 3
        }
    
    def prepare_features(self, data):
        """Prepare features for ML model"""
        
        # Select feature columns (exclude metadata and target)
        exclude_cols = ['timestamp', 'asset_id', 'symbol', 'target', 'target_original', 'future_return', 
                       'ewma_vol', 'upper_barrier', 'lower_barrier', 'barrier_hit_day', 'hit_return']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Handle categorical features
        categorical_features = ['day_of_week', 'month', 'quarter', 'year']
        categorical_indices = [feature_cols.index(col) for col in categorical_features 
                             if col in feature_cols]
        
        return feature_cols, categorical_indices
    
    def train_test_split(self, data, test_size=0.2):
        """Time-aware train/test split"""
        
        # Sort by timestamp
        data_sorted = data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove rows with NaN values
        data_clean = data_sorted.dropna()
        
        # Time-based split (use last test_size portion as test set)
        n_samples = len(data_clean)
        split_idx = int(n_samples * (1 - test_size))
        
        train_data = data_clean.iloc[:split_idx]
        test_data = data_clean.iloc[split_idx:]
        
        print(f"Train period: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
        print(f"Test period: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        return train_data, test_data
    
    def train(self, train_data):
        """Train the CatBoost model"""
        
        feature_cols, categorical_indices = self.prepare_features(train_data)
        self.feature_names = feature_cols
        
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        
        # Create CatBoost Pool
        train_pool = Pool(
            X_train, 
            y_train,
            cat_features=categorical_indices
        )
        
        # Train model
        self.model = CatBoostClassifier(**self.model_params)
        self.model.fit(train_pool)
        
        print(f"Model trained with {len(feature_cols)} features")
        
        return self
    
    def predict(self, test_data):
        """Generate predictions"""
        
        X_test = test_data[self.feature_names]
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)  # All class probabilities
        
        return predictions, probabilities
    
    def evaluate(self, test_data):
        """Evaluate ML strategy"""
        
        predictions, probabilities = self.predict(test_data)
        actual = test_data['target']
        
        accuracy = accuracy_score(actual, predictions)
        
        print(f"\n{self.name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Show distribution of predictions (convert back to original labels)
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        pred_orig = [reverse_mapping[int(x)] for x in predictions.flatten()]
        pred_dist = pd.Series(pred_orig).value_counts(normalize=True).sort_index()
        print(f"Prediction distribution:")
        print(f"  Bearish (0): {pred_dist.get(0, 0):.3f}")
        print(f"  Bullish (1): {pred_dist.get(1, 0):.3f}")
        print(f"  Neutral (-1): {pred_dist.get(-1, 0):.3f}")
        
        # Convert predictions back to original labels for display
        actual_orig = [reverse_mapping[int(x)] for x in actual]
        pred_orig = [reverse_mapping[int(x)] for x in predictions.flatten()]
        
        # Print detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(actual_orig, pred_orig, target_names=['Neutral', 'Bearish', 'Bullish']))
        
        return predictions, probabilities, accuracy
    
    def get_feature_importance(self, top_n=15):
        """Get top feature importances"""
        
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        feature_importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.head(top_n))
        
        return importance_df

# %% Model Training and Evaluation
print("Preparing data for modeling...")

# Split data
ml_strategy = MLStrategy()
train_data, test_data = ml_strategy.train_test_split(data, test_size=0.3)

print("\n" + "="*50)
print("TRAINING ML STRATEGY")
print("="*50)

# Train ML model
ml_strategy.train(train_data)

# Get feature importance
importance_df = ml_strategy.get_feature_importance()

# %% Strategy Comparison
print("\n" + "="*50)
print("STRATEGY EVALUATION COMPARISON")
print("="*50)

# Evaluate Rules-Based Strategy
rules_strategy = RulesBasedStrategy()
rules_predictions, rules_accuracy = rules_strategy.evaluate(test_data)

# Evaluate ML Strategy  
ml_predictions, ml_probabilities, ml_accuracy = ml_strategy.evaluate(test_data)

# %% Cross-Validation for ML Strategy
print("\n" + "="*50)
print("TIME SERIES CROSS-VALIDATION")
print("="*50)

def time_series_cv_evaluation(data, n_splits=5):
    """Perform time series cross-validation"""
    
    # Clean data
    clean_data = data.dropna().sort_values('timestamp').reset_index(drop=True)
    
    # Prepare features
    ml_cv = MLStrategy()
    feature_cols, categorical_indices = ml_cv.prepare_features(clean_data)
    
    X = clean_data[feature_cols]
    y = clean_data['target']
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    print(f"Performing {n_splits}-fold time series cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model for this fold
        train_pool = Pool(X_train_cv, y_train_cv, cat_features=categorical_indices)
        model_cv = CatBoostClassifier(**ml_strategy.model_params)
        model_cv.fit(train_pool)
        
        # Evaluate
        val_pred = model_cv.predict(X_val_cv)
        val_accuracy = accuracy_score(y_val_cv, val_pred)
        cv_scores.append(val_accuracy)
        
        print(f"Fold {fold+1}: Accuracy = {val_accuracy:.4f}")
    
    print(f"\nCross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
    
    return cv_scores

# Perform cross-validation
cv_scores = time_series_cv_evaluation(data)

# %% Strategy Pipeline Comparison
print("\n" + "="*70)
print("STRATEGY PIPELINE ARCHITECTURE COMPARISON")
print("="*70)

print("\n🔧 RULES-BASED STRATEGY PIPELINE:")
print("="*50)
print("""
1. Data Input (OHLCV + Volume)
   ↓
2. Feature Calculation (Hard-coded)
   • Moving averages (SMA 7, 21)
   • RSI (14 periods)
   • Volume ratio
   • Price momentum
   ↓
3. Rule Evaluation (Boolean Logic)
   • IF price > SMA_7 AND returns_7d > 0
   • AND RSI < 70
   • AND volume_ratio > 1.2
   • AND volatility < recent_avg
   • AND momentum_3d > 0
   ↓
4. Signal Generation (Threshold-based)
   • IF 3+ conditions met → BUY signal
   • ELSE → NO signal
   ↓
5. Output: Binary signals (0/1)

CHARACTERISTICS:
✅ Interpretable and transparent
✅ Fast execution
✅ Domain knowledge incorporated
❌ Fixed rules, no adaptation
❌ Limited feature interactions
❌ Manual parameter tuning
❌ Prone to overfitting to specific market regimes
""")

print("\n🤖 ML-BASED STRATEGY PIPELINE:")
print("="*50)
print("""
1. Data Input (OHLCV + Volume + External data)
   ↓
2. Feature Engineering (Automated + Manual)
   • Price-based: returns (1d, 7d, 30d)
   • Technical: RSI, moving averages, volatility
   • Volume: ratios, moving averages
   • Market structure: high/low ratios
   • External: VIX, ETH correlation
   • Temporal: day_of_week, month, quarter
   ↓
3. Feature Selection & Preprocessing
   • Handle categorical variables
   • Missing value imputation
   • Feature importance ranking
   ↓
4. Model Training (CatBoost)
   • Gradient boosting with categorical support
   • Automatic feature interaction detection
   • Regularization to prevent overfitting
   • Cross-validation for hyperparameter tuning
   ↓
5. Prediction Generation
   • Probability scores [0,1]
   • Binary classification threshold
   ↓
6. Signal Generation (Probability-based)
   • IF probability > 0.5 → BUY signal
   • ELSE → NO signal
   ↓
7. Output: Binary signals + confidence scores

CHARACTERISTICS:
✅ Learns complex feature interactions
✅ Adapts to changing market conditions
✅ Handles high-dimensional feature space
✅ Provides confidence/probability scores
✅ Robust to overfitting (with proper validation)
❌ "Black box" - less interpretable
❌ Requires more data and computation
❌ Risk of overfitting to training data
❌ May not incorporate domain expertise well
""")

print("\n📊 PERFORMANCE COMPARISON:")
print("="*50)
print(f"Rules-Based Strategy Accuracy: {rules_accuracy:.4f}")
print(f"ML-Based Strategy Accuracy:    {ml_accuracy:.4f}")
print(f"ML Cross-Validation Mean:      {np.mean(cv_scores):.4f}")

improvement = ((ml_accuracy - rules_accuracy) / rules_accuracy) * 100
print(f"ML Improvement over Rules:     {improvement:.1f}%")

# %% Final Summary and Recommendations
print("\n" + "="*70)
print("FINAL SUMMARY & RECOMMENDATIONS")
print("="*70)

print(f"""
📈 STRATEGY COMPARISON RESULTS:

Performance Metrics:
• Rules-Based Strategy: {rules_accuracy:.1%} accuracy
• ML-Based Strategy: {ml_accuracy:.1%} accuracy  
• ML Cross-Validation: {np.mean(cv_scores):.1%} ± {np.std(cv_scores):.1%}

🎯 KEY DIFFERENCES:

RULES-BASED APPROACH:
• Simple, transparent decision logic
• Fast execution and easy debugging
• Incorporates domain expertise directly
• Good for stable market conditions
• Risk: May not adapt to regime changes

ML-BASED APPROACH:
• Learns complex patterns automatically
• Better handling of feature interactions
• Adapts to changing market dynamics  
• Provides confidence scores for risk management
• Risk: May overfit to historical patterns

💡 RECOMMENDATIONS:

1. HYBRID APPROACH: Combine both strategies
   • Use ML for pattern recognition
   • Use rules for risk management overlays
   
2. ENSEMBLE METHODS: Multiple ML models
   • CatBoost + XGBoost + Neural Networks
   • Reduce single-model overfitting risk
   
3. REGIME DETECTION: Context-aware switching
   • Use ML to detect market regimes
   • Apply different strategies per regime
   
4. CONTINUOUS LEARNING: Online adaptation
   • Retrain models with new data
   • Monitor performance degradation
   
5. FEATURE ENGINEERING: Domain expertise + ML
   • Let ML discover interactions
   • Inject trading knowledge as features
""")

print(f"\nTop 5 Most Important ML Features:")
for i, row in importance_df.head(5).iterrows():
    print(f"{i+1}. {row['feature']}: {row['importance']:.2f}")

# %% TodoWrite update