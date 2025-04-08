# **Step-by-Step Implementation Plan**

Below is a grounded, chronological approach for building the *MISSIONCONTROL* system from the ground up. Each step builds on the previous, ensuring coherence and modularity. 

---

## **Phase 1: Foundation & Data Layer**

### **Step 1.1: Project Setup**
1. **Initialize Repository** ✅ 
   - Create a clean Git repository (e.g., `MISSIONCONTROL`).
   - Establish a basic folder structure (docs, geckoAPI, macro, micro, pages, portfolios, etc.).
2. **Configure Environment**  ✅
   - Set up a `venv` or Conda environment.
   - Install required libraries: `requests`, `pandas`, `numpy`, `plotly`, `streamlit`, etc.
3. **Add .env & Credentials**  ✅
   - Place API keys in `.env`, including Google Sheets credentials if required.

**Outcome**: A ready-to-go skeleton with environment & version control.

---

### **Step 1.2: Data Ingestion Scripts**

1. **CoinGecko Integration** ✅
   - Implemented `getAssetsCandleData.py` with robust features:
     - Historical OHLCV data fetching with chunking.
     - Automatic gap detection and filling.
     - Rate limit handling and retry logic.
     - Data validation and interpolation.
   - Data stored in CSV format under `./micro/candleData`.
2. **FRED Integration** ✅
   - `getFredData.py` script implemented to pull key macro metrics (e.g., M2, interest rates).  
   - Data stored in CSV under `./macro/fredData`.  
   - `computeFredChanges.py` calculates MoM, YoY changes and is used in the Streamlit dashboard.
3. **Additional Macro Inputs** ✅ 
   - **Yield Curve Regimes & Central Bank Liquidity:**  
     - Implemented yield curve regime extraction with enhanced handling (using last-observation-carried-forward to persist regimes into weekends/holidays).

**Outcome**: Data ingestion routines now robustly capture not only traditional market data but also extended macro signals that include yield curve regimes and central bank liquidity.

---

## **Phase 2: RORO Modelling**

### **Step 2.1: Identifying Momentum Turning Points** ✅
- Implemented initial HMM-based regime detection
- Successfully developed feature engineering pipeline
- Created modular structure for model training and prediction
- Implemented versioning system for models with metadata tracking

### **Step 2.2: Model Version Control & Registry**
- **Model Versioning System:** ✅
  - Implemented versioned model saving with timestamps
  - Added metadata storage for each model version
  - Created organized directory structure for models, scalers, and metadata

- **Model Registry & Comparison Tools:** (Planned)
  - Create a model registry to track all versions and their performance
  - Develop comparison tools to evaluate different model versions
  - Implement feature version control system
  - Add rollback mechanism for reverting to previous model versions
  - Build automated performance comparison reports between versions
  - Create visualization tools for model version comparison

### **Step 2.3: On-Chain & Macro Overlays**
- Integrate on-chain metrics (e.g., Glassnode's MVRV, SOPR, NUPL) into the RORO logic.
- Enhance the macro dashboard signals by incorporating advanced yield curve regimes and central bank liquidity data.
- Combine these macro signals with technical momentum/volatility to refine bullish/bearish probabilities.

**Outcome**: A more nuanced RORO model that benefits from a holistic view of both current market conditions and broader macroeconomic contexts.

### **Step 2.4: Ensemble Bayesian HMM for Regime Classification** ✅
- Developed initial HMM implementation for regime classification
- Added support for both long-only and long-short strategies
- Implemented Kalman filtering for price smoothing
- Created visualization tools for regime analysis and performance metrics

### **Step 2.5: Advanced Trading Strategies**
- **MarkovKAMA Implementation:** ✅
  - Implemented Markov Switching Regression (MSR) for volatility regime detection
  - Implemented Kaufman's Adaptive Moving Average (KAMA) for trend identification
  - Combined MSR and KAMA to classify market into four distinct regimes:
    - Bullish Low Volatility (Buy signal)
    - Bullish High Volatility
    - Bearish Low Volatility
    - Bearish High Volatility (Sell signal)
  - Developed trading strategy based on regime classification
  - Added performance metrics calculation and visualization tools
  - Created parameter optimization framework using Hyperopt
  - Added model saving and loading functionality

### **Step 2.6: Code Refactoring and Modularization** ✅
- **SOROS System Implementation:**
  - Refactored monolithic TrendAnalyzer class into modular components:
    - Created data module for data loading and preprocessing
    - Created indicators module for technical indicators calculation
    - Created analysis module for metrics and Markov analysis
    - Created portfolio module for portfolio management and backtesting
    - Created visualization module for performance plotting
  - Organized code into a clean, maintainable structure
  - Improved separation of concerns and component reusability
  - Enhanced maintainability and testability
  - Implemented proper caching and optimization

### **Step 2.7: Model Maintenance & Evolution**
- **Regular Retraining Pipeline:**
  - Implement automated feature importance analysis
  - Create scheduled retraining triggers based on performance metrics
  - Develop data drift detection
  - Add automated model performance monitoring
  - Build A/B testing framework for new features

### **Step 2.8: Finalizing RORO Classification**
- Establish a method to output a single numeric regime probability (e.g., through a weighted average of signals).
- Label market states clearly (e.g., Bull/High Vol, Bull/Low Vol, Bear/High Vol, Bear/Low Vol).
- Validate the predictive power of the composite signals against historical turning points.
- Adjust weightings and thresholds based on ongoing backtesting.

**Outcome**: A refined RORO engine that outputs daily regime probabilities with clear market state labels.

---

## **Phase 3: Risk Management & Portfolio Integration**

### **Step 3.1: Drawdown & Stop-Loss Logic** ✅
- Implemented rolling drawdown calculation and risk level categorization
- Added ATR-based stop-loss calculation and dynamic profit targets
- Integrated risk management with trend classification and portfolio backtesting
- Fixed critical issues in portfolio backtesting to ensure proper risk management

**Outcome**: A robust risk management module with dynamic stop-loss recommendations based on market conditions.

### **Step 3.2: Portfolio Sheets Integration** ✅
- Implemented portfolio management system with backtesting capabilities
- Added support for reading and tracking portfolio allocations
- Developed comprehensive asset-level and portfolio-wide performance metrics
- Fixed critical errors in backtest_portfolio method to ensure reliable calculations

**Outcome**: Reliable portfolio tracking with accurate performance metrics and backtesting capabilities.

### **Step 3.3: Enhanced Portfolio Backtesting** ✅
- Removed hardcoded signal weights, thresholds, and trading parameters from the backtester
- Implemented fully configurable portfolio criteria parameters:
  - **Signal Calculation Parameters**: Customizable signal weights, RSI thresholds, and volatility filtering
  - **Trading Parameters**: Configurable trade mode, position sizing, scale in/out options, and min trade size
  - **Decision Criteria**: Flexible threshold settings for trade signals
- Updated docstrings to clearly document all available configuration options
- Improved the main script to demonstrate proper parameter usage with examples
- Enhanced results reporting with detailed trade metrics (win rate, avg win/loss)

**Outcome**: A flexible and transparent backtesting system that allows complete customization of trading parameters without hardcoded assumptions, increasing reliability and avoiding misleading signals.

### **Step 3.4: Asset-Specific Signal Optimization Framework** ✅
- Designed and implemented a data-driven signal evaluation framework:
  - **Forward Returns Analysis**: Created tools to analyze how signals shift return distributions
  - **Statistical Testing**: Implemented rigorous statistical tests to validate signal effectiveness
  - **Effect Size Calculation**: Measured the magnitude of signal impact using Cohen's d and other metrics
  - **Signal Weighting**: Developed a formula combining effect size (50%), confidence (30%), and mean difference (20%)
- Developed signal selection and combination infrastructure:
  - **SignalBase Interface**: Created a common interface for all trading signals
  - **Signal Registry**: Implemented a central registry for discovering available signals
  - **SignalEvaluator**: Built component to analyze signal effectiveness on asset-specific data
  - **SignalCombiner**: Created module to combine multiple signals using calculated weights
  - **SignalSelector**: Developed bridge component between Portfolio Manager and signal framework
- Integrated the framework with portfolio management:
  - Updated PortfolioManager to leverage asset-specific signal selection
  - Maintained backward compatibility with existing portfolio configuration
  - Added a toggle to enable/disable dynamic signal selection
- Created example scripts demonstrating the new framework in action
- Added placeholder for future meta-labeling implementation

**Outcome**: A sophisticated, data-driven framework that dynamically selects and weighs the most effective signals for each asset based on empirical evidence, improving trading performance through asset-specific signal optimization.

### **Step 3.5: Event-Driven Signal Framework** ✅
- Implemented a more sophisticated event-driven signal execution logic:
  - **Signal Event Tracking**: Created system to track when signals activate (0→1) and maintain exposure for empirically optimal holding periods
  - **Optimal Holding Period Analysis**: Added capability to determine the best forward return window for each signal using Sortino ratio
  - **Weight Normalization**: Implemented proper normalization of weights across all active signal events for each asset
  - **Half-Life Resetting**: Developed logic to reset holding periods when signals re-activate while still active
- Created new components for event-driven architecture:
  - **SignalEvent Class**: Implemented a class to represent signal activation events with holding periods and weights
  - **SignalEventTracker**: Built a system to manage the lifecycle of signal events, calculate combined weights, and make trading decisions
  - **EventDrivenBacktester**: Developed an event-driven portfolio backtester that follows signal events rather than daily signal values
- Added comparative analysis capabilities:
  - **Backtest Comparison**: Created tools to compare traditional binary logic with event-driven logic performance
  - **Trade Frequency Analysis**: Implemented metrics to measure trade reduction and performance improvements
- Refactored the signal evaluation to support the new event-driven framework:
  - Enhanced the `SignalEvaluator` to determine optimal holding periods
  - Modified the weight calculation formula to use `effect_size × confidence × direction`
  - Improved the statistical significance threshold to better capture persistent effects

**Outcome**: A more sophisticated signal framework that better captures the persistent effects of signal activations, maintaining signal exposure for optimal holding periods while reducing unnecessary trading activity.

### **Step 3.6: Attribution & Reporting**
- Implement multi-period attribution analysis (monthly or as needed).
- Develop routines to compile daily metrics into comprehensive, user-friendly reports.

**Outcome**: Robust reporting for both portfolio performance and the underpinning macro signals, aiding timely decision-making.

### **Step 3.7: Unified Interface Implementation** ✅
- **Core Data Structures**: 
  - Created `AssetData` and `SignalData` container classes for centralized data management
  - Implemented `PortfolioAnalyzer` as a unified facade over existing components
- **Integration with Existing Modules**:
  - Connected to SignalEvaluator for calculating signal weights
  - Integrated with SignalEventTracker for tracking activations
  - Added compatibility with EventDrivenBacktester for backtesting
  - Enhanced SignalEventTracker with improved meta-labeling support
- **Simplified User Experience**:
  - Created a consistent high-level API for all operations
  - Implemented caching for performance optimization
  - Added parallel processing support for batch operations
  - Created unified state saving/loading functionality
- **Example Usage**:
  - Created `examples/unified_interface_example.py` to demonstrate all key features
  - Showed step-by-step workflow from data loading to backtesting
  - Demonstrated recommendation generation for current date

**Outcome**: A comprehensive, user-friendly interface that simplifies daily operations while leveraging the full power of the underlying signal evaluation, event tracking, and backtesting components. This unified interface reduces cognitive overhead and code duplication, allowing for faster development and more maintainable code.

---

## **Phase 4: Streamlit UI & Backtesting**

### **Step 4.1: Multi-Page Streamlit Dashboard**
- Create dedicated pages for:
  - **Home / MissionControl Page:** High-level summaries (e.g., portfolio returns vs. BTC, macro regime status).
  - **RORO & Risk Page:** Detailed regime probabilities and associated risk measures.
  - **Macro Overview Page:** Visualizations of yield curve regimes, central bank liquidity proxies, and correlations.
  - **Portfolio Page:** Real-time allocations and performance metrics, with data sourced from Google Sheets.

**Outcome**: An intuitive dashboard integrating all critical metrics for real-time decision-making.

### **Step 4.2: Backtesting Framework**
- Develop a historical simulation engine that feeds in both RORO signals and risk rules.
- Compute key performance metrics (e.g., CAGR, maximum drawdown, Sharpe/Sortino ratios).

**Outcome**: A validated backtesting module to optimize and continuously refine the overall approach.

### **Step 4.3: Final Polish & Future Enhancements**
- Refine visualizations (using Plotly, heatmaps, waterfall charts) for clearer insights.
- Lay the groundwork for incorporating advanced machine learning models and sentiment analysis.
- Ensure the system is modular and ready for continual updates and enhancements.

**Outcome**: A robust, evolving system capable of incorporating future developments and modular enhancements.

---

---

This update establishes a robust foundation for our RORO decision support system by integrating various approaches, including:

1. Ensemble Bayesian HMM approach (as outlined in [Andrew Hyde's article](https://andrew-hyde.medium.com/the-ensemble-of-hidden-markov-bayesian-models-for-regime-switching-in-equity-markets-a2a7dc109a39)), with advanced filtering and multi-model fusion.

2. Markov Switching with Kaufman's Adaptive Moving Average (KAMA+MSR) as described by Piotr Pomorski, providing a refined approach to regime classification and trading strategy implementation.

3. Modular SOROS system design that separates concerns into data handling, indicator calculation, trend analysis, portfolio management, and visualization components for enhanced maintainability and flexibility.

These techniques enhance the system's predictive power, smooth regime transitions, and improve overall decision-making reliability by providing multiple perspectives on market regimes.
