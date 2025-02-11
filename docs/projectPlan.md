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

### **Step 1.3: Data Normalization & Daily Pipeline**

1. **Resampling to Daily**  
   - Develop utility functions (e.g., in `data_utils.py`) to standardize data frames to daily frequency.  
   - Handle missing dates, weekends, and forward-fill for lower-frequency macro data—particularly important for yield curve regimes.
2. **Consolidated Daily Pipeline**  
   - Create an initial data pipeline script (e.g., `main.py`) to:
     1. Fetch and update data from all sources.
     2. Normalize data sets.
     3. Log successful updates and prepare data for further analysis.

**Outcome**: A single command (`python main.py`) now updates and prepares the entire data environment each day, incorporating enhanced macro data.

---

## **Phase 2: RORO Modelling**

### **Step 2.1: Basic Momentum & Variance**

1. **Momentum Calculation**  
   - Implement technical indicators and time-series momentum (TSM) in `roroClassifier.py`.  
   - Consider simple trend indicators (e.g., Bollinger/Keltner channels) for signal generation.
2. **Variance & Volatility Filters**  
   - Compute rolling volatility or ATR for each asset (especially BTC).  
   - Differentiate between "high vol" vs. "low vol" states (using threshold or percentile measures).

**Outcome**: A foundational RORO (Risk-On/Risk-Off) output, combining technical signals with basic volatility metrics.

---

### **Step 2.2: On-Chain & Macro Overlays**

1. **BTC On-Chain Integration**  
   - Integrate Glassnode metrics (MVRV, SOPR, NUPL, etc.) into the RORO logic.  
   - Decide on whether to treat these indicators as separate signals or combine them into a composite measure.
2. **Macro Regime Input Enhancement**  
   - Extend the macro dashboard signals to include the enhanced yield curve regime analysis and central bank liquidity data.
   - Combine these macro signals with technical momentum/volatility to refine bullish/bearish probabilities.
   - Consider using a Taylor Rule or alternative policy-based indicators.

**Outcome**: A more nuanced RORO model that benefits from a holistic view of both current market conditions and the broader macroeconomic context.

---

### **Step 2.3: Finalizing RORO Classification**

1. **Scoring & Probabilities**  
   - Establish a method to output a single numeric regime probability (e.g., through a weighted average of signals).
   - Label market states (e.g., Bull/High Vol, Bull/Low Vol, Bear/High Vol, Bear/Low Vol).
2. **Performance Review & Optimization**  
   - Validate the predictive power of the composite signals against historical turning points.
   - Adjust weightings and thresholds based on ongoing backtesting.

**Outcome**: A robust RORO engine that outputs daily regime probabilities for BTC (and possibly other assets).

---

## **Phase 3: Risk Management & Portfolio Integration**

### **Step 3.1: Drawdown & Stop-Loss Logic**

1. **Daily Drawdown Tracker**  
   - Calculate rolling drawdowns from asset peaks.
   - Categorize risk levels (e.g., low, medium, high).
2. **ATR-Based Stop-Loss & Profit Targets**  
   - Recompute ATR daily; define dynamic stop-loss and profit-taking zones, possibly tied to the macro regime.

**Outcome**: A dynamic risk management module providing daily recommendations for stop-losses and exits.

---

### **Step 3.2: Portfolio Sheets Integration**

1. **Reading Allocations**  
   - Integrate Google Sheets to capture daily portfolio allocations.
2. **Performance Calculation & Attribution**  
   - Monitor and calculate asset-level and overall portfolio performance.

**Outcome**: Seamless sync between real-world allocations and our performance metrics dashboard.

---

### **Step 3.3: Attribution & Reporting**

1. **Attribution Analysis**  
   - Implement multi-period attribution analysis (monthly, or as desired).
2. **Dashboard Reporting**  
   - Develop routines that compile daily metrics into comprehensive reports.

**Outcome**: Robust and timely reporting for both portfolio performance and the underpinning macro signals.

---

## **Phase 4: Streamlit UI & Backtesting**

### **Step 4.1: Multi-Page Streamlit Dashboard**

1. **Home / MissionControl Page**  
   - High-level summary including portfolio returns vs. BTC, and current macro regime status.
2. **RORO & Risk Page**  
   - Detailed regime probabilities and risk measures.
3. **Macro Overview Page**  
   - Visualize yield curve regimes, central bank liquidity proxies, and correlations with BTC.
4. **Portfolio Page**  
   - Display real-time allocations and performance metrics captured from Google Sheets.

**Outcome**: An intuitive dashboard integrating all critical metrics for real-time decision making.

---

### **Step 4.2: Backtesting Framework**

1. **Historical Simulation Engine**  
   - Feed the historical data along with RORO signals and risk rules into a backtest model.
2. **Performance Metrics Computation**  
   - Evaluate CAGR, max drawdown, Sharpe/Sortino ratios, etc.

**Outcome**: A validated backtesting module to optimize and refine our overall approach.

---

### **Step 4.3: Final Polish & Future Enhancements**

1. **Visualization & Reporting Enhancements**  
   - Refine visualizations (Plotly, heatmaps, waterfall charts).
2. **Advanced Modelling & ML**  
   - Lay the groundwork for incorporating advanced machine learning models and sentiment analysis.

**Outcome**: A robust, evolving system with continual updates and modular enhancements.

---

# **Conclusion & Next Steps**

**Evaluation of Macro Dashboard Enhancements:**  
- **Yield Curve Regimes:** The recent improvements allow us to assign a regime to every BTC day—even when the yield curve isn't reported—by carrying forward the last known regime.  
- **Central Bank Liquidity:** Early add-ons for central bank liquidity are being integrated to provide a broader context for the macro environment, complementing traditional FRED data.
- **Data Handling Adjustments:** Filtering BTC data from 2017 onward and annualizing volatility with 365 days improves our analysis accuracy for a 24/7 market.

**What's Next for Us:**  
1. **Integration into RORO Modelling:**  
   - Combine the enhanced macro dashboard signals with on-chain and technical momentum indicators in the RORO model.
   - Run initial backtests to check the correlation between regime changes (as captured with yield curve/central bank data) and subsequent BTC performance.

2. **Dashboard & Visualization Enhancements:**  
   - Expand the macro overview page in the Streamlit dashboard to visualize these new signals.
   - Enhance user feedback regarding the impact of macro regimes on trading decisions.

3. **Risk Management & Portfolio Links:**  
   - Integrate risk management modules that adapt stop-loss and profit targets based on evolving macro regimes.
   - Sync real portfolio data via Google Sheets for dynamic tracking.

4. **Further Data Enrichment:**  
   - Complete the ingestion and integration of central bank liquidity data.
   - Test additional macro variables, such as policy announcements and sentiment indicators, as further overlays.

In summary, the macro dashboard now robustly captures yield curve regimes and central bank liquidity signals. The next step is to integrate these enriched macro signals into our forecasting and risk management modules—refining our RORO engine and preparing for a more dynamic, responsive portfolio management system.

---

*This update will be crucial for transitioning to advanced modeling (Phase 2 and beyond), where we fully explore regime-switching models and their impact on crypto asset performance.*
