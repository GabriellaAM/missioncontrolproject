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

### **Step 2.1: Identifying Momentum Turning Points**

- **Objective**: Apply the ideas from the [Momentum Turning Points paper](https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/637bd3ef-b512-493b-93c3-4f223a45b55b.pdf) to detect momentum turning points in the time series data.
- **Method**:
  - Compute a momentum metric (e.g., percentage change over a defined window).
  - Calculate a rolling mean (or other baseline measure) of the momentum.
  - Identify turning points by detecting when the difference between the momentum and its rolling mean exceeds a specified threshold and when the deviation changes sign.
  - Explore ideas on market timing vs. volatility timing.

- **Outcome**: Generate features indicating turning points that will be used in the RORO suite.


### **Step 2.2: On-Chain & Macro Overlays**
- Integrate on-chain metrics (e.g., Glassnode's MVRV, SOPR, NUPL) into the RORO logic.
- Enhance the macro dashboard signals by incorporating advanced yield curve regimes and central bank liquidity data.
- Combine these macro signals with technical momentum/volatility to refine bullish/bearish probabilities.

**Outcome**: A more nuanced RORO model that benefits from a holistic view of both current market conditions and broader macroeconomic contexts.

### **Step 2.3: Ensemble Bayesian HMM for Regime Classification**
- Develop an ensemble model that combines Hidden Markov Model (HMM) predictions with a Bayesian approach (e.g., leveraging a Kalman filter) to classify regimes.
- This ensemble method will smooth transitions and reduce the overconfidence and rapid state switching typical in solitary HMMs.
- Reference: [Ensembling Hidden Markov & Bayesian Models for Regime Switching](https://andrew-hyde.medium.com/the-ensemble-of-hidden-markov-bayesian-models-for-regime-switching-in-equity-markets-a2a7dc109a39)

**Outcome**: A robust ensemble model that yields more stable regime classifications by integrating Bayesian smoothing with HMM state predictions.

### **Step 2.4: Finalizing RORO Classification**
- Establish a method to output a single numeric regime probability (e.g., through a weighted average of signals).
- Label market states clearly (e.g., Bull/High Vol, Bull/Low Vol, Bear/High Vol, Bear/Low Vol).
- Validate the predictive power of the composite signals against historical turning points.
- Adjust weightings and thresholds based on ongoing backtesting.

**Outcome**: A refined RORO engine that outputs daily regime probabilities with clear market state labels.

### **Step 2.5: Advanced Filtering & Multi-Model Fusion**
- Incorporate a Kalman filter into the primary regime model to provide adaptive smoothing and improved state estimation.
- Develop supplementary models that offer alternative perspectives and confidence levels.
- Fuse the outputs of the Bayesian ensemble, Kalman filter, and supplementary models to generate a unified regime signal and risk confidence metric.
- Backtest the ensemble approach and tune parameters based on performance metrics such as accuracy, drawdown, and Sharpe/Sortino ratios.

**Outcome**: An advanced, multi-model RORO system that enhances decision support and provides robust confidence levels for market regimes.


---

## **Phase 3: Risk Management & Portfolio Integration**

### **Step 3.1: Drawdown & Stop-Loss Logic**
- Calculate rolling drawdowns and categorize risk levels.
- Implement ATR-based stop-loss and dynamic profit targets, potentially tied to the current macro regime.

**Outcome**: A dynamic risk management module providing daily recommendations for stop-losses and exits.

### **Step 3.2: Portfolio Sheets Integration**
- Integrate Google Sheets for reading daily portfolio allocations.
- Monitor and calculate asset-level and overall portfolio performance.

**Outcome**: Seamless sync between real-world allocations and performance metrics, facilitating dynamic tracking.

### **Step 3.3: Attribution & Reporting**
- Implement multi-period attribution analysis (monthly or as needed).
- Develop routines to compile daily metrics into comprehensive, user-friendly reports.

**Outcome**: Robust reporting for both portfolio performance and the underpinning macro signals, aiding timely decision-making.

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

# **Conclusion & Next Steps**

**Evaluation of Macro Dashboard Enhancements:**  
- **Yield Curve Regimes:** The recent improvements allow assigning a regime to every BTC day—even when the yield curve isn't reported—by carrying forward the last known regime.  
- **Central Bank Liquidity:** Early add-ons for central bank liquidity provide a broader context for the macro environment, complementing traditional FRED data.  
- **Data Handling Adjustments:** Filtering BTC data from 2017 onward and annualizing volatility with 365 days enhances analysis accuracy for a 24/7 market.

**What's Next for Us:**  
1. **Integration into RORO Modelling:**  
   - Combine enhanced macro dashboard signals with on-chain and technical momentum indicators in the RORO model.
   - Run initial backtests to gauge the correlation between regime changes and subsequent BTC performance.
2. **Advanced Filtering & Model Fusion:**  
   - Integrate a Kalman filter into the primary model and develop supplementary models for decision-making support.
   - Fuse these models to generate a unified regime signal along with a risk confidence metric.
3. **Dashboard & Reporting Enhancements:**  
   - Expand the macro overview page in the Streamlit dashboard to reflect the new filtering and ensemble approaches.
   - Enhance user feedback on the impact of advanced models on trading decisions.
4. **Risk Management & Portfolio Links:**  
   - Adapt risk management modules for dynamic stop-loss and profit targets in line with evolving macro regimes.
   - Sync real portfolio data via Google Sheets for real-time tracking.
5. **Further Data Enrichment:**  
   - Complete the ingestion and integration of central bank liquidity data.
   - Explore additional macro variables (e.g., policy announcements, sentiment indicators) for further overlays.

---

This update establishes a robust foundation for our RORO decision support system by integrating an ensemble Bayesian HMM approach (as outlined in [Andrew Hyde's article](https://andrew-hyde.medium.com/the-ensemble-of-hidden-markov-bayesian-models-for-regime-switching-in-equity-markets-a2a7dc109a39)), along with advanced filtering and multi-model fusion. While the preliminary results are very promising, these techniques will further enhance the system's predictive power, smooth regime transitions, and improve overall decision-making reliability.
