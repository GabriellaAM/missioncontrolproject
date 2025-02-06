# **Step-by-Step Implementation Plan**

Below is a grounded, chronological approach for building the *MISSIONCONTROL* system from the ground up. Each step builds on the previous, ensuring coherence and modularity. 

---

## **Phase 1: Foundation & Data Layer**

### **Step 1.1: Project Setup**
1. **Initialize Repository**  
   - Create a clean Git repository (e.g., `MISSIONCONTROL`).
   - Establish a basic folder structure (docs, geckoAPI, macro, micro, pages, portfolios, etc.).
2. **Configure Environment**  
   - Set up a `venv` or Conda environment.
   - Install required libraries: `requests`, `pandas`, `numpy`, `plotly`, `streamlit`, etc.
3. **Add .env & Credentials**  
   - Place API keys in `.env`, including Google Sheets credentials if required.

**Outcome**: A ready-to-go skeleton with environment & version control.

---

### **Step 1.2: Data Ingestion Scripts**

1. **CoinGecko Integration** ✅
   - Implemented `getAssetsCandleData.py` with robust features:
     - Historical OHLCV data fetching with chunking
     - Automatic gap detection and filling
     - Rate limit handling and retry logic
     - Data validation and interpolation
   - Data stored in CSV format under `./micro/candleData`
2. **FRED Integration** ✅
   - `getFredData.py` script implemented to pull key macro metrics (e.g., M2, interest rates).  
   - Data stored in CSV under `./macro/fredData`.  
   - `computeFredChanges.py` calculates MoM, YoY changes and is used in the Streamlit dashboard.
3. **Glassnode Integration (BTC On-Chain)**  
   - Not yet started; will include MVRV, SOPR, NUPL, etc.  
   - Validate rate limits and data coverage; store in CSV/SQLite.

**Outcome**: You can run `python getFredData.py`, `python getAssetsCandleData.py`, etc., to fetch and store daily snapshots.

---

### **Step 1.3: Data Normalization & Daily Pipeline**

1. **Resampling to Daily**  
   - Write a small utility (e.g., `data_utils.py`) to standardize data frames to daily frequency.  
   - Handle missing dates, weekends, and forward-filling for lower-frequency macro data.
2. **main.py (Initial Version)**  
   - Orchestrate a simple flow:  
     1. Fetch/update data from each source.  
     2. Normalize and store.  
     3. Print/log a success message.  
   - Keep it minimal for now—no modeling logic yet.

**Outcome**: A single command (`python main.py`) updates and prepares the data environment each day.

*Note:* This step is partially complete. Basic date filtering is in the macro dashboard, but a consolidated daily pipeline script is still pending.

---

## **Phase 2: RORO Modelling**

### **Step 2.1: Basic Momentum & Variance**

1. **Momentum Calculation**  
   - Implement time-series momentum (TSM) in `roroClassifier.py`.  
   - Possibly add Bollinger/Keltner or other simple indicators to classify trend.
2. **Variance & Volatility Filters**  
   - Compute rolling volatility or ATR for each asset (especially BTC).  
   - Tag "high vol" vs. "low vol" states using thresholds (e.g., percentile-based).

**Outcome**: A foundational RORO output (e.g., a dictionary with `{'date': ..., 'asset': ..., 'roro_prob_bull_high_vol': 0.8, ...}`).

---

### **Step 2.2: On-Chain & Macro Overlays**

1. **BTC On-Chain Integration**  
   - Incorporate Glassnode metrics (MVRV, SOPR, etc.) into RORO logic.  
   - Decide if these remain separate signals or feed into a single "composite probability."
2. **Macro Regime Input**  
   - Derive a simple "macro stance" from FRED data (e.g., Growth, Inflation, Policy signals).  
   - Combine macro signals with momentum/variance to refine bullish/bearish probabilities.  
   - Consider adding a Taylor Rule or other policy-based metrics.

**Outcome**: RORO classification that accounts for short/medium-term technical signals + on-chain + broad macro environment.

---

### **Step 2.3: Finalizing RORO Classification**

1. **Scoring & Probabilities**  
   - Decide on a mechanism to produce a single numeric probability for each regime (e.g., weighted average of signals).  
   - Label states: `(Bull/High Vol)`, `(Bull/Low Vol)`, `(Bear/High Vol)`, `(Bear/Low Vol)`, etc.
2. **Check & Optimize**  
   - Perform a brief historical check to see if the signals align with known market turning points.  
   - Tweak thresholds or weighting if needed.

**Outcome**: A robust RORO engine that, on any given day, outputs regime probabilities for BTC and possibly other major altcoins.

---

## **Phase 3: Risk Management & Portfolio Integration**

### **Step 3.1: Drawdown & Stop-Loss Logic**

1. **Daily Drawdown Tracker**  
   - For each asset, compute rolling drawdown from previous peaks.  
   - Color-code or categorize risk (e.g., low, medium, high).
2. **ATR-Based Stop-Loss & Profit Targets**  
   - Recompute ATR daily; define recommended stop level.  
   - Optionally define profit-taking thresholds (e.g., x% above entry or x*ATR).  
   - Integrate the RORO regime to widen/narrow stops.

**Outcome**: For each asset, a recommended stop-loss & profit-taking range updated daily.

---

### **Step 3.2: Portfolio Sheets Integration**

1. **Reading Allocations**  
   - Use the Google Sheets API to pull daily allocations for each portfolio.  
   - Compare day-over-day changes to detect partial sells/buys.
2. **Performance Calculation**  
   - Track daily PnL and cumulative returns for each asset in the portfolio.  
   - Summarize overall portfolio performance vs. BTC as a benchmark.

**Outcome**: The system shows how real (manually updated) allocations perform and suggests risk-based adjustments.

---

### **Step 3.3: Attribution & Reporting**

1. **Attribution Analysis**  
   - Implement multi-period attribution (e.g., monthly or user-defined).  
   - Show each asset's contribution to the total return in bar/waterfall charts.
2. **Reporting**  
   - Build a routine that compiles daily metrics (RORO state, risk flags, portfolio performance) into a quick summary.  
   - Option for Markdown/PDF export within Streamlit.

**Outcome**: Comprehensive reporting on portfolio returns, risk status, and recommended next steps.

---

## **Phase 4: Streamlit UI & Backtesting**

### **Step 4.1: Multi-Page Streamlit Dashboard**

1. **Home / MissionControl Page**  
   - High-level summary: portfolio vs. BTC returns, daily changes, top signals.
2. **RORO & Risk Page**  
   - Display regime probabilities, volatility overlays, stop-loss levels, color-coded risk status.
3. **Portfolio Page**  
   - Current allocations (from Sheets), asset-level PnL, monthly attribution.
4. **Macro Overview Page**  
   - GIP signals, yield curves, liquidity proxies, correlation with BTC.

**Outcome**: A user-friendly interface to check all vital metrics in one place, with drill-down capabilities.

*Note:* Currently, we have partial multi-page support (dashboard, macro_overview, regressions). The macro page includes toggles for MoM and YoY calculations.

---

### **Step 4.2: Backtesting Framework**

1. **Basic Backtest Engine**  
   - Feed historical data + RORO signals + risk rules into a test harness.  
   - Compare performance to BTC buy-and-hold from the same period.
2. **Performance Metrics**  
   - Evaluate CAGR, max drawdown, Sharpe/Sortino, etc.  
   - Record reallocation signals and simulated PnL with transaction cost assumptions.

**Outcome**: Verified historical performance of your strategy, guiding further model refinements.

---

### **Step 4.3: Final Polish & Expansion Hooks**

1. **Refine Outputs**  
   - Tweak visualizations (Plotly charts, color-coded heatmaps, or waterfall charts).  
   - Ensure daily log files or data snapshots are archived properly.
2. **Prepare for Advanced Features**  
   - Lay groundwork for future ML integrations (random forests, sentiment analysis, etc.).  
   - Explore advanced allocation frameworks (risk parity, volatility targeting) as separate modules.

**Outcome**: A stable, modular system with daily usage and historical validation.

---

## **Ongoing: Testing & Maintenance**

- **Continuous Testing**  
  1. **Unit Tests**: For data ingestion, RORO calculations, risk logic.  
  2. **Integration Tests**: Validate `main.py` end-to-end.  
  3. **Manual Spot Checks**: Cross-check daily dashboards with external references.

- **Version Control & Logging**  
  - Keep code changes atomic, track them in Git, and maintain concise logs.  
  - Ensure the system gracefully handles missed data fetches or partial API failures.

**Outcome**: Confidence in your system's reliability and correctness, essential for real-world portfolio decisions.

---

# **Conclusion**

This updated plan ensures you tackle the *MISSIONCONTROL* system in a logical progression:

1. **Foundation & Data Layer** (Phase 1)  
2. **RORO Modelling** (Phase 2)  
3. **Risk & Portfolio Integration** (Phase 3)  
4. **UI & Backtesting** (Phase 4)

Each phase culminates in a working subset of features, providing iterative validation and reducing complexity. Long-term, you can integrate advanced ML-driven signals or deeper allocation frameworks, keeping robust testing and version control in place for reliability. 
