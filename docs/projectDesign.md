# **MISSIONCONTROL: Four-Phase Workflow for Your Crypto Decision-Support System**

This roadmap is tailored to AI-assisted development while maintaining engineering rigor. The plan is divided into four key phases:

---

## **1. Problem Statement & Requirements**

### **Problem Space & Goals**

- **Objective**:  
  Build a decision-support system (*MISSION CONTROL*) that:
  1. Integrates macro-economic (FRED), market (CoinGecko), and on-chain (Glassnode) data.  
  2. Utilizes Regime-On / Regime-Off (RORO) modeling to classify market states for BTC and altcoins.  
  3. Manages risk via drawdown analysis, ATR-based stops, and profit-taking signals.  
  4. Pulls portfolio allocations from Google Sheets and reports performance, attribution, and recommended rebalancing daily.

- **Primary Users**:  
  - You (portfolio manager) and your internal team.

- **Key Constraints**:
  - Must refresh data and run models on a **daily** frequency.  
  - Visualization and user interaction in **Streamlit** (multi-page).  
  - Integration with **Google Sheets** for allocation logs.  
  - Final decisions remain manual; system provides suggestions.

- **Acceptance Criteria**:
  - **Daily Dashboard**: Summaries of RORO regimes, portfolio returns vs. BTC, and risk signals.  
  - **Attribution & Performance**: Must display multi-period performance with asset-level contribution.  
  - **Extensibility**: Modular design to incorporate future ML/AI, new data sources, and advanced allocation methods.

**Goal**: Reach a **clear alignment** on the project's scope, inputs, outputs, and constraints before proceeding with development.

---

## **2. Design Documentation**

### **2.1 System Overview**

1. **Data Ingestion**:  
   - CoinGecko for crypto price/candlestick data.  
   - FRED for macro metrics (e.g., M2, interest rates).  
   - Glassnode (initially for BTC) on-chain metrics like MVRV, SOPR, NUPL.  

2. **Model & Analysis**:  
   - **RORO Classification**: Combining momentum, variance, on-chain metrics, and macro (GIP) factors.  
   - **Risk Module**: Rolling drawdowns, ATR-based stop-losses, regime-adjusted profit-taking thresholds.  

3. **Portfolio Reporting**:  
   - Pull daily allocations from Google Sheets.  
   - Calculate performance vs. BTC, attribute returns by asset, and propose rebalancing moves.

4. **Streamlit Dashboard**:  
   - Multi-page interface with dedicated views for RORO status, risk indicators, and portfolio metrics.

### **2.2 Key Requirements**

- **Daily Workflow**:
  1. Fetch new data from APIs.  
  2. Resample/align data (daily frequency).  
  3. Run RORO model + risk calculations.  
  4. Pull actual allocations from Sheets; compare vs. recommended changes.  
  5. Render Streamlit dashboard with updated metrics and logs.

- **Core Interactions**:
  - **Portfolio Manager**: Reviews the dashboard each morning, checks recommended changes, updates Google Sheets if needed.  
  - **System**: Logs partial sells/buys by comparing day-over-day differences in allocations.

### **2.3 Constraints & Considerations**

- **Medium/Long-Term**: Real-time or intraday updates not critical; daily or weekly granularity is sufficient.  
- **Modularity**: Future expansions (e.g., multiple allocation algorithms, ML-driven predictions) should not break existing functionality.  
- **Data Quality & API Limits**: Must handle occasional outages or data mismatches gracefully.

**Goal**: Provide a **structured design document** that captures both the functional and technical aspects so you (and the AI) share the same roadmap.

---

## **3. Phased Implementation**

### **Phase 1: Data Layer & Basic Pipeline**

1. **API Integration**  
   - Implement scripts for CoinGecko, FRED, Glassnode (BTC metrics).  
   - Store or cache data locally in CSV/SQLite (simplest approach first).

2. **Daily Orchestrator**  
   - A `main.py` script that:  
     1. Fetches & normalizes data.  
     2. Runs initial RORO classification (stub logic).  
     3. Loads Google Sheets allocations for logging.  
     4. Outputs simple daily summary.

3. **Initial Streamlit Dashboard**  
   - Barebones page showing fetched data, raw price charts, and partial metrics.

**Validation**: Confirm data correctness and daily consistency. No complex signals yet—just ensure the pipeline works end to end.

### **Phase 2: RORO Modeling & Risk Module**

1. **Regime Classification**  
   - Incorporate momentum/variance signals for BTC and altcoins.  
   - Integrate on-chain metrics (SOPR, MVRV) as a separate "layer."  
   - Output numeric probabilities (e.g., "Bull/High Vol" = 60%, "Bear/Low Vol" = 40%).

2. **Risk Calculations**  
   - Daily rolling drawdowns, ATR-based stop-loss thresholds.  
   - Color-coded or flagged indicators for caution vs. severe risk.

3. **Refined Dashboard**  
   - Present RORO states, risk overlays on price charts, recommended risk adjustments.

**Validation**: Compare historical signals to known market regimes. Begin partial backtests to ensure signals align with intuition.

### **Phase 3: Portfolio Reporting & Attribution**

1. **Google Sheets Integration**  
   - Pull each day's allocations. Track partial sells/buys by spotting day-over-day changes.  
   - Summarize daily returns and total equity curve.

2. **Attribution**  
   - Multi-period (monthly or user-defined range) performance analysis.  
   - Visualize each asset's contribution to total PnL using bar/waterfall charts.

3. **Recommended Rebalancing Logic**  
   - (Optional) Propose re-weighting or partial shifts based on RORO and risk signals.  
   - Manager manually updates Sheets; system logs final state.

**Validation**: Ensure performance calculations match expected numbers. Cross-verify with manual computations for accuracy.

### **Phase 4: Backtesting & Enhanced Visualization**

1. **Backtesting Framework**  
   - Feed historical data into the RORO + risk logic.  
   - Compare results to a BTC buy-and-hold baseline for key metrics (CAGR, MDD, Sharpe).
   - **Asset-Specific Backtesting**: Test individual assets with various signal combinations, including USD-only, BTC-only, or combined approaches, with comprehensive metrics for strategy evaluation.
   - **Portfolio-Level Backtesting**: Simulate portfolio performance with various allocation strategies and rebalancing methodologies.

2. **Advanced Visuals & Reporting**  
   - Multi-page Streamlit app, including a "Home" summary and dedicated pages for RORO, Risk, Portfolio, and Macro.  
   - Exportable daily or weekly reports in Markdown/PDF.
   - **Backtest Visualization**: Equity curves, drawdown charts, and comparative metrics for strategy assessment.

3. **Polishing & Expansion**  
   - Tweak signals, refine ATR stop-loss widths, manage alert thresholds.  
   - Lay groundwork for optional ML integrations (e.g., random forests).

**Validation**: Confirm backtested strategies match or exceed baseline. Gather feedback from real-world usage, iterate accordingly.

**Goal**: Deliver *manageable, reviewable chunks of work*, ensuring each phase meets documented requirements.

---

### **Testing Strategy**

1. **Unit Tests**  
   - Validate data-fetching scripts and transformations (e.g., correct daily resampling).  
   - Check RORO calculations for edge cases (no data, zero volume, etc.).

2. **Integration Tests**  
   - Ensure that the daily pipeline (`main.py`) processes data seamlessly end to end.  
   - Simulate different market regimes with historical data to test classification logic.

3. **Manual Validation**  
   - Cross-check daily dashboards with actual market conditions.  
   - Compare reported portfolio returns with reference calculations (in Excel, Sheets, etc.).

4. **Backward Compatibility**  
   - Keep data schemas consistent; if new columns or signals are added, older modules should still run without crashing.  
   - If using a SQLite or CSV approach, maintain versioning for backward compatibility.

**Goal**: Achieve confidence in every code change by blending **automated tests** with **manual inspection**—especially critical for trading and portfolio logic.

---

# **Conclusion & Next Steps**

1. **Document Requirements Clearly**  
   - Finalize acceptance criteria and constraints from all stakeholders.  
2. **Complete Design Review**  
   - Make sure each piece (RORO model, risk module, portfolio reporting) is fully specced before coding.  
3. **Implement in Phases**  
   - Begin with data ingestion and orchestrator scripts, then gradually add complexity.  
4. **Test Continually**  
   - Validate every phase with unit tests, integration tests, and real daily usage.

Following this **Four-Phase Workflow** will help ensure a robust, flexible system that evolves as your investment strategies grow.


## **Folder Structure & Modules**

Below is a proposed structure, reflecting your current layout and where new components will fit:

MISSIONCONTROL/
│
├─ docs/
│   └─ projectPlan.md               # High-level documentation & plan
│
├─ geckoAPI/                        # CoinGecko data ingestion
│   ├─ __init__.py
│   ├─ assetsRoster.py              # Lists & manages assets of interest
│   ├─ getAssetsCandleData.py       # Fetch & store candlestick data
│   ├─ getAssetsData.py             # Fetch & store basic price info
│   └─ utils.py                     # Shared utilities (API calls, cleaning)
│
├─ macro/                           # Macro data & modeling
│   ├─ __init__.py
│   ├─ fredData/
│   │   └─ getFredData.py           # Scripts to fetch FRED data (M2, rates, etc.)
│   └─ macro_overview.py            # Potential Streamlit page or macros summary
│
├─ micro/
│   ├─ __init__.py
│   ├─ assetData/                   # Folder for on-chain & micro fundamentals
│   └─ roroClassifier.py            # Classification logic (momentum, variance, etc.)
│
├─ pages/
│   ├─ chart_descriptions.txt       # Text references for chart tooltips
│   ├─ macro_overview.py            # Possibly a dedicated Streamlit page
│   ├─ regressions.py               # For any regression-based analytics
│   └─ ...                          # Additional pages as needed
│
├─ portfolios/
│   └─ (Allocation logs, or modules to handle portfolio logic & partial sells)
│
├─ venv/                            # Virtual environment
├─ .env                             # Environment variables (API keys, etc.)
├─ google_client.json               # Credentials for Sheets API
├─ __init__.py
└─ main.py                          # Primary orchestrator for daily runs

