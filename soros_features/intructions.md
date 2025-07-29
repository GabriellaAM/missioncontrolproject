How Soros Features Works 🏗️

  Core Components

  1. feature_store.yaml - Configuration
  - Defines project name: soros_features
  - Sets offline store: duckdb (for fast parquet queries)
  - Registry: registry.db (SQLite metadata storage)

  2. features/entities.py - Entity Definitions
  - Defines asset_id entity (bitcoin, ethereum, etc.)
  - Entities are the "join keys" for features

  3. features/crypto_market_features.py - Feature Views
  - Maps parquet files to feature definitions
  - Specifies which columns are features
  - Points to data source: data_parquet/crypto_data/*/data.parquet

  4. feature_repo.py - Registry
  - Imports all entities and feature views
  - Makes them discoverable by Feast CLI

  5. registry.db - Metadata Storage
  - Stores feature schemas, data source locations
  - Created by feast apply

  ---
  How to Register New Features

  Workflow:

  1. Calculate features → Save as parquet in data_parquet/ structure
  2. Define feature view → Create new .py file in features/
  3. Register with Feast → Run feast apply

  Example: Adding RSI features

  1. Calculate RSI values → Save to data_parquet/technical/rsi/*.parquet
  2. Create features/rsi_features.py → Define RSI feature view
  3. Import in feature_repo.py → Add to __all__ list
  4. feast apply → Register with Feast

