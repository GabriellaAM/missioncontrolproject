from feast import Entity, ValueType

"""
Entity definitions for the feature store.

This module defines the core entities used in the feature store, which represent
the key objects that features are associated with.

Currently defines:
- asset_id: Entity representing a cryptocurrency asset
- macro_asset_id: Entity representing a macro/economic asset
"""

asset_id = Entity(
    name="asset_id",
    value_type=ValueType.STRING,
    description="Cryptocurrency asset identifier (e.g., bitcoin, ethereum)"
)

macro_asset_id = Entity(
    name="macro_asset_id",
    value_type=ValueType.STRING,
    description="Macro/economic asset identifier (e.g., vix, treasury10Y, fedFundsRate)"
)