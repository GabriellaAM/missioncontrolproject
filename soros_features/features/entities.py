from feast import Entity, ValueType

"""
Entity definitions for the feature store.

This module defines the core entities used in the feature store, which represent
the key objects that features are associated with.

Currently defines:
- asset_id: Entity representing a cryptocurrency asset
"""

asset_id = Entity(
    name="asset_id",
    value_type=ValueType.STRING,
    description="Cryptocurrency asset identifier (e.g., bitcoin, ethereum)"
)