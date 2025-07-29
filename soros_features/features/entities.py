from feast import Entity, ValueType

# Crypto asset entity
asset_id = Entity(
    name="asset_id",
    value_type=ValueType.STRING,
    description="Cryptocurrency asset identifier (e.g., bitcoin, ethereum)"
)