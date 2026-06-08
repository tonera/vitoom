from __future__ import annotations

from .types import ModelCatalog


def validate_catalog(cat: ModelCatalog) -> None:
    # families must be non-empty
    if not cat.families:
        raise ValueError("ModelCatalog has no families")

    # basic sanity: canonical families should map to themselves
    for fam in cat.families.keys():
        if cat.to_family(fam) != fam:
            raise ValueError(f"Catalog to_family is inconsistent for canonical family: {fam}")

    # ensure model_index_rules are reachable
    for class_name, rule in cat.model_index_rules.items():
        if not class_name.strip():
            raise ValueError("Empty model_index_rules key")
        if rule.family not in cat.families:
            raise ValueError(f"model_index_rules references unknown family: {rule.family}")

