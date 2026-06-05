from __future__ import annotations

from .types import ModelCatalog, ModelFamilySpec, ModelIndexRule


def build_catalog(specs: list[ModelFamilySpec]) -> ModelCatalog:
    families: dict[str, ModelFamilySpec] = {}
    alias_to_family: dict[str, str] = {}
    model_index_rules: dict[str, ModelIndexRule] = {}
    flowmatch_families: set[str] = set()

    for spec in specs:
        fam = str(spec.family).strip().lower()
        if not fam:
            raise ValueError("ModelFamilySpec.family must be non-empty")
        if fam in families:
            raise ValueError(f"Duplicate family: {fam}")

        # normalize aliases
        aliases = {str(a).strip().lower() for a in (spec.aliases or set()) if str(a).strip()}
        # canonical 也作为 alias（便于直接命中）
        aliases.add(fam)

        families[fam] = ModelFamilySpec(
            family=fam,
            aliases=aliases,
            default_text2img=spec.default_text2img,
            default_img2img=spec.default_img2img,
            model_index_rules=spec.model_index_rules or {},
            is_flowmatch=bool(spec.is_flowmatch),
        )

    # build alias_to_family（检测冲突）
    for fam, spec in families.items():
        if spec.is_flowmatch:
            flowmatch_families.add(fam)
        for a in spec.aliases:
            prev = alias_to_family.get(a)
            if prev is not None and prev != fam:
                raise ValueError(f"Alias conflict: '{a}' belongs to both '{prev}' and '{fam}'")
            alias_to_family[a] = fam

    # build model_index_rules（检测冲突/一致性）
    for fam, spec in families.items():
        for class_name, rule in (spec.model_index_rules or {}).items():
            if not class_name:
                raise ValueError(f"Empty model_index class_name in family={fam}")
            if rule.family != fam:
                raise ValueError(
                    f"ModelIndexRule.family mismatch for class_name={class_name}: rule.family={rule.family} spec.family={fam}"
                )
            prev = model_index_rules.get(class_name)
            if prev is not None:
                # 如果重复定义，必须完全一致
                if prev != rule:
                    raise ValueError(f"Duplicate model_index rule for '{class_name}' with different content")
            model_index_rules[class_name] = rule

    return ModelCatalog(
        families=families,
        alias_to_family=alias_to_family,
        model_index_rules=model_index_rules,
        flowmatch_families=flowmatch_families,
    )

