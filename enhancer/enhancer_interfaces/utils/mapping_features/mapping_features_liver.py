feature_alias_map = {
    # Albumin / Protein related
    "a/g ratio": "albuminGlobulinRatio",

    # Bilirubin
    "bilirubin level": "totalBilirubin",
    "bilirubin levels": "totalBilirubin",
    "conjugated bilirubin": "directBilirubin",

    # Liver enzymes (specific ratios → components exist but ratio itself not directly represented)
    # We map to closest meaningful components ONLY when reasonable
    "alt/ast ratio": "alanineAminotransferase",
    "ast/alt ratio": "aspartateAminotransferase",
    "ast/altRatio": "aspartateAminotransferase",

    # General liver-related panels → mapped conservatively to most representative markers
    "liver enzymes": "alanineAminotransferase",
    "liver function tests": "totalBilirubin",
}