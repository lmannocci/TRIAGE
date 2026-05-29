feature_alias_map = {
    "mpt": "mean_platelet_volume",
    "rbc count": "red_blood_cells",
    "wbc count": "leukocytes",

    # No direct equivalents → best approximations / proxies
    "blood smear examination": "rdw",

    # Infection-related (mapped to inflammation marker or viral presence proxy)
    "history of infections": "c_reactive_protein_mg_dl",
    "infection history": "c_reactive_protein_mg_dl",
    "infection status": "c_reactive_protein_mg_dl",
    "infections": "c_reactive_protein_mg_dl",
    "medical history": "c_reactive_protein_mg_dl",

    # Respiratory viruses → mapped to one representative virus feature
    "history of respiratory viruses": "respiratory_syncytial_virus",
    "presence of respiratory viral infections": "respiratory_syncytial_virus",

    # Kidney
    "kidney function": "creatinine",
}