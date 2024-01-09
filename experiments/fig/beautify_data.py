ordered_model_names = {
    "tabtransformer": "TabTransformer",
    "torchrln": "RLN",
    "vime": "VIME",
    "stg": "STG",
    "tabnet": "TabNet",
    "saint": "SAINT",
    "deepfm": "DeepFM",
}

ordered_dataset_names = {
    "url": "URL",
    "lcld_v2_iid": "LCLD",
    "ctu_13_neris": "CTU",
    "wids": "WIDS",
}

ordered_attack_names = {
    "no_attack": "Clean",
    "pgdl2": "CPGD",
    "apgd": "CAPGD",
    "moeva": "MOEVA",
    "caa3": "CAA",
}

ordered_model_training_names = {
    "default": "Standard",
    "madry": "Robust",
    "subset": "Subset",
    "dist": "Distribution",
    "Unknown": "ERROR",
}

order_scenario = {
    "A": "A",
    "AB": "AB",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
}


column_names = {}

beautify = {"col_data": {}}

data_order = {
    "dataset": ordered_dataset_names,
    "source_model_arch": ordered_model_names,
    "target_model_arch": ordered_model_names,
    "target_model_training": ordered_model_training_names,
    "source_model_training": ordered_model_training_names,
    "attack": ordered_attack_names,
    "scenario": order_scenario,
}

column_names = {
    "dataset": "Dataset",
    "target_model_arch": "Model",
    "attack": "Attack",
    "target_model_training": "Training",
    "source_model_training": "Training",
    "source_model_arch": "Model",
    "scenario": "Scenario",
    # "robust_acc": "Accuracy",
}


def beautify_col_name(col):
    if col in column_names:
        return column_names[col]
    if col is None:
        return None
    return col
