ordered_model_names = {
    "tabtransformer": "TabTr.",
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
    "malware": "Malware",
}

ordered_attack_names = {
    "no_attack": "Clean",
    "pgdl2": "CPGD",
    "apgd": "CAPGD",
    "moeva": "MOEVA",
    "caa3": "CAA",
}

ordered_model_training_names = {
    "default": "Std.",
    "madry": "Rob.",
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
    "A_STEPS": "A_STEPS",
    "B_STEPS": "B_STEPS",
    "AB_EPS": "AB_EPS",
}

ordered_is_constrained = {
    True: "Yes",
    False: "No",
}
column_names = {}

ordered_eps = {
    0.25: "0.25",
    0.5: "0.5",
    1.0: "1.0",
    5.0: "5.0",
}

ordered_steps = {
    -1: "1",
    0: "0",
    10: "10",
    20: "20",
    50: "50",
    100: "100",
    200: "200",
}

data_order = {
    "dataset": ordered_dataset_names,
    "source_model_arch": ordered_model_names,
    "target_model_arch": ordered_model_names,
    "target_model_training": ordered_model_training_names,
    "source_model_training": ordered_model_training_names,
    "attack": ordered_attack_names,
    "scenario": order_scenario,
    "is_constrained": ordered_is_constrained,
    "eps": ordered_eps,
    "n_iter": ordered_steps,
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
    "model_arch": "Model",
    "is_constrained": "Constrained",
    "scenario_constrained": "Scenario",
    "n_iter": "Steps",
}


def beautify_col_name(col: str) -> str:
    if col in column_names:
        return column_names[col]
    if col is None:
        return None
    return col
