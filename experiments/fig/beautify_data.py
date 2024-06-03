import pandas as pd

ordered_model_names = {
    "tabtransformer": "TabTr.",
    "torchrln": "RLN",
    "vime": "VIME",
    "stg": "STG",
    "tabnet": "TabNet",
    "saint": "SAINT",
    "deepfm": "DeepFM",
    "Whitebox": "Whitebox",
    "Transfer": "Transfer",
    "Clean": "Clean",
}

ordered_dataset_names = {
    "url": "URL",
    "lcld_v2_iid": "LCLD",
    "ctu_13_neris": "CTU",
    "wids": "WIDS",
    "malware": "Malware",
}

# ordered_attack_names = {
#     "no_attack": "Clean",
#     "pgdl2org": "CPGD-IJCAI",
#     "pgdl2": "CPGD",
#     "pgdl2nrsnae": "PGD-NRSNAE",
#     "pgdl2rsae": "PGD-RSAE",
#     "apgd": "CAPGD",
#     "apgd2": "CAPGD2",
#     "moeva": "MOEVA",
#     "caa3": "CAA",
# }

ordered_attack_names = {
    "no_attack": "Clean",
    "lowprofool": "LowProFool",
    "bfs": "BFS",
    "pgdl2ijcai": "CPGD",
    "pgdl2org": "CPGD+R01",
    "pgdl2": "CPGD+R10",
    "pgdl2nrsnae": "CPGD+R00",
    "pgdl2rsae": "CPGD+R11",
    # "apgd": "CAPGD",
    "apgd2-nrep": "NREP2",
    "apgd2-nini": "NINI2",
    "apgd2-nran": "NRAN2",
    "apgd2-nbes": "NBES2",
    "apgd2-nada": "NADA2",
    "apgd2": "CAPGD2",
    "apgd3-nrep": "NREP",
    "apgd3-nini": "NINI",
    "apgd3-nran": "NRAN",
    "apgd3-nada": "NADA",
    "apgd3": "CAPGD",
    "ucs": "BF",
    "moeva": "MOEVA",
    "caa4": "CAA4",
    "caa5": "CAA",
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
    "budget": "A_BUDGET",
    "ablation": "Ablation",
    "default": "default",
    "capgd_ablation": "capgd_ablation",
    "caa_eps": "caa_eps",
    "caa_iter_search": "caa_iter_search",
    "caa_iter_gradient": "caa_iter_gradient",
    "caa_transferability": "caa_transferability",
}

ordered_is_constrained = {
    True: "Yes",
    False: "No",
}
column_names = {}

ordered_eps = {
    0.25: 0.25,
    0.5: 0.5,
    1.0: 1.0,
    5.0: 5.0,
    50.0: 50.0,
    100.0: 100.0,
    200.0: 200.0,
}

ordered_steps = {
    -1: 1,
    0: 0,
    10: 10,
    20: 20,
    50: 50,
    100: 100,
    200: 200,
    1000: 1000,
}

ordered_budget = {
    1: "1",
    2: "2",
    3: "3",
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
    "n_gen": ordered_steps,
    # "budget": ordered_budget,
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
    "budget": "Budget",
}


def beautify_col_name(col: str) -> str:
    if col in column_names:
        return column_names[col]
    if col is None:
        return None
    return col


def auto_beautify_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in data_order:
            df[col] = df[col].map(data_order[col])
    return df
