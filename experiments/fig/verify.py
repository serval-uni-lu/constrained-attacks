import itertools
import pandas as pd


N_SEED = 5

DATASETS = ["ctu_13_neris"]

ATTACKS = {
    # "no_attack": "Clean",
    "pgdl2": "CPGD",
    "apgd": "CAPGD",
    "moeva": "MOEVA",
    "caa3": "CAA",
}.keys()

MODEL_ARCH = [
    "vime",
    "torchrln",
    "tabtransformer",
    # "saint",
    "tabnet",
    "stg",
]
MODEL_TRAINING = ["madry", "default"]
CONSTRAINED = [False, True]

def run():
    missing = 0
    df = pd.read_csv("./data_tmp.csv")
    df = df[df["scenario"] == "AB"]
    for (
        dataset,
        attack,
        model_arch,
        model_training,
        is_constrained,
    ) in itertools.product(
        DATASETS, ATTACKS, MODEL_ARCH, MODEL_TRAINING, CONSTRAINED
    ):
        df_l = df[
            (df["dataset"] == dataset)
            & (df["attack"] == attack)
            & (df["target_model_arch"] == model_arch)
            & (df["target_model_training"] == model_training)
            & (df["is_constrained"] == is_constrained)
        ]
        if (df_l.shape[0] != N_SEED) or df_l["robust_acc"].isna().any():
            
            if attack == "no_attack":
                print("No attack")
            else:
                print(
                    dataset,
                    attack,
                    model_arch,
                    model_training,
                    is_constrained,
                    df_l.shape[0],
                )
                # print(df_l.shape)
                # print("As some experiments are run together, please delete all experiments.")
                # print(dataset, model_arch, model_training, is_constrained)
                # print(df_l["seed"].values)
                missing = missing + 1
    print(missing)                


if __name__ == "__main__":
    run()
