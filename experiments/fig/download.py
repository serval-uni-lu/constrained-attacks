import itertools
import json
import os
from datetime import datetime
from typing import Dict, List

import comet_ml
from joblib import Parallel, delayed
from mlc.logging.comet_config import COMET_APIKEY, COMET_WORKSPACE
from tqdm import tqdm


def generate_time_name():
    now = datetime.now()
    time_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    return time_name


def get_xp_data(xp, scenario_name):
    out = {
        "scenario_name": scenario_name,
        **{e["name"]: e["valueCurrent"] for e in xp.get_parameters_summary()},
        **{e["name"]: e["valueCurrent"] for e in xp.get_metrics_summary()},
    }
    # if len(out.keys()) == 1:
    #     print(xp.get_name())
    return out


def get_xp_data_json(project_path, scenario_name):
    with open(project_path) as f:
        data = json.load(f)
    out = {
        "scenario_name": scenario_name,
        **data["parameters"],
        **data["metrics"],
    }
    return out


def download_data(scenarios: Dict[str, List[str]]):
    out = []
    for scenario_name, project_names in scenarios.items():
        for p in project_names:
            print(f"------- Project {p}")
            experiments = comet_ml.api.API(COMET_APIKEY).get(
                COMET_WORKSPACE, p
            )
            out.extend(
                Parallel(n_jobs=-1)(
                    delayed(get_xp_data)(experiments[i], scenario_name)
                    for i in tqdm(range(len(experiments)))
                )
            )

    print(len(out))
    return out


def file_import(scenarios: Dict[str, List[str]]):
    out = []
    for scenario_name, list_project_path in scenarios.items():
        for p in list_project_path:
            json_files = [
                pos_json
                for pos_json in os.listdir(p)
                if pos_json.endswith(".json")
            ]
            # print(json_files)
            out.extend(
                Parallel(n_jobs=-1)(
                    delayed(get_xp_data_json)(
                        os.path.join(p, relative_path), scenario_name
                    )
                    for relative_path in tqdm(json_files)
                )
            )
    # print(len(out))
    return out


TO_GET = {
    "AB": [
        "scenario-ab-url-v3",
        "scenario-ab-wids-v3",
        "scenario-ab-lcld-v2-iid-v3",
        "scenario-ab-ctu-13-neris-v4",
    ],
    "C": [
        "scenario-c-url-v4",
        "scenario-c-wids-v4",
        "scenario-c-lcld-v2-iid-v4",
        "scenario-c-ctu-13-neris-v4",
    ],
    "D": [
        "scenario-d-url-v4",
        "scenario-d-wids-v4",
        "scenario-d-lcld-v2-iid-v4",
        "scenario-d-ctu-13-neris-v4",
    ],
    "E": [
        "scenario-e-url-v4",
        "scenario-e-wids-v4",
        "scenario-e-lcld-v2-iid-v4",
        "scenario-e-ctu-13-neris-v4",
    ],
    "AB_EPS": [
        "scenario-ab-lcld-v2-iid-eps",
        "scenario-ab-url-eps",
        "scenario-ab-wids-eps",
    ],
}

TO_GET_JSON = {
    "A_STEPS": [
        "./data/xp/thibaultsmnt/scenario_A_wids_STEPS",
        "./data/xp/thibaultsmnt/scenario_A_lcld_v2_iid_STEPS",
        "./data/xp/thibaultsmnt/scenario_A_url_STEPS",
        "./data/xp/thibaultsmnt/scenario_A_ctu_13_neris_STEPS",
    ],
    "B_STEPS": [
        "./data/xp/thibaultsmnt/scenario_B_wids_STEPS",
        "./data/xp/thibaultsmnt/scenario_B_lcld_v2_iid_STEPS",
        "./data/xp/thibaultsmnt/scenario_B_url_STEPS",
        "./data/xp/thibaultsmnt/scenario_B_ctu_13_neris_STEPS",
    ],
    "AB_EPS": [
        "./data/xp/thibaultsmnt/scenario_AB_ctu_13_neris_EPS",
    ],
}


TO_GET_OLD = {
    "B_STEPS": [
        "scenario-b1v11",
        "scenario-b2v11",
    ],
    "A_STEPS": [
        "scenario-a1v18-steps-2",
        "scenario-a2v18-steps-2",
    ],
    "AB_EPS": [
        "scenario-a1v18-eps",
        "scenario-a2v18-eps",
        "scenario-b1v11-eps",
        "scenario-b2v11-eps",
    ],
}

TO_GET_CORRECTED = {
    "AB": [
        "./data/xp/constrained-robustbench/gradient_attacks_lcld_v2_iid",
        "./data/xp/constrained-robustbench/gradient_attacks_url",
        "./data/xp/constrained-robustbench/gradient_attacks_wids",
        "./data/xp/constrained-robustbench/gradient_attacks_ctu_13_neris",
    ],
}

TO_GET_LAST = {
    "AB": [
        f"./data/xp/currated/{e}"
        for e in [
            "all_wids",
            "caa4_ctu_13_neris",
            "caa4_lcld_v2_iid",
            "caa4_url",
            "gradient_attacks_ctu_13_neris",
            "gradient_attacks_lcld_v2_iid",
            "gradient_attacks_url",
            "gradient_caa4_malware",
            "moeva2_ctu_13_neris",
            "moeva2_malware",
            "moeva2_url",
            # "scenario_AB_corrected_lcld_v2_iid_v9",
            "sota_ctu_13_neris",
            "sota_lcld_v2_iid",
            "sota_malware",
            "sota_url",
            "sota_wids",
            "ucs",
        ]
    ],
    "C": [
        f"./data/xp/currated/{e}"
        for e in [
            "transferability_ctu_13_neris_v1",
            "transferability_lcld_v2_iid_v1",
            "transferability_malware_v1",
            "transferability_url_v1",
            "transferability_wids_v1",
        ]
    ],
    "D": [
        f"data/xp/constrained-robustbench/{e}"
        for e in [
            "scenario_D_lcld_v2_iid",
            "scenario_D_ctu_13_neris",
            "scenario_D_url",
            "scenario_D_wids",
            "scenario_D_malware",
        ]
    ],
    "E": [
        f"data/xp/constrained-robustbench/{e}"
        for e in [
            "scenario_E_lcld_v2_iid",
            "scenario_E_ctu_13_neris",
            "scenario_E_url",
            "scenario_E_wids",
            "scenario_E_malware",
        ]
    ],
    "budget": [
        f"data/xp/constrained-robustbench/{e}"
        for e in [
            "eps_lcld_v2_iid",
            "eps_ctu_13_neris",
            "eps_url",
            "eps_wids",
            "eps_malware",
        ]
    ],
    "ablation": [
        f"data/xp/constrained-robustbench/{e}"
        for e in [
            "capgd_ablation_lcld_v2_iid",
            "capgd_ablation_url",
            "capgd_ablation_wids",
            "capgd_ablation_ctu_13_neris",
            "capgd_ablation_malware",
        ]
    ],
}


TO_GET_NEURIPS = {
    "default": [
        f"./data/xp/neurips24/{folder}_{ds}"
        for ds, folder in itertools.product(
            [
                "ctu_13_neris",
                "lcld_v2_iid",
                "url",
                "wids",
            ],
            ["sota", "moeva2", "apgd3_v3", "caa5_v3", "caa5_defense_v3"],
        )
    ],
    "capgd_ablation": [
        f"./data/xp/neurips24/{folder}_{ds}"
        for ds, folder in itertools.product(
            [
                "ctu_13_neris",
                "lcld_v2_iid",
                "url",
                "wids",
            ],
            ["apgd3_ablation_v3"],
        )
    ],
    "caa_eps": [
        f"./data/xp/neurips24/{folder}_{ds}"
        for ds, folder in itertools.product(
            [
                "ctu_13_neris",
                "lcld_v2_iid",
                "url",
                "wids",
            ],
            ["caa5_eps_v3"],
        )
    ],
    "caa_iter_search": [
        f"./data/xp/neurips24/{folder}_{ds}"
        for ds, folder in itertools.product(
            [
                "ctu_13_neris",
                "lcld_v2_iid",
                "url",
                "wids",
            ],
            ["caa5_iter_search_v3"],
        )
    ],
    "caa_iter_gradient": [
        f"./data/xp/neurips24/{folder}_{ds}"
        for ds, folder in itertools.product(
            [
                "ctu_13_neris",
                "lcld_v2_iid",
                "url",
                "wids",
            ],
            ["caa5_iter_gradient_v3"],
        )
    ],
    "caa_transferability": [
        f"./data/xp/neurips24/{folder}_{ds}"
        for ds, folder in itertools.product(
            [
                "ctu_13_neris",
                "lcld_v2_iid",
                "url",
                "wids",
            ],
            ["caa5_transferability_v3"],
        )
    ],
}


def run() -> None:
    path = f"./data/xp_results/data_{generate_time_name()}.json"
    out = []
    # out.extend(download_data(TO_GET_OLD))
    # out.extend(download_data(TO_GET))
    # out.extend(file_import(TO_GET_JSON))
    out.extend(file_import(TO_GET_NEURIPS))

    out = {
        "download_time": generate_time_name(),
        "experiments": out,
    }
    with open(path, "w") as f:
        json.dump(out, f)

    path_latest = "./data/xp_results/data_latest.json"
    with open(path_latest, "w") as f:
        json.dump(out, f)

    print("Path: ", path)


if __name__ == "__main__":
    run()
