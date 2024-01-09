import json
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
    if len(out.keys()) == 1:
        print(xp.get_name())
    return out


def download_data(scenarios: Dict[str, List[str]], path: str):
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

    out = {
        "download_time": generate_time_name(),
        "experiments": out,
    }
    print(len(out["experiments"]))
    with open(path, "w") as f:
        json.dump(out, f)


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
}


def run() -> None:
    path = f"./data/xp_results/data_{generate_time_name()}.json"
    download_data(TO_GET, path)


if __name__ == "__main__":
    run()
