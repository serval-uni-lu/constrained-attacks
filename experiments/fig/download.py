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
    "AB": ["scenario-ab-url-v3"],
    "C": [],
    "D": [],
    "E": [],
}


def run() -> None:
    path = f"./data/xp_results/data_{generate_time_name()}.json"
    download_data(TO_GET, path)


if __name__ == "__main__":
    run()
