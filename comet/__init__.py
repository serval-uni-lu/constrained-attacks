import json
import os
from pathlib import Path
import uuid
from mlc.logging.comet_config import (
    COMET_APIKEY,
    COMET_PROJECT,
    COMET_WORKSPACE,
)
from comet_ml import Experiment, OfflineExperiment
import time
import uuid

import numpy as np


class XP(object):
    def __init__(
        self, args, project_name=COMET_PROJECT, workspace=COMET_WORKSPACE
    ):
        if project_name == "" or project_name == None:
            self.xp = None
        else:
            timestamp = time.time()
            args["timestamp"] = timestamp
            experiment_name = "mlc_{}_{}_{}_{}".format(
                args.get("model_name", ""),
                args.get("dataset_name", ""),
                args.get("attack_name", ""),
                timestamp,
            )
            experiment = Experiment(
                api_key=COMET_APIKEY,
                project_name=project_name,
                workspace=workspace,
                auto_param_logging=False,
                auto_metric_logging=False,
                parse_args=False,
                display_summary=False,
                disabled=False,
            )

            # experiment = OfflineExperiment(
            #     project_name=project_name,
            #     workspace=workspace,
            #     auto_param_logging=False,
            #     auto_metric_logging=False,
            #     parse_args=False,
            #     display_summary=False,
            #     disabled=False,
            # )

            experiment.set_name(experiment_name)
            experiment.log_parameters(args)

            self.xp = experiment

    def log_parameters(self, *args, **kwargs):
        if self.xp is not None:
            self.xp.log_parameters(*args, **kwargs)
        else:
            print("logging parameters", *args)
            print("logging parameters", kwargs)

    def log_metrics(self, *args, **kwargs):
        if self.xp is not None:
            self.xp.log_metrics(*args, **kwargs)
        else:
            print("logging metrics", *args)
            print("logging metrics", kwargs)

    def log_metric(self, name, value, **kwargs):
        if self.xp is not None:
            self.xp.log_metric(name, value, **kwargs)
        else:
            print("logging metric", name, value)
            print("logging metric", kwargs)

    def end(self):
        if self.xp is not None:
            self.xp.end()

    def log_model(self, name, path):
        if self.xp is not None:
            self.xp.log_model(name, path, copy_to_tmp=False)
        else:
            print("Logging model", name, path)

    def log_asset(self, name, path):
        if self.xp is not None:
            self.xp.log_asset(name, path, copy_to_tmp=False)
        else:
            print("Logging asset", name, path)

    def get_name(self):
        if self.xp is not None:
            return self.xp.get_name() + "_"
        else:
            return ""

    def flush(self):
        if self.xp is not None:
            return self.xp.flush()
        else:
            return None


class LocalXp:
    def __init__(
        self, args, project_name=COMET_PROJECT, workspace=COMET_WORKSPACE
    ):
        # Add a timestamp
        timestamp = time.time()
        args["timestamp"] = timestamp
        args["uuid"] = str(uuid.uuid4())[:8]

        self.experiment_name = "mlc_{}_{}_{}_{}_{}".format(
            args.get("model_name", ""),
            args.get("dataset_name", ""),
            args.get("attack_name", ""),
            timestamp,
            args.get("uuid", ""),
        )
        self.parameters = {}
        self.metrics = {}

        self.log_parameters(**args)

        self.workspace = workspace
        self.project_name = project_name
        self.path = os.path.join(
            os.getenv("XP_ROOT", "./data/xp"),
            self.workspace,
            self.project_name,
        )

    def log_parameters(self, *args, **kwargs):
        self.parameters.update(kwargs)

    def log_metrics(self, *args, **kwargs):
        self.metrics.update(kwargs)

    def get_name(self):
        return self.experiment_name + "_"

    def log_metric(self, name, value, **kwargs):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        self.log_metrics(**{name: value})

    def log_model(self, name, path):
        pass

    def log_asset(self, name, path):
        pass

    def save_json(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)
        with open(
            os.path.join(self.path, self.experiment_name + ".json"), "w"
        ) as f:
            json.dump(
                {
                    "parameters": self.parameters,
                    "metrics": self.metrics,
                },
                f,
                indent=4,
            )

    def end(self):
        self.save_json()

    def flush(self):
        self.save_json()
