import numpy as np
import torch
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import (
    AndConstraint,
    BaseRelationConstraint,
    EqualConstraint,
    Feature,
)
from mlc.datasets.dataset_factory import load_dataset
from mlc.metrics.compute import compute_metric
from mlc.metrics.metric_factory import create_metric
from mlc.models.model_factory import load_model
from mlc.transformers.tab_scaler import TabScaler

from constrained_attacks.attacks.cta.capgd import CAPGD
from constrained_attacks.attacks.cta.cpgdl2 import CPGDL2
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)
from constrained_attacks.utils import fix_immutable, fix_types


def run(dataset_name="url", model_name="deepfm") -> None:

    # Load data
    dataset = load_dataset(dataset_name)
    x, y = dataset.get_x_y()
    splits = dataset.get_splits()
    x_test = x.iloc[splits["train"]]
    y_test = y[splits["train"]]
    metadata = dataset.get_metadata(only_x=True)

    # Scaler
    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )

    # Load model
    model_class = load_model(model_name)
    save_path = f"../models/constrained/{dataset_name}_{model_name}.model"
    model = model_class.load_class(
        save_path, x_metadata=metadata, scaler=scaler
    )

    # Verify model
    metric = create_metric("auc")
    auc = compute_metric(
        model,
        metric,
        x_test.values,
        y_test,
    )
    print("Test AUC: ", auc)
    metric = create_metric("accuracy")
    acc = compute_metric(model, metric, x_test.values, y_test)
    print("Test acc: ", acc)

    # Constraints

    constraints = dataset.get_constraints()
    constraints_executor = ConstraintsExecutor(
        AndConstraint(constraints.relation_constraints),
        PytorchBackend(),
        feature_names=constraints.feature_names,
    )
    constraints_val = constraints_executor.execute(torch.Tensor(x_test.values))
    constraints_ok = (constraints_val <= 0.01).float().mean()
    print(f"Constraints ok: {constraints_ok*100:.2f}%")

    print("--------- End of verification ---------")

    # Attack

    model_attack = model.wrapper_model.cpu()

    constraints_attack = dataset.get_constraints()
    constraints_attack.relation_constraints = None
    EPS = 0.3
    attack = CAPGD(
        constraints_attack,
        scaler,
        model_attack,
        model.predict_proba,
        verbose=True,
        steps=10,
        n_restarts=1,
        eps=EPS,
        norm="L2",
        eps_margin=0.01,
        loss="ce",
        fix_equality_constraints_iter=False,
        fix_equality_constraints_end=False
    )
    """
    attack = CPGDL2(
        constraints_attack,
        scaler,
        model_attack,
        model.predict_proba,
        eps=EPS,
        alpha=EPS / 3,
        steps=100,
        random_start=True,
        eps_for_division=1e-10,
        adaptive_eps=False,
    )
    """
    adv = attack(
        torch.Tensor(x_test.values),
        torch.tensor(y_test, dtype=torch.long),
    )
    # adv = scaler.inverse_transform(adv)
    # adv = fix_types(torch.Tensor(x_test.values), adv, metadata["type"])

    constraints_val = constraints_executor.execute(adv)
    constraints_ok = (constraints_val <= 0).float().mean()
    print(f"Constraints ok: {constraints_ok*100:.2f}%")

    model.device = "cpu"
    model.to_device()

    objective_calculator = ObjectiveCalculator(
        classifier=model.predict_proba,
        constraints=constraints,
        thresholds={
            "distance": EPS,
            "constraints": 0.01,
        },
        fun_distance_preprocess=scaler.transform,
    )
    print("Before fix")
    print(
        objective_calculator.get_success_rate(
            x_test.to_numpy().astype(np.float32),
            y_test,
            adv.unsqueeze(1).detach().numpy(),
        )
    )


if __name__ == "__main__":
    torch.set_warn_always(True)

    for e in ["deepfm", "tabtransformer", "torchrln", "saint", "vime"]:
        run(model_name=e)
    run()
