import torch
import numpy as np
from torch import nn


class EnsembleNN(nn.Module):
    def __init__(self, list_models) -> None:
        super().__init__()
        self.list_models = list_models

    def forward(self, x):
        out = [model(x) for model in self.list_models]
        return torch.mean(torch.stack(out), axis=0)

    def parameters(self):
        return self.list_models[0].parameters()


class Ensemble:
    def __init__(self, list_models) -> None:

        self.list_models = list_models
        self.objective = list_models[0].objective
        self.wrapper_model = EnsembleNN([e.wrapper_model for e in list_models])
        self.device = list_models[0].device

    def predict_proba(self, X):

        predictions = np.array(
            [model.predict_proba(X) for model in self.list_models]
        )

        if isinstance(predictions[0], torch.Tensor):
            return torch.mean(torch.stack(predictions), axis=0)

        if isinstance(predictions[0], np.ndarray):
            return np.mean(predictions, axis=0)

    def predict(self, X):

        out = self.predict_proba(X)
        if isinstance(out, torch.Tensor):
            return torch.argmax(out, axis=1)
        if isinstance(out, np.ndarray):
            return np.argmax(out, axis=1)
        return np.argmax(self.predict_proba(X), axis=1)

    def get_logits(self, x):
        logits = [model.get_logits(x) for model in self.list_models]
        return torch.mean(torch.stack(logits), axis=0)

    def parameters(self):
        return self.list_models[0].parameters()

    @property
    def training(self):
        return self.wrapper_model.training
    
    def eval(self):
        self.wrapper_model.eval()
        
    def train(self):
        pass