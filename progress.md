# Progress for paper

## Models

### Documentation
A general recommendation is to use `objective="classification"` and `num_classes=2` for the binary case.
All `model.wrapper_model` will produce a multi-logit output, which is required by the attacks.
A new feature of TabScaler is that one can retrieve the data it has been fit on, to create a new TabScaler, with the same data but towards a different representation.
For instance, the "default" scaler uses one hot encoding but most models use the non-encoded representation as inputs.
Hence in the constructor of such models, you will find
```python
if scaler is not None:
    self.scaler = TabScaler(num_scaler="min_max", one_hot_encode=False)
    self.scaler.fit_scaler_data(scaler.get_scaler_data())
```

To compute the gradient from an input `x`, one must avoid non-differentiable operations (which are common in the default implementation of TabSurvey). 
Hence some adaptations were required.
When "straight forward" we adapted the `self.model` attribute and the fit function (e.g. TorchRLN)
Otherwise we kept the code for the training process and use a wrapper (defined above the model object in the code). We store the wrapper in `self.wrapper_model`. (e.g. DeepFM). 
The code of the wrapper is inspired by the `predict_helper` function of the model.
In both cases, the first layer of the model is the scaler, if a scaler was provided.

At the level of the TabSurvey model, we create a get_logits function that takes as input a tensor X in the original representation (not scaler) and returns the logits of the model.
Because this `get_logits` function uses the wrapper_model and because this function was inspired by the `predict_helper` function, the `predict_helper` function is updated and uses `get_logits` to avoid code duplication (and countless sources of errors).

### Implementation progress and caveat

- [x] TorchRln
- [x] DeepFM 

This model does not support "classification" as a value for the `objective` parameter.
However, if passed this parameter and num_classes=2, it will adapt automatically.
The default model has a single output. 
Because the attacks are designed for at least 2 logits in output, an extra layer is added to the `wrapper_model`.
Please confirm it is ok to add a [1-y, y] layer after a sigmoid layer y.

- [x] TabTransformer

The recommended way to train a binary classifier is to pass the parameters `objective="classification"` and `num_classes=2`.
In some cases, other combinations will work but we do not recommend them.
For TabTransformer, using `objective="binary"` and `num_classes=2` produce bad performance.

- [x] SAINT 

Very slow to attack. No GPU support yet.

- [x] VIME 

## Constrained Attacks

### Documentation

The attacks are defined in `mlc.attacks.cta`.  
Check `run.*` for examples.

### Constrained Whitebox Implementation progress and caveat

Untested with GPU.

- [x] Moeva2
- [x] CAPGD
- [x] CPGD
- [x] CPGD Adaptive step

This is the version proposed in the IJCAI paper where every X iteration, the `alpha` step is reduced.
Use `CPGDL2(..., adaptive_step=True)` to use.
We could add hyperparameters to control the decrease in the step, so far we use the exact implementation of IJCAI paper (hardcoded).

- [ ] CFAB

We adapt FAB to tabular data, but we do not attempt to satisfy the constraints (yet).

- [ ] Autoattack

Not implemented yet.
For autoattack, we simply need to chain the attacks. 
To determine if the previous attack succeeded and retrieve the previous attack successes, use the `ObjectiveCalculator`. 

### Transferable Attacks Implementation progress and caveat
- [x] C-MAPGD(surrogate based, Momentum APGD)
- [x] C-LGV (surrogate based, Large Geometric Vicinity)
- [x] C-SGM (surrogate based, Skip Gradient Method)
- [] C-BASES (query based, Surrogate Ensemble Search)
- [] C-UAP (query based, Universal Adversarial Perturbation)
- [] L-MoEva (query based, using only label and not probability)

## Scripts

### Implementation progress 

- [ ] Model tuning
- [ ] Model attacking

## Datasets

### Integration progress

- [x] Botnet
- [x] LCLD
- [x] Malware
- [x] URL
- [x] Wids
